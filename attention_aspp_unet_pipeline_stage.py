# -*- coding: utf-8 -*-
"""
Attention-ASPP-UNet – unified train / predict / calibrate script
===============================================================

Stage-A → checkpoints/ckpt_main/best_*.pt
Stage-B → checkpoints/ckpt_finetune/best_*.pt
"""

import argparse, json, os, random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2, numpy as np, torch
import torch.nn as nn; import torch.nn.functional as F
from albumentations import (CLAHE, Compose, HorizontalFlip, MedianBlur,
    RandomBrightnessContrast, RandomGamma, Resize, ToFloat)
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.measure import label
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
try: import SimpleITK as sitk
except ImportError: sitk = None
import csv

# ----- globals -----
SEED, IMG_SIZE, WEIGHT_DECAY, GRAD_CLIP = 2025, 512, 5e-4, 1.0
EARLY_STOP_PATIENCE = 15
LOSS_TYPE, TV_ALPHA, TV_BETA = "combo", 0.7, 0.3

def set_seed(seed: int = SEED):
    import random, numpy as np, torch, os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # Albumentations
    A_set_seed = None
    try:
        from albumentations import set_seed as A_set_seed        
    except (ImportError, AttributeError):
        try:
            from albumentations.core.utils import set_seed as A_set_seed  
        except (ImportError, AttributeError):
            pass

    if A_set_seed is not None:
        A_set_seed(seed)
    else:
        print("albumentations.set_seed not exist")

def seed_worker(wid):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

# ----- model -----
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=k//2, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True))
    def forward(self,x): return self.block(x)

class ASPP(nn.Module):
    def __init__(self,in_c,out_c=256,rates=(6,12,18)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c,out_c,1,bias=False),
                          nn.BatchNorm2d(out_c),nn.ReLU(True)),
            *[nn.Sequential(nn.Conv2d(in_c,out_c,3,padding=r,dilation=r,bias=False),
                            nn.BatchNorm2d(out_c),nn.ReLU(True)) for r in rates]])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(in_c,out_c,1,bias=False),
                                  nn.BatchNorm2d(out_c),nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_c*5,out_c,1,bias=False),
                                     nn.BatchNorm2d(out_c),nn.ReLU(True),nn.Dropout(0.1))
    def forward(self,x):
        h,w = x.shape[2:]; feats=[b(x) for b in self.blocks]
        feats.append(F.interpolate(self.pool(x),(h,w),mode="bilinear",align_corners=False))
        return self.project(torch.cat(feats,1))

class AttentionGate(nn.Module):
    def __init__(self,Fg,Fl,Fint):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(Fg,Fint,1,bias=False),nn.BatchNorm2d(Fint))
        self.Wx = nn.Sequential(nn.Conv2d(Fl,Fint,1,bias=False),nn.BatchNorm2d(Fint))
        self.psi= nn.Sequential(nn.Conv2d(Fint,1,1,bias=False),nn.BatchNorm2d(1),nn.Sigmoid())
        self.relu=nn.ReLU(True)
    def forward(self,g,x): return x * self.psi(self.relu(self.Wg(g)+self.Wx(x)))

#Placeholder module, used to replace the actual Attention Gate when disabling the attention mechanism of certain layers
class DummyAttention(nn.Module):
    def forward(self,g,x): return x

class UpBlock(nn.Module):
    def __init__(self,in_c,out_c,use_att=True):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_c,out_c,2,2)
        self.att=AttentionGate(out_c,out_c,out_c//2) if use_att else DummyAttention()
        self.conv=nn.Sequential(ConvBNReLU(in_c,out_c),ConvBNReLU(out_c,out_c))
    def forward(self,g,x):
        g=self.up(g)
        if g.shape[-2:]!=x.shape[-2:]:
            g=F.interpolate(g,size=x.shape[-2:],mode="bilinear",align_corners=False)
        x=self.att(g,x)
        return self.conv(torch.cat([x,g],1))

class AttentionASPPUNet(nn.Module):
    def __init__(self,in_channels=1,num_classes=1,base_c=32):
        super().__init__()
        self.d1=nn.Sequential(ConvBNReLU(in_channels,base_c),ConvBNReLU(base_c,base_c))
        self.p1=nn.MaxPool2d(2)
        self.d2=nn.Sequential(ConvBNReLU(base_c,base_c*2),ConvBNReLU(base_c*2,base_c*2)); self.p2=nn.MaxPool2d(2)
        self.d3=nn.Sequential(ConvBNReLU(base_c*2,base_c*4),ConvBNReLU(base_c*4,base_c*4)); self.p3=nn.MaxPool2d(2)
        self.d4=nn.Sequential(ConvBNReLU(base_c*4,base_c*8),ConvBNReLU(base_c*8,base_c*8)); self.p4=nn.MaxPool2d(2)
        self.bridge=ASPP(base_c*8,base_c*16)
        self.u4=UpBlock(base_c*16,base_c*8); self.u3=UpBlock(base_c*8,base_c*4)
        self.u2=UpBlock(base_c*4,base_c*2); self.u1=UpBlock(base_c*2,base_c,use_att=False)
        self.out_conv=nn.Conv2d(base_c,num_classes,1)
    def forward(self,x):
        x1=self.d1(x); x2=self.d2(self.p1(x1)); x3=self.d3(self.p2(x2)); x4=self.d4(self.p3(x3))
        b=self.bridge(self.p4(x4))
        d4=self.u4(b,x4); d3=self.u3(d4,x3); d2=self.u2(d3,x2); d1=self.u1(d2,x1)
        return self.out_conv(d1)

# ----- dataset / aug -----
def SafeCLAHE(*a,**k): k.pop("always_apply",None); return CLAHE(*a,**k)
def SafeMedianBlur(*a,**k): k.pop("always_apply",None); return MedianBlur(*a,**k)
# ---------- helper: load pretrained with key-renaming ----------
def load_state_dict_compat(model: nn.Module, ckpt_path: str | Path):
    sd = torch.load(ckpt_path, map_location="cpu")
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace(".W_g.", ".Wg.").replace(".W_x.", ".Wx.")  
        new_sd[nk] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[i] loaded with {len(missing)} missing & {len(unexpected)} unexpected keys")

class FetalACDataset(Dataset):
    def __init__(self, imgs:List[Path], msks:List[Optional[Path]], train=True):
        self.imgs, self.msks, self.train = imgs, msks, train
        self.t = self._tfm()
    def _tfm(self):
        from albumentations import Affine, ElasticTransform
        train_t=[ Resize(IMG_SIZE,IMG_SIZE), HorizontalFlip(0.5),
            Affine(scale=(0.92,1.08),rotate=(-7,7),translate_percent=(0,0.02),shear=0,p=0.7),
                  RandomGamma(gamma_limit=(80, 120), p=0.3),
                  RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),

                  ElasticTransform(8,3,p=0.25), SafeCLAHE(1.0,(8,8)), SafeMedianBlur(3),
            ToFloat(max_value=255), ToTensorV2()]
        val_t=[Resize(IMG_SIZE,IMG_SIZE),SafeCLAHE(1.0,(8,8)),SafeMedianBlur(3),
               ToFloat(max_value=255),ToTensorV2()]
        return Compose(train_t if self.train else val_t)
    @staticmethod
    def _read(p:Path):
        if p.suffix.lower()=='.mha':
            arr=sitk.GetArrayFromImage(sitk.ReadImage(str(p)));
            if arr.ndim==3: arr=arr[arr.shape[0]//2]
        else: arr=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        return arr.astype(np.uint8)
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        img=self._read(self.imgs[i]); msk=np.zeros_like(img)
        if self.msks[i] is not None: msk=self._read(self.msks[i])
        s=self.t(image=img,mask=msk)
        return s["image"].float(),(s["mask"].unsqueeze(0).float()/255)

# ----- losses & metrics -----
class DiceLoss(nn.Module):
    def __init__(self,smooth=1.): super().__init__(); self.s=smooth
    def forward(self,l,t):
        p=torch.sigmoid(l); num=2*(p*t).sum((2,3))+self.s
        den=p.sum((2,3))+t.sum((2,3))+self.s
        return (1-num/den).mean()

class TverskyLoss(nn.Module):
    def __init__(self,a=0.7,b=0.3,s=1.): super().__init__(); self.a,self.b,self.s=a,b,s
    def forward(self,l,t):
        p=torch.sigmoid(l); tp=(p*t).sum((2,3)); fp=(p*(1-t)).sum((2,3)); fn=((1-p)*t).sum((2,3))
        tv=(tp+self.s)/(tp+self.a*fp+self.b*fn+self.s)
        return (1-tv).mean()

class ComboLoss(nn.Module):
    def __init__(self): super().__init__(); self.d=DiceLoss()
    def forward(self,l,t): return self.d(l,t)+F.binary_cross_entropy_with_logits(l,t)

def iou_score(l,t,thr=0.5):
    p=(torch.sigmoid(l)>thr).float(); inter=(p*t).sum((2,3))
    union=p.sum((2,3))+t.sum((2,3))-inter
    return (inter/(union+1e-7)).mean().item()
# ----- Edge / Surface / criterion builder -----
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32).view(1,1,3,3)
        ky=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx",kx); self.register_buffer("ky",ky)
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)                      # B×1×H×W
        dtype, device = p.dtype, p.device
        kx = self.kx.to(device=device, dtype=dtype)
        ky = self.ky.to(device=device, dtype=dtype)
        t  = targets.to(device=device, dtype=dtype)

        gx_p = F.conv2d(p, kx, padding=1)
        gy_p = F.conv2d(p, ky, padding=1)
        gx_t = F.conv2d(t, kx, padding=1)
        gy_t = F.conv2d(t, ky, padding=1)

        grad_p = torch.sqrt(gx_p**2 + gy_p**2 + 1e-8)
        grad_t = torch.sqrt(gx_t**2 + gy_t**2 + 1e-8)
        return F.l1_loss(grad_p, grad_t)

def build_criterion(args,base,edge):
    def crit(l,t):
        l,t=l.float(),t.float(); B=t.size(0)
        is_empty=(t.sum((2,3),keepdim=True)==0).float()
        w=torch.ones_like(t)
        if args.stage=="finetune": w=torch.where(is_empty==1,args.neg_bce_w,1.)
        bce=F.binary_cross_entropy_with_logits(l,t,weight=w)
        pos_idx=(is_empty.view(B)==0).nonzero(as_tuple=True)[0]
        dice=edge_loss=torch.tensor(0.,device=l.device)
        if len(pos_idx)>0:
            dice=base(l[pos_idx],t[pos_idx])
            if args.edge_w>0: edge_loss=edge(l[pos_idx],t[pos_idx])*args.edge_w
        return dice+bce+edge_loss
    return crit

# ---------- evaluate ----------
@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval(); d=i=0.
    for x,y in loader:
        x,y=x.to(device),y.to(device); l=model(x)
        d+=1-DiceLoss()(l,y).item(); i+=iou_score(l,y)
    return d/len(loader), i/len(loader)

# ---------- training routine ----------
def train(args):
    set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collect_pair(img_dir:Path,msk_dir:Optional[Path]):
        exts={'.png','.jpg','.jpeg','.tif','.bmp','.mha'}
        imgs,msks=[],[]
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in exts: continue
            imgs.append(p)
            q=msk_dir/p.name if msk_dir else None
            msks.append(q if (q and q.exists()) else None)
        return imgs,msks

    train_imgs,train_msks=collect_pair(Path(args.train_dir)/'images',Path(args.train_dir)/'masks')
    if args.neg_dir:
        neg_imgs,_=collect_pair(Path(args.neg_dir)/'images',None)
        train_imgs+=neg_imgs; train_msks+=[None]*len(neg_imgs)
    pos_cnt = sum(m is not None for m in train_msks)
    neg_cnt = len(train_msks) - pos_cnt
    print(f"Train samples: pos={pos_cnt}, neg={neg_cnt} (ratio={neg_cnt / (pos_cnt + 1e-6):.2f})")

    if args.val_dir:
        val_imgs,val_msks=collect_pair(Path(args.val_dir)/'images',Path(args.val_dir)/'masks')
        train_ds=FetalACDataset(train_imgs,train_msks,True)
        val_ds  =FetalACDataset(val_imgs,val_msks,False)
    else:
        pos_idx = [i for i, m in enumerate(train_msks) if m is not None]

        candidate_idx = pos_idx if len(pos_idx) > 0 else list(range(len(train_imgs)))

        rng = np.random.default_rng(args.seed)
        rng.shuffle(candidate_idx)
        val_len = max(1, int(0.1 * len(candidate_idx)))
        val_sel = set(candidate_idx[:val_len])

        train_idx = [i for i in range(len(train_imgs)) if i not in val_sel]
        val_idx = list(val_sel)

        train_imgs2 = [train_imgs[i] for i in train_idx]
        train_msks2 = [train_msks[i] for i in train_idx]
        val_imgs = [train_imgs[i] for i in val_idx]
        val_msks = [train_msks[i] for i in val_idx]

        train_ds = FetalACDataset(train_imgs2, train_msks2, train=True)
        val_ds = FetalACDataset(val_imgs, val_msks, train=False)

    g=torch.Generator().manual_seed(args.seed)
    train_ld=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,
                        num_workers=0,generator=g,worker_init_fn=seed_worker,drop_last=True )
    val_ld  =DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,
                        num_workers=0,generator=g,worker_init_fn=seed_worker)

    model=AttentionASPPUNet(base_c=args.base_c).to(device)
    if args.stage=="finetune":
        load_state_dict_compat(model, args.pretrained)
        print(f"loaded pretrained {args.pretrained}")

    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=WEIGHT_DECAY)
    tot_ep=args.epochs; warm=0 if args.stage=="finetune" else max(1,int(0.05*tot_ep))
    cos=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=tot_ep-warm)
    sch=cos if warm==0 else torch.optim.lr_scheduler.SequentialLR(
        opt,[torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, total_iters=warm),cos],[warm])

    base_loss=ComboLoss() if LOSS_TYPE=="combo" else TverskyLoss(TV_ALPHA,TV_BETA)
    edge_loss=EdgeLoss(); crit=build_criterion(args,base_loss,edge_loss)
    scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    out_dir=Path(args.output_dir)/("ckpt_main" if args.stage=="main" else "ckpt_finetune")
    out_dir.mkdir(parents=True,exist_ok=True)
    best,best_p=0.,out_dir/f"best_{datetime.now():%Y%m%d-%H%M%S}.pt"; noimp=0

    for ep in range(1,tot_ep+1):
        model.train(); run=0.
        for x,y in tqdm(train_ld,desc=f"Epoch {ep}/{tot_ep}"):
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                l=model(x); loss=crit(l,y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            scaler.step(opt); scaler.update(); run+=loss.item()
        sch.step()
        d,i=evaluate(model,val_ld,device)
        print(f"Dice {d:.4f} | IoU {i:.4f}")
        if d>best:
            best=d; noimp=0; torch.save(model.state_dict(),best_p)
            print(f"best saved → {best_p}")
        else:
            noimp+=1
            if noimp>=EARLY_STOP_PATIENCE: print("Early stop"); break

# ---------- predict helpers ----------
def predict_prob_tta(model,x):
    l=model(x); l_flip=model(torch.flip(x,[-1])); l_flip=torch.flip(l_flip,[-1])
    return torch.sigmoid((l+l_flip)/2)[0,0].cpu().numpy()

def refine_mask(m):
    if m.sum()==0: return m
    lab=label(m); cnt=np.bincount(lab.ravel()); cnt[0]=0
    min_area=max(20,int(0.0015*m.size)); keep=[i for i,c in enumerate(cnt) if c>=min_area]
    if not keep: return np.zeros_like(m)
    m=(np.isin(lab,keep)).astype(np.uint8); lab2=label(m); cmax=(np.bincount(lab2.ravel())[1:]).argmax()+1
    m=(lab2==cmax).astype(np.uint8)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k)
    return binary_fill_holes(m).astype(np.uint8)

def select_best(pred_stack,topk=5):
    areas=np.array([(p>0).sum() for p in pred_stack]); idx=areas.argsort()[::-1][:max(1,min(topk,len(areas)))]
    circ=lambda m: (lambda c,A,P:(0 if P==0 else 4*np.pi*A/(P*P)))(*max(cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea))
    return int(max(idx,key=lambda i:circ(pred_stack[i])))

import math
def _ellipse_circum(a, b):
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + 3*h / (10 + math.sqrt(4 - 3*h)))
def measure_ac_mm(mask01, spacing):
    """
    mask01: 0/1 numpy, spacing=(sx,sy) mm/px
    return: AC (mm)
    """
    import cv2, numpy as np
    cnts, _ = cv2.findContours(mask01.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    if len(c) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(c)
        a_mm, b_mm = MA/2*spacing[0], ma/2*spacing[1]
        return _ellipse_circum(a_mm, b_mm)
    return cv2.arcLength(c, True) * float(sum(spacing)/2)
# ---------- calibrate ----------
@torch.inference_mode()
def calibrate(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=AttentionASPPUNet(base_c=args.base_c).to(device)
    model.load_state_dict(torch.load(args.weights,map_location=device)); model.eval()
    val_dir=Path(args.val_dir); imgs=sorted((val_dir/'images').glob('*.png'))
    thrs=np.linspace(0.35,0.6,11); scores=[]
    for thr in tqdm(thrs, desc="Scanning thresholds", unit="thr"):
        ds=[]
        for p in imgs:
            sl=cv2.imread(str(p),0); sl_u8=cv2.normalize(sl,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            e=cv2.medianBlur(cv2.createCLAHE(clipLimit=1.0,tileGridSize=(8,8)).apply(sl_u8),3)
            x=Compose([Resize(IMG_SIZE,IMG_SIZE),ToFloat(max_value=255),ToTensorV2()])(image=e)['image'].unsqueeze(0).to(device)
            prob=predict_prob_tta(model,x); prob=cv2.resize(prob,sl.shape[::-1]); prob=cv2.GaussianBlur(prob,(5,5),0)
            m=(prob>thr).astype(np.uint8); gt=(cv2.imread(str(val_dir/'masks'/p.name),0)>127).astype(np.uint8)
            inter=(m&gt).sum(); ds.append(2*inter/(m.sum()+gt.sum()+1e-7))
        scores.append(np.mean(ds))
    best_thr=float(thrs[int(np.argmax(scores))]); out={'best_thr':best_thr}
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    json.dump(out,open(Path(args.output_dir)/'thr.json','w'),indent=2)
    print(f"Calibrated thr={best_thr:.3f}")

# ---------- predict ----------
@torch.inference_mode()
def predict(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Read the global threshold ----
    thr_cfg = Path('./checkpoints/thr.json'); THR = 0.48
    if thr_cfg.exists():
        try:
            THR = float(json.load(open(thr_cfg))['best_thr'])
            print(f"use thr {THR:.3f}")
        except Exception:
            pass

    # ---- read spacing_json ----
    spacing_map = {}
    if args.spacing_json:
        try:
            spacing_map = json.load(open(args.spacing_json, "r"))
            print(f"loaded spacing map for PNG ({len(spacing_map)})")
        except Exception as e:
            print(f"cannot load spacing_json: {e}")

    def _spacing_from_map(case_id: str) -> Tuple[float, float] | None:
        """Extract from JSON (sx, sy). It is compatible with both {'spacing':[sx,sy]} and [sx,sy]."""
        if case_id not in spacing_map:
            return None
        v = spacing_map[case_id]
        if isinstance(v, dict) and "spacing" in v:
            sx, sy = v["spacing"][:2]
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            sx, sy = v[:2]
        else:
            return None
        return float(sx), float(sy)

    # ---- model ----
    model = AttentionASPPUNet(base_c=args.base_c).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device)); model.eval()

    inp = Path(args.input_dir)
    od  = Path(args.out_dir); od.mkdir(exist_ok=True, parents=True)

    rows = []  # [(case_id, frame_idx, ac_mm)]

    for p in tqdm(sorted(inp.iterdir()), desc="Processing files", unit="file"):
        ext = p.suffix.lower()

        # ---------------- PNG / JPG ----------------
        if ext in {'.png', '.jpg', '.jpeg'}:
            sl = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            e = cv2.medianBlur(cv2.createCLAHE(1.0, (8, 8)).apply(sl_u8), 3)
            x = Compose([Resize(IMG_SIZE, IMG_SIZE), ToFloat(max_value=255), ToTensorV2()])(image=e)['image'].unsqueeze(0).to(device)

            prob = predict_prob_tta(model, x)
            prob = cv2.resize(prob, sl.shape[::-1])
            prob = cv2.GaussianBlur(prob, (5, 5), 0)
            mask = refine_mask((prob > THR).astype(np.uint8))

            # save mask
            cv2.imwrite(str(od / f"{p.stem}_mask.png"), mask * 255)

            # ---- calculate AC (spacing_json required)----
            stem = p.stem
            if "_s" in stem:
                case_id = stem.split("_s")[0]
                try:
                    frame_idx = int(stem.split("_s")[1])
                except Exception:
                    frame_idx = -1
            else:
                case_id = stem
                frame_idx = -1

            spacing_xy = _spacing_from_map(case_id)
            if spacing_xy is None:
                print(f"no spacing for {case_id}, skip AC")
            else:
                ac_mm = round(measure_ac_mm(mask, spacing_xy), 1)
                rows.append((case_id, frame_idx, ac_mm))
                print(f"{stem}: AC={ac_mm:.1f} mm")

        # ---------------- MHA ----------------
        elif ext == '.mha':
            if sitk is None:
                raise ImportError("SimpleITK needed for .mha")

            ref_img = sitk.ReadImage(str(p))
            vol = sitk.GetArrayFromImage(ref_img)  # (N,H,W)
            preds = []
            for sl in tqdm(vol, desc=f"{p.name} slices", unit="slice",
                           total=vol.shape[0], leave=False):
                sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                e = cv2.medianBlur(cv2.createCLAHE(1.0, (8, 8)).apply(sl_u8), 3)
                x = Compose([Resize(IMG_SIZE, IMG_SIZE), ToFloat(max_value=255), ToTensorV2()])(image=e)['image'].unsqueeze(0).to(device)
                prob = predict_prob_tta(model, x)
                prob = cv2.resize(prob, sl.shape[::-1])
                prob = cv2.GaussianBlur(prob, (5, 5), 0)
                preds.append(refine_mask((prob > THR).astype(np.uint8)))

            preds = np.stack(preds)                # (N,H,W)
            bf = select_best(preds, 5)
            bm = preds[bf]

            write_output_mha_and_json(bm, bf, p, od)

            # SimpleITK spacing: (sx, sy, sz) 单位 mm
            sx, sy = float(ref_img.GetSpacing()[0]), float(ref_img.GetSpacing()[1])
            ac_mm = round(measure_ac_mm(bm, (sx, sy)), 1)
            case = p.stem
            rows.append((case, int(bf), ac_mm))
            print(f"{case}: best_frame={bf}, AC={ac_mm:.1f} mm")

        else:
            continue

    # ---- CSV ----
    if rows:
        csv_path = od / "ac_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "frame_idx", "ac_mm"])
            w.writerows(rows)
        print(f"\n AC already saved → {csv_path}  (number {len(rows)})")


def convert_mask_2d_to_3d(mask,frame,nf):
    m=(mask>0).astype(np.uint8)*2; vol=np.zeros((nf,*m.shape),np.uint8)
    if 0<=frame<nf: vol[frame]=m
    return vol
def write_output_mha_and_json(mask,frame,ref,od):
    ref_img=sitk.ReadImage(str(ref)); nf=ref_img.GetSize()[2]
    arr=convert_mask_2d_to_3d(mask,frame,nf); out_img=sitk.GetImageFromArray(arr); out_img.CopyInformation(ref_img)
    case=ref.stem; cd=od/case; (cd/'images/fetal-abdomen-segmentation').mkdir(parents=True,exist_ok=True)
    sitk.WriteImage(out_img,str(cd/'images/fetal-abdomen-segmentation/output.mha'))
    json.dump(frame,open(cd/'fetal-abdomen-frame-number.json','w'),indent=2)
    print(f"{case} frame {frame}")

# ---------- CLI ----------
def get_args():
    p=argparse.ArgumentParser("A-ASPP-UNet unified"); sp=p.add_subparsers(dest="cmd",required=True)
    t=sp.add_parser("train"); t.add_argument("--stage",choices=['main','finetune'],default='main'); t.add_argument("--seed",type=int,default=SEED)
    t.add_argument("--train_dir",required=True); t.add_argument("--neg_dir"); t.add_argument("--val_dir")
    t.add_argument("--output_dir",default="./checkpoints"); t.add_argument("--pretrained")
    t.add_argument("--epochs",type=int,default=120); t.add_argument("--batch_size",type=int,default=8); t.add_argument("--lr",type=float,default=3e-4)
    t.add_argument("--base_c",type=int,default=48); t.add_argument("--edge_w",type=float,default=0.05); t.add_argument("--neg_bce_w",type=float,default=0.05)
    pr=sp.add_parser("predict"); pr.add_argument("--weights",required=True); pr.add_argument("--input_dir",required=True)
    pr.add_argument("--out_dir",default="./preds");  pr.add_argument("--spacing_json",required=True);pr.add_argument("--base_c",type=int,default=48)
    ca=sp.add_parser("calibrate"); ca.add_argument("--weights",required=True); ca.add_argument("--val_dir",required=True)
    ca.add_argument("--output_dir",default="./checkpoints"); ca.add_argument("--base_c",type=int,default=48)
    return p.parse_args()

if __name__=="__main__":
    args=get_args(); torch.backends.cudnn.benchmark=True
    if args.cmd=="train": train(args)
    elif args.cmd=="predict": predict(args)
    elif args.cmd=="calibrate": calibrate(args)
