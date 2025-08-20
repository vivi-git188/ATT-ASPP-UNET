# -*- coding: utf-8 -*-
"""
Attention-ASPP-UNet – unified train / predict / calibrate script
带 --deterministic 支持，可复现实验与消融。
"""

import argparse, json, os, random, csv, math
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2, numpy as np, torch
import torch.nn as nn; import torch.nn.functional as F
from albumentations import (CLAHE, Compose, HorizontalFlip, MedianBlur,
    RandomBrightnessContrast, RandomGamma, Resize, ToFloat)
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

# ---------- GLOBALS ----------
SEED, IMG_SIZE, WEIGHT_DECAY, GRAD_CLIP = 2025, 512, 5e-4, 1.0
EARLY_STOP_PATIENCE = 15
LOSS_TYPE, TV_ALPHA, TV_BETA = "combo", 0.7, 0.3

# ---------- DETERMINISTIC ----------
def set_seed(seed: int = SEED, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # albumentations
    try:
        from albumentations import set_seed as A_set_seed
    except (ImportError, AttributeError):
        try:
            from albumentations.core.utils import set_seed as A_set_seed
        except (ImportError, AttributeError):
            A_set_seed = None
    if A_set_seed: A_set_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(wid):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

# ---------- MODEL ----------
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
        h,w = x.shape[2:]
        feats=[b(x) for b in self.blocks]
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

class DummyAttention(nn.Module):
    def forward(self,g,x): return x

class UpBlock(nn.Module):
    def __init__(self,in_c,out_c,use_att=True):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_c,out_c,2,2)
        self.att=(AttentionGate if use_att else DummyAttention()) if isinstance(use_att,bool) else DummyAttention()
        if isinstance(self.att,type): self.att=self.att(out_c,out_c,out_c//2)
        self.conv=nn.Sequential(ConvBNReLU(in_c,out_c),ConvBNReLU(out_c,out_c))
    def forward(self,g,x):
        g=self.up(g)
        if g.shape[-2:]!=x.shape[-2:]:
            g=F.interpolate(g,size=x.shape[-2:],mode="bilinear",align_corners=False)
        x=self.att(g,x)
        return self.conv(torch.cat([x,g],1))

class AttentionASPPUNet(nn.Module):
    def __init__(self,in_channels=1,num_classes=1,base_c=32,use_att=True,use_aspp=True):
        super().__init__()
        self.d1=nn.Sequential(ConvBNReLU(in_channels,base_c),ConvBNReLU(base_c,base_c)); self.p1=nn.MaxPool2d(2)
        self.d2=nn.Sequential(ConvBNReLU(base_c,base_c*2),ConvBNReLU(base_c*2,base_c*2)); self.p2=nn.MaxPool2d(2)
        self.d3=nn.Sequential(ConvBNReLU(base_c*2,base_c*4),ConvBNReLU(base_c*4,base_c*4)); self.p3=nn.MaxPool2d(2)
        self.d4=nn.Sequential(ConvBNReLU(base_c*4,base_c*8),ConvBNReLU(base_c*8,base_c*8)); self.p4=nn.MaxPool2d(2)
        self.bridge = ASPP(base_c*8,base_c*16) if use_aspp \
                      else nn.Sequential(ConvBNReLU(base_c*8,base_c*16,3),nn.Dropout(0.1))
        self.u4=UpBlock(base_c*16,base_c*8,use_att)
        self.u3=UpBlock(base_c*8,base_c*4,use_att)
        self.u2=UpBlock(base_c*4,base_c*2,use_att)
        self.u1=UpBlock(base_c*2,base_c   ,False)
        self.out_conv=nn.Conv2d(base_c,num_classes,1)
    def forward(self,x):
        x1=self.d1(x); x2=self.d2(self.p1(x1)); x3=self.d3(self.p2(x2)); x4=self.d4(self.p3(x3))
        b=self.bridge(self.p4(x4))
        d4=self.u4(b,x4); d3=self.u3(d4,x3); d2=self.u2(d3,x2); d1=self.u1(d2,x1)
        return self.out_conv(d1)

# ---------- PRETRAINED ----------
def load_state_dict_compat(model: nn.Module, ckpt_path: Union[str, Path]):
    sd=torch.load(ckpt_path,map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd=sd["state_dict"]
    new_sd={k.replace(".W_g.",".Wg.").replace(".W_x.",".Wx."):v for k,v in sd.items()}
    model.load_state_dict(new_sd, strict=False)

# ---------- DATASET ----------
def SafeCLAHE(*a,**k): k.pop("always_apply",None); return CLAHE(*a,**k)
def SafeMedianBlur(*a,**k): k.pop("always_apply",None); return MedianBlur(*a,**k)

class FetalACDataset(Dataset):
    def __init__(self, imgs:List[Path], msks:List[Optional[Path]], train=True):
        self.imgs,self.msks,self.train=imgs,msks,train
        self.t=self._tfm()
    def _tfm(self):
        from albumentations import Affine, ElasticTransform
        if self.train:
            t = [
                Resize(IMG_SIZE, IMG_SIZE),
                HorizontalFlip(p=0.5),
                Affine(scale=(0.92, 1.08),
                       rotate=(-7, 7),
                       translate_percent=(0, 0.02),
                       p=0.7),
                RandomGamma(gamma_limit=(80, 120), p=0.3),
                RandomBrightnessContrast(brightness_limit=0.1,
                                         contrast_limit=0.1,
                                         p=0.3),
                # 仅保留 alpha、sigma（其他多余参数会触发 warning）
                ElasticTransform(alpha=8, sigma=3, p=0.25),
                SafeCLAHE(1.0, (8, 8)),
                SafeMedianBlur(3),
                ToFloat(max_value=255),
                ToTensorV2(),
            ]
        else:
            t=[Resize(IMG_SIZE,IMG_SIZE), SafeCLAHE(1.0,(8,8)), SafeMedianBlur(3),
               ToFloat(max_value=255), ToTensorV2()]
        return Compose(t)
    @staticmethod
    def _read(p:Path):
        if p.suffix.lower()=='.mha':
            assert sitk is not None,"Need SimpleITK"
            arr=sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
            if arr.ndim==3: arr=arr[arr.shape[0]//2]
        else:
            arr=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        return arr.astype(np.uint8)
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        img=self._read(self.imgs[i]); msk=np.zeros_like(img)
        if self.msks[i] is not None: msk=self._read(self.msks[i])
        s=self.t(image=img,mask=msk)
        return s["image"].float(), (s["mask"].unsqueeze(0).float()/255.0)

# ---------- LOSSES ----------
class DiceLoss(nn.Module):
    def __init__(self,smooth=1.): super().__init__(); self.s=smooth
    def forward(self,l,t):
        p=torch.sigmoid(l); num=2*(p*t).sum((2,3))+self.s
        den=p.sum((2,3))+t.sum((2,3))+self.s
        return (1-num/den).mean()

class ComboLoss(nn.Module):
    def __init__(self): super().__init__(); self.d=DiceLoss()
    def forward(self,l,t): return self.d(l,t)+F.binary_cross_entropy_with_logits(l,t)

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32).view(1,1,3,3)
        ky=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx",kx); self.register_buffer("ky",ky)
    def forward(self,logits,targets):
        p=torch.sigmoid(logits); t=targets.float()
        gx_p=F.conv2d(p,self.kx.to(p),padding=1); gy_p=F.conv2d(p,self.ky.to(p),padding=1)
        gx_t=F.conv2d(t,self.kx.to(p),padding=1); gy_t=F.conv2d(t,self.ky.to(p),padding=1)
        return F.l1_loss(torch.sqrt(gx_p**2+gy_p**2+1e-8),
                         torch.sqrt(gx_t**2+gy_t**2+1e-8))

def iou_score(l,t,thr=0.5):
    p=(torch.sigmoid(l)>thr).float(); inter=(p*t).sum((2,3))
    union=p.sum((2,3))+t.sum((2,3))-inter
    return (inter/(union+1e-7)).mean().item()

def build_criterion(base, edge, edge_w, args):
    def crit(l,t):
        l,t=l.float(),t.float(); B=t.size(0)
        is_empty=(t.sum((2,3),keepdim=True)==0).float()
        w=torch.where(is_empty==1,args.neg_bce_w if args.stage=='finetune' else 1.,1.).to(l)
        bce=F.binary_cross_entropy_with_logits(l,t,weight=w)
        dice=edge_loss=torch.tensor(0.,device=l.device)
        pos=(is_empty.view(B)==0).nonzero(as_tuple=True)[0]
        if len(pos)>0:
            dice=base(l[pos],t[pos])
            if edge_w>0: edge_loss=edge(l[pos],t[pos])*edge_w
        return dice+bce+edge_loss
    return crit

# ---------- POST-PROCESS ----------
@torch.inference_mode()
def predict_prob_tta(model,x):
    l=model(x); l_flip=model(torch.flip(x,[-1])); l_flip=torch.flip(l_flip,[-1])
    return torch.sigmoid((l+l_flip)/2)[0,0].cpu().numpy()

def refine_mask(m):
    if m.sum()==0: return m
    lab=label(m); cnt=np.bincount(lab.ravel()); cnt[0]=0
    keep=[i for i,c in enumerate(cnt) if c>=max(20,int(0.0015*m.size))]
    if not keep: return np.zeros_like(m)
    m=(np.isin(lab,keep)).astype(np.uint8); lab2=label(m); bc=np.bincount(lab2.ravel()); bc[0]=0
    m=(lab2==np.argmax(bc)).astype(np.uint8)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    return binary_fill_holes(cv2.morphologyEx(m,cv2.MORPH_CLOSE,k)).astype(np.uint8)

def _circularity_score(mask):
    cnts,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    c=max(cnts,key=cv2.contourArea); area=cv2.contourArea(c); peri=cv2.arcLength(c,True)
    return 0.0 if peri<=1e-6 else 4*np.pi*area/(peri**2)

def select_best(stack,topk=5):
    if len(stack)==0: return 0
    areas=np.array([(m>0).sum() for m in stack])
    idx=areas.argsort()[::-1][:max(1,min(topk,len(areas)))]
    return int(max(idx,key=lambda i:_circularity_score(stack[i])))

def _ellipse_circum(a,b):
    h=((a-b)**2)/((a+b)**2)
    return math.pi*(a+b)*(1+3*h/(10+math.sqrt(4-3*h)))

def measure_ac_mm(mask01,spacing):
    cnts,_=cv2.findContours(mask01.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cnts: return 0.0
    c=max(cnts,key=cv2.contourArea)
    if len(c)>=5:
        (_, _),(MA,ma),_=cv2.fitEllipse(c)
        a_mm,b_mm=MA/2*spacing[0],ma/2*spacing[1]
        return _ellipse_circum(a_mm,b_mm)
    return cv2.arcLength(c,True)*float(sum(spacing)/2)
# ---------- IO HELPERS ----------
def convert_mask_2d_to_3d(mask,frame,nf):
    m=(mask>0).astype(np.uint8)*2
    vol=np.zeros((nf,*m.shape),np.uint8)
    if 0<=frame<nf: vol[frame]=m
    return vol

def write_output_mha_and_json(mask,frame,ref,od):
    ref_img=sitk.ReadImage(str(ref)); nf=ref_img.GetSize()[2]
    arr=convert_mask_2d_to_3d(mask,frame,nf)
    out_img=sitk.GetImageFromArray(arr); out_img.CopyInformation(ref_img)
    case=ref.stem; cd=od/case; (cd/'images/fetal-abdomen-segmentation').mkdir(parents=True,exist_ok=True)
    sitk.WriteImage(out_img,str(cd/'images/fetal-abdomen-segmentation/output.mha'))
    json.dump(frame,open(cd/'fetal-abdomen-frame-number.json','w'),indent=2)
    print(f"[✓] {case} frame {frame}")

# ---------- EVALUATE ----------
@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval(); d=i=0.
    for x,y in loader:
        x,y=x.to(device),y.to(device); l=model(x)
        d+=1-DiceLoss()(l,y).item(); i+=iou_score(l,y)
    return d/len(loader), i/len(loader)

def _save_topk_viz(imgs_u8, probs, masks, topk_idx, best_idx, ac_mm, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = len(topk_idx)
    fig, axes = plt.subplots(2, K, figsize=(3.2*K, 6), dpi=180)
    for j, idx in enumerate(topk_idx):
        img = imgs_u8[idx]
        prob = probs[idx]
        m = masks[idx].astype(bool)
        # 上排：概率热度
        ax = axes[0, j]
        ax.imshow(img, cmap="gray")
        ax.imshow(prob, cmap="jet", alpha=0.35, vmin=0, vmax=1)
        ax.set_title(f"s{idx}  circ={_circularity_score(masks[idx]):.2f}\narea={int(m.sum())}")
        ax.axis("off")
        # 下排：mask 叠加
        ax = axes[1, j]
        ax.imshow(img, cmap="gray")
        ax.imshow(m, cmap="spring", alpha=0.35)
        ax.axis("off")
        # 高亮最佳
        if idx == best_idx:
            for a in (axes[0, j], axes[1, j]):
                for sp in a.spines.values():
                    sp.set_edgecolor("lime"); sp.set_linewidth(3)
    fig.suptitle(f"Top-{K} candidates; best = s{best_idx}; AC = {ac_mm:.1f} mm", y=0.98)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ---------- TRAIN ----------
def train(args):
    set_seed(args.seed,args.deterministic)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collect(img_dir:Path,msk_dir:Optional[Path]):
        exts={'.png','.jpg','.jpeg','.tif','.bmp','.mha'}
        imgs,msks=[],[]
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in exts: continue
            imgs.append(p); q=msk_dir/p.name if msk_dir else None
            msks.append(q if (q and q.exists()) else None)
        return imgs,msks

    train_imgs,train_msks=collect(Path(args.train_dir)/'images',Path(args.train_dir)/'masks')
    if args.neg_dir:
        neg_imgs,_=collect(Path(args.neg_dir)/'images',None)
        train_imgs+=neg_imgs; train_msks+=[None]*len(neg_imgs)

    if args.val_dir:
        val_imgs,val_msks=collect(Path(args.val_dir)/'images',Path(args.val_dir)/'masks')
    else:
        pos=[i for i,m in enumerate(train_msks) if m is not None]
        cand=pos if pos else list(range(len(train_imgs)))
        rng=np.random.default_rng(args.seed); rng.shuffle(cand)
        vset=set(cand[:max(1,int(0.1*len(cand)))])
        val_imgs=[train_imgs[i] for i in vset]; val_msks=[train_msks[i] for i in vset]
        train_imgs=[train_imgs[i] for i in range(len(train_imgs)) if i not in vset]
        train_msks=[train_msks[i] for i in range(len(train_msks)) if i not in vset]

    train_ds=FetalACDataset(train_imgs,train_msks,True)
    val_ds  =FetalACDataset(val_imgs ,val_msks ,False)
    g=torch.Generator().manual_seed(args.seed)
    train_ld=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,
                        worker_init_fn=seed_worker,generator=g,drop_last=True)
    val_ld  =DataLoader(val_ds  ,batch_size=args.batch_size,shuffle=False,
                        worker_init_fn=seed_worker,generator=g)

    model=AttentionASPPUNet(base_c=args.base_c,use_att=not args.no_att,
                            use_aspp=not args.no_aspp).to(device)
    if args.stage=='finetune': load_state_dict_compat(model,args.pretrained)

    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=WEIGHT_DECAY)
    warm=0 if args.stage=='finetune' else max(1,int(0.05*args.epochs))
    cos=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs-warm)
    sch = cos if warm == 0 else torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.2, end_factor=1.0, total_iters=warm),
         cos],
        [warm])

    base_loss=ComboLoss()
    crit=build_criterion(base_loss,EdgeLoss(),
                         0.0 if args.no_edge_loss else args.edge_w, args)
    scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    out_dir=Path(args.output_dir)/('ckpt_main' if args.stage=='main' else 'ckpt_finetune')
    out_dir.mkdir(parents=True,exist_ok=True)
    best,noimp,best_p=0.,0,out_dir/f"best_{datetime.now():%Y%m%d-%H%M%S}.pt"

    # --- 新增：准备 metrics.csv ---
    metrics_csv = out_dir/'metrics.csv'
    mfp = open(metrics_csv, 'w', newline='')
    writer = csv.writer(mfp)
    writer.writerow(["epoch","train_loss","val_loss","train_dice","val_dice","train_iou","val_iou"])
    dice_metric = DiceLoss()

    for ep in range(1,args.epochs+1):
        model.train()
        train_loss_sum = 0.0
        train_dice_sum = 0.0
        train_iou_sum  = 0.0
        n_batches = 0

        for x,y in tqdm(train_ld,f"Epoch {ep}/{args.epochs}"):
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                l=model(x); loss=crit(l,y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            scaler.step(opt); scaler.update()

            # 在线统计训练指标
            train_loss_sum += loss.item()
            with torch.no_grad():
                train_dice_sum += (1 - dice_metric(l, y).item())
                train_iou_sum  += iou_score(l, y)
            n_batches += 1

        sch.step()

        # 验证（含 val_loss）
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum  = 0.0
        n_val = 0
        with torch.no_grad():
            for x,y in val_ld:
                x,y=x.to(device),y.to(device)
                l=model(x)
                val_loss_sum += crit(l,y).item()
                val_dice_sum += (1 - dice_metric(l, y).item())
                val_iou_sum  += iou_score(l, y)
                n_val += 1

        train_loss = train_loss_sum / max(1,n_batches)
        train_dice = train_dice_sum / max(1,n_batches)
        train_iou  = train_iou_sum  / max(1,n_batches)
        val_loss   = val_loss_sum   / max(1,n_val)
        val_dice   = val_dice_sum   / max(1,n_val)
        val_iou    = val_iou_sum    / max(1,n_val)

        print(f"[Val] Dice {val_dice:.4f} | IoU {val_iou:.4f} | Loss {val_loss:.4f}")

        # 写入 CSV
        writer.writerow([ep, f"{train_loss:.6f}", f"{val_loss:.6f}",
                            f"{train_dice:.6f}", f"{val_dice:.6f}",
                            f"{train_iou:.6f}", f"{val_iou:.6f}"])
        mfp.flush()

        # 早停/保存最好
        if val_dice>best:
            best=val_dice; noimp=0; torch.save(model.state_dict(),best_p)
            print(f"[✓] best saved → {best_p}")
        else:
            noimp+=1
            if noimp>=EARLY_STOP_PATIENCE:
                print("Early stop")
                break

    mfp.close()


# ---------- CALIBRATE ----------
@torch.inference_mode()
def calibrate(args):
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=AttentionASPPUNet(base_c=args.base_c,use_att=not args.no_att,
                            use_aspp=not args.no_aspp).to(device)
    model.load_state_dict(torch.load(args.weights,map_location=device)); model.eval()

    val_dir=Path(args.val_dir)
    imgs=sorted((val_dir/'images').glob('*.png'))
    assert len(imgs)>0, f"No PNGs under {val_dir/'images'}"
    thrs=np.linspace(0.35,0.60,11)

    thr_means, thr_stds, thr_medians = [], [], []
    all_rows=[]  # 可选：逐图逐阈值的原始分数

    for thr in tqdm(thrs,desc="Scanning"):
        ds=[]
        for p in imgs:
            sl=cv2.imread(str(p),0)
            sl_u8=cv2.normalize(sl,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            e=cv2.medianBlur(cv2.createCLAHE(1.0,(8,8)).apply(sl_u8),3)
            x=Compose([Resize(IMG_SIZE,IMG_SIZE),ToFloat(max_value=255),ToTensorV2()])(image=e)['image'].unsqueeze(0).to(device)
            prob=predict_prob_tta(model,x)
            prob=cv2.resize(prob,sl.shape[::-1])
            prob=cv2.GaussianBlur(prob,(5,5),0)
            m=(prob>float(thr)).astype(np.uint8)
            gt=(cv2.imread(str(val_dir/'masks'/p.name),0)>127).astype(np.uint8)
            inter=(m&gt).sum()
            dice=2*inter/(m.sum()+gt.sum()+1e-7)
            ds.append(dice)
            all_rows.append((p.name,float(thr),float(dice)))
        ds=np.array(ds, dtype=np.float32)
        thr_means.append(ds.mean()); thr_stds.append(ds.std()); thr_medians.append(np.median(ds))

    thr_means=np.array(thr_means); thr_stds=np.array(thr_stds)
    best_thr=float(thrs[int(np.argmax(thr_means))])

    # 保存结果
    out_root=Path(args.output_dir); out_root.mkdir(parents=True,exist_ok=True)
    json.dump({'best_thr':best_thr}, open(out_root/'thr.json','w'), indent=2)

    import pandas as pd
    df_curve=pd.DataFrame({
        'thr': thrs,
        'dice_mean': thr_means,
        'dice_std': thr_stds,
        'dice_median': thr_medians
    })
    df_curve.to_csv(out_root/'calibrate_curve.csv', index=False)

    # （可选）逐图明细
    pd.DataFrame(all_rows, columns=['case','thr','dice']).to_csv(out_root/'calibrate_raw.csv', index=False)

    # 画线图
    plt.figure(figsize=(7,4), dpi=200)
    plt.plot(thrs, thr_means, marker='o', label='Mean Dice')
    plt.fill_between(thrs, thr_means-thr_stds, thr_means+thr_stds, alpha=0.15, label='±1σ')
    plt.axvline(best_thr, linestyle='--', label=f'best={best_thr:.3f}')
    plt.xlabel('Threshold'); plt.ylabel('Dice'); plt.title('Threshold–Dice on Validation')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_root/'thr_dice_curve.png'); plt.close()

    # 画柱状图
    plt.figure(figsize=(7,4), dpi=200)
    plt.bar(thrs, thr_means, width=(thrs[1]-thrs[0])*0.8)
    plt.axvline(best_thr, linestyle='--')
    plt.xlabel('Threshold'); plt.ylabel('Mean Dice'); plt.title('Threshold–Dice (bars)')
    plt.tight_layout(); plt.savefig(out_root/'thr_dice_bars.png'); plt.close()

    print(f"Calibrated thr={best_thr:.3f} → saved to {out_root}")

# ---------- PREDICT ----------
@torch.inference_mode()
def predict(args):
    set_seed(deterministic=args.deterministic)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    THR=0.48; cfg=Path('./checkpoints/thr.json')
    if cfg.exists():
        try: THR=float(json.load(open(cfg))['best_thr']); print(f"[i] use thr {THR:.3f}")
        except: pass

    spacing_map=json.load(open(args.spacing_json)) if args.spacing_json else {}
    def _sp(case):
        v=spacing_map.get(case);
        return None if v is None else tuple(map(float,(v['spacing'] if isinstance(v,dict) else v)[:2]))

    model=AttentionASPPUNet(base_c=args.base_c,use_att=not args.no_att,
                            use_aspp=not args.no_aspp).to(device)
    model.load_state_dict(torch.load(args.weights,map_location=device)); model.eval()

    inp,od=Path(args.input_dir),Path(args.out_dir); od.mkdir(parents=True,exist_ok=True)
    rows=[]
    for p in tqdm(sorted(inp.iterdir()),desc="Processing"):
        ext=p.suffix.lower()
        # ---------- 2D ----------
        if ext in {'.png','.jpg','.jpeg'}:
            sl=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
            sl_u8=cv2.normalize(sl,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            e=cv2.medianBlur(cv2.createCLAHE(1.0,(8,8)).apply(sl_u8),3)
            x=Compose([Resize(IMG_SIZE,IMG_SIZE),ToFloat(max_value=255),ToTensorV2()])(image=e)['image'].unsqueeze(0).to(device)
            prob=predict_prob_tta(model,x); prob=cv2.resize(prob,sl.shape[::-1]); prob=cv2.GaussianBlur(prob,(5,5),0)
            mask=refine_mask((prob>THR).astype(np.uint8)); cv2.imwrite(str(od/f"{p.stem}_mask.png"),mask*255)
            case,frame=(p.stem.split("_s")[0],int(p.stem.split("_s")[1]) if "_s" in p.stem else -1)
            sp=_sp(case)
            if sp:
                ac=round(measure_ac_mm(mask,sp),1); rows.append((case,frame,ac))
                print(f"[✓] {p.stem}: AC={ac:.1f} mm")
            else:
                print(f"[warn] no spacing for {case}")

        # ---------- 3D ----------
        elif ext == '.mha':
            if sitk is None: raise ImportError("SimpleITK needed")
            ref = sitk.ReadImage(str(p))
            vol = sitk.GetArrayFromImage(ref)

            imgs_u8, probs, preds = [], [], []
            for sl in tqdm(vol, leave=False, desc=p.name):
                sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                e = cv2.medianBlur(cv2.createCLAHE(1.0, (8, 8)).apply(sl_u8), 3)
                x = Compose([Resize(IMG_SIZE, IMG_SIZE), ToFloat(max_value=255), ToTensorV2()])(image=e)[
                    'image'].unsqueeze(0).to(device)
                prob = predict_prob_tta(model, x)  # (H,W), float in [0,1]
                prob = cv2.resize(prob, sl.shape[::-1])  # 回到原尺寸
                prob = cv2.GaussianBlur(prob, (5, 5), 0)
                m = refine_mask((prob > THR).astype(np.uint8))

                imgs_u8.append(sl_u8)
                probs.append(prob)
                preds.append(m)

            preds_np = np.stack(preds)
            # 先取面积 Top-K，再在其中按圆度选最佳
            areas = np.array([m.sum() for m in preds_np])
            K = min(5, len(areas))
            topk_idx = areas.argsort()[::-1][:K]
            best_idx = topk_idx[np.argmax([_circularity_score(preds_np[i]) for i in topk_idx])]

            # 原有保存（3D 输出 + best 帧 JSON）
            write_output_mha_and_json(preds[best_idx], int(best_idx), p, od)
            ac = round(measure_ac_mm(preds[best_idx], ref.GetSpacing()[:2]), 1)
            rows.append((p.stem, int(best_idx), ac))
            print(f"[✓] {p.stem}: best={best_idx}, AC={ac:.1f} mm")

            # 新增：保存 Top-K 可视化 + 每帧指标
            out_png = od / f"{p.stem}_top{K}_viz.png"
            _save_topk_viz(imgs_u8, probs, preds, topk_idx, int(best_idx), ac, str(out_png))
            # 保存每帧的面积/圆度（可放论文补充）
            import pandas as pd
            import numpy as np
            circ = [float(_circularity_score(m)) for m in preds]
            pd.DataFrame({
                "slice": np.arange(len(preds)),
                "area": [int(m.sum()) for m in preds],
                "circularity": circ
            }).to_csv(od / f"{p.stem}_slice_metrics.csv", index=False)

    if rows:
        with open(od/'ac_results.csv','w',newline='') as f:
            csv.writer(f).writerows([("case_id","frame_idx","ac_mm"),*rows])
        print(f"[✓] AC results saved → {od/'ac_results.csv'}")

# ---------- CLI ----------
def get_args():
    p=argparse.ArgumentParser("A-ASPP-UNet deterministic")
    sp=p.add_subparsers(dest="cmd",required=True)

    t=sp.add_parser("train")
    t.add_argument("--stage",choices=['main','finetune'],default='main')
    t.add_argument("--train_dir",required=True); t.add_argument("--neg_dir"); t.add_argument("--val_dir")
    t.add_argument("--output_dir",default="./checkpoints"); t.add_argument("--pretrained")
    t.add_argument("--epochs",type=int,default=120); t.add_argument("--batch_size",type=int,default=8)
    t.add_argument("--lr",type=float,default=3e-4); t.add_argument("--base_c",type=int,default=48)
    t.add_argument("--edge_w",type=float,default=0.05); t.add_argument("--neg_bce_w",type=float,default=0.05)
    t.add_argument("--seed",type=int,default=SEED); t.add_argument("--deterministic",action="store_true")
    t.add_argument("--no_att",action="store_true"); t.add_argument("--no_aspp",action="store_true"); t.add_argument("--no_edge_loss",action="store_true")

    pr=sp.add_parser("predict")
    pr.add_argument("--weights",required=True); pr.add_argument("--input_dir",required=True)
    pr.add_argument("--out_dir",default="./preds"); pr.add_argument("--spacing_json",required=True)
    pr.add_argument("--base_c",type=int,default=48); pr.add_argument("--deterministic",action="store_true")
    pr.add_argument("--no_att",action="store_true"); pr.add_argument("--no_aspp",action="store_true")

    ca=sp.add_parser("calibrate")
    ca.add_argument("--weights",required=True); ca.add_argument("--val_dir",required=True)
    ca.add_argument("--output_dir",default="./checkpoints"); ca.add_argument("--base_c",type=int,default=48)
    ca.add_argument("--deterministic",action="store_true"); ca.add_argument("--no_att",action="store_true"); ca.add_argument("--no_aspp",action="store_true")

    return p.parse_args()

if __name__ == "__main__":
    args=get_args(); torch.backends.cudnn.benchmark=True
    if args.cmd=="train":      train(args)
    elif args.cmd=="predict":  predict(args)
    elif args.cmd=="calibrate": calibrate(args)
