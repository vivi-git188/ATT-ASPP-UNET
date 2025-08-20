#!/usr/bin/env python3
"""
attention_aspp_unet_pipeline_v2.py
----------------------------------

简化版 Attention-ASPP-UNet 训练 / 推理流水线。
- 训练阶段：Flip + CLAHE + Bright/Contrast
- 验证 / 推理：Resize + Normalize
- 损失：ComboLoss(Dice+BCE)，可选 EdgeLoss
- 后处理：最大连通域 + 填洞
"""

import argparse, json, time
from pathlib import Path

import albumentations as A
import cv2, numpy as np, torch
import pandas as pd
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- Dataset -------------------- #
class PNGSegDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, split: str):
        self.img_paths = sorted(img_dir.glob("*.png"))
        self.mask_dir = mask_dir
        self.split = split
        self.tf = self._build_tf()

    def _build_tf(self):
        base = [A.Resize(512, 512),
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=65535.0)]
        if self.split == "train":
            base = [A.HorizontalFlip(p=.5),
                    A.CLAHE(p=.3),
                    A.RandomBrightnessContrast(p=.2)] + base
        return A.Compose(base + [ToTensorV2()])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        ip = self.img_paths[i]
        mp = self.mask_dir / ip.name
        img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        msk = (msk > 0).astype(np.uint8)
        t = self.tf(image=img, mask=msk)
        return t["image"], t["mask"].float().unsqueeze(0)

# -------------------- Model -------------------- #
def C(n_i, n_o):  # conv block
    return nn.Sequential(
        nn.Conv2d(n_i, n_o, 3, 1, 1), nn.BatchNorm2d(n_o), nn.ReLU(True),
        nn.Conv2d(n_o, n_o, 3, 1, 1), nn.BatchNorm2d(n_o), nn.ReLU(True))

class AttGate(nn.Module):
    def __init__(s, g, x, inter):
        super().__init__()
        s.Wg=nn.Sequential(nn.Conv2d(g,inter,1),nn.BatchNorm2d(inter))
        s.Wx=nn.Sequential(nn.Conv2d(x,inter,1),nn.BatchNorm2d(inter))
        s.psi=nn.Sequential(nn.Conv2d(inter,1,1),nn.BatchNorm2d(1),nn.Sigmoid())
        s.relu=nn.ReLU(True)
    def forward(s,g,x): a=s.relu(s.Wg(g)+s.Wx(x)); a=s.psi(a); return x*a

class ASPP(nn.Module):
    def __init__(s, n_i, n_o):
        super().__init__(); rates=[1,6,12,18]
        s.blocks=nn.ModuleList([nn.Sequential(
            nn.Conv2d(n_i,n_o,1 if r==1 else 3,1,r,padding=r,dilation=r),
            nn.BatchNorm2d(n_o), nn.ReLU(True)) for r in rates])
        s.out=nn.Sequential(nn.Conv2d(n_o*4,n_o,1), nn.BatchNorm2d(n_o), nn.ReLU(True))
    def forward(s,x): return s.out(torch.cat([b(x) for b in s.blocks],1))

class Net(nn.Module):
    """
    Attention-ASPP-UNet
    use_attn  = False → 三层 Gate 退化为恒等映射
    use_aspp  = False → 桥接层使用普通 3×3 Conv
    """
    def __init__(s, F=[64,128,256,512], use_attn=True, use_aspp=True):
        super().__init__()
        # ------- 编码器 ------- #
        s.e1, s.e2, s.e3, s.e4 = C(1,F[0]), C(F[0],F[1]), C(F[1],F[2]), C(F[2],F[3])
        s.pool = nn.MaxPool2d(2)

        # ------- 桥接层 ------- #
        s.bridge = ASPP(F[3],F[3]) if use_aspp else C(F[3],F[3])

        # ------- Attention Gate ------- #
        if use_attn:
            s.g3 = AttGate(F[3],F[2],F[2]//2)
            s.g2 = AttGate(F[2],F[1],F[1]//2)
            s.g1 = AttGate(F[1],F[0],F[0]//2)
        else:          # 恒等映射
            s.g3 = s.g2 = s.g1 = lambda g,x: x

        # ------- 解码器 ------- #
        s.u3, s.u2, s.u1 = C(F[3]+F[2],F[2]), C(F[2]+F[1],F[1]), C(F[1]+F[0],F[0])
        s.out = nn.Conv2d(F[0],1,1)

    def forward(s,x):
        e1 = s.e1(x)
        e2 = s.e2(s.pool(e1))
        e3 = s.e3(s.pool(e2))
        e4 = s.e4(s.pool(e3))

        b  = s.bridge(e4)

        d3 = torch.cat([F.interpolate(b,scale_factor=2,mode="bilinear",align_corners=False),
                        s.g3(b,e3)],1); d3 = s.u3(d3)
        d2 = torch.cat([F.interpolate(d3,scale_factor=2,mode="bilinear",align_corners=False),
                        s.g2(d3,e2)],1); d2 = s.u2(d2)
        d1 = torch.cat([F.interpolate(d2,scale_factor=2,mode="bilinear",align_corners=False),
                        s.g1(d2,e1)],1); d1 = s.u1(d1)

        return torch.sigmoid(s.out(d1))


# -------------------- Loss -------------------- #
class DiceLoss(nn.Module):
    def __init__(s):
        super().__init__()
    def forward(s,p,y):
        p,y=p.flatten(1),y.flatten(1)
        return 1-(2*(p*y).sum(1)+1)/(p.sum(1)+y.sum(1)+1).mean()

class Combo(nn.Module):
    def __init__(s,a=.5): super().__init__(); s.a=a; s.d=DiceLoss(); s.b=nn.BCELoss()
    def forward(s,p,y): return s.a*s.b(p,y)+(1-s.a)*s.d(p,y)

def edge_loss(p,y):
    k=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32,device=p.device).view(1,1,3,3)
    g=lambda t: torch.sqrt(F.conv2d(t,k,1)+F.conv2d(t,k.transpose(2,3),1)**2)
    return F.l1_loss(g(p),g(y))

# -------------------- Utils -------------------- #
def postproc(pred, area=100):
    m=(pred>0.5).astype(np.uint8)
    lbl=measure.label(m,2)
    if lbl.max()==0: return m
    m=lbl==(np.bincount(lbl.flat)[1:].argmax()+1)
    m=morphology.remove_small_objects(m,area)
    m=binary_fill_holes(m)
    return m.astype(np.uint8)

# ------------------ Train / Eval ------------------ #
def train_ep(net,ldr,opt,crit,ew):
    net.train(); tot=0
    for x,y in tqdm(ldr,desc="train",leave=False):
        x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
        p=net(x); loss=crit(p,y)+ (ew*edge_loss(p,y) if ew else 0)
        loss.backward(); opt.step(); tot+=loss.item()*x.size(0)
    return tot/len(ldr.dataset)

@torch.no_grad()
def val_ep(net,ldr):
    net.eval(); ds=0
    for x,y in tqdm(ldr,"val",leave=False):
        x,y=x.to(DEVICE),y.to(DEVICE); p=(net(x)>0.5).float()
        inter=(p*y).sum([1,2,3]); uni=p.sum([1,2,3])+y.sum([1,2,3])
        ds+=((2*inter+1e-7)/(uni+1e-7)).sum().item()
    return ds/len(ldr.dataset)

# -------------------- CLI -------------------- #
def train(args):
    d=Path(args.data); tr=PNGSegDataset(d/"train"/"images",d/"train"/"masks","train")
    vl=PNGSegDataset(d/"val"/"images",d/"val"/"masks","val")
    tl=DataLoader(tr,args.bs,True,4,pin_memory=True); vl=DataLoader(vl,args.bs,False,4,pin_memory=True)
    net = Net(use_attn=(not args.no_attention),use_aspp = (not args.no_aspp)).to(DEVICE)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr)
    crit=Combo(args.alpha); best=0; ck=Path(args.ckpt); ck.mkdir(parents=True,exist_ok=True)
    for e in range(1,args.epochs+1):
        t0=time.time(); loss=train_ep(net,tl,opt,crit,args.edge_w)
        dice=val_ep(net,vl); print(f"ep{e}/{args.epochs} loss{loss:.4f} dice{dice:.4f} "
                                   f"{(time.time()-t0)/60:.1f}m")
        if dice>best: best=dice; torch.save(net.state_dict(),ck/"best.pth")
    print("best dice:",best)

# ---------- 辅助函数 ---------- #
def comp_ac(mask: np.ndarray, spacing: float) -> float:
    """根据二值掩膜计算 AC (mm)。 spacing=像素间距 (mm/px)"""
    if mask.sum() == 0: return 0.0
    cnt,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    peri = max(cv2.arcLength(c,True) for c in cnt)
    return peri * spacing          # mm

def dice_coef(a,b):
    a, b = a.astype(bool), b.astype(bool)
    inter = (a & b).sum()
    return 1.0 if inter==0 and a.sum()==0 and b.sum()==0 else (2*inter)/(a.sum()+b.sum()+1e-7)

def iou_coef(a,b):
    a, b = a.astype(bool), b.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return 1.0 if union==0 else inter/union


def hd95(a,b):
    if a.sum()==0 or b.sum()==0: return np.nan
    from scipy.spatial.distance import cdist
    a_pts = np.argwhere(a)
    b_pts = np.argwhere(b)
    d1 = cdist(a_pts,b_pts).min(1)
    d2 = cdist(b_pts,a_pts).min(1)
    return np.percentile(np.hstack([d1,d2]),95)

# ---------- 推理 ---------- #
def infer(args):
    net = Net().to(DEVICE)
    net.load_state_dict(torch.load(args.w,map_location=DEVICE))
    net.eval()

    img_dir = Path(args.img)
    out_dir = Path(args.out); out_dir.mkdir(parents=True,exist_ok=True)

    # 需要指标时必须提供真值 masks
    calc_metrics = args.metrics
    if calc_metrics and args.mask == ".":
        raise ValueError("打开 --metrics 时必须指定 --mask <gt_mask_dir>")

    ds = PNGSegDataset(img_dir, Path(args.mask), split="test")
    dl = DataLoader(ds,1,False)

    dice_all, iou_all, hd_all, ac_pred, ac_gt, ac_err, img_names = [], [], [], [], [], []

    for i, (x, gt) in enumerate(tqdm(dl, "infer")):
        x = x.to(DEVICE)
        pred = net(x)[0, 0].cpu().numpy()
        pmask = postproc(pred)

        # —— 保存 PNG —— #
        stem = ds.img_paths[i].stem
        out_name = f"{stem}_mask.png"
        cv2.imwrite(str(out_dir / out_name), pmask * 255)

        if not calc_metrics:
            continue  # 提速：无指标直接下一张

        # —— 指标 —— #
        gmask = gt[0, 0].numpy().astype(np.uint8)

        ac_p = comp_ac(pmask, args.spacing)
        ac_g = comp_ac(gmask, args.spacing)

        dice_all.append(dice_coef(pmask, gmask))
        iou_all.append(iou_coef(pmask, gmask))
        hd_all.append(hd95(pmask, gmask))
        ac_pred.append(ac_p)
        ac_gt.append(ac_g)
        ac_err.append(abs(ac_p - ac_g))
        img_names.append(out_name)

    if calc_metrics:
        print("\n===  Metrics  ===")
        print(f"Dice  : {np.nanmean(dice_all):.4f}")
        print(f"IoU   : {np.nanmean(iou_all):.4f}")
        print(f"HD95  : {np.nanmean(hd_all) :.2f} px")
        print(f"AC MAE: {np.nanmean(ac_err):.2f} mm")
        df = pd.DataFrame({
            "image": img_names,
            "Dice": dice_all,
            "IoU": iou_all,
            "HD95_px": hd_all,
            "AC_pred_mm": ac_pred,
            "AC_gt_mm": ac_gt,
            "AC_abs_err_mm": ac_err,
        })
        mean_row = df.mean(numeric_only=True)
        mean_row["image"] = "mean"
        df = pd.concat([df, mean_row.to_frame().T], ignore_index=True)

        csv_path = out_dir / "metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n 详细指标已写入: {csv_path}")


def parse():
    p=argparse.ArgumentParser("Attn-ASPP-Unet v2"); sub=p.add_subparsers(dest="cmd",required=True)
    tr=sub.add_parser("train"); tr.add_argument("data"); tr.add_argument("--epochs",type=int,default=150)
    tr.add_argument("--bs",type=int,default=8); tr.add_argument("--lr",type=float,default=1e-4)
    tr.add_argument("--alpha",type=float,default=.5)
    tr.add_argument("--edge_w",type=float,default=0.)
    tr.add_argument("--no_attention", action="store_true",help = "关闭三层 Attention Gate")
    tr.add_argument("--no_aspp", action="store_true",help = "桥接层不用 ASPP，而是 3×3 Conv")
    tr.add_argument("--ckpt",default="ckpt"); tr.set_defaults(func=train)
    inf=sub.add_parser("infer"); inf.add_argument("w"); inf.add_argument("img")
    inf.add_argument("--mask",default=".")
    inf.add_argument("--spacing", type=float, default=1.0,help = "像素尺寸 (mm/px) 用于计算 AC，默认 1.0")
    inf.add_argument("--metrics", action="store_true",help = "若开启，将输出 Dice/IoU/HD95/AC 误差")
    inf.add_argument("--out",default="preds"); inf.set_defaults(func=infer)
    return p.parse_args()

if __name__ == "__main__":
    a=parse(); a.func(a)
