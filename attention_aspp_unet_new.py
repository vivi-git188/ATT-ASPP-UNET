# =========================================
#  Fast Attention‑ASPP U‑Net  (改进版)
#  ✓ Combo Loss  ✓ AMP  ✓ EarlyStopping  ✓ CosineLR
#  ⚠ 依赖:  pip install monai tqdm opencv-python
# =========================================

""" 分两部分展示 (Part‑1 / Part‑2) ，方便阅读。
Part‑1: 网络结构、数据集、度量、损失
Part‑2: 训练脚本 (argparse、main) 与 run_epoch()
"""

# ---------------  Part 1  ---------------
import argparse, os, random
from pathlib import Path

import cv2, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from monai.losses import DiceLoss   # ★ 新增

# ---------- Model Blocks ----------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))
    def forward(self, x):
        return self.block(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_g, in_x, inter):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(in_g, inter, 1, bias=False), nn.BatchNorm2d(inter))
        self.W_x = nn.Sequential(nn.Conv2d(in_x, inter, 1, bias=False), nn.BatchNorm2d(inter))
        self.psi = nn.Sequential(nn.Conv2d(inter, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(True)

    def forward(self, g, x):
        g1, x1 = self.W_g(g), self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.psi(self.relu(g1 + x1))
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x * psi

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ReLU(True)) for r in rates])
        self.project = nn.Sequential(nn.Conv2d(out_ch * len(rates), out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, x):
        return self.project(torch.cat([c(x) for c in self.convs], dim=1))

class AttentionASPPUNet(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 1, base: int = 32):  # ★ base=32
        super().__init__()
        b = base
        self.inc = DoubleConv(in_ch, b)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b, b * 2))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 2, b * 4))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 4, b * 8))
        self.aspp = ASPP(b * 8, b * 16)
        self.u3 = nn.ConvTranspose2d(b * 16, b * 8, 2, 2)
        self.a3, self.c3 = AttentionBlock(b * 8, b * 4, b * 4), DoubleConv(b * 8 + b * 4, b * 8)
        self.u2 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.a2, self.c2 = AttentionBlock(b * 4, b * 2, b * 2), DoubleConv(b * 4 + b * 2, b * 4)
        self.u1 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.a1, self.c1 = AttentionBlock(b * 2, b, b), DoubleConv(b * 2 + b, b * 2)
        self.outc = nn.Conv2d(b * 2, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        center = self.aspp(x4)
        d3 = self.u3(center)
        d3 = F.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.c3(torch.cat([d3, self.a3(d3, x3)], 1))
        d2 = self.u2(d3)
        d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.c2(torch.cat([d2, self.a2(d2, x2)], 1))
        d1 = self.u1(d2)
        d1 = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.c1(torch.cat([d1, self.a1(d1, x1)], 1))
        return self.outc(d1)                       # logits

# ---------- Dataset ----------
class FetalAbdomenDataset(Dataset):
    def __init__(self, root, size=224, keep_empty=False):
        self.root = Path(root); self.size = size
        imgs = sorted((self.root / 'images').glob('*.png'))
        if keep_empty:
            self.imgs = imgs
        else:
            self.imgs = [p for p in imgs if (cv2.imread(str(self.root/'masks'/p.name), 0) > 0).any()]
        self.masks = [self.root/'masks'/p.name for p in self.imgs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.imgs[idx]), 0)
        msk = cv2.imread(str(self.masks[idx]), 0)
        img = cv2.resize(img, (self.size, self.size), cv2.INTER_AREA)
        msk = cv2.resize(msk, (self.size, self.size), cv2.INTER_NEAREST)
        img = torch.from_numpy(img.astype(np.float32) / 255.).unsqueeze(0)
        msk = torch.from_numpy((msk > 0).astype(np.float32)).unsqueeze(0)
        return img, msk

# ---------- Metrics ----------
@torch.no_grad()
def dice_coe(pred, tgt, eps=1e-6):
    pred = (pred > 0.25).float()         # ★ 阈值 0.25
    inter = (pred * tgt).sum()
    return (2 * inter + eps) / (pred.sum() + tgt.sum() + eps)

@torch.no_grad()
def iou_coe(pred, tgt, eps=1e-6):
    pred = (pred > 0.25).float()
    inter = (pred * tgt).sum()
    union = pred.sum() + tgt.sum() - inter
    return (inter + eps) / (union + eps)

@torch.no_grad()
def hausdorff95(pred, tgt):
    p = (pred.squeeze().cpu().numpy() > 0.25).astype(np.uint8)
    t = tgt.squeeze().cpu().numpy().astype(np.uint8)
    if p.sum() == 0 or t.sum() == 0:
        return np.nan
    pp, tt = np.column_stack(np.where(p)), np.column_stack(np.where(t))
    return max(directed_hausdorff(pp, tt)[0], directed_hausdorff(tt, pp)[0])

# ===============  Part 2  ===============

def combo_loss(pred, target):
    # 0.5 BCE + 0.5 Dice (sigmoid 内部开)
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = DiceLoss(sigmoid=True, reduction='mean')(pred, target)
    return 0.5 * bce + 0.5 * dice


def run_epoch(model, loader, opt, dev, train=True, amp=False, scaler=None):
    model.train() if train else model.eval()
    loop = tqdm(loader, leave=False, desc='Train' if train else 'Val')
    loss_l, dice_l, iou_l, hd_l = [], [], [], []
    for img, mask in loop:
        img, mask = img.to(dev), mask.to(dev)
        with torch.set_grad_enabled(train):
            with autocast(enabled=amp):
                logits = model(img)
                loss = combo_loss(logits, mask)
        if train:
            scaler.scale(loss).backward() if amp else loss.backward()
            scaler.step(opt) if amp else opt.step()
            scaler.update() if amp else None
            opt.zero_grad()
        else:
            prob = torch.sigmoid(logits)
            dice_l.append(dice_coe(prob, mask).item())
            iou_l.append(iou_coe(prob, mask).item())
            hd_l.append(hausdorff95(prob, mask))
        loss_l.append(loss.item())
    if train:
        return np.mean(loss_l)
    return np.mean(dice_l), np.mean(iou_l), np.nanmean(hd_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', required=True)
    ap.add_argument('--val_dir', required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--base_ch', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--out_dir', default='./checkpoints')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = FetalAbdomenDataset(args.train_dir, args.size)
    val_ds = FetalAbdomenDataset(args.val_dir, args.size)
    tl = DataLoader(train_ds, args.batch_size, True, num_workers=args.num_workers, pin_memory=True)
    vl = DataLoader(val_ds, 1, False, num_workers=2, pin_memory=True)

    model = AttentionASPPUNet(base=args.base_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler(enabled=args.amp)

    best, wait = 0, 0
    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, tl, opt, dev, True, args.amp, scaler)
        dice, iou, hd = run_epoch(model, vl, opt, dev, False, args.amp)
        sched.step()
        print(f"Epoch {ep:03d} | Loss {tr_loss:.4f} | Dice {dice:.4f} | IoU {iou:.4f} | HD {hd:.2f}")
        if dice > best:
            best, wait = dice, 0
            torch.save(model.state_dict(), Path(args.out_dir) / 'best_model.pth')
            print('✔️  Saved new best model')
        else:
            wait += 1
            if wait >= args.patience:
                print('⏹️  Early stopping')
                break


if __name__ == '__main__':
    main()
