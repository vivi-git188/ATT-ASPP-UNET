# attention_aspp_unet_pipeline.py (updated for folder structure train/images, train/masks, val/...)
"""
End‑to‑end pipeline for fetal abdominal circumference segmentation using an
Attention‑ASPP U‑Net.  **This revision supports the common folder layout**

<root>/
 ├─ train/
 │   ├─ images/  (PNG/JPG, grayscale)
 │   └─ masks/   (binary masks; same filename as image)
 ├─ val/          (optional – otherwise script will auto‑split)
 │   ├─ images/
 │   └─ masks/
 └─ test/images/  (no masks)  ←  used by `predict`.

The network remains 2‑D; SimpleITK now optional – used only if a *.mha* file is
encountered.
"""
from __future__ import annotations
import argparse, os, cv2, math, json
import pathlib
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from albumentations import (Compose, CLAHE, MedianBlur, HorizontalFlip,
                            RandomGamma, ToFloat)
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

try:
    import SimpleITK as sitk  # optional – only needed for .mha
except ImportError:
    sitk = None

# -----------------------------------------------------------------------------
# Model (identical to first version)
# -----------------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, p: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=p if p is not None else k // 2, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.rel  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.rel(self.bn(self.conv(x)))

class ASPP(nn.Module):
    def __init__(self, in_c: int, out_c: int = 256, rates=(6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True)),
            *[nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=r, dilation=r, bias=False),
                            nn.BatchNorm2d(out_c), nn.ReLU(True)) for r in rates]
        ])
        self.img_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_c, out_c, 1, bias=False),
                                      nn.BatchNorm2d(out_c), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_c*(len(self.blocks)+1), out_c, 1, bias=False),
                                     nn.BatchNorm2d(out_c), nn.ReLU(True), nn.Dropout(0.1))
    def forward(self, x):
        h, w = x.shape[2:]
        feats = [blk(x) for blk in self.blocks]
        pool = self.img_pool(x)
        pool = F.interpolate(pool, size=(h, w), mode='bilinear', align_corners=False)
        feats.append(pool)
        return self.project(torch.cat(feats, dim=1))

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(True)
    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, att=True):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.att = AttentionGate(out_c, out_c, out_c//2) if att else nn.Identity()
        self.conv = nn.Sequential(ConvBNReLU(in_c, out_c), ConvBNReLU(out_c, out_c))
    def forward(self, g, x):
        g = self.up(g)
        x = self.att(g, x)
        g = torch.cat([x, g], dim=1)
        return self.conv(g)

class AttentionASPPUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_c=32):
        super().__init__()
        # encoder
        self.d1 = nn.Sequential(ConvBNReLU(in_channels, base_c), ConvBNReLU(base_c, base_c))
        self.p1 = nn.MaxPool2d(2)
        self.d2 = nn.Sequential(ConvBNReLU(base_c, base_c*2), ConvBNReLU(base_c*2, base_c*2))
        self.p2 = nn.MaxPool2d(2)
        self.d3 = nn.Sequential(ConvBNReLU(base_c*2, base_c*4), ConvBNReLU(base_c*4, base_c*4))
        self.p3 = nn.MaxPool2d(2)
        self.d4 = nn.Sequential(ConvBNReLU(base_c*4, base_c*8), ConvBNReLU(base_c*8, base_c*8))
        self.p4 = nn.MaxPool2d(2)
        # bridge
        self.bridge = ASPP(base_c*8, base_c*16)
        # decoder
        self.u4 = UpBlock(base_c*16, base_c*8, att=True)
        self.u3 = UpBlock(base_c*8,  base_c*4, att=True)
        self.u2 = UpBlock(base_c*4,  base_c*2, att=True)
        self.u1 = UpBlock(base_c*2,  base_c,   att=False)
        self.out_conv = nn.Conv2d(base_c, num_classes, 1)
    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))
        b  = self.bridge(self.p4(x4))
        d4 = self.u4(b,  x4)
        d3 = self.u3(d4, x3)
        d2 = self.u2(d3, x2)
        d1 = self.u1(d2, x1)
        return self.out_conv(d1)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class FetalACDataset(Dataset):
    def __init__(self, img_paths: List[Path], msk_paths: List[Path] | None, train=True):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.train     = train
        self.transform = self._build_transform()
    def _build_transform(self):
        if self.train:
            return Compose([
                CLAHE(1.0, (8,8), always_apply=True),
                MedianBlur(3, always_apply=True),
                HorizontalFlip(p=0.5),
                RandomGamma((80,120), p=0.5),
                ToFloat(max_value=255.0),
                ToTensorV2(),
            ])
        else:
            return Compose([
                CLAHE(1.0, (8,8), always_apply=True),
                MedianBlur(3, always_apply=True),
                ToFloat(max_value=255.0),
                ToTensorV2(),
            ])
    def __len__(self):
        return len(self.img_paths)
    def _read(self, path: Path):
        ext = path.suffix.lower()
        if ext == '.mha':
            if sitk is None:
                raise RuntimeError('SimpleITK required for .mha files')
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
            if arr.ndim == 3:
                arr = arr[arr.shape[0]//2]  # middle slice
        else:  # png/jpg
            arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return arr.astype(np.uint8)
    def __getitem__(self, idx):
        img = self._read(self.img_paths[idx])
        mask = None
        if self.msk_paths is not None:
            mask = self._read(self.msk_paths[idx])
        else:
            mask = np.zeros_like(img)
        aug = self.transform(image=img, mask=mask)
        x  = aug['image']          # (1,H,W)
        y  = aug['mask'].unsqueeze(0).float()  # (1,H,W)
        return x.float(), y

# -----------------------------------------------------------------------------
# Loss & metrics
# -----------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        num = 2*((p*targets).sum((2,3))) + self.smooth
        den = p.sum((2,3)) + targets.sum((2,3)) + self.smooth
        return (1 - num/den).mean()

def iou_score(logits, targets, thresh=.5):
    preds = (torch.sigmoid(logits) > thresh).float()
    inter = (preds*targets).sum((2,3))
    union = preds.sum((2,3))+targets.sum((2,3))-inter
    return (inter/(union+1e-7)).mean().item()

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); d, i = 0., 0.
    for img, msk in loader:
        img, msk = img.to(device), msk.to(device)
        logits = model(img)
        d += 1 - DiceLoss()(logits, msk).item()
        i += iou_score(logits, msk)
    n = len(loader)
    return d/n, i/n

# -----------------------------------------------------------------------------
# Training / inference helpers
# -----------------------------------------------------------------------------

def _collect_pairs(img_dir: Path, msk_dir: Path | None):
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.mha'}
    img_paths, msk_paths = [], []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        img_paths.append(p)
        if msk_dir is not None:
            msk_paths.append(msk_dir/p.name)
    return img_paths, msk_paths if msk_dir is not None else None


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_train = pathlib.Path('./train')
    root_val = pathlib.Path('./val')
    train_imgs, train_msks = _collect_pairs(root_train/'images', root_train/'masks')

    # If a separate val folder exists, use it; else random split
    val_folder = root_val/'images'
    if val_folder.exists():
        val_imgs, val_msks   = _collect_pairs(val_folder, root_val/'masks')
        train_ds = FetalACDataset(train_imgs, train_msks, train=True)
        val_ds   = FetalACDataset(val_imgs,   val_msks,   train=False)
    else:
        full_ds = FetalACDataset(train_imgs, train_msks, train=True)
        val_len = int(len(full_ds)*0.1)
        train_len = len(full_ds)-val_len
        train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = AttentionASPPUNet().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit  = DiceLoss()

    best = 0.
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train(); running = 0.
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}")
        for img, msk in pbar:
            img, msk = img.to(device), msk.to(device)
            opt.zero_grad(); logit = model(img); loss = crit(logit, msk)
            loss.backward(); opt.step(); running += loss.item()
            pbar.set_postfix(loss=f"{running/(pbar.n+1):.4f}")
        sch.step()
        d_val, i_val = evaluate(model, val_loader, device)
        print(f"\nVal Dice {d_val:.4f} | IoU {i_val:.4f}")
        if d_val>best:
            best=d_val; torch.save(model.state_dict(), out/'best.pt')
            print(f"[✓] saved best model (Dice {best:.4f})")

@torch.inference_mode()
def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionASPPUNet(); model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    img_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.mha'}
    transform = Compose([
        CLAHE(1.0,(8,8),always_apply=True),
        MedianBlur(3,always_apply=True),
        ToFloat(max_value=255.0),
        ToTensorV2(),
    ])
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in exts: continue
        # read
        if img_path.suffix.lower()=='.mha':
            if sitk is None: raise RuntimeError('SimpleITK required for .mha')
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
            if arr.ndim==3: arr=arr[arr.shape[0]//2]  # single slice
        else:
            arr = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        sample = transform(image=arr)
        x = sample['image'].unsqueeze(0).to(device)
        mask = (torch.sigmoid(model(x)).cpu().numpy()>0.5).astype(np.uint8)[0,0]
        # save
        save_path = out_dir/img_path.name.replace(img_path.suffix, '.png')
        cv2.imwrite(str(save_path), mask*255)
        print(f"[✓] {save_path.name} saved")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _get_args():
    p = argparse.ArgumentParser('Attention‑ASPP U‑Net pipeline')
    sp = p.add_subparsers(dest='cmd', required=True)
    t = sp.add_parser('train')
    t.add_argument('--data_dir', required=True)
    t.add_argument('--output_dir', default='./checkpoints')
    t.add_argument('--epochs', type=int, default=120)
    t.add_argument('--batch_size', type=int, default=8)
    t.add_argument('--lr', type=float, default=3e-4)
    pr = sp.add_parser('predict')
    pr.add_argument('--weights', required=True)
    pr.add_argument('--input_dir', required=True)
    pr.add_argument('--out_dir', default='./preds')
    return p.parse_args()

if __name__=='__main__':
    args=_get_args(); torch.backends.cudnn.benchmark=True
    if args.cmd=='train': train(args)
    else: predict(args)
