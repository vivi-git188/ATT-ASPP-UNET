# attention_aspp_unet_pipeline.py
"""
End‑to‑end pipeline for fetal abdominal circumference (AC) segmentation on the
ACOUSLIC‑AI dataset using an Attention‑ASPP U‑Net backbone.

Main entry points
-----------------
# Train
python attention_aspp_unet_pipeline.py train \
    --data_dir /path/to/ACOUSLIC-AI \
    --output_dir ./checkpoints \
    --epochs 120 --batch_size 8 --lr 3e-4

# Inference (produces a .mha mask per study in <out_dir>)
python attention_aspp_unet_pipeline.py predict \
    --weights ./checkpoints/best.pt \
    --input_dir /input/images/stacked-fetal-ultrasound \
    --out_dir /output/images

The script expects the ACOUSLIC folder structure:
ACOUSLIC-AI/
 ├─ imagesTr/  (training volumes *.mha)
 ├─ labelsTr/  (binary masks *.mha)
 ├─ imagesVal/ (optional validation set)
 └─ imagesTs/  (test set)

The network is **2‑D** – it sees a single frame at a time but uses ASPP to
broaden the receptive field. A simple 3‑frame pseudo‑RGB stack variant can be
implemented by replacing the Dataset's _get_slice() method.

Dependencies
------------
* Python ≥3.9
* torch ≥2.1
* torchvision ≥0.16
* albumentations
* opencv‑python
* SimpleITK
* tqdm

"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import (Compose, HorizontalFlip, RandomGamma, CLAHE,
                            MedianBlur, ToFloat)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Model components
# ----------------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, p: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=p if p is not None else k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling used as the bridge between encoder/decoder"""
    def __init__(self, in_c: int, out_c: int = 256, rates: Tuple[int, int, int] = (6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)),
            *[
                nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ) for r in rates
            ],
        ])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_c * (len(self.blocks) + 1), out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [b(x) for b in self.blocks]
        img_pool = self.image_pool(x)
        img_pool = F.interpolate(img_pool, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(img_pool)
        x = torch.cat(feats, dim=1)
        return self.project(x)


class AttentionGate(nn.Module):
    """Attention gate from Attention U‑Net (Oktay et al., 2018)"""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, use_attention: bool):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.att = AttentionGate(F_g=out_c, F_l=out_c, F_int=out_c // 2) if use_attention else nn.Identity()
        self.conv = nn.Sequential(ConvBNReLU(in_c, out_c), ConvBNReLU(out_c, out_c))

    def forward(self, g, x):
        g = self.up(g)
        x = self.att(g, x)
        g = torch.cat([x, g], dim=1)
        return self.conv(g)


class AttentionASPPUNet(nn.Module):
    """Attention U‑Net enhanced with ASPP in the bottleneck"""
    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_c: int = 32):
        super().__init__()
        # Encoder
        self.down1 = nn.Sequential(ConvBNReLU(in_channels, base_c), ConvBNReLU(base_c, base_c))
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(ConvBNReLU(base_c, base_c * 2), ConvBNReLU(base_c * 2, base_c * 2))
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = nn.Sequential(ConvBNReLU(base_c * 2, base_c * 4), ConvBNReLU(base_c * 4, base_c * 4))
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = nn.Sequential(ConvBNReLU(base_c * 4, base_c * 8), ConvBNReLU(base_c * 8, base_c * 8))
        self.pool4 = nn.MaxPool2d(2)
        # Bridge ASPP
        self.aspp = ASPP(base_c * 8, base_c * 16)
        # Decoder with attention gates
        self.up4 = UpBlock(base_c * 16, base_c * 8, use_attention=True)
        self.up3 = UpBlock(base_c * 8, base_c * 4, use_attention=True)
        self.up2 = UpBlock(base_c * 4, base_c * 2, use_attention=True)
        self.up1 = UpBlock(base_c * 2, base_c, use_attention=False)
        # Output
        self.out_conv = nn.Conv2d(base_c, num_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))
        # Bridge
        bridge = self.aspp(self.pool4(x4))
        # Decoder
        d4 = self.up4(bridge, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        return self.out_conv(d1)


# ----------------------------------------------------------------------------
# Dataset & transforms
# ----------------------------------------------------------------------------
class FetalACDataset(Dataset):
    """ACOUSLIC‑AI single‑frame dataset (2‑D slices)."""

    def __init__(self, img_paths: List[Path], msk_paths: List[Path], train: bool = True):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.train = train
        self.transform = self._get_transform()

    def _get_transform(self):
        if self.train:
            return Compose([
                CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), always_apply=True),
                MedianBlur(blur_limit=3, always_apply=True),
                HorizontalFlip(p=0.5),
                RandomGamma(gamma_limit=(80, 120), p=0.5),
                ToFloat(max_value=255.0),
                ToTensorV2(),
            ])
        else:
            return Compose([
                CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), always_apply=True),
                MedianBlur(blur_limit=3, always_apply=True),
                ToFloat(max_value=255.0),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(self.img_paths[idx]))).astype(np.uint8)
        msk = sitk.GetArrayFromImage(sitk.ReadImage(str(self.msk_paths[idx]))).astype(np.uint8)
        # Assume (H, W) for each; if 3‑D, take middle slice ⚠️
        if img.ndim == 3:
            mid = img.shape[0] // 2
            img, msk = img[mid], msk[mid]
        augmented = self.transform(image=img, mask=msk)
        x, y = augmented["image"], augmented["mask"].unsqueeze(0)
        return x.float(), y.float()


# ----------------------------------------------------------------------------
# Losses & metrics
# ----------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        targets = targets
        num = 2 * (logits * targets).sum(dim=(2, 3)) + self.smooth
        den = logits.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        loss = 1 - num / den
        return loss.mean()


def iou_score(logits, targets, thresh: float = 0.5):
    preds = (torch.sigmoid(logits) > thresh).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    return (inter / (union + 1e-7)).mean().item()


# ----------------------------------------------------------------------------
# Training & validation loops
# ----------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    dice_meter, iou_meter = 0.0, 0.0
    for imgs, msks in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        logits = model(imgs)
        dice_meter += 1 - DiceLoss()(logits, msks).item()
        iou_meter += iou_score(logits, msks)
    n = len(loader)
    return dice_meter / n, iou_meter / n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Collect all .mha pairs (frame‑wise) → flatten into 2‑D slices list
    img_dir = Path(args.data_dir) / "imagesTr"
    msk_dir = Path(args.data_dir) / "labelsTr"

    img_paths, msk_paths = [], []
    for img_p in sorted(img_dir.glob("*.mha")):
        msk_p = msk_dir / img_p.name
        if not msk_p.exists():
            continue
        img_paths.append(img_p)
        msk_paths.append(msk_p)

    dataset = FetalACDataset(img_paths, msk_paths, train=True)
    val_len = int(len(dataset) * 0.1)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = AttentionASPPUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DiceLoss()

    best_dice = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, msks in pbar:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, msks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / (pbar.n + 1):.4f}")
        scheduler.step()
        # Validation
        dice_val, iou_val = evaluate(model, val_loader, device)
        print(f"\nValidation → Dice: {dice_val:.4f} | IoU: {iou_val:.4f}")
        if dice_val > best_dice:
            best_dice = dice_val
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"[✓] New best model saved with Dice {best_dice:.4f}")


@torch.inference_mode()
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    model = AttentionASPPUNet()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over each volume, slice‑wise inference
    for vol_path in sorted(input_dir.glob("*.mha")):
        vol = sitk.ReadImage(str(vol_path))
        arr = sitk.GetArrayFromImage(vol).astype(np.uint8)  # (N, H, W)
        pred_stack = np.zeros_like(arr, dtype=np.uint8)

        transform = Compose([
            CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), always_apply=True),
            MedianBlur(blur_limit=3, always_apply=True),
            ToFloat(max_value=255.0),
            ToTensorV2(),
        ])
        for i, sl in enumerate(arr):
            sample = transform(image=sl)
            x = sample["image"].unsqueeze(0).to(device)
            logits = model(x)
            mask = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(np.uint8)[0, 0]
            pred_stack[i] = mask * 2  # 0 background, 2 abdomen for ITK‑SNAP palette

        # Save 3‑D stack as .mha
        out_img = sitk.GetImageFromArray(pred_stack)
        out_img.CopyInformation(vol)
        sitk.WriteImage(out_img, str(out_dir / vol_path.name))
        print(f"[✓] Saved prediction for {vol_path.name}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Attention‑ASPP U‑Net pipeline")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Train args
    p_train = subparsers.add_parser("train")
    p_train.add_argument("--data_dir", type=str, required=True)
    p_train.add_argument("--output_dir", type=str, default="./checkpoints")
    p_train.add_argument("--epochs", type=int, default=120)
    p_train.add_argument("--batch_size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=3e-4)

    # Predict args
    p_pred = subparsers.add_parser("predict")
    p_pred.add_argument("--weights", type=str, required=True)
    p_pred.add_argument("--input_dir", type=str, required=True)
    p_pred.add_argument("--out_dir", type=str, default="./preds")

    return parser.parse_args()


def main():
    args = _parse_args()
    torch.backends.cudnn.benchmark = True
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)
    else:
        raise ValueError("Unsupported command")


if __name__ == "__main__":
    main()
