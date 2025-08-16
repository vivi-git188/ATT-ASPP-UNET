# -*- coding: utf-8 -*-
"""
Attention‑ASPP‑UNet training / inference script – now with **two‑stage curriculum**
---------------------------------------------------------------------------
Stage‑A (main):
  • Train on mostly positive (has‐mask) frames – a handful of empty frames is
    fine.  Objective is high recall / avoid predicting all‑empty.
  • Output ckpt_main/best.pt

Stage‑B (finetune):
  • Continue from A‑weights, add a *small* number of negative (empty) frames –
    preferably the ones the model still confuses (hard‑negatives).
  • 10–20 epochs, lower LR, empty frames are only trained with
        BCE × NEG_BCE_W   (NEG_BCE_W≈0.1‑0.2)
    while positive frames keep Dice + BCE (and optional edge loss).
  • Output ckpt_finetune/best.pt   ← **final model** for deployment / paper.

Typical commands
----------------
# Stage‑A
python attention_aspp_unet_pipeline_stage.py train \
       --stage main \
       --train_dir train_png_main \
       --val_dir   val_png_main \
       --output_dir checkpoints

# Stage‑B
python attention_aspp_unet_pipeline_stage.py train \
       --stage finetune \
       --train_dir train_png_main \
       --neg_dir   train_png_finetune  # <- contains hard‑neg images (no masks) \
       --pretrained checkpoints/ckpt_main/best.pt \
       --epochs 15 --lr 1e-4 \
       --neg_bce_w 0.15 \
       --output_dir checkpoints

Other sub‑commands (predict / calibrate) keep the same CLI.
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import (CLAHE, Compose, HorizontalFlip, MedianBlur, RandomGamma,
                            RandomBrightnessContrast, Resize, ToFloat)
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

# --------------------------- Global Hyper‑params --------------------------- #
SEED = 2025
IMG_SIZE = 512
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 15

# Loss switches
LOSS_TYPE = "combo"        # "combo"(Dice+BCE) | "tversky"
TV_ALPHA, TV_BETA = 0.7, 0.3

# --------------------------- Reproducibility ------------------------------ #

def set_seed(seed: int):
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass
    # Albumentations 2.x：不再用全局 set_seed；在 Compose 中传 seed 即可
    try:
        import albumentations as A
        if hasattr(A, "set_seed"):
            A.set_seed(seed)  # 若旧版仍有，顺带调用；新版无则忽略
    except Exception:
        pass



def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --------------------------- Model definition ----------------------------- #


class ConvBNReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_c: int, out_c: int = 256, rates: Tuple[int, int, int] = (6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True)
                ),
                *[
                    nn.Sequential(
                        nn.Conv2d(in_c, out_c, 3, padding=r, dilation=r, bias=False),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(True),
                    )
                    for r in rates
                ],
            ]
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_c * 5, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [b(x) for b in self.blocks]
        pooled = F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False)
        feats.append(pooled)
        return self.project(torch.cat(feats, dim=1))


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


class DummyAttention(nn.Module):
    def forward(self, g, x):
        return x


class UpBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, use_att: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.att = AttentionGate(out_c, out_c, out_c // 2) if use_att else DummyAttention()
        self.conv = nn.Sequential(ConvBNReLU(in_c, out_c), ConvBNReLU(out_c, out_c))

    def forward(self, g, x):
        g = self.up(g)
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.att(g, x)
        return self.conv(torch.cat([x, g], dim=1))


class AttentionASPPUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_c: int = 32):
        super().__init__()
        self.d1 = nn.Sequential(ConvBNReLU(in_channels, base_c), ConvBNReLU(base_c, base_c))
        self.p1 = nn.MaxPool2d(2)
        self.d2 = nn.Sequential(ConvBNReLU(base_c, base_c * 2), ConvBNReLU(base_c * 2, base_c * 2))
        self.p2 = nn.MaxPool2d(2)
        self.d3 = nn.Sequential(ConvBNReLU(base_c * 2, base_c * 4), ConvBNReLU(base_c * 4, base_c * 4))
        self.p3 = nn.MaxPool2d(2)
        self.d4 = nn.Sequential(ConvBNReLU(base_c * 4, base_c * 8), ConvBNReLU(base_c * 8, base_c * 8))
        self.p4 = nn.MaxPool2d(2)
        self.bridge = ASPP(base_c * 8, base_c * 16)
        self.u4 = UpBlock(base_c * 16, base_c * 8)
        self.u3 = UpBlock(base_c * 8, base_c * 4)
        self.u2 = UpBlock(base_c * 4, base_c * 2)
        self.u1 = UpBlock(base_c * 2, base_c, use_att=False)
        self.out_conv = nn.Conv2d(base_c, num_classes, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))
        b = self.bridge(self.p4(x4))
        d4 = self.u4(b, x4)
        d3 = self.u3(d4, x3)
        d2 = self.u2(d3, x2)
        d1 = self.u1(d2, x1)
        return self.out_conv(d1)

# --------------------------- Dataset & Augment ----------------------------- #


def SafeCLAHE(*args, **kwargs):
    kwargs.pop("always_apply", None)
    return CLAHE(*args, **kwargs)


def SafeMedianBlur(*args, **kwargs):
    kwargs.pop("always_apply", None)
    return MedianBlur(*args, **kwargs)


class FetalACDataset(Dataset):
    """Return (image_tensor, mask_tensor[0‑1])"""

    def __init__(self, img_paths: List[Path], msk_paths: Optional[List[Optional[Path]]], train: bool = True):
        assert len(img_paths) == len(msk_paths)
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.train = train
        self.transform = self._build_transform()

    # ----------- Albumentations pipeline ----------- #
    def _build_transform(self):
        from albumentations import Affine, ElasticTransform

        t_train = [
            Resize(IMG_SIZE, IMG_SIZE),
            HorizontalFlip(p=0.5),
            Affine(scale=(0.92, 1.08), rotate=(-7, 7), translate_percent=(0.0, 0.02), shear=0, p=0.7),
            RandomGamma((80, 120), p=0.3),
            RandomBrightnessContrast(0.1, 0.1, p=0.3),
            ElasticTransform(alpha=8, sigma=3, p=0.25),
            SafeCLAHE(1.0, (8, 8)),
            SafeMedianBlur(3),
            ToFloat(max_value=255.0),
            ToTensorV2(),
        ]

        t_val = [
            Resize(IMG_SIZE, IMG_SIZE),
            SafeCLAHE(1.0, (8, 8)),
            SafeMedianBlur(3),
            ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
        return Compose(t_train if self.train else t_val)

    # ----------- helper I/O ----------- #
    @staticmethod
    def _read_img(path: Path):
        if path.suffix.lower() == ".mha":
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
            if img.ndim == 3:
                img = img[img.shape[0] // 2]
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img.astype(np.uint8)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self._read_img(self.img_paths[idx])
        msk = np.zeros_like(img)
        if self.msk_paths and self.msk_paths[idx] is not None:
            msk = self._read_img(self.msk_paths[idx])
        sample = self.transform(image=img, mask=msk)
        return sample["image"].float(), (sample["mask"].unsqueeze(0).float() / 255.0)

# --------------------------- Losses --------------------------------------- #


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        num = 2 * (p * targets).sum((2, 3)) + self.smooth
        den = p.sum((2, 3)) + targets.sum((2, 3)) + self.smooth
        return (1 - num / den).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p * targets).sum((2, 3))
        fp = (p * (1 - targets)).sum((2, 3))
        fn = ((1 - p) * targets).sum((2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


class ComboLoss(nn.Module):
    """Dice + BCE"""

    def __init__(self, dice_w: float = 1.0, bce_w: float = 1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.dw, self.bw = dice_w, bce_w

    def forward(self, logits, targets):
        d = self.dice(logits, targets)
        b = F.binary_cross_entropy_with_logits(logits, targets)
        return self.dw * d + self.bw * b


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        dtype, device = p.dtype, p.device
        kx, ky = self.kx.to(device=device, dtype=dtype), self.ky.to(device=device, dtype=dtype)
        t = targets.to(dtype=dtype, device=device)
        gx_p = F.conv2d(p, kx, padding=1)
        gy_p = F.conv2d(p, ky, padding=1)
        gx_t = F.conv2d(t, kx, padding=1)
        gy_t = F.conv2d(t, ky, padding=1)
        grad_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + 1e-8)
        grad_t = torch.sqrt(gx_t ** 2 + gy_t ** 2 + 1e-8)
        return F.l1_loss(grad_p, grad_t)


# --------------------------- Metrics -------------------------------------- #

def iou_score(logits, targets, thresh: float = 0.5):
    preds = (torch.sigmoid(logits) > thresh).float()
    inter = (preds * targets).sum((2, 3))
    union = preds.sum((2, 3)) + targets.sum((2, 3)) - inter
    return (inter / (union + 1e-7)).mean().item()

# --------------------------- Train / Val ---------------------------------- #


def evaluate(model, loader, device):
    model.eval()
    dice, iou = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
        dice += 1 - DiceLoss()(logits, y).item()
        iou += iou_score(logits, y)
    return dice / len(loader), iou / len(loader)


# ---------- helper for loss weighting on empty / non‑empty --------------- #

def build_criterion(args, base_loss, edge_loss):
    """Return a closure taking (logits, targets)"""

    def criterion_fn(logits, targets):
        logits = logits.float()
        targets = targets.float()
        B = targets.shape[0]
        # is_empty shape (B, 1, 1, 1) so it broadcasting properly
        is_empty = (targets.sum(dim=(2, 3), keepdim=True) == 0).float()

        # BCE part – different weight on empty masks during finetune
        bce_weight = torch.ones_like(targets)
        if args.stage == "finetune":
            bce_weight = torch.where(is_empty == 1, args.neg_bce_w, 1.0)
        bce = F.binary_cross_entropy_with_logits(logits, targets, weight=bce_weight)

        # Dice + Edge only for positive samples (if any)
        pos_idx = (is_empty.view(B) == 0).nonzero(as_tuple=True)[0]
        dice = torch.tensor(0.0, device=logits.device)
        edge = torch.tensor(0.0, device=logits.device)
        if len(pos_idx) > 0:
            dice = base_loss(logits[pos_idx], targets[pos_idx])
            if args.edge_w > 0:
                edge = edge_loss(logits[pos_idx], targets[pos_idx]) * args.edge_w

        return dice + bce + edge

    return criterion_fn


# --------------------------- Main train routine --------------------------- #

def train(args):
    set_seed(args.seed)

    # ---------------- Datasets ---------------- #
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir) if args.val_dir else None
    neg_dir = Path(args.neg_dir) if args.neg_dir else None

    def collect_pairs(img_dir: Path, msk_dir: Optional[Path]):
        exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp", ".mha"}
        imgs, msks = [], []
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            imgs.append(p)
            q = msk_dir / p.name if msk_dir else None
            msks.append(q if (q and q.exists()) else None)
        return imgs, msks

    # positive images (with GT mask)
    train_imgs, train_msks = collect_pairs(train_dir / "images", train_dir / "masks")

    # optional negatives
    if neg_dir and neg_dir.exists():
        neg_imgs, _ = collect_pairs(neg_dir / "images", None)
        train_imgs += neg_imgs
        train_msks += [None] * len(neg_imgs)

    # validation set
    if val_dir and (val_dir / "images").exists():
        val_imgs, val_msks = collect_pairs(val_dir / "images", val_dir / "masks")
        train_ds = FetalACDataset(train_imgs, train_msks, train=True)
        val_ds = FetalACDataset(val_imgs, val_msks, train=False)
    else:
        full_ds = FetalACDataset(train_imgs, train_msks, train=True)
        val_len = max(1, int(0.1 * len(full_ds)))
        train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_len, val_len])

    # ---------------- Dataloaders ---------------- #
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
        worker_init_fn=seed_worker,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        generator=g,
        worker_init_fn=seed_worker,
    )

    # ---------------- Model & Optim ---------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionASPPUNet(base_c=args.base_c).to(device)

    if args.stage == "finetune":
        assert args.pretrained is not None, "Finetune stage requires --pretrained weights from Stage‑A"
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
        print(f"[i] Loaded Stage‑A weights from {args.pretrained}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    # Warmup + Cosine for Stage‑A, plain cosine for Stage‑B (few epochs)
    total_epochs = args.epochs
    warmup_epochs = 0 if args.stage == "finetune" else max(1, int(0.05 * total_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs),
                cosine,
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine

    # losses
    base_loss = ComboLoss() if LOSS_TYPE == "combo" else TverskyLoss(TV_ALPHA, TV_BETA)
    edge_loss = EdgeLoss()
    criterion_fn = build_criterion(args, base_loss, edge_loss)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ---------------- Checkpoint dir ---------------- #
    stage_dir = "ckpt_main" if args.stage == "main" else "ckpt_finetune"
    out_dir = Path(args.output_dir) / stage_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_path = out_dir / f"best_{ts}.pt"
    best_dice = 0.0
    no_improve = 0

    # ---------------- Training loop ---------------- #
    for ep in range(1, total_epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{total_epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / (pbar.n + 1):.4f}")
        scheduler.step()

        dice, iou = evaluate(model, val_loader, device)
        print(f"\n[Val] Dice: {dice:.4f} | IoU: {iou:.4f}")

        if dice > best_dice:
            best_dice = dice
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"[✓] Best model saved → {best_path}")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print("[EarlyStop] stopping – no improvement.")
                break


# --------------------------- Argparse ------------------------------------- #

def get_args():
    p = argparse.ArgumentParser("Attention‑ASPP‑UNet two‑stage trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- train ---- #
    t = sub.add_parser("train")
    t.add_argument("--stage", choices=["main", "finetune"], default="main", help="training stage")
    t.add_argument("--seed", type=int, default=SEED)
    t.add_argument("--train_dir", required=True, help="dir with images/ & masks/ (positive frames)")
    t.add_argument("--neg_dir", help="dir with images/ only (negative frames)")
    t.add_argument("--val_dir", help="optional dir for validation set")
    t.add_argument("--output_dir", default="./checkpoints")
    t.add_argument("--pretrained", help="Stage‑B: path to Stage‑A best.pt")
    t.add_argument("--epochs", type=int, default=120)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--base_c", type=int, default=48)
    t.add_argument("--edge_w", type=float, default=0.05, help="edge loss weight")
    t.add_argument("--neg_bce_w", type=float, default=0.15, help="BCE weight for empty masks in finetune")

    # ---- predict (unchanged) ---- #
    pr = sub.add_parser("predict")
    pr.add_argument("--seed", type=int, default=SEED)
    pr.add_argument("--weights", required=True)
    pr.add_argument("--input_dir", required=True)
    pr.add_argument("--out_dir", default="./preds")
    pr.add_argument("--base_c", type=int, default=48)

    # ---- calibrate (unchanged) ---- #
    ca = sub.add_parser("calibrate")
    ca.add_argument("--seed", type=int, default=SEED)
    ca.add_argument("--weights", required=True)
    ca.add_argument("--val_dir", required=True)
    ca.add_argument("--output_dir", default="./checkpoints")
    ca.add_argument("--base_c", type=int, default=48)

    return p.parse_args()


# --------------------------- entry‑point ---------------------------------- #

if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.benchmark = True

    if args.cmd == "train":
        train(args)
    else:
        raise NotImplementedError("Only the train pipeline has been updated for two‑stage training. Use the original script for predict/calibrate (or port them similarly if needed).")
