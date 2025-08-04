import argparse
import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff

# -------------------------
#  Model Components
# -------------------------
class DoubleConv(nn.Module):
    """(Conv ⇒ BN ⇒ ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionBlock(nn.Module):
    """Attention Gate from "Attention U-Net" (Oktay et al.)"""

    def __init__(self, in_g, in_x, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_g, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_x, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)) for r in rates]
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(rates), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        x = torch.cat(feats, dim=1)
        return self.project(x)


class AttentionASPPUNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=1, base_ch=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch*2, base_ch*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch*4, base_ch*8))

        self.aspp = ASPP(base_ch*8, base_ch*16)

        self.up3 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.att3 = AttentionBlock(in_g=base_ch*8, in_x=base_ch*8, inter_ch=base_ch*4)
        self.conv3 = DoubleConv(base_ch*16, base_ch*8)

        self.up2 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.att2 = AttentionBlock(in_g=base_ch*4, in_x=base_ch*4, inter_ch=base_ch*2)
        self.conv2 = DoubleConv(base_ch*8, base_ch*4)

        self.up1 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.att1 = AttentionBlock(in_g=base_ch*2, in_x=base_ch*2, inter_ch=base_ch)
        self.conv1 = DoubleConv(base_ch*4, base_ch*2)

        self.outc = nn.Conv2d(base_ch*2, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)          # (bs, 32)
        x2 = self.down1(x1)       # (bs, 64)
        x3 = self.down2(x2)       # (bs, 128)
        x4 = self.down3(x3)       # (bs, 256)

        center = self.aspp(x4)    # (bs, 512)

        d3 = self.up3(center)
        x4_att = self.att3(g=d3, x=x4)
        d3 = torch.cat([d3, x4_att], dim=1)
        d3 = self.conv3(d3)

        d2 = self.up2(d3)
        x3_att = self.att2(g=d2, x=x3)
        d2 = torch.cat([d2, x3_att], dim=1)
        d2 = self.conv2(d2)

        d1 = self.up1(d2)
        x2_att = self.att1(g=d1, x=x2)
        d1 = torch.cat([d1, x2_att], dim=1)
        d1 = self.conv1(d1)

        out = self.outc(d1)
        return out

# -------------------------
#  Dataset & Augmentation
# -------------------------
class FetalAbdomenDataset(Dataset):
    """Assumes directory with 'images' and 'masks' subfolders (grayscale PNG)."""
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.img_paths = sorted(glob.glob(str(self.root / 'images' / '*.png')))
        self.mask_paths = [str(self.root / 'masks' / Path(p).name) for p in self.img_paths]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        # To tensor
        img = torch.from_numpy(img).unsqueeze(0)  # (1,H,W)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

# -------------------------
#  Metrics
# -------------------------

def dice_coef(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersect = (pred * target).sum()
    return (2*intersect + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersect = (pred * target).sum()
    union = pred.sum() + target.sum() - intersect
    return (intersect + eps) / (union + eps)

def hausdorff_distance(pred, target):
    pred = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    target = target.squeeze().cpu().numpy().astype(np.uint8)
    if pred.sum() == 0 or target.sum() == 0:
        return np.nan
    pred_pts = np.column_stack(np.where(pred))
    target_pts = np.column_stack(np.where(target))
    return max(directed_hausdorff(pred_pts, target_pts)[0],
               directed_hausdorff(target_pts, pred_pts)[0])

# -------------------------
#  Training & Validation
# -------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for img, mask in tqdm(loader, leave=False):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        out = model(img)
        out = torch.sigmoid(out)
        loss = criterion(out, mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate(model, loader, device):
    model.eval()
    dices, ious, hds = [], [], []
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            out = torch.sigmoid(model(img))
            dices.append(dice_coef(out, mask).item())
            ious.append(iou_score(out, mask).item())
            hds.append(hausdorff_distance(out, mask))
    return np.nanmean(dices), np.nanmean(ious), np.nanmean(hds)


# -------------------------
#  Main script
# -------------------------

def main(args):
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset splits
    train_ds = FetalAbdomenDataset(args.train_dir)
    val_ds   = FetalAbdomenDataset(args.val_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = AttentionASPPUNet(in_ch=1, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_dice = 0
    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dice, iou, hd = validate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, Dice={dice:.4f}, IoU={iou:.4f}, HD={hd:.2f}")
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), args.out_dir / 'best_model.pth')
            print("✔️  Saved new best model")

    print("Training finished. Best Dice:", best_dice)


def _parse_args():
    parser = argparse.ArgumentParser(description="Attention-ASPP U-Net for Fetal Abdomen Segmentation")
    parser.add_argument('--train_dir', type=Path, required=True, help='path to training dataset')
    parser.add_argument('--val_dir',   type=Path, required=True, help='path to validation dataset')
    parser.add_argument('--out_dir',   type=Path, default=Path('./checkpoints'))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main(args)
