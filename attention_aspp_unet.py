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
        self.att3 = AttentionBlock(in_g=base_ch*8, in_x=base_ch*4, inter_ch=base_ch*4)
        self.conv3 = DoubleConv(base_ch*8 + base_ch*4, base_ch*8)

        self.up2 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.att2 = AttentionBlock(in_g=base_ch*4, in_x=base_ch*2, inter_ch=base_ch*2)
        self.conv2 = DoubleConv(base_ch*4 + base_ch*2, base_ch*4)

        self.up1 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.att1 = AttentionBlock(in_g=base_ch*2, in_x=base_ch, inter_ch=base_ch)
        self.conv1 = DoubleConv(base_ch*2 + base_ch, base_ch*2)

        self.outc = nn.Conv2d(base_ch*2, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)          # 32
        x2 = self.down1(x1)       # 64
        x3 = self.down2(x2)       # 128
        x4 = self.down3(x3)       # 256

        center = self.aspp(x4)    # 512

        d3 = self.up3(center)
        x3_att = self.att3(g=d3, x=x3)
        d3 = torch.cat([d3, x3_att], dim=1)
        d3 = self.conv3(d3)

        d2 = self.up2(d3)
        x2_att = self.att2(g=d2, x=x2)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.conv2(d2)

        d1 = self.up1(d2)
        x1_att = self.att1(g=d1, x=x1)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.conv1(d1)

        out = self.outc(d1)
        return out


# -------------------------
#  Main Training Script
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class FetalAbdomenDataset(Dataset):
        def __init__(self, root):
            self.root = Path(root)
            self.img_paths = sorted((self.root/'images').glob('*.png'))
            self.mask_paths = [self.root/'masks'/p.name for p in self.img_paths]

        def __len__(self): return len(self.img_paths)

        def __getitem__(self, idx):
            img = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            img = img.astype(np.float32) / 255.0
            mask = (mask > 0).astype(np.float32)
            img = torch.from_numpy(img).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)
            return img, mask

    def dice_coef(pred, target, eps=1e-6):
        pred = (pred > 0.5).float()
        intersect = (pred * target).sum()
        return (2 * intersect + eps) / (pred.sum() + target.sum() + eps)

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

    train_ds = FetalAbdomenDataset(args.train_dir)
    val_ds = FetalAbdomenDataset(args.val_dir)
    print(f"✅ Loaded {len(train_ds)} training samples, {len(val_ds)} validation samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = AttentionASPPUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_dice = 0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(torch.sigmoid(pred), mask)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        dices, ious, hds = [], [], []
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                img, mask = img.to(device), mask.to(device)
                out = torch.sigmoid(model(img))
                dices.append(dice_coef(out, mask).item())
                ious.append(iou_score(out, mask).item())
                hds.append(hausdorff_distance(out, mask))

        avg_train_loss = np.mean(train_losses)
        avg_dice = np.nanmean(dices)
        avg_iou = np.nanmean(ious)
        avg_hd = np.nanmean(hds)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | HD: {avg_hd:.2f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))
            print("✅ Saved new best model")

if __name__ == '__main__':
    main()
