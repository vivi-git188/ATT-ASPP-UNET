import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from albumentations import CLAHE, Compose, HorizontalFlip, MedianBlur, RandomGamma, ToFloat
from albumentations.pytorch import ToTensorV2

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

# --- Patch: remove deprecated "always_apply" ---
def SafeCLAHE(*args, **kwargs):
    kwargs.pop("always_apply", None)
    return CLAHE(*args, **kwargs)

def SafeMedianBlur(*args, **kwargs):
    kwargs.pop("always_apply", None)
    return MedianBlur(*args, **kwargs)

# --- Model ---
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=k//2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class ASPP(nn.Module):
    def __init__(self, in_c, out_c=256, rates=(6,12,18)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True)),
            *[nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=r, dilation=r, bias=False),
                            nn.BatchNorm2d(out_c), nn.ReLU(True)) for r in rates]
        ])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_c, out_c, 1, bias=False),
                                  nn.BatchNorm2d(out_c), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_c*5, out_c, 1, bias=False),
                                     nn.BatchNorm2d(out_c), nn.ReLU(True), nn.Dropout(0.1))
    def forward(self, x):
        h, w = x.shape[2:]
        feats = [b(x) for b in self.blocks]
        pooled = F.interpolate(self.pool(x), size=(h, w), mode='bilinear', align_corners=False)
        feats.append(pooled)
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
        return x * self.psi(psi)

class DummyAttention(nn.Module):
    def forward(self, g, x):
        return x


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, use_att=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.att = AttentionGate(out_c, out_c, out_c // 2) if use_att else DummyAttention()
        self.conv = nn.Sequential(ConvBNReLU(in_c, out_c), ConvBNReLU(out_c, out_c))

    def forward(self, g, x):
        g = self.up(g)
        if g.shape[-2:] != x.shape[-2:]:  # 防止尺寸不匹配
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.att(g, x)
        return self.conv(torch.cat([x, g], dim=1))

class AttentionASPPUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_c=32):
        super().__init__()
        self.d1 = nn.Sequential(ConvBNReLU(in_channels, base_c), ConvBNReLU(base_c, base_c))
        self.p1 = nn.MaxPool2d(2)
        self.d2 = nn.Sequential(ConvBNReLU(base_c, base_c*2), ConvBNReLU(base_c*2, base_c*2))
        self.p2 = nn.MaxPool2d(2)
        self.d3 = nn.Sequential(ConvBNReLU(base_c*2, base_c*4), ConvBNReLU(base_c*4, base_c*4))
        self.p3 = nn.MaxPool2d(2)
        self.d4 = nn.Sequential(ConvBNReLU(base_c*4, base_c*8), ConvBNReLU(base_c*8, base_c*8))
        self.p4 = nn.MaxPool2d(2)
        self.bridge = ASPP(base_c*8, base_c*16)
        self.u4 = UpBlock(base_c*16, base_c*8)
        self.u3 = UpBlock(base_c*8, base_c*4)
        self.u2 = UpBlock(base_c*4, base_c*2)
        self.u1 = UpBlock(base_c*2, base_c, use_att=False)
        self.out_conv = nn.Conv2d(base_c, num_classes, 1)
    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))
        b  = self.bridge(self.p4(x4))
        d4 = self.u4(b, x4)
        d3 = self.u3(d4, x3)
        d2 = self.u2(d3, x2)
        d1 = self.u1(d2, x1)
        return self.out_conv(d1)
# --- Dataset ---
class FetalACDataset(Dataset):
    def __init__(self, img_paths: List[Path], msk_paths: List[Path] | None, train=True):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.train = train
        self.transform = self._build_transform()

    def _build_transform(self):
        tf = [SafeCLAHE(1.0, (8,8)), SafeMedianBlur(3), ToFloat(max_value=255.0), ToTensorV2()]
        if self.train:
            tf.insert(2, HorizontalFlip(p=0.5))
            tf.insert(3, RandomGamma((80,120), p=0.5))
        return Compose(tf)

    def __len__(self): return len(self.img_paths)

    def _read(self, path: Path):
        if path.suffix.lower() == '.mha':
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
            if img.ndim == 3: img = img[img.shape[0] // 2]
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img.astype(np.uint8)

    def __getitem__(self, idx):
        img = self._read(self.img_paths[idx])
        msk = np.zeros_like(img)
        if self.msk_paths: msk = self._read(self.msk_paths[idx])
        sample = self.transform(image=img, mask=msk)
        return sample["image"].float(), (sample["mask"].unsqueeze(0).float() / 255.0)


# --- Utility ---
def collect_pairs(img_dir: Path, msk_dir: Path | None):
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.bmp', '.mha'}
    imgs, msks = [], []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in exts: continue
        imgs.append(p)
        if msk_dir: msks.append(msk_dir/p.name)
    return imgs, msks if msk_dir else None

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.): super().__init__(); self.smooth = smooth
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        num = 2*(p*targets).sum((2,3)) + self.smooth
        den = p.sum((2,3)) + targets.sum((2,3)) + self.smooth
        return (1 - num/den).mean()

def iou_score(logits, targets, thresh=0.5):
    preds = (torch.sigmoid(logits) > thresh).float()
    inter = (preds * targets).sum((2, 3))
    union = preds.sum((2, 3)) + targets.sum((2, 3)) - inter
    return (inter / (union + 1e-7)).mean().item()

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    dice, iou = 0., 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        dice += 1 - DiceLoss()(logits, y).item()
        iou  += iou_score(logits, y)
    return dice/len(loader), iou/len(loader)

# --- Train ---
def train(args):
    root_train = Path("train_png")
    root_val = Path("val_png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_imgs, train_msks = collect_pairs(root_train/"images", root_train/"masks")

    if (root_val/"images").exists():
        val_imgs, val_msks = collect_pairs(root_val/"images", root_val/"masks")
        train_ds = FetalACDataset(train_imgs, train_msks, train=True)
        val_ds   = FetalACDataset(val_imgs, val_msks, train=False)
    else:
        full_ds = FetalACDataset(train_imgs, train_msks, train=True)
        val_len = int(0.1 * len(full_ds))
        train_ds, val_ds = random_split(full_ds, [len(full_ds)-val_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = AttentionASPPUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DiceLoss()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0

    for ep in range(1, args.epochs+1):
        model.train(); running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss/(pbar.n+1):.4f}")
        scheduler.step()
        dice, iou = evaluate(model, val_loader, device)
        print(f"\n[Val] Dice: {dice:.4f} | IoU: {iou:.4f}")
        if dice > best:
            best = dice
            torch.save(model.state_dict(), out_dir / "best.pt")
            print("[✓] Best model saved.")

# --- Predict ---
from skimage.measure import label

@torch.inference_mode()
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionASPPUNet(); model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(input_dir.glob("*.mha")):
        arr3d = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))  # (N,H,W)
        pred_masks = []

        for i, sl in enumerate(arr3d):
            sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
            enhanced = cv2.medianBlur(clahe.apply(sl_u8), 3)
            sample = Compose([ToFloat(max_value=255.0), ToTensorV2()])(image=enhanced)
            x = sample["image"].unsqueeze(0).to(device)
            pred = torch.sigmoid(model(x)).cpu().numpy()[0, 0]
            mask = (pred > 0.5).astype(np.uint8)
            pred_masks.append(mask)

        pred_stack = np.stack(pred_masks)  # shape: (N,H,W)

        # 选帧：找最大面积的 mask 帧
        best_frame = max(range(pred_stack.shape[0]), key=lambda i: (pred_stack[i] > 0).sum())
        best_mask = pred_stack[best_frame]

        # 保存 .mha + .json
        write_output_mha_and_json(best_mask, best_frame, img_path, out_dir)


# --- CLI ---
def get_args():
    p = argparse.ArgumentParser("Attention-ASPP-UNet")
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--data_dir", required=True)
    t.add_argument("--output_dir", default="./checkpoints")
    t.add_argument("--epochs", type=int, default=120)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    pr = sp.add_parser("predict")
    pr.add_argument("--weights", required=True)
    pr.add_argument("--input_dir", required=True)
    pr.add_argument("--out_dir", default="./preds")
    return p.parse_args()


import SimpleITK as sitk
import json

def convert_mask_2d_to_3d(mask_2d: np.ndarray, frame: int, total_frames: int):
    mask_2d = (mask_2d > 0).astype(np.uint8) * 2  # 保持 ITK-SNAP 绿色
    mask_3d = np.zeros((total_frames, *mask_2d.shape), dtype=np.uint8)
    if 0 <= frame < total_frames:
        mask_3d[frame] = mask_2d
    return mask_3d


def write_output_mha_and_json(mask_2d: np.ndarray, frame: int, reference_mha_path: Path, output_dir: Path):
    case_name = reference_mha_path.stem
    case_out_dir = output_dir / case_name

    # 读取参考原图
    ref_img = sitk.ReadImage(str(reference_mha_path))
    total_frames = ref_img.GetSize()[2]  # 获取 Z 轴帧数（必须匹配）

    # 构造掩码 3D
    mask_3d = convert_mask_2d_to_3d(mask_2d, frame, total_frames)

    # 构造 sitk image + metadata
    output_img = sitk.GetImageFromArray(mask_3d)
    output_img.CopyInformation(ref_img)

    # 写入 .mha
    output_mha_path = case_out_dir / "images/fetal-abdomen-segmentation/output.mha"
    output_mha_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_img, str(output_mha_path))

    # 写入 JSON
    json_path = case_out_dir / "fetal-abdomen-frame-number.json"
    with open(json_path, "w") as f:
        json.dump(frame, f, indent=2)

    print(f"[✓] {case_name} → output.mha (frame {frame})")




if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.benchmark = True
    if args.cmd == "train": train(args)
    elif args.cmd == "predict": predict(args)
