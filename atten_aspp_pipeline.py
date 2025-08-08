
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# === Third-party optional dependency ===
try:
    import SimpleITK as sitk
except Exception as e:
    sitk = None

# Albumentations (image transforms)
from albumentations import CLAHE, Compose, HorizontalFlip, MedianBlur, RandomGamma, ToFloat
from albumentations.pytorch import ToTensorV2


# =========================
# Helpers for Albumentations
# =========================
def SafeCLAHE(*args, **kwargs):
    """Drop deprecated always_apply arg if present."""
    kwargs.pop("always_apply", None)
    return CLAHE(*args, **kwargs)


def SafeMedianBlur(*args, **kwargs):
    """Drop deprecated always_apply arg if present."""
    kwargs.pop("always_apply", None)
    return MedianBlur(*args, **kwargs)


# ==============
# Model blocks
# ==============
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_c, out_c=256, rates=(6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(True),
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
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.att(g, x)
        return self.conv(torch.cat([x, g], dim=1))


class AttentionASPPUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_c=32):
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


# ==============
# Dataset
# ==============
class FetalACDataset(Dataset):
    def __init__(self, img_paths: List[Path], msk_paths: Optional[List[Path]], train: bool = True):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.train = train
        self.transform = self._build_transform()

        if any(p.suffix.lower() == ".mha" for p in img_paths):
            assert sitk is not None, "Reading .mha requires SimpleITK. Please install SimpleITK."

    def _build_transform(self):
        tf = [SafeCLAHE(1.0, (8, 8)), SafeMedianBlur(3), ToFloat(max_value=255.0), ToTensorV2()]
        if self.train:
            tf.insert(2, HorizontalFlip(p=0.5))
            tf.insert(3, RandomGamma((80, 120), p=0.5))
        return Compose(tf)

    def __len__(self):
        return len(self.img_paths)

    def _read(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".mha":
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
            if img.ndim == 3:
                img = img[img.shape[0] // 2]
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img.astype(np.uint8)

    def __getitem__(self, idx):
        img = self._read(self.img_paths[idx])
        msk = np.zeros_like(img)
        if self.msk_paths:
            msk = self._read(self.msk_paths[idx])
        sample = self.transform(image=img, mask=msk)
        # image: (1,H,W) float in [0,1]; mask: (H,W) -> add channel
        return sample["image"].float(), (sample["mask"].unsqueeze(0).float() / 255.0)


# ==============
# Utils
# ==============
def collect_pairs(img_dir: Path, msk_dir: Optional[Path]) -> Tuple[List[Path], Optional[List[Path]]]:
    # PNG-only training by folder, but allow common 2D formats
    exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
    imgs, msks = [], []
    if not img_dir.exists():
        return imgs, (msks if msk_dir else None)
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        imgs.append(p)
        if msk_dir:
            msks.append(msk_dir / p.name)
    return imgs, (msks if msk_dir else None)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        num = 2 * (p * targets).sum((2, 3)) + self.smooth
        den = p.sum((2, 3)) + targets.sum((2, 3)) + self.smooth
        return (1 - num / den).mean()


def iou_score(logits, targets, thresh: float = 0.5):
    preds = (torch.sigmoid(logits) > thresh).float()
    inter = (preds * targets).sum((2, 3))
    union = preds.sum((2, 3)) + targets.sum((2, 3)) - inter
    return (inter / (union + 1e-7)).mean().item()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    dice, iou = 0.0, 0.0
    criterion = DiceLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        dice += 1 - criterion(logits, y).item()
        iou += iou_score(logits, y)
    n = max(1, len(loader))
    return dice / n, iou / n


# ==============
# Train (uses fixed train_png_best/ and val_png_best/)
# ==============
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_train = Path("train_png_best")
    root_val   = Path("val_png_best")
    print(f"[Train] train_png_best: {root_train.resolve()}")
    print(f"[Val]   val_png_best:   {root_val.resolve()}")

    train_imgs, train_msks = collect_pairs(root_train / "images", root_train / "masks")

    if (root_val / "images").exists():
        val_imgs, val_msks = collect_pairs(root_val / "images", root_val / "masks")
        train_ds = FetalACDataset(train_imgs, train_msks, train=True)
        val_ds   = FetalACDataset(val_imgs,   val_msks,   train=False)
    else:
        full_ds = FetalACDataset(train_imgs, train_msks, train=True)
        val_len = int(max(1, round(args.val_ratio * len(full_ds))))
        train_len = max(1, len(full_ds) - val_len)
        train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = AttentionASPPUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DiceLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / max(1, pbar.n):.4f}")
        scheduler.step()
        dice, iou = evaluate(model, val_loader, device)
        print(f"\n[Val] Dice: {dice:.4f} | IoU: {iou:.4f}")
        if dice > best:
            best = dice
            torch.save(model.state_dict(), out_dir / "best.pt")
            print("[✓] Best model saved.")


# ==============
# Predict
# ==============
def _connected_components_lcc(mask: np.ndarray) -> np.ndarray:
    """Keep largest connected component of a binary mask."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels <= 2:
        return mask
    areas = [np.sum(labels == k) for k in range(1, num_labels)]
    k = int(np.argmax(areas)) + 1
    return (labels == k).astype(np.uint8)


def _preprocess_slice(sl: np.ndarray) -> np.ndarray:
    """Normalize and enhance single 2D slice to uint8 [0,255]."""
    sl = sl.astype(np.float32)
    mn, mx = float(sl.min()), float(sl.max())
    if mx > mn:
        sl = (sl - mn) / (mx - mn)
    else:
        sl = np.zeros_like(sl, dtype=np.float32)
    u8 = (sl * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    u8 = clahe.apply(u8)
    u8 = cv2.medianBlur(u8, 3)
    return u8


def _tensor_from_u8(u8: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert uint8 image [H,W] to model tensor [1,1,H,W] in [0,1]."""
    aug = Compose([ToFloat(max_value=255.0), ToTensorV2()])
    x = aug(image=u8)["image"].unsqueeze(0).to(device)  # (1,1,H,W)
    return x


def _otsu_thresh(prob: np.ndarray) -> float:
    """Otsu threshold on probability map [0,1], returns scalar in [0,1]."""
    u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    # 全零或常数图像时，OTSU 不稳定，回退到 0.5
    if u8.max() == u8.min():
        return 0.5
    ret, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(ret / 255.0)



def _score_frames(probs: np.ndarray, mode: str, thresh: float, adaptive: bool, alpha: float, min_area: int):
    """
    probs: (M, H, W) float in [0,1]
    returns: scores (M,), bin_masks (list of np.uint8), thresholds (list of float)
    """
    M = probs.shape[0]
    scores = np.zeros(M, dtype=np.float64)
    bin_masks = []
    used_thresh = []

    for i in range(M):
        p = np.clip(probs[i], 0.0, 1.0)
        t = _otsu_thresh(p) if adaptive else thresh
        b = (p >= t).astype(np.uint8)

        if min_area > 0:
            if b.sum() < min_area:
                # too small -> suppress
                b[:] = 0

        if mode == "prob_sum":
            s = float(p.sum())
        elif mode == "bin_area":
            s = float(b.sum())
        elif mode == "hybrid":
            s = float(alpha * p.sum() + (1.0 - alpha) * b.sum())
        else:
            raise ValueError(f"Unknown frame_select mode: {mode}")

        scores[i] = s
        bin_masks.append(b)
        used_thresh.append(t)

    return scores, bin_masks, used_thresh


def convert_mask_2d_to_3d(mask_2d: np.ndarray, frame: int, total_frames: int) -> np.ndarray:
    mask_2d = (mask_2d > 0).astype(np.uint8) * 2  # 2 for ITK-SNAP green
    mask_3d = np.zeros((total_frames, *mask_2d.shape), dtype=np.uint8)
    if 0 <= frame < total_frames:
        mask_3d[frame] = mask_2d
    return mask_3d


def write_output_mha_and_json(mask_2d: np.ndarray, frame: int, reference_mha_path: Path, output_dir: Path):
    assert sitk is not None, "SimpleITK is required to write .mha outputs."
    case_name = Path(reference_mha_path).stem
    case_out_dir = Path(output_dir) / case_name

    # Read reference image to copy spatial meta
    ref_img = sitk.ReadImage(str(reference_mha_path))
    total_frames = ref_img.GetSize()[2]  # Z dimension

    # Build 3D label
    mask_3d = convert_mask_2d_to_3d(mask_2d, frame, total_frames)

    # Write .mha
    output_img = sitk.GetImageFromArray(mask_3d)
    output_img.CopyInformation(ref_img)
    output_mha_path = case_out_dir / "images/fetal-abdomen-segmentation/output.mha"
    output_mha_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_img, str(output_mha_path))

    # Write frame json
    json_path = case_out_dir / "fetal-abdomen-frame-number.json"
    with open(json_path, "w") as f:
        json.dump(int(frame), f, indent=2)

    print(f"[✓] {case_name} → output.mha (frame {frame})")


@torch.inference_mode()
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionASPPUNet()
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Threshold and postprocess flags
    base_thresh: float = float(getattr(args, "thresh", 0.5))
    keep_lcc: bool = bool(getattr(args, "lcc", True))

    for img_path in sorted(input_dir.iterdir()):
        suf = img_path.suffix.lower()

        # ---------- 2D images ----------
        if suf in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}:
            sl = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            sl_u8 = _preprocess_slice(sl)
            x = _tensor_from_u8(sl_u8, device)
            prob = torch.sigmoid(model(x)).cpu().numpy()[0, 0]
            # final thresholding for saving a 2D mask
            t = _otsu_thresh(prob) if args.adaptive_thresh else base_thresh
            mask = (prob >= t).astype(np.uint8)
            if keep_lcc and mask.any():
                mask = _connected_components_lcc(mask)

            out_path = out_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(out_path), mask * 255)
            print(f"[✓] Saved: {out_path}")

        # ---------- 3D .mha volumes ----------
        elif suf == ".mha":
            assert sitk is not None, "Reading .mha requires SimpleITK. Please install SimpleITK."
            vol = sitk.ReadImage(str(img_path))
            arr3d = sitk.GetArrayFromImage(vol)  # (N, H, W)
            total = arr3d.shape[0]

            # -------- frame window restriction --------
            max_frames = int(getattr(args, "max_frames", 128))
            start_frame = int(getattr(args, "start_frame", 0))
            if max_frames is None or max_frames < 0:
                end_frame = total
            else:
                end_frame = min(total, start_frame + max_frames)
            assert 0 <= start_frame < total, f"start_frame out of range: {start_frame}/{total}"
            assert end_frame > start_frame, f"Empty frame window: [{start_frame}, {end_frame})"

            probs = []
            for i in range(start_frame, end_frame):
                sl = arr3d[i]
                sl_u8 = _preprocess_slice(sl)
                x = _tensor_from_u8(sl_u8, device)
                prob = torch.sigmoid(model(x)).cpu().numpy()[0, 0]  # (H, W)
                probs.append(prob)

            probs = np.stack(probs, axis=0)  # (M, H, W), M=end_frame-start_frame

            # -------- frame selection --------
            scores, bin_masks, used_thresh = _score_frames(
                probs,
                mode=args.frame_select,
                thresh=base_thresh,
                adaptive=bool(args.adaptive_thresh),
                alpha=float(args.alpha),
                min_area=int(args.min_area),
            )
            best_idx_rel = int(np.argmax(scores))  # 0..M-1
            best_frame = start_frame + best_idx_rel  # absolute index

            # Build final 2D mask from best frame
            best_prob = probs[best_idx_rel]
            t_final = used_thresh[best_idx_rel]
            best_mask = (best_prob >= t_final).astype(np.uint8)
            if keep_lcc and best_mask.any():
                best_mask = _connected_components_lcc(best_mask)

            write_output_mha_and_json(best_mask, best_frame, img_path, out_dir)

        else:
            print(f"[!] Skip unsupported: {img_path.name}")


# ==============
# CLI
# ==============
def get_args():
    p = argparse.ArgumentParser("Attention-ASPP-UNet")
    sp = p.add_subparsers(dest="cmd", required=True)

    # Train (no data_dir needed; fixed folders are used)
    t = sp.add_parser("train")
    t.add_argument("--output_dir", default="./checkpoints")
    t.add_argument("--epochs", type=int, default=120)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--val_ratio", type=float, default=0.1)
    t.add_argument("--num_workers", type=int, default=0)

    # Predict
    pr = sp.add_parser("predict")
    pr.add_argument("--weights", required=True)
    pr.add_argument("--input_dir", required=True, help="Folder of images or .mha volumes")
    pr.add_argument("--out_dir", default="./preds")
    pr.add_argument("--thresh", type=float, default=0.5)
    pr.add_argument("--lcc", action="store_true", help="Keep only largest connected component")
    pr.add_argument("--max_frames", type=int, default=128,
                    help="Only evaluate the first N frames of each .mha (use -1 for all frames)")
    pr.add_argument("--start_frame", type=int, default=0, help="Start index within the volume (default 0)")
    pr.add_argument("--frame_select", choices=["prob_sum", "bin_area", "hybrid"], default="prob_sum",
                    help="How to select best frame within the window")
    pr.add_argument("--alpha", type=float, default=0.7, help="Hybrid weight: score = alpha*sum(prob) + (1-alpha)*area")
    pr.add_argument("--min_area", type=int, default=0, help="Ignore frames whose binary area < min_area")
    pr.add_argument("--adaptive_thresh", action="store_true",
                    help="Use Otsu threshold per-frame instead of fixed --thresh")

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.benchmark = True
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)
