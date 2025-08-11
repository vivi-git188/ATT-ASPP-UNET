# attention_aspp_unet_pipeline.py
# 改进要点：统一Resize到512；更强的数据增强；Dice+BCE(可切Tversky)；
# AdamW+weight_decay；AMP+梯度裁剪；TTA；全局阈值校准；稳健后处理与帧选择；早停

import argparse
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from albumentations import (
    CLAHE, Compose, HorizontalFlip, MedianBlur, RandomGamma, ToFloat,
    Resize, ShiftScaleRotate, RandomBrightnessContrast,
    ElasticTransform, GridDistortion
)

from albumentations.pytorch import ToTensorV2

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

# ---------------- Common Config ----------------
SEED = 2025
IMG_SIZE = 512
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 15

# 损失：'combo'使用 Dice+BCE；'tversky'使用Tversky
LOSS_TYPE = 'combo'   # 'combo' | 'tversky'
TV_ALPHA, TV_BETA = 0.7, 0.3

def set_seed(seed=SEED):
    import random, os
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# 处理老版本 albumentations 的 always_apply 参数
def SafeCLAHE(*args, **kwargs):
    kwargs.pop("always_apply", None)
    return CLAHE(*args, **kwargs)
def SafeMedianBlur(*args, **kwargs):
    kwargs.pop("always_apply", None)
    return MedianBlur(*args, **kwargs)

# ---------------- Model ----------------
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
    def forward(self, g, x): return x

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

# ---------------- Dataset ----------------
class FetalACDataset(Dataset):
    def __init__(self, img_paths: List[Path], msk_paths: List[Path] | None, train=True):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.train = train
        self.transform = self._build_transform()

    def _build_transform(self):
        # 几何增强先于强度增强
        # 训练增强里：用 Affine 代替 ShiftScaleRotate；去掉 ElasticTransform 的 alpha_affine
        from albumentations import Affine, ElasticTransform

        t_train = [
            Resize(IMG_SIZE, IMG_SIZE),
            HorizontalFlip(p=0.5),
            Affine(
                scale=(0.92, 1.08),
                rotate=(-7, 7),
                translate_percent=(0.0, 0.02),
                shear=0,  # ← 关键：不要用 None
                fit_output=False,
                p=0.7
            ),
            RandomGamma((80, 120), p=0.3),
            RandomBrightnessContrast(0.1, 0.1, p=0.3),
            ElasticTransform(alpha=8, sigma=3, p=0.25),  # 不要再传 alpha_affine
            SafeCLAHE(1.0, (8, 8)),
            SafeMedianBlur(3),
            ToFloat(max_value=255.0),
            ToTensorV2()
        ]

        t_val = [
            Resize(IMG_SIZE, IMG_SIZE),
            SafeCLAHE(1.0, (8, 8)),
            SafeMedianBlur(3),
            ToFloat(max_value=255.0),
            ToTensorV2()
        ]
        return Compose(t_train if self.train else t_val)

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
        # 归一化到[0,1]后转 tensor
        return sample["image"].float(), (sample["mask"].unsqueeze(0).float() / 255.0)

# ---------------- Utils ----------------
def collect_pairs(img_dir: Path, msk_dir: Path | None):
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.bmp', '.mha'}
    imgs, msks = [], []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in exts: continue
        imgs.append(p)
        if msk_dir:
            q = msk_dir/p.name
            if q.exists(): msks.append(q)
            else: msks.append(None)
    return imgs, msks if msk_dir else None

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.): super().__init__(); self.smooth = smooth
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        num = 2*(p*targets).sum((2,3)) + self.smooth
        den = p.sum((2,3)) + targets.sum((2,3)) + self.smooth
        return (1 - num/den).mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p*targets).sum((2,3))
        fp = (p*(1-targets)).sum((2,3))
        fn = ((1-p)*targets).sum((2,3))
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return (1 - tversky).mean()

class ComboLoss(nn.Module):
    """Dice + BCE：提升召回且抑制边界收缩"""
    def __init__(self, dice_w=1.0, bce_w=1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.dw, self.bw = dice_w, bce_w
    def forward(self, logits, targets):
        d = self.dice(logits, targets)
        b = F.binary_cross_entropy_with_logits(logits, targets)
        return self.dw*d + self.bw*b

def iou_score(logits, targets, thresh=0.5):
    preds = (torch.sigmoid(logits) > thresh).float()
    inter = (preds * targets).sum((2, 3))
    union = preds.sum((2, 3)) + targets.sum((2, 3)) - inter
    return (inter / (union + 1e-7)).mean().item()

from scipy.ndimage import distance_transform_edt

class EdgeLoss(nn.Module):
    """用Sobel对概率图与GT的梯度做L1，AMP 兼容"""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, logits, targets):
        # logits 可能是 float16（AMP）；把核和 targets 都匹配到同 dtype/device
        p = torch.sigmoid(logits)
        dtype = p.dtype
        device = p.device
        kx = self.kx.to(device=device, dtype=dtype)
        ky = self.ky.to(device=device, dtype=dtype)
        t = targets.to(dtype=dtype, device=device)

        gx_p = F.conv2d(p, kx, padding=1); gy_p = F.conv2d(p, ky, padding=1)
        gx_t = F.conv2d(t, kx, padding=1); gy_t = F.conv2d(t, ky, padding=1)
        grad_p = torch.sqrt(gx_p**2 + gy_p**2 + 1e-8)
        grad_t = torch.sqrt(gx_t**2 + gy_t**2 + 1e-8)
        return F.l1_loss(grad_p, grad_t)


def _dist_weight(targets):
    """基于距离变换的权重，边界附近权重大，降低HD95"""
    y = targets.detach().cpu().numpy()
    ws = []
    for t in y:  # t: (1,H,W)
        t2 = t[0]
        pos = distance_transform_edt(t2)
        neg = distance_transform_edt(1 - t2)
        dist = pos + neg
        w = dist / (dist.max() + 1e-8)
        ws.append(w)
    w = torch.from_numpy(np.stack(ws)).to(targets.device).unsqueeze(1).float()
    return w

class SurfaceLoss(nn.Module):
    """距离加权的BCE，逼近surface/boundary类损失"""
    def __init__(self, lam=0.5):
        super().__init__(); self.lam = lam
    def forward(self, logits, targets):
        w = 1.0 + self.lam * _dist_weight(targets)
        return F.binary_cross_entropy_with_logits(logits, targets, weight=w)


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

def train(args):
    set_seed()
    root_train = Path("train_png_best")
    root_val = Path("val_png_best")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_imgs, train_msks = collect_pairs(root_train/"images", root_train/"masks")

    if (root_val/"images").exists():
        val_imgs, val_msks = collect_pairs(root_val/"images", root_val/"masks")
        train_ds = FetalACDataset(train_imgs, train_msks, train=True)
        val_ds   = FetalACDataset(val_imgs, val_msks, train=False)
    else:
        full_ds = FetalACDataset(train_imgs, train_msks, train=True)
        val_len = max(1, int(0.1 * len(full_ds)))
        train_ds, val_ds = random_split(full_ds, [len(full_ds)-val_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = AttentionASPPUNet(base_c=args.base_c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    # Warmup + Cosine
    total_epochs = args.epochs
    warmup_epochs = max(1, int(0.05 * total_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs), cosine],
        milestones=[warmup_epochs]
    )

    # 损失（基础+边界），SurfaceLoss 先不启用，稳定后再加
    base_loss = ComboLoss() if LOSS_TYPE == 'combo' else TverskyLoss(TV_ALPHA, TV_BETA)
    edge_loss = EdgeLoss()

    def criterion_fn(logit_map, y):
        logit_map = logit_map.float()
        y = y.float()
        return base_loss(logit_map, y) + args.edge_w * edge_loss(logit_map, y)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0; best_path = out_dir/"best.pt"
    no_improve = 0

    for ep in range(1, total_epochs + 1):
        model.train(); running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{total_epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logit_map = model(x)
                loss = criterion_fn(logit_map, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/(pbar.n+1):.4f}")
        scheduler.step()

        dice, iou = evaluate(model, val_loader, device)
        print(f"\n[Val] Dice: {dice:.4f} | IoU: {iou:.4f}")

        if dice > best:
            best = dice; no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"[✓] Best model saved to {best_path}")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print("[EarlyStop] no improvement, stop training.")
                break

# ---------------- Predict ----------------
from skimage.measure import label
from scipy.ndimage import binary_fill_holes

def predict_prob_tta(model, x):
    """TTA：原图+水平翻转"""
    logits = model(x)
    x_flip = torch.flip(x, dims=[-1])
    logits_flip = model(x_flip)
    logits_flip = torch.flip(logits_flip, dims=[-1])
    logits = (logits + logits_flip) / 2.0
    return torch.sigmoid(logits).cpu().numpy()[0, 0]

def otsu_on_prob(prob):
    """备用：对[0,1]概率图做 Otsu，自适应阈值并夹到[0.35,0.6]"""
    p8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    ret, _ = cv2.threshold(p8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = ret / 255.0
    thr = max(0.35, min(0.6, float(thr)))
    return thr

def refine_mask(mask01: np.ndarray) -> np.ndarray:
    """小面积滤除 + 最大连通域 + (7,7)闭运算 + 填洞"""
    if mask01.sum()==0: return mask01
    lab = label(mask01)
    cnt = np.bincount(lab.ravel()); cnt[0] = 0

    H, W = mask01.shape
    min_area = max(20, int(0.0015 * H * W))   # 过滤极小噪点（~0.15%图像）
    keep_ids = [i for i,c in enumerate(cnt) if c >= min_area]
    if not keep_ids: return np.zeros_like(mask01)
    m = np.isin(lab, keep_ids).astype(np.uint8)

    # 只保留面积最大的那块
    lab2 = label(m); cnt2 = np.bincount(lab2.ravel()); cnt2[0]=0
    m = (lab2 == int(np.argmax(cnt2))).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = binary_fill_holes(m).astype(np.uint8)
    return m

def circularity(mask01: np.ndarray) -> float:
    m = (mask01>0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    c = max(cnts, key=cv2.contourArea)
    A = cv2.contourArea(c); P = cv2.arcLength(c, True)
    return 0.0 if P==0 else float(4*np.pi*A/(P*P))

def select_best_frame(pred_stack: np.ndarray, topk: int = 5) -> int:
    """先按面积取 topk，再按圆度最大挑一帧"""
    areas = np.array([(m>0).sum() for m in pred_stack])
    idxs = areas.argsort()[::-1][:max(1, min(topk, len(areas)))]
    best = max(idxs, key=lambda i: circularity(pred_stack[i]))
    return int(best)

@torch.inference_mode()
def calibrate(args):
    """在验证集上网格搜索全局阈值，写 thr.json"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionASPPUNet(base_c=args.base_c)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    val_img_dir = Path(args.val_dir)/"images"
    val_msk_dir = Path(args.val_dir)/"masks"
    img_paths = sorted(val_img_dir.glob("*.png"))

    thrs = np.linspace(0.35, 0.60, 11)
    scores = []

    for thr in thrs:
        dices = []
        for p in img_paths:
            sl = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
            enhanced = cv2.medianBlur(clahe.apply(sl_u8), 3)
            sample = Compose([Resize(IMG_SIZE, IMG_SIZE), ToFloat(max_value=255.0), ToTensorV2()])(image=enhanced)
            x = sample["image"].unsqueeze(0).to(device)
            prob = predict_prob_tta(model, x)
            prob = cv2.resize(prob, (sl.shape[1], sl.shape[0]))
            prob = cv2.GaussianBlur(prob, (5,5), 0)   # 阈值前平滑
            m = (prob > float(thr)).astype(np.uint8)

            gt = cv2.imread(str(val_msk_dir/p.name), cv2.IMREAD_GRAYSCALE)
            gt = (gt>127).astype(np.uint8)
            inter = (m & gt).sum()
            dice = (2*inter) / (m.sum() + gt.sum() + 1e-7)
            dices.append(dice)
        scores.append(np.mean(dices))

    best_thr = float(thrs[int(np.argmax(scores))])
    out = {"best_thr": best_thr}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir)/"thr.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[✓] Calibrated threshold = {best_thr:.3f} (saved to thr.json)")

@torch.inference_mode()
def predict(args):
    set_seed()

    # 读取全局阈值
    thr_cfg = Path("./checkpoints/thr.json")
    GLOBAL_THR = 0.48
    if thr_cfg.exists():
        try:
            with open(thr_cfg, "r") as f:
                GLOBAL_THR = float(json.load(f)["best_thr"])
            print(f"[i] Use calibrated threshold: {GLOBAL_THR:.3f}")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionASPPUNet(base_c=args.base_c)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(input_dir.iterdir()):
        ext = img_path.suffix.lower()
        if ext in {'.png', '.jpg', '.jpeg'}:
            # 读2D
            sl = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
            enhanced = cv2.medianBlur(clahe.apply(sl_u8), 3)

            # Resize到训练分辨率
            sample = Compose([Resize(IMG_SIZE, IMG_SIZE), ToFloat(max_value=255.0), ToTensorV2()])(image=enhanced)
            x = sample["image"].unsqueeze(0).to(device)

            prob = predict_prob_tta(model, x)                    # (H',W')
            prob = cv2.resize(prob, (sl.shape[1], sl.shape[0]))  # 还原
            prob = cv2.GaussianBlur(prob, (5,5), 0)              # 阈值前平滑

            thr = GLOBAL_THR
            m = (prob > thr).astype(np.uint8)
            mask = refine_mask(m)

            cv2.imwrite(str(out_dir / f"{img_path.stem}_mask.png"), mask * 255)
            print(f"[✓] Saved: {img_path.stem}_mask.png")

        elif ext == '.mha':
            if sitk is None:
                raise ImportError("SimpleITK is required for .mha processing.")
            # 读3D并逐帧推理
            vol = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))  # (N,H,W)
            pred_masks = []
            for sl in vol:
                sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
                enhanced = cv2.medianBlur(clahe.apply(sl_u8), 3)
                sample = Compose([Resize(IMG_SIZE, IMG_SIZE), ToFloat(max_value=255.0), ToTensorV2()])(image=enhanced)
                x = sample["image"].unsqueeze(0).to(device)

                prob = predict_prob_tta(model, x)                        # (H',W')
                prob = cv2.resize(prob, (sl.shape[1], sl.shape[0]))      # 还原
                prob = cv2.GaussianBlur(prob, (5,5), 0)                  # 阈值前平滑

                thr = GLOBAL_THR
                m = (prob > thr).astype(np.uint8)
                mask = refine_mask(m)
                pred_masks.append(mask)

            pred_stack = np.stack(pred_masks)                            # (N,H,W)
            best_frame = select_best_frame(pred_stack, topk=5)
            best_mask = pred_stack[best_frame]
            write_output_mha_and_json(best_mask, best_frame, img_path, out_dir)

# ---------------- I/O helpers for challenge ----------------
def convert_mask_2d_to_3d(mask_2d: np.ndarray, frame: int, total_frames: int):
    mask_2d = (mask_2d > 0).astype(np.uint8) * 2  # ITK-SNAP绿色标签=2
    mask_3d = np.zeros((total_frames, *mask_2d.shape), dtype=np.uint8)
    if 0 <= frame < total_frames:
        mask_3d[frame] = mask_2d
    return mask_3d

def write_output_mha_and_json(mask_2d: np.ndarray, frame: int, reference_mha_path: Path, output_dir: Path):
    case_name = reference_mha_path.stem
    case_out_dir = output_dir / case_name
    ref_img = sitk.ReadImage(str(reference_mha_path))
    total_frames = ref_img.GetSize()[2]

    mask_3d = convert_mask_2d_to_3d(mask_2d, frame, total_frames)
    output_img = sitk.GetImageFromArray(mask_3d)
    output_img.CopyInformation(ref_img)

    output_mha_path = case_out_dir / "images/fetal-abdomen-segmentation/output.mha"
    output_mha_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_img, str(output_mha_path))

    json_path = case_out_dir / "fetal-abdomen-frame-number.json"
    with open(json_path, "w") as f:
        json.dump(frame, f, indent=2)

    print(f"[✓] {case_name} → output.mha (frame {frame})")

# ---------------- CLI ----------------
def get_args():
    p = argparse.ArgumentParser("Attention-ASPP-UNet")
    sp = p.add_subparsers(dest="cmd", required=True)

    t = sp.add_parser("train")
    t.add_argument("--data_dir", required=True)
    t.add_argument("--output_dir", default="./checkpoints")
    t.add_argument("--epochs", type=int, default=120)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--base_c", type=int, default=48)
    t.add_argument("--edge_w", type=float, default=0.05, help="边界损失权重，0~0.1之间可调")

    pr = sp.add_parser("predict")
    pr.add_argument("--weights", required=True)
    pr.add_argument("--input_dir", required=True)
    pr.add_argument("--out_dir", default="./preds")
    pr.add_argument("--base_c", type=int, default=48)

    ca = sp.add_parser("calibrate")
    ca.add_argument("--weights", required=True)
    ca.add_argument("--val_dir", required=True)    # e.g. ./val_png_best
    ca.add_argument("--output_dir", default="./checkpoints")
    ca.add_argument("--base_c", type=int, default=48)

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.benchmark = True
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)
    elif args.cmd == "calibrate":
        calibrate(args)
