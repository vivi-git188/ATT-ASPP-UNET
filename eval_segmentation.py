#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    import SimpleITK as sitk
except Exception:
    sitk = None  # 只有在用毫米单位时才需要


# ========= 这里填你的路径/选项 =========
GT_DIR        = Path("output_png/gt")
NEW_DIR       = Path("output_png/aspp")      # 新模型（Attention-ASPP-UNet）PNG
BASE_DIR      = Path("output_png/baseline")  # baseline PNG
OUT_CSV       = Path("output_png/seg_eval.csv")

NEW_SUFFIX    = "_mask"   # 新模型文件名结尾后缀（如 *_mask.png）；如果没有就设成 ""
BASE_SUFFIX   = ""        # 基线文件名后缀（一般为空）

# HD95 单位：像素 or 毫米（读取一个 MHA 的 spacing，作为全局采样）
USE_MM        = False
SPACING_MHA   = Path("output_mha/baseline/output_frame304.mha")  # 仅当 USE_MM=True 时生效
# =====================================


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------- Metrics ----------------------
def _binarize(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return (a > 0).astype(np.uint8)

def dice_coeff(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = _binarize(pred); gt = _binarize(gt)
    inter = (pred & gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)

def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = _binarize(pred); gt = _binarize(gt)
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + eps) / (union + eps)

def hd95(pred: np.ndarray, gt: np.ndarray, sampling: Tuple[float, float] = (1.0, 1.0)) -> float:
    """
    Symmetric 95th percentile Hausdorff distance with optional anisotropic sampling.
    Returns NaN if either mask is empty.
    sampling: (dy, dx) in physical units (e.g., mm per pixel for row and column).
    """
    pred = _binarize(pred); gt = _binarize(gt)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")

    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    pred_b = pred - cv2.erode(pred, kernel, iterations=1)
    gt_b   = gt   - cv2.erode(gt,   kernel, iterations=1)

    dt_pred = distance_transform_edt(1 - pred_b, sampling=sampling)
    dt_gt   = distance_transform_edt(1 - gt_b,   sampling=sampling)

    dists_p2g = dt_gt[pred_b.astype(bool)]
    dists_g2p = dt_pred[gt_b.astype(bool)]
    if dists_p2g.size == 0 or dists_g2p.size == 0:
        return float("nan")

    return float(max(np.percentile(dists_p2g, 95), np.percentile(dists_g2p, 95)))


# ---------------------- Spacing (mm) ----------------------
def read_global_spacing_from_mha(mha_path: Path) -> Optional[Tuple[float, float]]:
    """
    从一个 .mha 读取 spacing，返回 (dy, dx) in mm。若失败返回 None。
    这里采用“全局一个 spacing”应用到所有样本，简单稳定。
    """
    if not USE_MM:
        return None
    if sitk is None:
        raise RuntimeError("SimpleITK 未安装，无法读取 spacing。")
    if not mha_path.exists():
        print(f"[WARN] spacing MHA 不存在：{mha_path}，将退回像素单位。")
        return None
    try:
        img = sitk.ReadImage(str(mha_path))
        sp = img.GetSpacing()  # (sx, sy, [sz])
        sx = float(sp[0]) if len(sp) > 0 else 1.0  # 列方向
        sy = float(sp[1]) if len(sp) > 1 else 1.0  # 行方向
        return (sy, sx)  # (dy, dx)
    except Exception as e:
        print(f"[WARN] 读取 spacing 失败：{e}，将退回像素单位。")
        return None


# ---------------------- Matching helpers ----------------------
def read_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img

def remove_suffix(stem: str, suffix: str) -> str:
    return stem[:-len(suffix)] if (suffix and stem.endswith(suffix)) else stem

def extract_slice_suffix(stem: str) -> Optional[str]:
    """
    从 stem 中提取 '_s###' 这样的帧后缀，例如 'a38f_s304' -> '_s304'。
    """
    m = re.search(r"_s\d+$", stem)
    return m.group(0) if m else None

def build_pred_index(pred_dir: Path, pred_suffix: str) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    """
    返回两个索引：
    - by_stem: 去后缀后的完整 stem -> Path
    - by_slice: 仅按 '_s###' -> [Path, ...]（可能多个，靠调用方处理冲突）
    """
    by_stem: Dict[str, Path] = {}
    by_slice: Dict[str, List[Path]] = {}
    for p in pred_dir.iterdir():
        if p.suffix.lower() not in IMG_EXTS:
            continue
        stem_norm = remove_suffix(p.stem, pred_suffix)
        by_stem[stem_norm] = p
        sfx = extract_slice_suffix(stem_norm)
        if sfx:
            by_slice.setdefault(sfx, []).append(p)
    return by_stem, by_slice

def match_pairs(gt_dir: Path, pred_dir: Path, pred_suffix: str) -> Dict[str, Path]:
    """
    返回映射：gt_stem -> pred_path
    先用去后缀后的完整 stem 精确匹配；不行则退回按 '_s###' 匹配（若唯一）。
    """
    pred_by_stem, pred_by_slice = build_pred_index(pred_dir, pred_suffix)
    mapping: Dict[str, Path] = {}
    for g in gt_dir.iterdir():
        if g.suffix.lower() not in IMG_EXTS:
            continue
        gt_stem = g.stem
        # 1) 完整 stem 匹配
        if gt_stem in pred_by_stem:
            mapping[gt_stem] = pred_by_stem[gt_stem]
            continue
        # 2) 按帧后缀 '_s###' 匹配（需要是唯一）
        sfx = extract_slice_suffix(gt_stem)
        if sfx and sfx in pred_by_slice and len(pred_by_slice[sfx]) == 1:
            mapping[gt_stem] = pred_by_slice[sfx][0]
        else:
            print(f"[WARN] 无法匹配：{g.name}")
    return mapping


# ---------------------- Main ----------------------
def main():
    if not GT_DIR.exists() or not NEW_DIR.exists() or not BASE_DIR.exists():
        raise SystemExit("[ERROR] 请检查 GT_DIR / NEW_DIR / BASE_DIR 路径是否存在。")

    # 全局 spacing（mm）；若失败则回到像素单位
    sampling = (1.0, 1.0)
    unit = "px"
    if USE_MM:
        sp = read_global_spacing_from_mha(SPACING_MHA)
        if sp is not None:
            sampling = sp
            unit = "mm"

    # 构建匹配（新模型 & 基线）
    map_new  = match_pairs(GT_DIR, NEW_DIR,  NEW_SUFFIX)
    map_base = match_pairs(GT_DIR, BASE_DIR, BASE_SUFFIX)

    rows = []
    for g in sorted(GT_DIR.iterdir()):
        if g.suffix.lower() not in IMG_EXTS:
            continue
        stem = g.stem
        if stem not in map_new or stem not in map_base:
            # 已在 match_pairs 里打印过 warn
            continue

        gt = read_gray(g)
        pn = read_gray(map_new[stem])
        pb = read_gray(map_base[stem])

        d_new = dice_coeff(pn, gt)
        i_new = iou_score(pn, gt)
        h_new = hd95(pn, gt, sampling=sampling)

        d_b   = dice_coeff(pb, gt)
        i_b   = iou_score(pb, gt)
        h_b   = hd95(pb, gt, sampling=sampling)

        rows.append((stem, d_new, i_new, h_new, d_b, i_b, h_b))

    if not rows:
        raise SystemExit("[ERROR] 没有匹配到任何 (GT, 新模型, 基线) 三方对。请检查文件名后缀或目录。")

    arr = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows], dtype=float)
    m = np.nanmean(arr, axis=0)
    print(f"[Summary] New  → Dice {m[0]:.4f}, IoU {m[1]:.4f}, HD95 {m[2]:.2f} {unit}")
    print(f"[Summary] Base → Dice {m[3]:.4f}, IoU {m[4]:.4f}, HD95 {m[5]:.2f} {unit}")

    # 写 CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "dice_new", "iou_new", f"hd95_new_{unit}",
                    "dice_base", "iou_base", f"hd95_base_{unit}"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.2f}",
                              f"{r[4]:.4f}", f"{r[5]:.4f}", f"{r[6]:.2f}"])
    print(f"[SAVED] {OUT_CSV}")

if __name__ == "__main__":
    main()
