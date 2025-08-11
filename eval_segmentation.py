#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# \"\"\"
# Evaluate segmentation predictions against GT using Dice / IoU / HD95 (95th percentile Hausdorff).
#
# Now supports millimetre (mm) units for HD95 via optional spacing lookup:
#   --spacing_source {none|mha_root|baseline_out}
#     none         : HD95 reported in pixels (default)
#     mha_root     : read spacing from <mha_root>/images/<case>.mha
#     baseline_out : read spacing from <baseline_out>/<case>/images/fetal-abdomen-segmentation/output.mha
#
# Examples:
#   (pixel units)
#   python eval_segmentation.py \
#     --gt_dir val_png_best/masks \
#     --pred_new_dir preds_new \
#     --pred_base_dir preds_base \
#     --pred_new_suffix _mask \
#     --out_csv seg_eval_val.csv
#
#   (mm units via baseline outputs)
#   python eval_segmentation.py \
#     --gt_dir val_png_best/masks \
#     --pred_new_dir preds_new \
#     --pred_base_dir preds_base \
#     --pred_new_suffix _mask \
#     --spacing_source baseline_out \
#     --baseline_out /path/to/baseline_outputs \
#     --out_csv seg_eval_val_mm.csv
# \"\"\"

import argparse
from pathlib import Path
import numpy as np
import cv2
import csv
from typing import Tuple, Dict, List, Optional
from scipy.ndimage import distance_transform_edt

try:
    import SimpleITK as sitk
except Exception:
    sitk = None  # Only required if spacing_source != 'none'

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# ---------------------- Metrics ----------------------
def _binarize(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return (a > 0).astype(np.uint8)

def dice_coeff(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = _binarize(pred); gt = _binarize(gt)
    inter = (pred & gt).sum()
    return (2*inter + eps) / (pred.sum() + gt.sum() + eps)

def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = _binarize(pred); gt = _binarize(gt)
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + eps) / (union + eps)

def hd95(pred: np.ndarray, gt: np.ndarray, sampling: Tuple[float,float]=(1.0,1.0)) -> float:
    # "\"\"Symmetric 95th percentile Hausdorff distance under anisotropic sampling.
    Returns NaN if either mask is empty.
    sampling = (dy, dx) in *physical* units (e.g., mm per pixel in row and column).\"\"\"
    pred = _binarize(pred); gt = _binarize(gt)
    if pred.sum() == 0 or gt.sum() == 0:
        return float('nan')

    # extract boundaries (4-connected)
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    pred_b = pred - cv2.erode(pred, kernel, iterations=1)
    gt_b   = gt   - cv2.erode(gt,   kernel, iterations=1)

    # EDT with sampling to convert to physical distances
    dt_pred = distance_transform_edt(1 - pred_b, sampling=sampling)  # distance to pred boundary
    dt_gt   = distance_transform_edt(1 - gt_b,   sampling=sampling)  # distance to gt boundary

    dists_p2g = dt_gt[pred_b.astype(bool)]
    dists_g2p = dt_pred[gt_b.astype(bool)]
    if dists_p2g.size == 0 or dists_g2p.size == 0:
        return float('nan')
    return float(max(np.percentile(dists_p2g, 95), np.percentile(dists_g2p, 95)))

# ---------------------- Spacing lookup ----------------------
def _read_spacing_from_mha(mha_path: Path) -> Optional[Tuple[float,float]]:
    # \"\"\"Return (dy, dx) in physical units (mm), or None if not available.\"\"\"
    if sitk is None:
        return None
    if not mha_path.exists():
        return None
    try:
        img = sitk.ReadImage(str(mha_path))
        # SimpleITK spacing returns (sx, sy, sz) for (x=columns, y=rows, z=slices)
        sx, sy, *_ = img.GetSpacing() + (1.0,) * max(0, 3 - len(img.GetSpacing()))
        dy, dx = float(sy), float(sx)
        return (dy, dx)
    except Exception:
        return None

def get_spacing_for_case(case_stem: str,
                         spacing_source: str,
                         mha_root: Optional[Path],
                         baseline_out: Optional[Path]) -> Optional[Tuple[float,float]]:
    # \"\"\"Resolve spacing for a case stem (without _s###).\"\"\"
    if spacing_source == 'none':
        return None
    if spacing_source == 'mha_root' and mha_root is not None:
        # Expect <mha_root>/images/<case>.mha
        p = mha_root / "images" / f"{case_stem}.mha"
        return _read_spacing_from_mha(p)
    if spacing_source == 'baseline_out' and baseline_out is not None:
        # Expect <baseline_out>/<case>/images/fetal-abdomen-segmentation/output.mha
        p = baseline_out / case_stem / "images" / "fetal-abdomen-segmentation" / "output.mha"
        return _read_spacing_from_mha(p)
    return None

# ---------------------- I/O helpers ----------------------
def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def match_pairs(gt_dir: Path, pred_dir: Path, pred_suffix: str = "") -> List[Tuple[Path, Path]]:
    pairs = []
    preds_by_stem: Dict[str, Path] = {}
    for p in pred_dir.iterdir():
        if p.suffix.lower() not in IMG_EXTS: continue
        stem = p.stem
        if pred_suffix and stem.endswith(pred_suffix):
            stem = stem[:-len(pred_suffix)]
        preds_by_stem[stem] = p

    for g in gt_dir.iterdir():
        if g.suffix.lower() not in IMG_EXTS: continue
        if g.stem in preds_by_stem:
            pairs.append((g, preds_by_stem[g.stem]))
    return pairs

def summarize(rows: List[Tuple[str, float, float, float, float, float, float]], unit: str):
    if not rows:
        print("[WARN] No matched pairs; check naming and directories.")
        return
    arr = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows], dtype=float)
    m = np.nanmean(arr, axis=0)  # NaN-safe for HD95
    print(f"[Summary] New  → Dice {m[0]:.4f}, IoU {m[1]:.4f}, HD95 {m[2]:.2f} {unit}")
    print(f"[Summary] Base → Dice {m[3]:.4f}, IoU {m[4]:.4f}, HD95 {m[5]:.2f} {unit}")

def main():
    ap = argparse.ArgumentParser("Evaluate segmentation (Dice/IoU/HD95) with optional mm units")
    ap.add_argument("--gt_dir", required=True, help="Directory of GT masks (PNG/JPG/TIF)")
    ap.add_argument("--pred_new_dir", required=True, help="Directory of NEW model masks")
    ap.add_argument("--pred_base_dir", required=True, help="Directory of BASELINE masks")
    ap.add_argument("--pred_new_suffix", default="_mask", help="Suffix to strip from NEW masks when matching")
    ap.add_argument("--pred_base_suffix", default="", help="Suffix to strip from BASE masks when matching")
    ap.add_argument("--out_csv", default="seg_eval.csv", help="Output CSV path")

    ap.add_argument("--spacing_source", choices=["none", "mha_root", "baseline_out"], default="none",
                    help="Where to read pixel spacing to compute HD95 in mm")
    ap.add_argument("--mha_root", default=None, help="Root with original .mha (expects images/<case>.mha)")
    ap.add_argument("--baseline_out", default=None, help="Baseline outputs root (expects <case>/.../output.mha)")
    args = ap.parse_args()

    gt_dir   = Path(args.gt_dir)
    new_dir  = Path(args.pred_new_dir)
    base_dir = Path(args.pred_base_dir)

    mha_root = Path(args.mha_root) if args.mha_root else None
    baseline_out = Path(args.baseline_out) if args.baseline_out else None

    if args.spacing_source != "none" and sitk is None:
        raise RuntimeError("SimpleITK is required for spacing_source != 'none' but is not available.")

    pairs_new  = match_pairs(gt_dir, new_dir,  pred_suffix=args.pred_new_suffix)
    pairs_base = match_pairs(gt_dir, base_dir, pred_suffix=args.pred_base_suffix)

    map_new  = {g.stem: p for g, p in pairs_new}
    map_base = {g.stem: p for g, p in pairs_base}

    unit = "px" if args.spacing_source == "none" else "mm"

    rows = []
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "dice_new", "iou_new", f"hd95_new_{unit}",
                                  "dice_base", "iou_base", f"hd95_base_{unit}"])

        for g in sorted(gt_dir.iterdir()):
            if g.suffix.lower() not in IMG_EXTS: continue
            stem = g.stem
            if stem not in map_new or stem not in map_base:
                continue

            # Optional spacing lookup (by case stem without _s###)
            case_id = stem.split("_s")[0]
            spacing = get_spacing_for_case(case_id, args.spacing_source, mha_root, baseline_out)
            sampling = (1.0, 1.0) if spacing is None else spacing

            gt = read_gray(g)
            pn = read_gray(map_new[stem])
            pb = read_gray(map_base[stem])

            d_new = dice_coeff(pn, gt); i_new = iou_score(pn, gt); h_new = hd95(pn, gt, sampling=sampling)
            d_b   = dice_coeff(pb, gt); i_b   = iou_score(pb, gt); h_b   = hd95(pb, gt, sampling=sampling)

            writer.writerow([stem, f"{d_new:.4f}", f"{i_new:.4f}", f"{h_new:.2f}", f"{d_b:.4f}", f"{i_b:.4f}", f"{h_b:.2f}"])
            rows.append((stem, d_new, i_new, h_new, d_b, i_b, h_b))

    summarize(rows, unit)
    print(f"[SAVED] {args.out_csv}")

if __name__ == "__main__":
    main()
