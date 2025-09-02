#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual diagnostic for segmentation results.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# --------- 路径（与评估脚本保持一致）---------
GT_DIR   = Path("val_png_best_ac/masks")                    # GT
BASE_DIR = Path("test/output/images/fetal-abdomen-segmentation")  # baseline 
NEW_DIR  = Path("preds_finetune")                             
CSV_PATH = NEW_DIR / "seg_eval.csv"
OUT_DIR  = NEW_DIR / "vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_DIR  = GT_DIR.parent / "images"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def case_id(stem: str) -> str:
    return stem[:36].lower()

def read_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img

def index_dir(root: Path, suffix_strip: str = "") -> dict[str, Path]:
    idx: dict[str, Path] = {}
    if not root.exists():
        return idx
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            stem = p.stem.replace(suffix_strip, "")
            idx[case_id(stem)] = p
    return idx

def to_bgr(gray: np.ndarray) -> np.ndarray:
    # Convert grayscale to BGR three-channel format#
    if len(gray.shape) == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray

def overlay_and_border(img_bgr: np.ndarray,
                       mask_bin: np.ndarray,
                       color_bgr: tuple[int, int, int],
                       alpha: float = 0.35,
                       thickness: int = 2) -> np.ndarray:
    if mask_bin.dtype != np.uint8:
        mask_bin = (mask_bin > 0).astype(np.uint8)

    if mask_bin.any():
        # Translucent filling (mixed only within the mask area)
        idx = mask_bin.astype(bool)
        img_bgr[idx] = (img_bgr[idx] * (1 - alpha) + np.array(color_bgr) * alpha).astype(np.uint8)

        # Outer contour outline stroke
        cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(img_bgr, cnts, -1, color_bgr, thickness)
    return img_bgr


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--edge", type=int, default=2)
    ap.add_argument("--no_bg", action="store_true")
    return ap.parse_args()

def main():
    args = parse()
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    edge  = max(1, int(args.edge))

    if not CSV_PATH.exists():
        raise SystemExit(f"Please run the evaluation script to generate {CSV_PATH}")

    rows = []
    with open(CSV_PATH, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    if not rows:
        raise SystemExit("The seg_eval.csv file is empty, so visualization is not possible.")

    # Convert to array
    dice_diff = np.array([float(r["dice_diff"]) for r in rows])
    dice_new  = np.array([float(r["dice_new"])  for r in rows])
    dice_base = np.array([float(r["dice_base"]) for r in rows])
    case_ids  = [r["case"] for r in rows]

    # --- Dice(Base) vs Dice(New) ---
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=dice_base, y=dice_new)
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel("Dice (Base)")
    plt.ylabel("Dice (New)")
    plt.title("Dice scatter")
    plt.savefig(OUT_DIR / "dice_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Dice_diff ---
    plt.figure(figsize=(5, 4))
    sns.histplot(dice_diff, bins=20, kde=True, color='steelblue')
    plt.axvline(0, color='red', ls='--')
    plt.xlabel("Dice(New) − Dice(Base)")
    plt.title("Dice difference histogram")
    plt.savefig(OUT_DIR / "dice_diff_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"plots saved to {OUT_DIR}")

    worst_idx = np.argsort(dice_diff)[:args.top]

    gt_idx   = index_dir(GT_DIR)
    base_idx = index_dir(BASE_DIR)
    new_idx  = index_dir(NEW_DIR, suffix_strip="_mask")
    img_idx  = {} if args.no_bg else index_dir(IMG_DIR)   # 底图索引

    COL_NEW  = (255, 0,   0)   # blue：New-only
    COL_BASE = (0,   255, 0)   # green：Base-only
    COL_GT   = (0,   0, 255)   # red：GT

    missing = 0
    for rank, i in enumerate(worst_idx, 1):
        cid = case_ids[i]

        if cid not in gt_idx or cid not in base_idx or cid not in new_idx:
            print(f"  skip: {cid} "
                  f"(gt:{cid in gt_idx}, base:{cid in base_idx}, new:{cid in new_idx})")
            missing += 1
            continue

        # read masks
        gt = read_gray(gt_idx[cid])
        pb = read_gray(base_idx[cid])
        pn = read_gray(new_idx[cid])

        h, w = gt.shape

        if (not args.no_bg) and (cid in img_idx):
            try:
                bg = read_gray(img_idx[cid])
                if bg.shape != (h, w):
                    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                bg = np.zeros((h, w), np.uint8)
        else:
            bg = np.zeros((h, w), np.uint8)

        canvas = to_bgr(bg)

        # Region: GT, Only New, Only Baseline
        m_gt        = (gt > 0).astype(np.uint8)
        m_new_only  = ((pn > 0) & (gt == 0)).astype(np.uint8)
        m_base_only = ((pb > 0) & (gt == 0)).astype(np.uint8)

        # Region: GT, only new, only baseline overlay Order: Difference area → GT (both semi-transparent + stroke)
        canvas = overlay_and_border(canvas, m_new_only,  COL_NEW,  alpha=alpha, thickness=edge)
        canvas = overlay_and_border(canvas, m_base_only, COL_BASE, alpha=alpha, thickness=edge)
        canvas = overlay_and_border(canvas, m_gt,        COL_GT,   alpha=alpha, thickness=edge)

        out_path = OUT_DIR / f"{rank:02d}_{cid[:8]}_overlay.png"
        Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(out_path)

        print(f"  {rank:02d}. {cid}  Dice_new={dice_new[i]:.4f} | Dice_base={dice_base[i]:.4f}  →  {out_path.name}")

    if missing:
        print(f"\n {missing}")


if __name__ == "__main__":
    main()
