#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual diagnostic for segmentation results.

• 读取  eval_segmentation_batch.py  输出的  seg_eval.csv
• 画   ▸ Dice(New) vs Dice(Base)   散点图
      ▸ Dice_diff 直方图
• 自动导出 “最差 10 例” 的 3-通道叠加图：
      GT(红)  |  New-only(蓝) |  Base-only(绿)
  保存到  preds_aspp48/vis/

用法：
  python vis_error_analysis.py            # 生成图 + worst-case overlay
  python vis_error_analysis.py --top 5    # 只导出最差 5 例
"""

import argparse, csv, textwrap
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# --------- 路径（与评估脚本保持一致）---------
GT_DIR   = Path("val_png_best_ac/masks")
BASE_DIR = Path("test/output/images/fetal-abdomen-segmentation")
NEW_DIR  = Path("preds_aspp48")
CSV_PATH = NEW_DIR / "seg_eval.csv"
OUT_DIR  = NEW_DIR / "vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def case_id(stem: str) -> str:
    return stem[:36].lower()      # 已保证前 36 位为 uuid

def read_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img

def index_dir(root: Path, suffix_strip: str="") -> dict[str, Path]:
    idx = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            idx[case_id(p.stem.replace(suffix_strip,""))] = p
    return idx

# ---------- 主流程 ----------
def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10,
                    help="export N worst cases by Dice_diff (default 10)")
    return ap.parse_args()

def main():
    args = parse()
    if not CSV_PATH.exists():
        raise SystemExit("请先运行评估脚本生成 seg_eval.csv")

    # --- 读取 csv ---
    rows = []
    with open(CSV_PATH, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)

    dice_diff = np.array([float(r["dice_diff"]) for r in rows])
    dice_new  = np.array([float(r["dice_new"])  for r in rows])
    dice_base = np.array([float(r["dice_base"]) for r in rows])
    case_ids  = [r["case"] for r in rows]

    # --- 1) 散点图 : Dice(Base) vs Dice(New) ---
    plt.figure(figsize=(5,5))
    sns.scatterplot(x=dice_base, y=dice_new)
    plt.plot([0,1],[0,1],'--',color='grey')
    plt.xlabel("Dice (Base)")
    plt.ylabel("Dice (New)")
    plt.title("Dice scatter")
    plt.savefig(OUT_DIR/"dice_scatter.png", dpi=300, bbox_inches="tight")

    # --- 2) 直方图 : Dice_diff ---
    plt.figure(figsize=(5,4))
    sns.histplot(dice_diff, bins=20, kde=True, color='steelblue')
    plt.axvline(0,color='red',ls='--')
    plt.xlabel("Dice(New) − Dice(Base)")
    plt.title("Dice difference histogram")
    plt.savefig(OUT_DIR/"dice_diff_hist.png", dpi=300, bbox_inches="tight")

    print(f"[✓] plots saved to {OUT_DIR}")

    # --- 3) 导出最差 N 例叠加 ----
    worst_idx = np.argsort(dice_diff)[:args.top]
    print(f"\n导出 Dice_diff 最低 {args.top} 例至 {OUT_DIR}")

    gt_idx   = index_dir(GT_DIR)
    base_idx = index_dir(BASE_DIR)
    new_idx  = index_dir(NEW_DIR, suffix_strip="_mask")

    for rank, idx in enumerate(worst_idx, 1):
        cid = case_ids[idx]
        gt  = read_gray(gt_idx[cid])
        pb  = read_gray(base_idx[cid])
        pn  = read_gray(new_idx[cid])

        h,w = gt.shape
        overlay = np.zeros((h,w,3), np.uint8)
        overlay[:,:,2] = ((pn>0) & (gt==0)) * 255        # New-only → Blue
        overlay[:,:,1] = ((pb>0) & (gt==0)) * 255        # Base-only→ Green
        overlay[:,:,0] = (gt>0) * 255                    # GT → Red

        out_path = OUT_DIR/f"{rank:02d}_{cid[:8]}_overlay.png"
        Image.fromarray(overlay).save(out_path)

        print(f"  {rank:02d}. {cid}  Dice_new={dice_new[idx]:.4f} "
              f"| Dice_base={dice_base[idx]:.4f}  →  {out_path.name}")

if __name__ == "__main__":
    main()
