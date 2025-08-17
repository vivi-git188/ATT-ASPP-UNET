#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual diagnostic for segmentation results.

• 读取  eval_segmentation_batch.py  输出的  seg_eval.csv
• 画   ▸ Dice(New) vs Dice(Base)   散点图
      ▸ Dice_diff 直方图
• 自动导出 “最差 N 例” 的半透明边框叠加图：
      GT(红)  |  New-only(蓝) |  Base-only(绿)
  保存到  preds_aspp48/vis/
--alpha 0.5 --edge 3  透明度和边框/粗细的调节
用法：
  python vis_error_analysis.py                    # 生成图 + 导出最差 10 例叠加
  python vis_error_analysis.py --top 5            # 只导出最差 5 例
  python vis_error_analysis.py --alpha 0.5 --edge 3
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
GT_DIR   = Path("val_png_best_ac/masks")                    # GT 掩码
BASE_DIR = Path("test/output/images/fetal-abdomen-segmentation")  # baseline 预测
NEW_DIR  = Path("preds_finetune")                              # 新模型预测
CSV_PATH = NEW_DIR / "seg_eval.csv"
OUT_DIR  = NEW_DIR / "vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 底图目录（与 GT 同级，若不存在则自动用黑底）
IMG_DIR  = GT_DIR.parent / "images"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------- 工具函数 ----------
def case_id(stem: str) -> str:
    """文件名的前 36 位为 uuid，统一为小写作为 key。"""
    return stem[:36].lower()

def read_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img

def index_dir(root: Path, suffix_strip: str = "") -> dict[str, Path]:
    """递归索引目录下的图片文件，key 为 36 位 uuid。支持去除后缀（如 '_mask'）。"""
    idx: dict[str, Path] = {}
    if not root.exists():
        return idx
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            stem = p.stem.replace(suffix_strip, "")
            idx[case_id(stem)] = p
    return idx

def to_bgr(gray: np.ndarray) -> np.ndarray:
    """灰度转 BGR 三通道"""
    if len(gray.shape) == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray

def overlay_and_border(img_bgr: np.ndarray,
                       mask_bin: np.ndarray,
                       color_bgr: tuple[int, int, int],
                       alpha: float = 0.35,
                       thickness: int = 2) -> np.ndarray:
    """
    在 img_bgr 上对 mask_bin>0 的区域做半透明填充，并描边。
    color_bgr: (B,G,R)
    """
    if mask_bin.dtype != np.uint8:
        mask_bin = (mask_bin > 0).astype(np.uint8)

    if mask_bin.any():
        # 半透明填充（仅在 mask 区域内混合）
        idx = mask_bin.astype(bool)
        img_bgr[idx] = (img_bgr[idx] * (1 - alpha) + np.array(color_bgr) * alpha).astype(np.uint8)

        # 外轮廓描边
        cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(img_bgr, cnts, -1, color_bgr, thickness)
    return img_bgr


# ---------- 主流程 ----------
def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10,
                    help="导出 Dice_diff 最低的 N 例（默认 10）")
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="叠加透明度 0~1（默认 0.35）")
    ap.add_argument("--edge", type=int, default=2,
                    help="轮廓线粗细（像素，默认 2）")
    ap.add_argument("--no_bg", action="store_true",
                    help="不使用底图（强制黑底）")
    return ap.parse_args()

def main():
    args = parse()
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    edge  = max(1, int(args.edge))

    if not CSV_PATH.exists():
        raise SystemExit(f"请先运行评估脚本生成 {CSV_PATH}")

    # --- 读取 csv ---
    rows = []
    with open(CSV_PATH, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    if not rows:
        raise SystemExit("seg_eval.csv 为空，无法可视化。")

    # 转成数组
    dice_diff = np.array([float(r["dice_diff"]) for r in rows])
    dice_new  = np.array([float(r["dice_new"])  for r in rows])
    dice_base = np.array([float(r["dice_base"]) for r in rows])
    case_ids  = [r["case"] for r in rows]

    # --- 1) 散点图 : Dice(Base) vs Dice(New) ---
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=dice_base, y=dice_new)
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel("Dice (Base)")
    plt.ylabel("Dice (New)")
    plt.title("Dice scatter")
    plt.savefig(OUT_DIR / "dice_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- 2) 直方图 : Dice_diff ---
    plt.figure(figsize=(5, 4))
    sns.histplot(dice_diff, bins=20, kde=True, color='steelblue')
    plt.axvline(0, color='red', ls='--')
    plt.xlabel("Dice(New) − Dice(Base)")
    plt.title("Dice difference histogram")
    plt.savefig(OUT_DIR / "dice_diff_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[✓] plots saved to {OUT_DIR}")

    # --- 3) 导出最差 N 例叠加 ----
    worst_idx = np.argsort(dice_diff)[:args.top]
    print(f"\n导出 Dice_diff 最低 {args.top} 例至 {OUT_DIR}")

    gt_idx   = index_dir(GT_DIR)
    base_idx = index_dir(BASE_DIR)
    new_idx  = index_dir(NEW_DIR, suffix_strip="_mask")
    img_idx  = {} if args.no_bg else index_dir(IMG_DIR)   # 底图索引

    # 颜色(BGR) —— 注意 OpenCV 用 BGR
    COL_NEW  = (255, 0,   0)   # 蓝：New-only
    COL_BASE = (0,   255, 0)   # 绿：Base-only
    COL_GT   = (0,   0, 255)   # 红：GT

    missing = 0
    for rank, i in enumerate(worst_idx, 1):
        cid = case_ids[i]

        # 缺失检查（某些病例若不在某个索引中则跳过）
        if cid not in gt_idx or cid not in base_idx or cid not in new_idx:
            print(f"  [skip] 缺文件 → {cid} "
                  f"(gt:{cid in gt_idx}, base:{cid in base_idx}, new:{cid in new_idx})")
            missing += 1
            continue

        # 读 masks
        gt = read_gray(gt_idx[cid])
        pb = read_gray(base_idx[cid])
        pn = read_gray(new_idx[cid])

        h, w = gt.shape

        # 读底图（若缺失或 --no_bg 则用黑底）
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

        # 区域：GT、仅新、仅基线
        m_gt        = (gt > 0).astype(np.uint8)
        m_new_only  = ((pn > 0) & (gt == 0)).astype(np.uint8)
        m_base_only = ((pb > 0) & (gt == 0)).astype(np.uint8)

        # 叠加顺序：差异区 → GT（都半透明 + 描边）
        canvas = overlay_and_border(canvas, m_new_only,  COL_NEW,  alpha=alpha, thickness=edge)
        canvas = overlay_and_border(canvas, m_base_only, COL_BASE, alpha=alpha, thickness=edge)
        canvas = overlay_and_border(canvas, m_gt,        COL_GT,   alpha=alpha, thickness=edge)

        out_path = OUT_DIR / f"{rank:02d}_{cid[:8]}_overlay.png"
        Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(out_path)

        print(f"  {rank:02d}. {cid}  Dice_new={dice_new[i]:.4f} | Dice_base={dice_base[i]:.4f}  →  {out_path.name}")

    if missing:
        print(f"\n[!] 共跳过 {missing} 例（因缺少对应文件）。")


if __name__ == "__main__":
    main()
