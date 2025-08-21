#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 3-1 预处理管线三联图生成脚本
Original → CLAHE(+Median) → Mask Overlay（可选轮廓/椭圆/预测掩码）

用法示例：
  # 基本（GT 覆盖 + 椭圆）
  python make_fig3_1_triptych.py \
      --img  /path/to/images/CASE_s042.png \
      --mask /path/to/masks/CASE_s042.png \
      --out  fig3_1_CASE_s042.png \
      --clip 1.0 --tile 8 --median 3 \
      --ellipse

  # 同时叠加预测掩码（蓝色）
  python make_fig3_1_triptych.py \
      --img  /path/to/images/CASE_s042.png \
      --mask /path/to/masks/CASE_s042.png \
      --pred /path/to/preds/CASE_s042.png \
      --out  fig3_1_CASE_s042_pred.png \
      --ellipse
"""

from pathlib import Path
from typing import Optional, Tuple

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


def apply_clahe(u8_gray: np.ndarray,
                clip: float = 1.0,
                tile: int = 8,
                median_k: Optional[int] = 3) -> np.ndarray:
    """对灰度图应用 CLAHE，然后可选中值滤波；返回 uint8。"""
    clahe = cv2.createCLAHE(clipLimit=float(clip),
                            tileGridSize=(int(tile), int(tile)))
    out = clahe.apply(u8_gray)
    if median_k and int(median_k) >= 3:
        k = int(median_k)
        if k % 2 == 0:  # kernel 必须为奇数
            k += 1
        out = cv2.medianBlur(out, k)
    return out


def to_u8(img: np.ndarray) -> np.ndarray:
    """任意类型灰度图线性映射为 [0,255] 的 uint8。"""
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    u8 = (img - mn) / (mx - mn) * 255.0
    return np.clip(u8, 0, 255).astype(np.uint8)


def load_gray(path: Path) -> np.ndarray:
    """读取为灰度 uint8。"""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return to_u8(arr)


def binarize_mask(path: Path) -> np.ndarray:
    """将掩码二值化（>0 视为前景），返回 uint8 {0,255}。"""
    m = load_gray(path)
    return (m > 0).astype(np.uint8) * 255


def overlay_on_bgr(base_bgr: np.ndarray,
                   mask_u8: np.ndarray,
                   alpha: float = 0.35,
                   color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """在 BGR 图像上以给定颜色半透明叠加二值掩码（255=前景）。"""
    color_layer = np.zeros_like(base_bgr, dtype=np.uint8)
    color_layer[mask_u8 > 0] = color
    out = cv2.addWeighted(color_layer, alpha, base_bgr, 1.0, 0.0)
    return out


def largest_component(mask_u8: np.ndarray) -> np.ndarray:
    """仅保留最大连通域。"""
    num, lab = cv2.connectedComponents((mask_u8 > 0).astype(np.uint8))
    if num <= 1:
        return np.zeros_like(mask_u8)
    max_area, keep = 0, 0
    for i in range(1, num):
        area = int((lab == i).sum())
        if area > max_area:
            max_area, keep = area, i
    return ((lab == keep).astype(np.uint8) * 255)


def draw_contour(img_bgr: np.ndarray,
                 mask_u8: np.ndarray,
                 thickness: int = 2) -> np.ndarray:
    """在 BGR 图上绘制掩码外轮廓（绿色）。"""
    cnts, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img_bgr.copy()
    cv2.drawContours(out, cnts, -1, (0, 255, 0), thickness)
    return out


def draw_best_fit_ellipse(img_bgr: np.ndarray,
                          mask_u8: np.ndarray,
                          thickness: int = 2) -> np.ndarray:
    """对最大连通域拟合椭圆并绘制（黄色）。"""
    binmask = largest_component(mask_u8)
    ys, xs = np.where(binmask > 0)
    out = img_bgr.copy()
    if len(xs) >= 5:  # fitEllipse 至少需要 5 个点
        pts = np.column_stack([xs, ys]).astype(np.int32)
        ellipse = cv2.fitEllipse(pts)
        cv2.ellipse(out, ellipse, (255, 255, 0), thickness)
    return out


def make_triptych(img_path: Path,
                  mask_path: Path,
                  pred_path: Optional[Path],
                  out_path: Path,
                  clip: float = 1.0,
                  tile: int = 8,
                  median_k: Optional[int] = 3,
                  draw_contour_flag: bool = True,
                  draw_ellipse_flag: bool = False,
                  title: Optional[str] = None) -> str:
    # 读取
    img_u8 = load_gray(img_path)
    gt_mask = binarize_mask(mask_path)
    pred_mask = binarize_mask(pred_path) if pred_path is not None else None

    # 预处理
    clahe_u8 = apply_clahe(img_u8, clip=clip, tile=tile, median_k=median_k)

    # 叠加（GT 红；Pred 蓝）
    overlay = cv2.cvtColor(clahe_u8, cv2.COLOR_GRAY2BGR)
    overlay = overlay_on_bgr(overlay, gt_mask, alpha=0.35, color=(255, 0, 0))
    if pred_mask is not None:
        overlay = overlay_on_bgr(overlay, pred_mask, alpha=0.35, color=(0, 0, 255))
    if draw_contour_flag:
        overlay = draw_contour(overlay, gt_mask, thickness=2)
    if draw_ellipse_flag:
        overlay = draw_best_fit_ellipse(overlay, gt_mask, thickness=2)

    # 拼三联图
    fig = plt.figure(figsize=(9, 3))
    if title:
        fig.suptitle(title)

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img_u8, cmap="gray")
    ax1.set_title("Original"); ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(clahe_u8, cmap="gray")
    ax2.set_title("CLAHE (+Median)"); ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    subtitle = "Overlay: GT"
    if pred_mask is not None:
        subtitle += " + Pred"
    if draw_ellipse_flag:
        subtitle += " + Ellipse"
    ax3.set_title(subtitle); ax3.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Make Fig 3-1 (Original → CLAHE → Mask overlay triptych)"
    )
    ap.add_argument("--img", required=True, help="灰度超声 PNG 路径")
    ap.add_argument("--mask", required=True, help="GT 掩码 PNG 路径")
    ap.add_argument("--pred", default=None, help="(可选) 预测掩码 PNG 路径")
    ap.add_argument("--out", required=True, help="输出图路径 (PNG)")
    ap.add_argument("--clip", type=float, default=1.0, help="CLAHE clipLimit")
    ap.add_argument("--tile", type=int, default=8, help="CLAHE tileGridSize")
    ap.add_argument("--median", type=int, default=3,
                    help="MedianBlur kernel (奇数≥3；0 表示不做)")
    ap.add_argument("--no_contour", action="store_true", help="不画 GT 轮廓")
    ap.add_argument("--ellipse", action="store_true", help="在 GT 上拟合椭圆")
    ap.add_argument("--title", default=None, help="可选标题")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    make_triptych(
        img_path=Path(args.img),
        mask_path=Path(args.mask),
        pred_path=(Path(args.pred) if args.pred else None),
        out_path=out_path,
        clip=args.clip,
        tile=args.tile,
        median_k=(None if args.median <= 0 else args.median),
        draw_contour_flag=(not args.no_contour),
        draw_ellipse_flag=args.ellipse,
        title=args.title,
    )
    print(f"Saved figure to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
