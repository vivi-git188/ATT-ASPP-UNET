#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert 3D .mha volumes (images + masks) into per-slice PNG pairs.
• Positive frames: GT foreground area >= threshold (px or mm²)
• Negative frames: others (GT==0 or small blobs below threshold)
• Save images to out_root/images, masks to out_root/masks
• Write indices JSON: {case: {"pos":[...], "neg":[...]}, ...}

Supports:
  - Top-K positive selection + neighbor padding
  - Negative export strategies: all | random | stride
  - Two-stage presets: main (focus positives), finetune (add negatives)
  - NEW: Global negative cap (--neg_total_cap)

Usage examples
~~~~~~~~~~~~~~
主训练（少量或不含负帧）：
  python convert_mha_to_png.py \
    --mha_root train --out_root train_png_main \
    --topk 3 --neighbor_pad 1 \
    --stage main --min_area_mm2 80

假阳性微调（全局负样本上限 73 张）：
  python convert_mha_to_png.py \
    --mha_root train --out_root train_png_finetune_small \
    --topk 3 --neighbor_pad 1 \
    --stage finetune \
    --neg_strategy random --neg_ratio 2.0 --neg_cap 999 \
    --neg_total_cap 73 \
    --min_area_mm2 80
"""

from pathlib import Path
import argparse
import json
import numpy as np
import SimpleITK as sitk
import cv2, imageio
from tqdm import tqdm


def get_xy_spacing_mm(img_sitk) -> tuple[float, float] | None:
    """
    Return in-plane spacing (sx, sy) in mm from a SITK image.
    For 3D (Z, Y, X) arrays, SITK spacing order is (sx, sy, sz).
    """
    try:
        sp = img_sitk.GetSpacing()
        if len(sp) >= 2 and sp[0] > 0 and sp[1] > 0:
            return float(sp[0]), float(sp[1])
    except Exception:
        pass
    return None


def normalize_slice_to_u8(sl: np.ndarray) -> np.ndarray:
    """Min-max normalize a slice to uint8 [0,255]."""
    sl = np.asarray(sl, dtype=np.float32)
    mn, mx = float(sl.min()), float(sl.max())
    if mx <= mn:
        return np.zeros_like(sl, dtype=np.uint8)
    u8 = (sl - mn) / (mx - mn) * 255.0
    return u8.clip(0, 255).astype(np.uint8)


def decide_threshold_px(min_area_mm2: float, min_area_px: int,
                        sx_sy_mm: tuple[float, float] | None) -> int:
    """
    Compute the positive threshold in pixels.
    Priority: use mm² when provided and spacing available; otherwise fallback to px.
    """
    if min_area_mm2 is not None and min_area_mm2 > 0 and sx_sy_mm is not None:
        sx, sy = sx_sy_mm
        px_area_mm2 = sx * sy
        thr_px = int(np.ceil(min_area_mm2 / px_area_mm2))
        return max(1, thr_px)
    # fallback to px
    return max(1, int(min_area_px))


def convert_frames_with_negatives(
    mha_root: str,
    out_root: str,
    topk: int = 3,
    neighbor_pad: int = 0,
    # 判定阈值：二选一（优先 mm²）
    min_area_mm2: float | None = 80.0,
    min_area_px: int = 100,
    # 负样本导出策略
    neg_strategy: str = "random",   # "all" | "random" | "stride"
    neg_ratio: float = 0.0,         # 当 strategy="random" 时：负样本 : 正样本（主训练建议 0~0.3）
    neg_cap: int = 5,               # 单病例负样本上限
    neg_stride: int = 5,            # 当 strategy="stride" 时：每隔多少帧取一帧
    seed: int = 2025,
    # 只导出负样本（做 FP 评估）
    export_neg_only: bool = False,
    # 全局负样本总上限（新增）
    neg_total_cap: int = 0,         # 0 表示不启用全局上限
):
    """
    主函数：读取 mha_root/{images,masks}/*.mha，导出 PNG 到 out_root/{images,masks}/
    并写出 out_root/masks/frame_indices.json
    """
    assert neg_strategy in ("all", "random", "stride"), "neg_strategy must be all|random|stride"
    rng = np.random.default_rng(seed)

    mha_root = Path(mha_root)
    out_img = Path(out_root, 'images'); out_img.mkdir(parents=True, exist_ok=True)
    out_msk = Path(out_root, 'masks');  out_msk.mkdir(parents=True, exist_ok=True)

    image_files = list((mha_root / 'images').glob('*.mha'))
    # 为避免全局上限时偏向前几个病例：按 seed 打乱病例顺序
    image_files = list(rng.permutation(image_files))

    index_dict: dict[str, dict[str, list[int]]] = {}
    neg_total_saved = 0  # 全局已导出的负样本计数

    for f_img in tqdm(image_files, desc=f"{mha_root.name} [TopK={topk}, pad={neighbor_pad}, neg={neg_strategy}]"):
        name = f_img.stem
        f_msk = mha_root / 'masks' / f'{name}.mha'
        if not f_msk.exists():
            print(f"⚠️ 掩码不存在，跳过: {name}")
            continue

        img_itk = sitk.ReadImage(str(f_img))
        msk_itk = sitk.ReadImage(str(f_msk))
        img3d = sitk.GetArrayFromImage(img_itk)  # (Z,H,W)
        msk3d = sitk.GetArrayFromImage(msk_itk)  # (Z,H,W)
        Z = int(img3d.shape[0])

        # 判定阈值（像素）
        sx_sy = get_xy_spacing_mm(img_itk) or get_xy_spacing_mm(msk_itk)
        thr_px = decide_threshold_px(min_area_mm2, min_area_px, sx_sy)

        # 前景像素计数（显式二值化）
        areas_px = (msk3d > 0).reshape(Z, -1).sum(axis=1)
        is_pos = areas_px >= thr_px
        pos_pool = np.where(is_pos)[0]
        neg_pool = np.where(~is_pos)[0]

        # 选择正样本：Top-K + 邻帧（过滤掉被邻帧引入的负帧）
        pos_idxs = np.array([], dtype=int)
        if not export_neg_only and len(pos_pool) > 0 and topk > 0:
            order = np.argsort(areas_px[pos_pool])[::-1]
            top = pos_pool[order[:min(topk, len(pos_pool))]]
            if neighbor_pad > 0:
                extra = []
                for i in top:
                    for d in range(1, neighbor_pad + 1):
                        if i - d >= 0: extra.append(i - d)
                        if i + d <  Z: extra.append(i + d)
                pos_idxs = np.unique(np.concatenate([top, extra]))
                pos_idxs = pos_idxs[is_pos[pos_idxs]]  # 过滤邻帧中的负帧
            else:
                pos_idxs = np.unique(top)

        # 选择负样本
        neg_idxs = np.array([], dtype=int)
        if len(neg_pool) > 0:
            if neg_strategy == "all":
                neg_idxs = neg_pool  # ⚠️ 可能很多，评估用；训练建议 random/stride
            elif neg_strategy == "random":
                if len(pos_idxs) > 0:
                    n_neg = int(min(np.ceil(neg_ratio * len(pos_idxs)), neg_cap, len(neg_pool)))
                else:
                    # 全负病例或 export_neg_only
                    n_neg = int(min(neg_cap if neg_cap > 0 else len(neg_pool), len(neg_pool)))
                if n_neg > 0:
                    neg_idxs = rng.choice(neg_pool, n_neg, replace=False)
            elif neg_strategy == "stride":
                stride = max(1, int(neg_stride))
                neg_idxs = neg_pool[::stride]
                if neg_cap > 0:
                    neg_idxs = neg_idxs[:neg_cap]

        # ---- 全局负样本上限控制（新增） ----
        if neg_total_cap and neg_total_cap > 0:
            remaining = neg_total_cap - neg_total_saved
            if remaining <= 0:
                neg_idxs = np.array([], dtype=int)
            elif len(neg_idxs) > remaining:
                neg_idxs = rng.choice(neg_idxs, size=remaining, replace=False)

        saved_pos, saved_neg = [], []

        # 保存正样本
        for idx in pos_idxs:
            sl = img3d[idx]
            msk = (msk3d[idx] > 0).astype(np.uint8)
            sl_u8 = normalize_slice_to_u8(sl)
            msk_u8 = (msk * 255).astype(np.uint8)
            fname  = f'{name}_s{int(idx):03d}.png'
            imageio.imwrite(out_img / fname, sl_u8)
            imageio.imwrite(out_msk / fname, msk_u8)
            saved_pos.append(int(idx))

        # 保存负样本（掩码全 0）
        for idx in neg_idxs:
            sl = img3d[idx]
            sl_u8 = normalize_slice_to_u8(sl)
            msk_u8 = np.zeros_like(sl_u8, dtype=np.uint8)
            fname  = f'{name}_s{int(idx):03d}.png'
            imageio.imwrite(out_img / fname, sl_u8)
            imageio.imwrite(out_msk / fname, msk_u8)
            saved_neg.append(int(idx))

        # 更新全局负样本计数（新增）
        neg_total_saved += len(saved_neg)

        if saved_pos or saved_neg:
            index_dict[name] = {
                "pos": sorted(saved_pos),
                "neg": sorted(saved_neg),
                "_meta": {
                    "thr_px": int(thr_px),
                    "spacing_xy_mm": None if sx_sy is None else [float(sx_sy[0]), float(sx_sy[1])],
                    "min_area_mm2": None if min_area_mm2 is None else float(min_area_mm2)
                }
            }

        # 若全局额度已用尽，可继续处理剩余病例的正样本；无需提前 break

    # 保存索引 JSON
    (out_msk / "frame_indices.json").write_text(
        json.dumps(index_dict, indent=2, ensure_ascii=False)
    )
    print(f"Total negatives exported (global): {neg_total_saved}")
    print(f'✅ {mha_root} → {out_root} 完成，处理 {len(index_dict)} 个病例')


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mha_root", type=str, required=True,
                    help="Folder containing images/*.mha and masks/*.mha")
    ap.add_argument("--out_root", type=str, required=True,
                    help="Output folder; will create images/ and masks/")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--neighbor_pad", type=int, default=1)
    # 阈值参数（二选一；优先 mm²）
    ap.add_argument("--min_area_mm2", type=float, default=80.0,
                    help="Positive threshold in mm² (requires spacing). Set <=0 to disable.")
    ap.add_argument("--min_area_px", type=int, default=100,
                    help="Fallback positive threshold in pixels if spacing is missing or mm² disabled.")
    # 负样本策略
    ap.add_argument("--neg_strategy", type=str, default="random",
                    choices=["all", "random", "stride"])
    ap.add_argument("--neg_ratio", type=float, default=0.0,
                    help="For random: negatives per positive (e.g., 0.3).")
    ap.add_argument("--neg_cap", type=int, default=5,
                    help="Per-case cap for negatives (random/stride). <=0 means no cap.")
    ap.add_argument("--neg_stride", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--export_neg_only", action="store_true",
                    help="Export only negatives (useful for FP evaluation).")
    # 两阶段预设：方便一键设置常用比例
    ap.add_argument("--stage", type=str, default=None, choices=["main", "finetune"],
                    help="If set, will override some neg defaults: "
                         "main → neg_ratio=0, neg_cap=5; finetune → neg_ratio=0.3, neg_cap=10")
    # 全局负样本总上限（新增）
    ap.add_argument("--neg_total_cap", type=int, default=0,
                    help="Global cap for total negative slices across all cases (0=disable).")
    return ap


def apply_stage_presets(args: argparse.Namespace):
    if args.stage == "main":
        # 强调正样本学习，避免负帧主导
        args.neg_strategy = "random" if args.neg_strategy is None else args.neg_strategy
        args.neg_ratio = 0.0 if args.neg_ratio is None else args.neg_ratio
        args.neg_cap = 5 if (args.neg_cap is None or args.neg_cap <= 0) else args.neg_cap
    elif args.stage == "finetune":
        # 轻量加入负样本（建议配合困难空帧集合）
        if args.neg_strategy is None:
            args.neg_strategy = "random"
        if args.neg_ratio is None or args.neg_ratio <= 0:
            args.neg_ratio = 0.3
        if args.neg_cap is None or args.neg_cap <= 0:
            args.neg_cap = 10
    return args


if __name__ == "__main__":
    args = build_argparser().parse_args()
    args = apply_stage_presets(args)
    convert_frames_with_negatives(
        mha_root=args.mha_root,
        out_root=args.out_root,
        topk=args.topk,
        neighbor_pad=args.neighbor_pad,
        min_area_mm2=args.min_area_mm2,
        min_area_px=args.min_area_px,
        neg_strategy=args.neg_strategy,
        neg_ratio=args.neg_ratio,
        neg_cap=args.neg_cap,
        neg_stride=args.neg_stride,
        seed=args.seed,
        export_neg_only=args.export_neg_only,
        neg_total_cap=args.neg_total_cap,  # 新增
    )
