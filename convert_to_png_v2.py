#!/usr/bin/env python3
"""
convert_to_png_v3.py  ―― 带进度条版
-----------------------------------

把 *.mha / *.mhd 体数据批量展开成 PNG。
"""

import argparse, random, shutil
from pathlib import Path
import numpy as np, SimpleITK as sitk
from skimage.io import imsave
from tqdm import tqdm            # NEW


# ---------------- util ---------------- #
def rescale_to_uint16(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    vol -= vol.min()
    if vol.max() > 0:
        vol /= vol.max()
    vol *= 65535
    return vol.astype(np.uint16)


def save_slice(img_z, msk_z, stem, img_dir: Path, msk_dir: Path):
    imsave(img_dir / f"{stem}.png", img_z, check_contrast=False)
    imsave(msk_dir / f"{stem}.png", msk_z, check_contrast=False)


# ------------- per-volume ------------- #
def process_volume(img_path: Path, msk_path: Path,
                   img_dir: Path, msk_dir: Path,
                   neg_ratio: float, slice_bar=False) -> None:
    arr_img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
    arr_msk = sitk.GetArrayFromImage(sitk.ReadImage(str(msk_path))).astype(bool)

    arr_img = rescale_to_uint16(arr_img)

    pos_idx = [i for i in range(arr_msk.shape[0]) if arr_msk[i].any()]
    neg_pool = [i for i in range(arr_msk.shape[0]) if not arr_msk[i].any()]
    random.shuffle(neg_pool)
    neg_idx = neg_pool[: int(len(pos_idx) * neg_ratio)]

    iterable = pos_idx + neg_idx
    if slice_bar:
        iterable = tqdm(iterable, desc=f"{img_path.stem}", leave=False)

    for i in iterable:
        if i in pos_idx:
            stem = f"{img_path.stem}_pos_{i:03d}"
            save_slice(arr_img[i], (arr_msk[i]*255).astype(np.uint8),
                       stem, img_dir, msk_dir)
        else:
            stem = f"{img_path.stem}_neg_{i:03d}"
            save_slice(arr_img[i], np.zeros_like(arr_msk[i], np.uint8),
                       stem, img_dir, msk_dir)


# ------------------ main ---------------- #
def main():
    ap = argparse.ArgumentParser("ACOUSLIC volume → PNG slicer")
    ap.add_argument("split_dir",  type=Path, help="split 目录 (含 images/ masks/)")
    ap.add_argument("output_dir", type=Path, help="输出根目录（images/ masks/）")
    ap.add_argument("--neg_ratio", type=float, default=0.1,
                    help="负片 ≈ 正片 × neg_ratio")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--slice_bar", action="store_true",
                    help="显示每卷切片进度条")           # NEW
    args = ap.parse_args()
    random.seed(args.seed)

    out_img = args.output_dir / "images"
    out_msk = args.output_dir / "masks"
    if args.output_dir.exists():
        print(f"[INFO] {args.output_dir} 已存在，将清空覆盖…")
        shutil.rmtree(args.output_dir)
    out_img.mkdir(parents=True)
    out_msk.mkdir(parents=True)

    img_dir, msk_dir = args.split_dir / "images", args.split_dir / "masks"
    assert img_dir.exists() and msk_dir.exists(), "split 下需有 images/ masks/"

    img_files = sorted(list(img_dir.glob("*.mha")) + list(img_dir.glob("*.mhd")))
    assert img_files, f"未在 {img_dir} 找到 *.mha / *.mhd"

    for img_path in tqdm(img_files, desc="Volumes"):        # NEW 进度条
        stem = img_path.stem
        msk_path = msk_dir / f"{stem}.mha"
        if not msk_path.exists():
            msk_path = msk_dir / f"{stem}.mhd"
        if not msk_path.exists():
            print(f"[WARN] 掩膜缺失：{stem}，已跳过"); continue
        process_volume(img_path, msk_path,
                       out_img, out_msk, args.neg_ratio,
                       slice_bar=args.slice_bar)

    print(f"✅ 完成！结果存至 {args.output_dir}")


if __name__ == "__main__":
    main()
