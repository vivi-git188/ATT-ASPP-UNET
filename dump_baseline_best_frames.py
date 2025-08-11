#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump baseline best-frame 2D masks from 3D output.mha to PNG images.

Expected baseline structure per case:
  <baseline_out>/<case_id>/
    fetal-abdomen-frame-number.json
    images/fetal-abdomen-segmentation/output.mha

This script reads the chosen frame index from JSON, slices the 3D mask, binarizes (>0),
and saves: <out_dir>/<prefix><case_id>_s{frame:03d}.png

Usage:
  python dump_baseline_best_frames.py     --baseline_out /path/to/baseline_outputs     --out_dir preds_base     [--prefix base_]
把 baseline 的 output.mha + frame-number.json 转成单帧 PNG 掩码，命名为 CaseID_s{frame}.png，
与 GT 命名对齐，之后就能和新模型 PNG 直接比。
"""
import argparse
from pathlib import Path
import json
import numpy as np
import cv2

try:
    import SimpleITK as sitk
except Exception as e:
    sitk = None

def dump_case(case_dir: Path, out_dir: Path, prefix: str = "") -> bool:
    """Return True on success, False on skip."""
    json_path = case_dir / "fetal-abdomen-frame-number.json"
    mha_path  = case_dir / "images" / "fetal-abdomen-segmentation" / "output.mha"
    case_name = case_dir.name

    if not json_path.exists():
        print(f"[WARN] {case_name}: missing frame json -> {json_path}")
        return False
    if not mha_path.exists():
        print(f"[WARN] {case_name}: missing output.mha -> {mha_path}")
        return False

    try:
        with open(json_path) as f:
            frame = int(json.load(f))
    except Exception as e:
        print(f"[WARN] {case_name}: cannot parse frame json: {e}")
        return False

    if sitk is None:
        print(f"[ERROR] SimpleITK not available; cannot read {mha_path}")
        return False

    try:
        itk_img = sitk.ReadImage(str(mha_path))
        arr3d = sitk.GetArrayFromImage(itk_img)  # (Z, H, W)
    except Exception as e:
        print(f"[WARN] {case_name}: cannot read MHA: {e}")
        return False

    z = arr3d.shape[0]
    if frame < 0 or frame >= z:
        print(f"[WARN] {case_name}: frame {frame} out of range (Z={z})")
        return False

    mask2d = (arr3d[frame] > 0).astype(np.uint8) * 255
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{prefix}{case_name}_s{frame:03d}.png"
    out_path = out_dir / out_name
    ok = cv2.imwrite(str(out_path), mask2d)
    if not ok:
        print(f"[WARN] {case_name}: failed to write {out_path}")
        return False

    print(f"[✓] {case_name}: frame {frame} -> {out_path}")
    return True

def main():
    ap = argparse.ArgumentParser("Dump baseline best-frame PNG masks")
    ap.add_argument("--baseline_out", required=True, help="Baseline output root containing per-case folders")
    ap.add_argument("--out_dir", required=True, help="Directory to save PNG masks")
    ap.add_argument("--prefix", default="", help="Optional filename prefix")
    args = ap.parse_args()

    base_root = Path(args.baseline_out)
    out_dir   = Path(args.out_dir)

    if not base_root.exists():
        print(f"[ERROR] baseline_out not found: {base_root}")
        return

    case_dirs = [p for p in base_root.iterdir() if p.is_dir()]
    if not case_dirs:
        print(f"[ERROR] No case folders found under {base_root}")
        return

    ok = 0
    for case_dir in sorted(case_dirs):
        ok += int(dump_case(case_dir, out_dir, args.prefix))

    print(f"[DONE] Converted {ok}/{len(case_dirs)} cases to PNG. Output -> {out_dir}")

if __name__ == "__main__":
    main()
