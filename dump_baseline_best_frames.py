#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-case best-frame PNG dumper with hardcoded paths.

Layout (example):
  output_mha/
    aspp/a38f_s304_pred_new.mha
    baseline/fetal-abdomen-frame-number.json
    baseline/output_frame304.mha

Usage:
  python dump_baseline_best_frames_single.py
"""
import re
import json
from pathlib import Path
import numpy as np
import cv2

try:
    import SimpleITK as sitk
except Exception as e:
    raise SystemExit(f"SimpleITK is required but not available: {e}")

# ====== EDIT THESE CONSTANTS (if needed) ======
BASELINE_JSON = Path("output_mha/baseline/fetal-abdomen-frame-number.json")
BASELINE_MHA  = Path("output_mha/baseline/output_frame304.mha")
ASPP_MHA      = Path("output_mha/aspp/a38f_s304_pred_new.mha")  # optional, used to infer CASE_ID if empty
OUT_DIR       = Path("output_png/baseline")
CASE_ID       = ""   # if empty, will try to infer from ASPP_MHA or BASELINE_MHA filename
PREFIX        = ""   # optional filename prefix
# =============================================

def infer_case_id() -> str:
    # Try from aspp filename like a38f_s304_pred_new.mha → "a38f"
    for p in [ASPP_MHA, BASELINE_MHA]:
        if p and p.exists():
            m = re.match(r"([A-Za-z0-9-]+)_s\d+", p.stem)
            if m:
                return m.group(1)
            # fallback: use stem before any suffix words
            return p.stem.split("_")[0]
    return "case"

def infer_frame_from_filename(p: Path):
    """Extract frame index from names like *_s304* or *frame304*; return int or None."""
    m = re.search(r"(?:_s|frame)(\d+)", p.stem, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def read_volume(mha_path: Path) -> np.ndarray:
    if not mha_path.exists():
        raise FileNotFoundError(mha_path)
    itk_img = sitk.ReadImage(str(mha_path))
    return sitk.GetArrayFromImage(itk_img)  # (Z, H, W)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    case_id = CASE_ID or infer_case_id()
    frame = None

    # 1) preferred: JSON frame
    if BASELINE_JSON.exists():
        try:
            with open(BASELINE_JSON) as f:
                frame = int(json.load(f))
            print(f"[info] frame from JSON: {frame}")
        except Exception as e:
            print(f"[warn] cannot parse JSON: {e}")

    # 2) try parse from filename
    if frame is None:
        frame = infer_frame_from_filename(BASELINE_MHA)
        if frame is not None:
            print(f"[info] frame from filename: {frame}")

    # 3) read baseline volume
    vol = read_volume(BASELINE_MHA)
    z = vol.shape[0]

    # 4) fallback: max area slice
    if frame is None:
        areas = [(vol[i] > 0).sum() for i in range(z)]
        frame = int(np.argmax(areas))
        print(f"[info] frame by max area: {frame}")

    if not (0 <= frame < z):
        raise ValueError(f"Frame {frame} out of range (Z={z})")

    # 5) write PNG
    mask2d = (vol[frame] > 0).astype(np.uint8) * 255
    out_path = OUT_DIR / f"{PREFIX}{case_id}_s{frame:03d}.png"
    ok = cv2.imwrite(str(out_path), mask2d)
    if not ok:
        raise RuntimeError(f"Failed to write {out_path}")
    print(f"[✓] Saved: {out_path}")

if __name__ == "__main__":
    main()
