#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate fetal abdomen segmentation (single case)
Metrics: Dice, IoU, Hausdorff Distance (HD) and 95% Hausdorff (HD95)

用法：
- 直接运行：python eval_ac_seg.py
- 先到下方 “USER CONFIG (预制路径)” 改成你图里的路径即可。
"""

import re
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import SimpleITK as sitk

# ========== USER CONFIG (预制路径) ==========
# 方案A：单目录自动识别（推荐）——三份 .mha 都在同一目录
# 识别规则：baseline 文件名包含 "output_frame"；你的模型包含 "pred"；剩下那个视为 GT
CASE_DIR = None  # ← 改成图中那个文件夹路径

# 方案B：显式三路径（如使用则把下面三个都改成真实路径；若留 None 则使用方案A）
MINE_PATH       = r"D:\MiniProject\ACOUSLIC\output_mha\aspp\a38f_s304_pred_new_1.mha"  # r"D:\...\xxx_gt.mha"
BASELINE_PATH = r"D:\MiniProject\ACOUSLIC\output_mha\baseline\output_frame304.mha"  # r"D:\...\output_frame304.mha"
GT_PATH     = r"D:\MiniProject\ACOUSLIC\output_mha\gt\0d7c4d8f-6e07-4f2b-aa76-8915ce15a38f.mha"  # r"D:\...\a38f_s304_pred_new_1.mha"
# ===========================================


# ---------- Utils ----------
def read_binary(path: str) -> sitk.Image:
    """Read mask and binarize (>0 -> 1, cast to UInt8)."""
    img = sitk.ReadImage(path)
    bin_img = sitk.BinaryThreshold(img, lowerThreshold=1, upperThreshold=2**31-1,
                                   insideValue=1, outsideValue=0)
    return sitk.Cast(bin_img, sitk.sitkUInt8)

def align_to_ref(img: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """Resample 'img' to the geometry of 'ref' using NN."""
    same_geom = (
        list(img.GetSize())      == list(ref.GetSize()) and
        list(img.GetSpacing())   == list(ref.GetSpacing()) and
        list(img.GetDirection()) == list(ref.GetDirection()) and
        list(img.GetOrigin())    == list(ref.GetOrigin())
    )
    if same_geom:
        return img
    return sitk.Resample(
        img, ref, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )

def dice_iou(pred: sitk.Image, gt: sitk.Image) -> Tuple[float, float]:
    p = sitk.GetArrayFromImage(pred).astype(bool)
    g = sitk.GetArrayFromImage(gt).astype(bool)
    inter = np.logical_and(p, g).sum()
    p_sum, g_sum = p.sum(), g.sum()
    union = p_sum + g_sum - inter
    if p_sum == 0 and g_sum == 0:
        return 1.0, 1.0
    if union == 0:
        return 0.0, 0.0
    dice = 2.0 * inter / (p_sum + g_sum)
    iou  = inter / union
    return float(dice), float(iou)

def hd_and_hd95(a: sitk.Image, b: sitk.Image) -> Tuple[float, float]:
    """Symmetric Hausdorff (max) + HD95（考虑物理间距）"""
    arr_a = sitk.GetArrayFromImage(a)
    arr_b = sitk.GetArrayFromImage(b)
    if arr_a.max()==0 and arr_b.max()==0:
        return 0.0, 0.0
    if (arr_a.max()==0) ^ (arr_b.max()==0):
        return float("inf"), float("inf")

    f = sitk.HausdorffDistanceImageFilter()
    f.Execute(a, b)
    hd = float(f.GetHausdorffDistance())

    a_surf = sitk.LabelContour(a)
    b_surf = sitk.LabelContour(b)
    dist_to_b = sitk.SignedMaurerDistanceMap(b_surf, insideIsPositive=False,
                                             squaredDistance=False, useImageSpacing=True)
    dist_to_a = sitk.SignedMaurerDistanceMap(a_surf, insideIsPositive=False,
                                             squaredDistance=False, useImageSpacing=True)

    da = np.abs(sitk.GetArrayFromImage(sitk.Mask(dist_to_b, a_surf)))
    db = np.abs(sitk.GetArrayFromImage(sitk.Mask(dist_to_a, b_surf)))
    da = da[sitk.GetArrayFromImage(a_surf) > 0]
    db = db[sitk.GetArrayFromImage(b_surf) > 0]
    all_d = np.concatenate([da, db])
    hd95 = 0.0 if all_d.size == 0 else float(np.percentile(all_d, 95))
    return hd, hd95

def evaluate_pair(pred_path: str, gt_path: str) -> Dict[str, float]:
    gt   = read_binary(gt_path)
    pred = read_binary(pred_path)
    pred = align_to_ref(pred, gt)
    dice, iou = dice_iou(pred, gt)
    hd, hd95  = hd_and_hd95(pred, gt)
    return {"Dice": dice, "IoU": iou, "HD": hd, "HD95": hd95}

# ---------- Auto-detect ----------
def autodetect_paths(case_dir: Path) -> Dict[str, Path]:
    files = [p for p in case_dir.iterdir() if p.suffix.lower()==".mha"]
    if not files:
        raise FileNotFoundError(f"该目录下未找到 .mha：{case_dir}")

    baseline = next((p for p in files if re.search(r"output_frame", p.name, re.I)), None)
    mine     = next((p for p in files if re.search(r"pred", p.name, re.I)), None)
    gt_candidates = [p for p in files if p != baseline and p != mine]

    if baseline is None:
        raise FileNotFoundError("未找到 baseline（文件名需包含 'output_frame'）。")
    if mine is None:
        raise FileNotFoundError("未找到你的模型结果（文件名需包含 'pred'）。")
    if not gt_candidates:
        raise FileNotFoundError("未找到 GT（不含 'pred'/'output_frame' 的那个）。")
    gt = max(gt_candidates, key=lambda p: p.stat().st_size) if len(gt_candidates)>1 else gt_candidates[0]
    return {"gt": gt, "baseline": baseline, "mine": mine}

# ---------- Main ----------
def main():
    # 路径解析：优先显式三路径，否则用 CASE_DIR 自动识别
    if GT_PATH and BASELINE_PATH and MINE_PATH:
        gt_path, base_path, mine_path = GT_PATH, BASELINE_PATH, MINE_PATH
    else:
        case_dir = Path(CASE_DIR)
        if not case_dir.exists():
            raise FileNotFoundError(f"CASE_DIR 不存在：{case_dir}")
        paths = autodetect_paths(case_dir)
        gt_path, base_path, mine_path = str(paths["gt"]), str(paths["baseline"]), str(paths["mine"])

    print("Resolved files:")
    print(f"  GT       : {gt_path}")
    print(f"  Baseline : {base_path}")
    print(f"  Mine     : {mine_path}")

    print("\n=== Baseline vs GT ===")
    res_b = evaluate_pair(base_path, gt_path)
    for k,v in res_b.items(): print(f"{k:>5}: {v:.6g}")

    print("\n=== Mine vs GT ===")
    res_m = evaluate_pair(mine_path, gt_path)
    for k,v in res_m.items(): print(f"{k:>5}: {v:.6g}")

    print("\n=== Summary ===")
    print("Metric     Baseline      Mine")
    for k in ["Dice","IoU","HD","HD95"]:
        bv, mv = res_b[k], res_m[k]
        print(f"{k:<8}  {bv:>10.6g}  {mv:>10.6g}")

if __name__ == "__main__":
    main()
