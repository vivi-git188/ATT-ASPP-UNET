#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import SimpleITK as sitk

# ========== USER CONFIG ==========
CASE_DIR = "output_mha"

# 显式路径（若三条都不为 None，则优先生效）
MINE_PATH       = r"/home/ubuntu/ACOUSLIC-AI-baseline/output_mha/aspp/a38f_s304_pred_new.mha"
BASELINE_PATH   = r"/home/ubuntu/ACOUSLIC-AI-baseline/output_mha/baseline/output_frame304.mha"
GT_PATH         = r"/home/ubuntu/ACOUSLIC-AI-baseline/output_mha/gt/0d7c4d8f-6e07-4f2b-aa76-8915ce15a38f.mha"

# ★ 新增：固定用哪一帧做对比（你要第304帧）
FIXED_FRAME = 305          # 304
FRAME_ONE_BASED = True     # ITK-SNAP 显示从1开始，这里设 True 会自动减1
# ================================


# ---------- Utils ----------
def read_binary(path: str) -> sitk.Image:
    img = sitk.ReadImage(path)
    bin_img = sitk.BinaryThreshold(img, lowerThreshold=1, upperThreshold=2**31-1,
                                   insideValue=1, outsideValue=0)
    return sitk.Cast(bin_img, sitk.sitkUInt8)

def align_to_ref(img: sitk.Image, ref: sitk.Image) -> sitk.Image:
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

def evaluate_pair(pred_img: sitk.Image, gt_img: sitk.Image) -> Dict[str, float]:
    dice, iou = dice_iou(pred_img, gt_img)
    hd, hd95  = hd_and_hd95(pred_img, gt_img)
    return {"Dice": dice, "IoU": iou, "HD": hd, "HD95": hd95}

# ★ 新增：只保留指定 z 帧（其他帧清零）
def keep_only_slice(img: sitk.Image, z_idx: int) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img)  # (Z,H,W)
    if z_idx < 0 or z_idx >= arr.shape[0]:
        raise ValueError(f"z_idx 超界：{z_idx} (Z={arr.shape[0]})")
    out = np.zeros_like(arr, dtype=arr.dtype)
    out[z_idx] = arr[z_idx]
    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(img)
    return out_img

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
    # 解析路径
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

    # 读取并对齐到 GT
    gt_img   = read_binary(gt_path)
    base_img = align_to_ref(read_binary(base_path), gt_img)
    mine_img = align_to_ref(read_binary(mine_path), gt_img)

    # ★ 固定在指定帧上做评估
    if FIXED_FRAME is not None:
        z = FIXED_FRAME - 1 if FRAME_ONE_BASED else FIXED_FRAME
        print(f"\n[Info] Evaluating on fixed frame z={z} "
              f"(输入={FIXED_FRAME}, one_based={FRAME_ONE_BASED})")
        gt_img   = keep_only_slice(gt_img, z)
        base_img = keep_only_slice(base_img, z)
        mine_img = keep_only_slice(mine_img, z)

    print("\n=== Baseline vs GT ===")
    res_b = evaluate_pair(base_img, gt_img)
    for k, v in res_b.items(): print(f"{k:>5}: {v:.6g}")

    print("\n=== Mine vs GT ===")
    res_m = evaluate_pair(mine_img, gt_img)
    for k, v in res_m.items(): print(f"{k:>5}: {v:.6g}")

    print("\n=== Summary ===")
    print("Metric     Baseline      Mine")
    for k in ["Dice", "IoU", "HD", "HD95"]:
        print(f"{k:<8}  {res_b[k]:>10.6g}  {res_m[k]:>10.6g}")

if __name__ == "__main__":
    main()
