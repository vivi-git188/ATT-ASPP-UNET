#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare GT  vs  Baseline  vs  New-model segmentation masks
metrics: Dice, IoU, HD95  (pixel unit)

Folder layout
├── val_png_best/masks/                   ← GT *.png
├── test/output/images/fetal-abdomen-segmentation/**/ *.png   ← baseline
└── preds_aspp48/*.png                    ← new model (_mask.png)

Run:  python eval_segmentation.py
"""

import csv, re
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

# ───── 目录 ─────
GT_DIR   = Path("val_png_best/masks")
BASE_DIR = Path("test/output/images/fetal-abdomen-segmentation")
NEW_DIR  = Path("preds_aspp48")
OUT_CSV  = NEW_DIR / "seg_eval.csv"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ───── case-id 提取（保留 uuid，丢掉后缀）─────
ID_RE = re.compile(r"^([0-9a-f-]{36})")

def case_id(stem: str) -> str:
    m = ID_RE.match(stem.lower())
    if not m:
        raise ValueError(f"文件名缺合法 UUID: {stem}")
    return m.group(1)

# ───── metrics ─────
def _bin(a): return (a > 0).astype(np.uint8)

def dice(a,b,eps=1e-7):
    a,b=_bin(a),_bin(b); inter=(a&b).sum()
    return (2*inter+eps)/(a.sum()+b.sum()+eps)

def iou(a,b,eps=1e-7):
    a,b=_bin(a),_bin(b); inter=(a&b).sum()
    return (inter+eps)/(a.sum()+b.sum()-inter+eps)

def hd95(a,b):
    a,b=_bin(a),_bin(b)
    if a.sum()==0 or b.sum()==0: return float("nan")
    ker=np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
    ab,bb=a-cv2.erode(a,ker), b-cv2.erode(b,ker)
    dta=distance_transform_edt(1-ab); dtb=distance_transform_edt(1-bb)
    d1, d2 = dtb[ab.astype(bool)], dta[bb.astype(bool)]
    return float(max(np.percentile(d1,95),np.percentile(d2,95)))

def read_gray(p:Path):
    img=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(p)
    return img

# ───── 索引三个目录 ─────
def index_dir(root:Path)->Dict[str,Path]:
    idx={}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            idx[case_id(p.stem)] = p
    return idx

def main():
    if not (GT_DIR.exists() and BASE_DIR.exists() and NEW_DIR.exists()):
        raise SystemExit("❌ 目录检查失败")

    gt_idx   = index_dir(GT_DIR)
    base_idx = index_dir(BASE_DIR)
    new_idx  = index_dir(NEW_DIR)

    rows=[]
    for cid, gt_path in gt_idx.items():
        if cid not in base_idx or cid not in new_idx:
            print(f"[WARN] {cid} 缺 baseline 或 new 预测")
            continue
        gt  = read_gray(gt_path)
        pb  = read_gray(base_idx[cid])
        pn  = read_gray(new_idx[cid])

        rows.append((
            cid,
            dice(pn,gt), iou(pn,gt), hd95(pn,gt),
            dice(pb,gt), iou(pb,gt), hd95(pb,gt)
        ))

    if not rows:
        raise SystemExit("❌ 无共同 case-id")

    arr=np.array([[*r[1:]] for r in rows],float)
    mean=np.nanmean(arr,0)
    print(f"\n[New ] Dice {mean[0]:.4f} | IoU {mean[1]:.4f} | HD95 {mean[2]:.2f} px")
    print(f"[Base] Dice {mean[3]:.4f} | IoU {mean[4]:.4f} | HD95 {mean[5]:.2f} px")

    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with open(OUT_CSV,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["case","dice_new","iou_new","hd95_new_px",
                    "dice_base","iou_base","hd95_base_px"])
        for r in rows: w.writerow([r[0],*(f"{x:.4f}" for x in r[1:])])
    print(f"[SAVED] {OUT_CSV}")

if __name__=="__main__":
    main()
