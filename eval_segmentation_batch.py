#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate segmentation masks: GT  vs  Baseline  vs  New model
Metrics: Dice · IoU · HD95 (pixel)
"""

import argparse, csv, re, statistics as st
from math import isnan
from pathlib import Path

import cv2                         # opencv-python
import numpy as np
import scipy.stats as ss
from scipy.ndimage import distance_transform_edt

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ModuleNotFoundError:
    HAS_PLT = False

GT_DIR   = Path("val_png_best_ac/masks")
BASE_DIR = Path("test/output/images/fetal-abdomen-segmentation")
NEW_DIR  = Path("preds_panel")
NEW_SUFFIX = "_png"

OUT_CSV  = NEW_DIR / "seg_eval.csv"
PLOT_DIR = NEW_DIR / "plots"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------- case-id ----------
ID_RE = re.compile(r"^([0-9a-f-]{36})", re.I)
def case_id(stem: str) -> str:
    m = ID_RE.match(stem)
    if not m:
        raise ValueError(f"error UUID: {stem}")
    return m.group(1).lower()

# ---------- metrics ----------
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
    ab=a-cv2.erode(a,ker); bb=b-cv2.erode(b,ker)
    dta=distance_transform_edt(1-ab); dtb=distance_transform_edt(1-bb)
    d1, d2 = dtb[ab.astype(bool)], dta[bb.astype(bool)]
    return float(max(np.percentile(d1,95), np.percentile(d2,95)))

def read_gray(p: Path):
    img=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img

def index_dir(root: Path, suffix_strip: str = "") -> dict[str, Path]:
    idx={}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            cid = case_id(p.stem.replace(suffix_strip,""))
            idx[cid]=p
    return idx

def describe(arr):
    arr=[x for x in arr if not isnan(x)]
    return st.mean(arr), st.stdev(arr), st.median(arr), min(arr), max(arr)

def show_metric(name, new_arr, base_arr, unit="", higher_is_better=True):
    mn, sn, mdn, minn, maxn = describe(new_arr)
    mb, sb, mdb, minb, maxb = describe(base_arr)
    cmp = (np.array(new_arr) > np.array(base_arr)) if higher_is_better \
        else (np.array(new_arr) < np.array(base_arr))  
    improve_count = int(cmp.sum())
    improve_ratio = 100.0 * improve_count / len(new_arr)

    w,p = ss.wilcoxon(new_arr, base_arr, alternative="two-sided")
    stars = "n.s."
    if p<.001: stars="***"
    elif p<.01: stars="**"
    elif p<.05: stars="*"

    print(f"\n{name} {unit}")
    print(f"  New  : {mn:.4f} ± {sn:.4f} | median {mdn:.4f} | "
          f"min {minn:.4f} | max {maxn:.4f}")
    print(f"  Base : {mb:.4f} ± {sb:.4f} | median {mdb:.4f} | "
          f"min {minb:.4f} | max {maxb:.4f}")
    print(f"  Improve ratio: {improve_ratio:.1f}% ({improve_count}/{len(new_arr)})")
    print(f"  Wilcoxon p={p:.4g}  {stars}")

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plot",action="store_true",help="save hist & box plots")
    return ap.parse_args()

def main():
    args=parse_args()
    print(GT_DIR)
    print(BASE_DIR)
    print(NEW_DIR)
    if not (GT_DIR.exists() and BASE_DIR.exists() and NEW_DIR.exists()):
        raise SystemExit("GT / BASE / NEW not exist")

    gt_idx   = index_dir(GT_DIR)
    base_idx = index_dir(BASE_DIR)
    new_idx  = index_dir(NEW_DIR, suffix_strip=NEW_SUFFIX)

    rows=[]
    for cid, gp in gt_idx.items():
        if cid not in base_idx or cid not in new_idx:
            print(f"{cid} 缺预测文件")
            continue
        gt  = read_gray(gp)
        pb  = read_gray(base_idx[cid])
        pn  = read_gray(new_idx[cid])

        d_n,i_n,h_n = dice(pn,gt), iou(pn,gt), hd95(pn,gt)
        d_b,i_b,h_b = dice(pb,gt), iou(pb,gt), hd95(pb,gt)
        rows.append((cid,d_n,i_n,h_n,d_b,i_b,h_b,
                     d_n-d_b,i_n-i_b,h_n-h_b))

    if not rows:
        raise SystemExit("No matching cases, check the file name or suffix settings")

    dice_n=[r[1] for r in rows]; dice_b=[r[4] for r in rows]
    iou_n =[r[2] for r in rows]; iou_b =[r[5] for r in rows]
    hd_n  =[r[3] for r in rows]; hd_b =[r[6] for r in rows]

    show_metric("Dice", dice_n, dice_b, higher_is_better=True)
    show_metric("IoU ", iou_n, iou_b, higher_is_better=True)
    show_metric("HD95", hd_n, hd_b, "px", higher_is_better=False)

    # Top / Worst
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
    print("\nTop-5 Dice(New):")
    for r in rows_sorted[:5]:
        print(f"  {r[0][:8]}… New {r[1]:.4f} | Base {r[4]:.4f}")
    print("Worst-5 Dice(New):")
    for r in rows_sorted[-5:]:
        print(f"  {r[0][:8]}… New {r[1]:.4f} | Base {r[4]:.4f}")

    # CSV
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with open(OUT_CSV,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["case","dice_new","iou_new","hd95_new_px",
                    "dice_base","iou_base","hd95_base_px",
                    "dice_diff","iou_diff","hd95_diff"])
        for r in rows: w.writerow(r)
    print(f"\n {OUT_CSV}")

    if args.plot:
        if not HAS_PLT:
            print("matplotlib is not installed, skipping the plotting.")
        else:
            PLOT_DIR.mkdir(parents=True,exist_ok=True)
            metrics=[("dice",dice_n,dice_b),
                     ("iou", iou_n ,iou_b ),
                     ("hd95",hd_n ,hd_b)]
            for name,new,base in metrics:
                plt.figure()
                plt.hist([base,new],label=["Base","New"],bins=20,alpha=.6)
                plt.legend(); plt.title(f"{name.upper()} distribution"); plt.xlabel(name.upper())
                plt.savefig(PLOT_DIR/f"{name}_hist.png",dpi=200)

                plt.figure()
                plt.boxplot([base,new],labels=["Base","New"])
                plt.title(f"{name.upper()} boxplot"); plt.ylabel(name.upper())
                plt.savefig(PLOT_DIR/f"{name}_box.png",dpi=200)
            print(f"saved to {PLOT_DIR}")

if __name__=="__main__":
    main()
