#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate segmentation masks: GT  vs  Baseline  vs  New model
⇢ Dice · IoU · HD95 (pixel)   +   Wilcoxon 双尾   +   Bootstrap CI
"""

from __future__ import annotations
import argparse, csv, re, statistics as st, random
from math import isnan
from pathlib import Path

import cv2, numpy as np, scipy.stats as ss
from scipy.ndimage import distance_transform_edt

# ====================== 路径修改区 ======================
GT_DIR   = Path("val_png_best_ac/masks")
BASE_DIR = Path("test/output/images/fetal-abdomen-segmentation")
NEW_DIR  = Path("preds_09")
NEW_SUFFIX = "_png"                    # 若无后缀留空
# =======================================================

OUT_CSV  = NEW_DIR / "seg_eval.csv"
PLOT_DIR = NEW_DIR / "plots"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------- case-id ----------
ID_RE = re.compile(r"^([0-9a-f-]{36})", re.I)
def case_id(stem: str) -> str:
    m = ID_RE.match(stem)
    if not m:
        raise ValueError(f"文件名缺合法 UUID: {stem}")
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

# ---------- 索引 ----------
def index_dir(root: Path, suffix_strip: str = "") -> dict[str, Path]:
    idx={}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            cid = case_id(p.stem.replace(suffix_strip,""))
            idx[cid]=p
    return idx

# ---------- bootstrap ----------
def bootstrap_ci(data: list[float], B:int=20_000,
                 ci: tuple[float,float]=(2.5,97.5)) -> tuple[float,float]:
    rng=random.Random(42)
    n=len(data)
    boots=[st.mean(rng.choices(data,k=n)) for _ in range(B)]
    lo,hi=np.percentile(boots,ci)
    return float(lo), float(hi)

# ---------- 输出格式 ----------
def describe(arr):
    arr=[x for x in arr if not isnan(x)]
    return st.mean(arr), st.stdev(arr), st.median(arr), min(arr), max(arr)

def show_metric(name, new_arr, base_arr, unit="", higher=True):
    mn,sn,medn,minn,maxn = describe(new_arr)
    mb,sb,medb,minb,maxb = describe(base_arr)
    diff = list(np.array(new_arr)-np.array(base_arr))
    lo,hi = bootstrap_ci(diff)

    better = (np.array(new_arr) > np.array(base_arr)) if higher \
             else (np.array(new_arr) < np.array(base_arr))
    ratio = 100*better.sum()/len(better)

    _,p = ss.wilcoxon(new_arr,base_arr)
    stars = "n.s.";   stars = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "n.s."))

    flag = "✅" if (lo>0 if higher else hi<0) else "⚠️"

    print(f"\n{name}{' ('+unit+')' if unit else ''}")
    print(f"  New  : {mn:.4f} ± {sn:.4f} | median {medn:.4f} | min {minn:.4f} | max {maxn:.4f}")
    print(f"  Base : {mb:.4f} ± {sb:.4f} | median {medb:.4f} | min {minb:.4f} | max {maxb:.4f}")
    print(f"  Improve ratio: {ratio:.1f}%")
    print(f"  Wilcoxon p={p:.4g}  {stars}")
    print(f"  95 % CI of Δ: [{lo:.4f},{hi:.4f}] {flag}")

# ---------- CLI ----------
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plot",action="store_true")
    return ap.parse_args()

# ---------- main ----------
def main():
    if not (GT_DIR.exists() and BASE_DIR.exists() and NEW_DIR.exists()):
        raise SystemExit("❌ GT / BASE / NEW 目录不存在")
    gt   = index_dir(GT_DIR)
    base = index_dir(BASE_DIR)
    new  = index_dir(NEW_DIR, NEW_SUFFIX)

    rows=[]
    for cid,gp in gt.items():
        if cid not in base or cid not in new:
            print(f"[WARN] 缺预测 {cid[:8]}…"); continue
        gt_m = read_gray(gp); pb=read_gray(base[cid]); pn=read_gray(new[cid])
        dn,in_,hn = dice(pn,gt_m), iou(pn,gt_m), hd95(pn,gt_m)
        db,ib,hb  = dice(pb,gt_m), iou(pb,gt_m), hd95(pb,gt_m)
        rows.append((cid,dn,in_,hn,db,ib,hb,dn-db,in_-ib,hn-hb))
    if not rows: raise SystemExit("❌ 无匹配病例")

    dice_n=[r[1] for r in rows]; dice_b=[r[4] for r in rows]
    iou_n =[r[2] for r in rows]; iou_b =[r[5] for r in rows]
    hd_n  =[r[3] for r in rows]; hd_b =[r[6] for r in rows]

    show_metric("Dice", dice_n, dice_b, higher=True)
    show_metric("IoU ", iou_n,  iou_b, higher=True)
    show_metric("HD95",hd_n,   hd_b,"px",higher=False)

    # ----- 保存 CSV -----
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with open(OUT_CSV,"w",newline="") as f:
        csv.writer(f).writerows([(
            "case","dice_new","iou_new","hd95_new_px",
            "dice_base","iou_base","hd95_base_px",
            "dice_diff","iou_diff","hd95_diff"),*rows])
    print("[SAVED]",OUT_CSV)

if __name__=="__main__":
    main()
