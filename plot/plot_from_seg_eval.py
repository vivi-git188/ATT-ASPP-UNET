#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def dropna_pair(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return a[m], b[m]

def summarize(new, base, higher_is_better=True):
    new, base = dropna_pair(new, base)
    n = len(new)
    if n == 0:
        return dict(n=0, new_mean=np.nan, new_std=np.nan, new_median=np.nan,
                    base_mean=np.nan, base_std=np.nan, base_median=np.nan,
                    improve_ = np.nan, wilcoxon_p=np.nan)
    improve = (new > base).mean()*100 if higher_is_better else (new < base).mean()*100
    p = wilcoxon(new, base, alternative="two-sided").pvalue
    return dict(
        n=n,
        new_mean=float(np.mean(new)),  new_std=float(np.std(new, ddof=1)),  new_median=float(np.median(new)),
        base_mean=float(np.mean(base)), base_std=float(np.std(base, ddof=1)), base_median=float(np.median(base)),
        improve_=float(improve), wilcoxon_p=float(p)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="path to seg_eval.csv (宽表)")
    ap.add_argument("--px2mm", type=float, default=1.0,
                    help="HD95 像素转毫米的比例（mm/px），默认1.0表示单位为px")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.csv).parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = "mm" if args.px2mm != 1.0 else "px"

    dice_base = df["dice_new"].to_numpy(float)
    dice_new = df["dice_base"].to_numpy(float)
    iou_base  = df["iou_new"].to_numpy(float)
    iou_new  = df["iou_base"].to_numpy(float)
    hd_base   = (df["hd95_new_px"]  * args.px2mm).to_numpy(float)
    hd_new= (df["hd95_base_px"] * args.px2mm).to_numpy(float)

    # ---- Table 4-1：总体统计保存到 seg_stats.csv ----
    table_rows = []
    for name, new, base, hib in [
        ("dice", dice_new, dice_base, True),
        ("iou",  iou_new,  iou_base,  True),
        (f"hd95({unit})", hd_new, hd_base, False),
    ]:
        s = summarize(new, base, higher_is_better=hib)
        table_rows.append({"metric": name, **s})
    stats_path = Path(args.csv).parent / "seg_stats.csv"
    pd.DataFrame(table_rows).to_csv(stats_path, index=False)
    print(f"[SAVED] {stats_path}")

    # ---- 画图（直方图 / 箱线图 / 均值±std 柱状图）----
    def plot_all(mname, base, new):
        safe = lambda x: x[~np.isnan(x)]
        b, n = safe(base), safe(new)

        # hist
        plt.figure()
        plt.hist([b, n], bins=20, alpha=0.6, label=["Base","New"])
        plt.xlabel(mname); plt.ylabel("Count"); plt.title(f"{mname} distribution")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"hist_{mname.replace('/','_')}.png", dpi=200)

        # box
        plt.figure()
        plt.boxplot([b, n], labels=["Base","New"], showfliers=False)
        plt.ylabel(mname); plt.title(f"{mname} boxplot"); plt.tight_layout()
        plt.savefig(out_dir / f"box_{mname.replace('/','_')}.png", dpi=200)

        # bar mean±std
        means = [np.mean(b), np.mean(n)]
        stds  = [np.std(b, ddof=1), np.std(n, ddof=1)]
        plt.figure()
        plt.bar(["Base","New"], means, yerr=stds, capsize=4)
        plt.ylabel(mname); plt.title(f"{mname} mean ± std"); plt.tight_layout()
        plt.savefig(out_dir / f"bar_{mname.replace('/','_')}.png", dpi=200)

    plot_all("dice", dice_base, dice_new)
    plot_all("iou",  iou_base,  iou_new)
    plot_all(f"hd95({unit})", hd_base, hd_new)

    print(f"[PLOTS] saved to {out_dir}")

if __name__ == "__main__":
    main()
