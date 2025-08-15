#!/usr/bin/env python3
"""
Analyze fetal abdominal circumference (AC) predictions vs ground‑truth — *sweep‑aware* version.

Update
======
* **Sweep index改为 1‑based**：
  * GT 侧：`sweep_1_ac_mm → sweep_idx = 1`（不再减 1）
  * 预测侧：`sweep_idx = frame_idx // FRAMES_PER_SWEEP + 1`

其余逻辑不变。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

CONFIG = {
    "GT": "ac_result/gt/fetal_abdominal_circumferences_per_sweep.csv",
    "BASELINE": "ac_result/baseline/ac_measurements.csv",
    "NEW": "ac_result/aspp/ac_results.csv",
    "OUT": "ac_result/ac_analysis_results",
    "FRAMES_PER_SWEEP": 140,
}

def parse_args():
    p = argparse.ArgumentParser(description="Sweep‑aware AC analysis (1‑based sweep_idx)")
    p.add_argument("--gt", default=CONFIG["GT"])
    p.add_argument("--baseline", default=CONFIG["BASELINE"])
    p.add_argument("--new", default=CONFIG["NEW"])
    p.add_argument("--out", default=CONFIG["OUT"])
    p.add_argument("--fps", type=int, default=CONFIG["FRAMES_PER_SWEEP"])
    return p.parse_args()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def melt_gt(df: pd.DataFrame) -> pd.DataFrame:
    sweep_cols = [c for c in df.columns if c.endswith("_ac_mm")]
    long = (
        df.melt(id_vars=[c for c in df.columns if c not in sweep_cols],
                 value_vars=sweep_cols,
                 var_name="sweep", value_name="gt_ac_mm")
          .dropna(subset=["gt_ac_mm"]).copy())
    if "uuid" in long.columns and "case_id" not in long.columns:
        long["case_id"] = long["uuid"]
    long["sweep_idx"] = long["sweep"].str.extract(r"(\d+)").astype("Int64")  # 1‑based
    return long[["case_id", "sweep_idx", "gt_ac_mm"]]


def read_pred(path: str, model: str, fps: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"case_id", "frame_idx", "ac_mm"}
    if not req.issubset(df.columns):
        raise ValueError(f"{path} needs columns {req}")
    df["model"] = model
    df["sweep_idx"] = (df["frame_idx"] // fps).astype(int) + 1  # 1‑based
    return df[["case_id", "sweep_idx", "frame_idx", "ac_mm", "model"]]


def add_err(d: pd.DataFrame) -> pd.DataFrame:
    d["abs_err"] = (d["ac_mm"] - d["gt_ac_mm"]).abs()
    d["sq_err"] = (d["ac_mm"] - d["gt_ac_mm"]) ** 2
    d["ape_%"] = d["abs_err"] / d["gt_ac_mm"] * 100
    return d

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def scatter(ax, gt, pred, title):
    ax.scatter(gt, pred, alpha=0.6)
    lim = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
    ax.plot(lim, lim, ls="--")
    ax.set_xlabel("GT (mm)"); ax.set_ylabel("Pred (mm)"); ax.set_title(title)

def bland_alt(ax, gt, pred, title):
    diff = pred - gt
    md, sd = diff.mean(), diff.std(ddof=1)
    loa = 1.96*sd
    ax.scatter((gt+pred)/2, diff, alpha=0.6)
    ax.axhline(md, ls="--"); ax.axhline(md-loa, ls="--", c="r"); ax.axhline(md+loa, ls="--", c="r")
    ax.set_xlabel("Mean (mm)"); ax.set_ylabel("Diff (mm)"); ax.set_title(title)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    a = parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    gt = melt_gt(pd.read_csv(a.gt))
    pred = pd.concat([
        read_pred(a.baseline, "baseline", a.fps),
        read_pred(a.new, "attention_aspp_unet", a.fps)
    ])

    data = pred.merge(gt, on=["case_id", "sweep_idx"], how="inner")
    if data.empty:
        raise RuntimeError("No matched (case_id, sweep_idx)")
    data = add_err(data)
    data.to_csv(out/"merged_ac_values.csv", index=False)

    metrics = (data.groupby("model")
               .agg(MAE_mm=("abs_err","mean"),
                    RMSE_mm=("sq_err", lambda x: np.sqrt(x.mean())),
                    MAPE_pct=("ape_%","mean"),
                    Corr_r=("ac_mm", lambda x: x.corr(data.loc[x.index, "gt_ac_mm"])))
               .round(3))
    metrics.to_csv(out/"metrics.csv", index=True)
    print("\n*** Metrics ***\n", metrics)

    be, ne = (data.query("model=='baseline'")["abs_err"],
               data.query("model=='attention_aspp_unet'")["abs_err"])
    t, p_t = stats.ttest_rel(be, ne); w, p_w = stats.wilcoxon(be, ne, zero_method="zsplit")
    with open(out/"stats.txt", "w") as f:
        f.write(f"Paired t-test: t={t:.3f}, p={p_t:.4g}\nWilcoxon: W={w:.1f}, p={p_w:.4g}\n")

    for m in ["baseline","attention_aspp_unet"]:
        sub = data.query("model==@m")
        fig, ax = plt.subplots(figsize=(5,5)); scatter(ax, sub["gt_ac_mm"], sub["ac_mm"], m); fig.savefig(out/f"scatter_{m}.png", dpi=300); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5,5)); bland_alt(ax, sub["gt_ac_mm"], sub["ac_mm"], m); fig.savefig(out/f"bland_alt_{m}.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    for m,c in zip(["baseline","attention_aspp_unet"],[0.5,0.5]):
        ax.hist(data.query("model==@m")["abs_err"], bins=25, alpha=c, label=m, histtype="stepfilled")
    ax.legend(); fig.savefig(out/"error_hist.png", dpi=300); plt.close(fig)
    print(f"\nDone. Results in {out.resolve()}")

if __name__ == "__main__":
    main()
