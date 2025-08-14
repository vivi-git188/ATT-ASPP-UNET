#!/usr/bin/env python3
"""
Analyze fetal abdominal circumference (AC) predictions vs ground‑truth — *sweep‑aware* version.

Key change
==========
• Predictions contain **frame_idx**.  We infer
      sweep_idx = frame_idx // FRAMES_PER_SWEEP  (0‑based)
  and require GT AC from the **same sweep** (sweep_1_ac_mm → sweep_idx 0 …).

Outputs (in OUT directory)
~~~~~~~~~~~~~~~~~~~~~~~~~~
  merged_ac_values.csv  – case_id · sweep_idx · frame_idx · ac_mm · gt_ac_mm · model
  metrics.csv           – MAE / RMSE / MAPE / r (per model)
  stats.txt             – paired t‑test & Wilcoxon result
  *.png                 – scatter / Bland‑Altman / error histogram
"""

import argparse
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ==========================================================================
# 📝 USER CONFIG – edit paths & sweep length if needed
# ==========================================================================
CONFIG = {
    "GT": "ac_result/gt/fetal_abdominal_circumferences_per_sweep.csv",
    "BASELINE": "ac_result/baseline/ac_measurements.csv",
    "NEW": "ac_result/aspp/ac_results.csv",
    "OUT": "ac_result/ac_analysis_results",
    "FRAMES_PER_SWEEP": 140,   # 🟡 adjust if sweep length differs
}
# ==========================================================================

# -------------------------------------------------------------------------
# CLI → fall back to CONFIG
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze AC predictions vs GT (sweep‑aware)")
    p.add_argument("--gt", default=CONFIG["GT"], help="Ground‑truth AC CSV (wide)")
    p.add_argument("--baseline", default=CONFIG["BASELINE"], help="Baseline prediction CSV")
    p.add_argument("--new", default=CONFIG["NEW"], help="New‑model prediction CSV")
    p.add_argument("--out", default=CONFIG["OUT"], help="Output directory")
    p.add_argument("--fps", type=int, default=CONFIG["FRAMES_PER_SWEEP"], help="Frames per sweep")
    return p.parse_args()

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def melt_gt(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Wide GT ➔ long [case_id, sweep_idx, gt_ac_mm]."""
    sweep_cols = [c for c in gt_df.columns if c.endswith("_ac_mm")]
    long = (
        gt_df.melt(id_vars=[c for c in gt_df.columns if c not in sweep_cols],
                    value_vars=sweep_cols,
                    var_name="sweep",
                    value_name="gt_ac_mm")
            .dropna(subset=["gt_ac_mm"]).copy()
    )
    if "uuid" in long.columns and "case_id" not in long.columns:
        long["case_id"] = long["uuid"]
    long["sweep_idx"] = long["sweep"].str.extract(r"(\\d+)").astype(int) - 1  # sweep_1 ➔ 0
    return long[["case_id", "sweep_idx", "gt_ac_mm"]]


def read_pred(path: str, model_name: str, fps: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model"] = model_name
    required = {"case_id", "frame_idx", "ac_mm"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {required}")
    df["sweep_idx"] = (df["frame_idx"] // fps).astype(int)
    return df[["case_id", "sweep_idx", "frame_idx", "ac_mm", "model"]]


def add_error(df: pd.DataFrame) -> pd.DataFrame:
    df["abs_err"] = (df["ac_mm"] - df["gt_ac_mm"]).abs()
    df["sq_err"] = (df["ac_mm"] - df["gt_ac_mm"]) ** 2
    df["ape_%"] = df["abs_err"] / df["gt_ac_mm"] * 100
    return df

# -------------------------------------------------------------------------
# Plot helpers (unchanged)
# -------------------------------------------------------------------------

def scatter_plot(ax, gt, pred, title):
    ax.scatter(gt, pred, alpha=0.6)
    lims = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
    ax.plot(lims, lims, ls="--", lw=1)
    ax.set_xlabel("GT AC (mm)")
    ax.set_ylabel("Predicted AC (mm)")
    ax.set_title(title)


def bland_altman(ax, gt, pred, title):
    diff = pred - gt
    mean = (gt + pred) / 2
    md, sd = diff.mean(), diff.std(ddof=1)
    loa = 1.96 * sd
    ax.scatter(mean, diff, alpha=0.6)
    ax.axhline(md, ls="--")
    ax.axhline(md - loa, ls="--", color="r")
    ax.axhline(md + loa, ls="--", color="r")
    ax.set_xlabel("Mean of GT & Pred (mm)")
    ax.set_ylabel("Difference (Pred – GT) (mm)")
    ax.set_title(f"Bland–Altman: {title}\nmean={md:.2f} ±1.96SD")

# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    gt_long = melt_gt(pd.read_csv(args.gt))
    baseline = read_pred(args.baseline, "baseline", args.fps)
    new      = read_pred(args.new, "attention_aspp_unet", args.fps)

    preds = pd.concat([baseline, new], ignore_index=True)
    data  = preds.merge(gt_long, on=["case_id", "sweep_idx"], how="inner")
    if data.empty:
        raise RuntimeError("No matching (case_id, sweep_idx) between predictions and GT.")

    data = add_error(data)
    data.to_csv(out_dir / "merged_ac_values.csv", index=False)

    # 2. Metrics
    metrics = (data.groupby("model")
                 .agg(MAE_mm=("abs_err", "mean"),
                      RMSE_mm=("sq_err", lambda x: np.sqrt(x.mean())),
                      MAPE_pct=("ape_%", "mean"),
                      Corr_r=("ac_mm", lambda x: x.corr(data.loc[x.index, "gt_ac_mm"])))
                 .round(3))
    metrics.to_csv(out_dir / "metrics.csv")
    print("\n*** Metrics ***\n", metrics, sep="\n")

    # 3. Paired tests
    base_err = data.query("model=='baseline'")["abs_err"].values
    new_err  = data.query("model=='attention_aspp_unet'")["abs_err"].values
    if len(base_err) != len(new_err):
        raise RuntimeError("Different number of matched sweeps between models.")
    t_stat, p_t = stats.ttest_rel(base_err, new_err)
    w_stat, p_w = stats.wilcoxon(base_err, new_err, zero_method="zsplit")
    with open(out_dir / "stats.txt", "w") as f:
        f.write(f"Paired t‑test:         t = {t_stat:.3f}, p = {p_t:.4g}\n")
        f.write(f"Wilcoxon signed‑rank: W = {w_stat:.1f}, p = {p_w:.4g}\n")
    print("Statistical tests written to stats.txt")

    # 4. Plots
    for model in ["baseline", "attention_aspp_unet"]:
        sub = data.query("model == @model")
        # scatter
        fig, ax = plt.subplots(figsize=(5,5))
        scatter_plot(ax, sub["gt_ac_mm"], sub["ac_mm"], f"{model} (n={len(sub)})")
        fig.tight_layout(); fig.savefig(out_dir / f"scatter_{model}.png", dpi=300); plt.close(fig)
        # BA
        fig, ax = plt.subplots(figsize=(5,5))
        bland_altman(ax, sub["gt_ac_mm"], sub["ac_mm"], model)
        fig.tight_layout(); fig.savefig(out_dir / f"bland_altman_{model}.png", dpi=300); plt.close(fig)

    # error hist
    fig, ax = plt.subplots(figsize=(6,4))
    for model, a in zip(["baseline", "attention_aspp_unet"], [0.5,0.5]):
        ax.hist(data.query("model == @model")["abs_err"], bins=25, alpha=a, label=model, histtype="stepfilled")
    ax.set_xlabel("|Pred – GT| (mm)"); ax.set_ylabel("Freq"); ax.set_title("Error distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "error_hist.png", dpi=300); plt.close(fig)

    print(f"\nAnalysis complete. Outputs saved in → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
