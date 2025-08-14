#!/usr/bin/env python3
"""
Analyze fetal abdominal circumference (AC) predictions vs ground‑truth.

Configuration
-------------
You can either:
  1. Simply **edit the paths in the CONFIG block below** and run
         $ python analyze_ac_results.py
     with *no* command‑line arguments; or
  2. Override any of them from the command line, e.g.
         $ python analyze_ac_results.py --baseline other.csv

Inputs
~~~~~~
  • Ground‑truth  (wide format) : fetal_abdominal_circumferences_per_sweep.csv
  • Baseline predictions        : ac_measurements.csv
  • New‑model predictions       : ac_results.csv

Each *prediction* CSV must have columns:
    case_id, frame_idx (optional), ac_mm
The GT CSV must have columns:
    uuid (or case_id), … , sweep_#_ac_mm

Outputs (to out_dir)
~~~~~~~~~~~~~~~~~~~~
  metrics.csv      – summary table (MAE, RMSE, …)
  stats.txt        – paired t‑test & Wilcoxon result
  *.png            – scatter, Bland–Altman, error histogram plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 📝 USER CONFIG – edit the four paths below as you like
# ---------------------------------------------------------------------------
CONFIG = {
    "GT": "ac_result/gt/fetal_abdominal_circumferences_per_sweep.csv",
    "BASELINE": "ac_result/baseline/ac_measurements.csv",
    "NEW": "ac_result/aspp/ac_results.csv",
    "OUT": "ac_result/ac_analysis_results",
}
# ---------------------------------------------------------------------------


def parse_args():
    """Parse CLI args but fall back to CONFIG defaults when omitted."""
    p = argparse.ArgumentParser(description="Analyze AC predictions vs GT")
    p.add_argument("--gt", default=CONFIG["GT"], help="Ground‑truth AC CSV (wide)")
    p.add_argument("--baseline", default=CONFIG["BASELINE"], help="Baseline prediction CSV")
    p.add_argument("--new", default=CONFIG["NEW"], help="New‑model prediction CSV")
    p.add_argument("--out", default=CONFIG["OUT"], help="Output directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def melt_gt(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide GT table to long format with columns [case_id, gt_ac_mm]."""
    sweep_cols = [c for c in gt_df.columns if c.endswith("_ac_mm")]
    long = (
        gt_df.melt(id_vars=[c for c in gt_df.columns if c not in sweep_cols],
                    value_vars=sweep_cols,
                    var_name="sweep",
                    value_name="gt_ac_mm")
            .dropna(subset=["gt_ac_mm"]).copy()
    )
    # harmonise key name
    if "uuid" in long.columns and "case_id" not in long.columns:
        long["case_id"] = long["uuid"]
    return long[["case_id", "gt_ac_mm"]]


def read_pred(path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model"] = model_name
    required = {"case_id", "ac_mm"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {required}")
    return df[["case_id", "ac_mm", "model"]]


def add_error(df: pd.DataFrame) -> pd.DataFrame:
    df["abs_err"] = (df["ac_mm"] - df["gt_ac_mm"]).abs()
    df["sq_err"] = (df["ac_mm"] - df["gt_ac_mm"]) ** 2
    df["ape_%"] = df["abs_err"] / df["gt_ac_mm"] * 100
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def scatter_plot(ax, gt, pred, title):
    ax.scatter(gt, pred, alpha=0.6)
    min_, max_ = np.nanmin(gt), np.nanmax(gt)
    ax.plot([min_, max_], [min_, max_], ls="--", lw=1)
    ax.set_xlabel("GT AC (mm)")
    ax.set_ylabel("Predicted AC (mm)")
    ax.set_title(title)


def bland_altman(ax, gt, pred, title):
    diff = pred - gt
    mean = (gt + pred) / 2
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loA, upA = md - 1.96 * sd, md + 1.96 * sd

    ax.scatter(mean, diff, alpha=0.6)
    ax.axhline(md, ls="--")
    ax.axhline(loA, ls="--", color="r")
    ax.axhline(upA, ls="--", color="r")
    ax.set_xlabel("Mean of GT & Pred (mm)")
    ax.set_ylabel("Difference (Pred – GT) (mm)")
    ax.set_title(f"Bland–Altman: {title}\nmean={md:.2f} mm, ±1.96SD")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read data
    gt_raw = pd.read_csv(args.gt)
    gt_long = melt_gt(gt_raw)

    baseline = read_pred(args.baseline, "baseline")
    new = read_pred(args.new, "attention_aspp_unet")

    # 2. Merge
    preds = pd.concat([baseline, new], ignore_index=True)
    data = preds.merge(gt_long, on="case_id", how="inner")
    if data.empty:
        raise RuntimeError("No matching case_id between predictions and GT.")

    # 3. Error metrics
    data = add_error(data)

    metrics = (
        data.groupby("model")
            .agg(MAE_mm=("abs_err", "mean"),
                 RMSE_mm=("sq_err", lambda x: np.sqrt(x.mean())),
                 MAPE_pct=("ape_%", "mean"),
                 Corr_r=("ac_mm", lambda x: x.corr(data.loc[x.index, "gt_ac_mm"])) )
            .round(3)
    )
    metrics.to_csv(out_dir / "metrics.csv")
    print("\n*** Metrics ***\n", metrics, sep="\n")

    # 4. Paired tests (abs error baseline vs new)
    base_err = data.query("model == 'baseline'")["abs_err"].values
    new_err = data.query("model == 'attention_aspp_unet'")["abs_err"].values

    if len(base_err) != len(new_err):
        raise RuntimeError("Baseline and new model have different number of cases; ensure 1 prediction per case.")

    t_stat, p_t = stats.ttest_rel(base_err, new_err)
    w_stat, p_w = stats.wilcoxon(base_err, new_err, zero_method="zsplit")

    with open(out_dir / "stats.txt", "w") as f:
        f.write(f"Paired t-test:         t = {t_stat:.3f}, p = {p_t:.4g}\n")
        f.write(f"Wilcoxon signed-rank: W = {w_stat:.1f}, p = {p_w:.4g}\n")

    print("\nStatistical tests written to stats.txt")

    # 5. Plots
    for model_name in ["baseline", "attention_aspp_unet"]:
        subset = data.query("model == @model_name")

        # Scatter
        fig, ax = plt.subplots(figsize=(5, 5))
        scatter_plot(ax, subset["gt_ac_mm"].values, subset["ac_mm"].values,
                     f"{model_name} (n={len(subset)})")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{model_name}.png", dpi=300)
        plt.close(fig)

        # Bland–Altman
        fig, ax = plt.subplots(figsize=(5, 5))
        bland_altman(ax, subset["gt_ac_mm"].values, subset["ac_mm"].values, model_name)
        fig.tight_layout()
        fig.savefig(out_dir / f"bland_altman_{model_name}.png", dpi=300)
        plt.close(fig)

    # Combined error histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name, alpha in zip(["baseline", "attention_aspp_unet"], [0.6, 0.6]):
        errs = data.query("model == @model_name")["abs_err"].values
        ax.hist(errs, bins=25, alpha=alpha, label=model_name, histtype="stepfilled")
    ax.set_xlabel("Absolute error |Pred – GT| (mm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "error_hist.png", dpi=300)
    plt.close(fig)

    print(f"\nAnalysis complete. All outputs in → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
