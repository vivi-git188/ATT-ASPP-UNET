#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset_reports.py   (flat images/masks 单目录 & 多目录 兼容版)

生成：
  • split_stats.csv
  • per_case_stats.csv
  • class_balance.png
  • frames_per_case_hist.png
  • latex_table_split_stats.tex

用法（任选其一）：
A) 有 frame_indices.json
   python make_dataset_reports.py --frame_index path/to/frame_indices.json --outdir reports

B) 目录带 pos/neg 子文件夹
   python make_dataset_reports.py --root path/to/dataset_root --outdir reports

C1) 扁平目录【单个 split】root/{images,masks}
   python make_dataset_reports.py --flat_root train_png_finetune_small --flat_split_name train --outdir reports

C2) 扁平目录【多个 split】root/{train,val}/{images,masks}
   python make_dataset_reports.py --flat_root data_png --outdir reports
"""

import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


# =============== 模式 A：frame_indices.json ===============
def load_from_frame_index(json_path: Path) -> pd.DataFrame:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []

    if isinstance(data, list):  # 逐帧列表
        for it in data:
            split = str(it.get("split", it.get("subset", "train"))).lower()
            label = str(it.get("label", it.get("cls", "pos"))).lower()
            case  = it.get("case") or it.get("case_id") or it.get("study_id")
            fname = it.get("path") or it.get("file") or it.get("filename")
            label = "pos" if label in ["1","true","pos","positive","foreground"] else "neg"
            if case is None and fname:
                case = Path(fname).stem.split("_")[0]
            rows.append({"split": split, "cls": label, "case": case, "file": fname})
    elif isinstance(data, dict):  # 按病例聚合
        # 尝试从路径猜 split；猜不到就用 "train"
        pstr = str(json_path.parent).lower()
        guess = "train" if "train" in pstr else ("val" if "val" in pstr else "train")
        for case, d in data.items():
            for idx in d.get("pos", []):
                rows.append({"split": guess, "cls": "pos", "case": case, "file": f"{case}_s{idx:03d}.png"})
            for idx in d.get("neg", []):
                rows.append({"split": guess, "cls": "neg", "case": case, "file": f"{case}_s{idx:03d}.png"})
    else:
        raise ValueError("Unsupported frame_indices.json structure")

    return pd.DataFrame(rows)


# =============== 模式 B：train/pos *.png, train/neg *.png ===============
def load_from_directory(root: Path, splits=("train","val"), classes=("pos","neg"), case_regex: str | None = None) -> pd.DataFrame:
    rx = re.compile(case_regex) if case_regex else re.compile(r"^([A-Za-z0-9\-]+)[_\-]?\d+")
    recs = []
    for sp in splits:
        for cl in classes:
            d = root / sp / cl
            if not d.exists():
                continue
            for p in d.glob("*.png"):
                m = rx.match(p.stem)
                case = m.group(1) if m else p.stem.split("_")[0]
                recs.append({"split": sp, "cls": cl, "case": case, "file": str(p)})
    return pd.DataFrame(recs)


# =============== 模式 C：扁平 images/masks 结构 ===============
def _list_flat_splits(root: Path) -> list[tuple[str, Path]]:
    """自动发现扁平 split。优先检测 root/{train,val,test}/{images,masks}；否则把 root 当作单 split。"""
    candidates = []
    # 多 split 情形
    for sub in root.iterdir() if root.exists() else []:
        if sub.is_dir() and (sub/"images").exists() and (sub/"masks").exists():
            candidates.append((sub.name, sub))
    if candidates:
        return sorted(candidates, key=lambda x: x[0])
    # 单 split 情形（root 直接含 images/masks）
    if (root/"images").exists() and (root/"masks").exists():
        return [("train", root)]
    return []


def load_flat_images_masks(root: Path, case_regex: str | None = None, split_name_override: str | None = None) -> pd.DataFrame:
    """
    支持：
      1) root/{images,masks}                  （单 split）
      2) root/{train,val}/{images,masks}      （多 split）
    如果传了 split_name_override，则当作单 split 并用该名字。
    """
    rx = re.compile(case_regex) if case_regex else re.compile(r"^([A-Za-z0-9\-]+)[_\-]?\d+")
    recs = []

    # 如果指定了 split_name_override，则强制按单 split 解析
    if split_name_override:
        splits = [(split_name_override, root)]
    else:
        splits = _list_flat_splits(root)

    for sp_name, sp_dir in splits:
        img_dir, msk_dir = sp_dir/"images", sp_dir/"masks"
        if not img_dir.exists() or not msk_dir.exists():
            continue
        for p in img_dir.glob("*.png"):
            msk_p = msk_dir / p.name
            if not msk_p.exists():
                continue
            msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
            cls = "pos" if (msk > 0).any() else "neg"
            m = rx.match(p.stem)
            case = m.group(1) if m else p.stem.split("_")[0]
            recs.append({"split": sp_name, "cls": cls, "case": case, "file": str(p)})
    return pd.DataFrame(recs)


# =============== 统计 / 绘图 / LaTeX ===============
def summarize(df: pd.DataFrame):
    per_case = (df.groupby(["split","case"]).size().reset_index(name="frames_per_case"))

    summary_rows = []
    for sp, g in df.groupby("split"):
        frames_total = len(g)
        pos = int((g["cls"]=="pos").sum())
        neg = int(frames_total - pos)
        fpc = per_case[per_case["split"]==sp]["frames_per_case"]
        summary_rows.append({
            "split": sp,
            "cases": int(g["case"].nunique()),
            "frames_total": frames_total,
            "pos_frames": pos,
            "neg_frames": neg,
            "pos_%": round(100*pos/frames_total,1) if frames_total else 0.0,
            "neg_%": round(100*neg/frames_total,1) if frames_total else 0.0,
            "frames_per_case_mean": round(float(fpc.mean()) if len(fpc)>0 else 0.0, 1),
            "frames_per_case_std":  round(float(fpc.std(ddof=1)) if len(fpc)>1 else 0.0, 1),
        })

    overall = {
        "split": "overall",
        "cases": int(df["case"].nunique()),
        "frames_total": int(len(df)),
        "pos_frames": int((df["cls"]=="pos").sum()),
        "neg_frames": int((df["cls"]=="neg").sum()),
    }
    if overall["frames_total"] > 0:
        overall["pos_%"] = round(100*overall["pos_frames"]/overall["frames_total"],1)
        overall["neg_%"] = round(100*overall["neg_frames"]/overall["frames_total"],1)
    f_all = per_case["frames_per_case"]
    overall["frames_per_case_mean"] = round(float(f_all.mean()) if len(f_all)>0 else 0.0, 1)
    overall["frames_per_case_std"]  = round(float(f_all.std(ddof=1)) if len(f_all)>1 else 0.0, 1)

    summary = pd.DataFrame(summary_rows + [overall])
    return per_case, summary


def save_latex_table(summary: pd.DataFrame, tex_path: Path):
    cols = ["split","cases","frames_total","pos_frames","neg_frames",
            "pos_%","neg_%","frames_per_case_mean","frames_per_case_std"]
    lines = [r"\begin{table}[t]",
             r"\centering",
             r"\caption{数据拆分统计}",
             r"\begin{tabular}{l r r r r r r r r}",
             r"\toprule",
             r"Split & \#Cases & \#Frames & Pos & Neg & Pos (\%) & Neg (\%) & Frames/Case mean & std \\",
             r"\midrule"]
    for _, r in summary[cols].iterrows():
        lines.append(f"{r['split']} & {int(r['cases'])} & {int(r['frames_total'])} & "
                     f"{int(r['pos_frames'])} & {int(r['neg_frames'])} & {r['pos_%']} & "
                     f"{r['neg_%']} & {r['frames_per_case_mean']} & {r['frames_per_case_std']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def plot_class_balance(summary: pd.DataFrame, out_png: Path):
    sub = summary[summary["split"].isin(["train","val","test"])]
    if sub.empty:
        return
    x = np.arange(len(sub)); w = 0.35
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(x-w/2, sub["pos_%"], w, label="Positive (%)")
    ax.bar(x+w/2, sub["neg_%"], w, label="Negative (%)")
    ax.set_xticks(x); ax.set_xticklabels(sub["split"])
    ax.set_ylabel("Percentage of frames"); ax.set_title("Class balance")
    ax.legend(); fig.tight_layout(); fig.savefig(out_png, dpi=300)


def plot_frames_per_case_hist(per_case: pd.DataFrame, out_png: Path):
    if per_case.empty:
        return
    fig, ax = plt.subplots(figsize=(5,4))
    for sp in sorted(per_case["split"].unique()):
        vals = per_case[per_case["split"]==sp]["frames_per_case"].values
        ax.hist(vals, bins=20, alpha=0.5, label=f"{sp} (n={len(vals)})")
    ax.set_xlabel("Frames per case"); ax.set_ylabel("Count")
    ax.set_title("Frames-per-case distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=300)


# =============== 主函数 ===============
def main():
    # ======= 写死路径配置 =======
    flat_root = Path("/home/ubuntu/ACOUSLIC-AI-baseline/data_png/train")

    print(flat_root)
    flat_split_name = "train"            # 给这个 split 起名字
    outdir = Path("reports_finetune")    # 输出目录
    outdir.mkdir(parents=True, exist_ok=True)

    # ======= 数据加载 =======
    df = load_flat_images_masks(flat_root,
                                case_regex=None,
                                split_name_override=flat_split_name)

    if df.empty:
        raise SystemExit("⚠️ 未找到任何帧，请检查路径/正则/扩展名（默认只扫 .png）。")

    # ======= 统计 + 保存 =======
    per_case, summary = summarize(df)
    (outdir/"split_stats.csv").write_text(summary.to_csv(index=False), encoding="utf-8")
    (outdir/"per_case_stats.csv").write_text(per_case.to_csv(index=False), encoding="utf-8")
    save_latex_table(summary, outdir/"latex_table_split_stats.tex")
    plot_class_balance(summary, outdir/"class_balance.png")
    plot_frames_per_case_hist(per_case, outdir/"frames_per_case_hist.png")

    print("=== 生成完成 ===")
    for f in ["split_stats.csv","per_case_stats.csv",
              "latex_table_split_stats.tex",
              "class_balance.png","frames_per_case_hist.png"]:
        print("  ", outdir/f)

if __name__ == "__main__":
    main()
