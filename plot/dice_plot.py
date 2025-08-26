# plot_dice_compare_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === 固定文件路径 ===
csv_path = "/Users/gaomeili/PycharmProjects/ACOUSLIC/plot/seg_eval.csv"       # 这里改成你的 csv 文件路径
outdir = Path("figs")           # 输出目录
outdir.mkdir(parents=True, exist_ok=True)

# === 读取数据 ===
df = pd.read_csv(csv_path)

# 自动匹配列名（避免大小写问题）
def pick(colnames):
    cols = {c.strip().lower(): c for c in df.columns}
    for name in colnames:
        key = name.strip().lower()
        if key in cols:
            return cols[key]
    raise KeyError(f"找不到列名，候选：{colnames}；文件包含列：{list(df.columns)}")

col_new  = pick(["dice_new","new_dice","dice_new_mean","new"])
col_base = pick(["dice_base","base_dice","dice_base_mean","base"])

new  = df[col_new].astype(float).values
base = df[col_base].astype(float).values
delta = new - base

# === Fig 4-7：散点图 ===
lim_min = np.floor(min(new.min(), base.min())*10)/10
lim_max = np.ceil (max(new.max(), base.max())*10)/10
plt.figure(figsize=(6,6))
plt.scatter(base, new, alpha=0.8)
plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
plt.xlabel("Dice (Baseline)")
plt.ylabel("Dice (New)")
plt.title("Fig 4-7  Dice(New) vs Dice(Baseline)")
plt.xlim(lim_min, lim_max); plt.ylim(lim_min, lim_max)
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig(outdir/"fig4_7_dice_scatter.png", dpi=300)
plt.close()

# === Fig 4-8：ΔDice 直方图 ===
improved = (delta > 0).sum()
worse    = (delta < 0).sum()
equal    = (delta == 0).sum()

plt.figure(figsize=(7,4.5))
plt.hist(delta, bins=15)
plt.axvline(0, linestyle="--")
plt.xlabel("ΔDice = Dice(New) − Dice(Baseline)")
plt.ylabel("Count")
plt.title("Fig 4-8  Histogram of ΔDice")
txt = f"Improved: {improved}  Worse: {worse}  Equal: {equal}\n" \
      f"Mean ΔDice = {delta.mean():.4f}"
plt.annotate(txt, xy=(0.98,0.98), xycoords="axes fraction",
             ha="right", va="top")
plt.tight_layout()
plt.savefig(outdir/"fig4_8_dice_delta_hist.png", dpi=300)
plt.close()

print("[Saved] figs/fig4_7_dice_scatter.png")
print("[Saved] figs/fig4_8_dice_delta_hist.png")
print(f"Improved: {improved}, Worse: {worse}, Equal: {equal}, "
      f"Mean ΔDice: {delta.mean():.4f}")
