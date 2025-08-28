import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Step 1. 文件路径 ===
with_att_path = "/Users/gaomeili/PycharmProjects/ACOUSLIC/plot/att_metrics/seg_eval.csv"         # 有 Attention 的模型评估结果
no_att_path = "/Users/gaomeili/PycharmProjects/ACOUSLIC/plot/att_metrics/no_att_aeg_vag.csv"     # 无 Attention 的模型评估结果

# === Step 2. 读取数据 ===
df_with = pd.read_csv(with_att_path)
df_noatt = pd.read_csv(no_att_path)

# === Step 3. 合并两个 DataFrame，按 case 对齐 ===
df_merge = pd.merge(
    df_with[["case", "dice_new", "iou_new", "hd95_new_px"]],
    df_noatt[["case", "dice_new", "iou_new", "hd95_new_px"]],
    on="case",
    suffixes=("_with", "_without")
)

# === Step 4. 计算差值 ===
df_merge["delta_dice"] = df_merge["dice_new_with"] - df_merge["dice_new_without"]
df_merge["delta_iou"] = df_merge["iou_new_with"] - df_merge["iou_new_without"]
df_merge["delta_hd95"] = df_merge["hd95_new_px_with"] - df_merge["hd95_new_px_without"]

# === Step 5. 图像保存函数设置 ===
os.makedirs("figures", exist_ok=True)  # 创建保存目录

# === Step 6. 配对图函数 ===
def plot_paired_lines(data, y_with, y_without, ylabel, title, save_name):
    x = np.arange(len(data))
    plt.figure(figsize=(10, 5))
    plt.scatter(x, data[y_without], label="w/o Attention", color="orange")
    plt.scatter(x, data[y_with], label="w/ Attention", color="blue")
    for i in range(len(data)):
        plt.plot([x[i], x[i]], [data[y_without].iloc[i], data[y_with].iloc[i]], color="gray", alpha=0.4)
    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{save_name}_paired.png", dpi=300)
    plt.close()

# === Step 7. 差值直方图函数 ===
def plot_delta_histogram(data, delta_col, xlabel, title, save_name):
    plt.figure(figsize=(7, 5))
    plt.hist(data[delta_col], bins=20, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"figures/{save_name}_delta_hist.png", dpi=300)
    plt.close()

# === Step 8. 绘图执行 ===
plot_paired_lines(df_merge, "dice_new_with", "dice_new_without", "Dice", "Paired Dice Comparison", "dice")
plot_delta_histogram(df_merge, "delta_dice", "ΔDice (with - without)", "ΔDice Histogram", "dice")

plot_paired_lines(df_merge, "iou_new_with", "iou_new_without", "IoU", "Paired IoU Comparison", "iou")
plot_delta_histogram(df_merge, "delta_iou", "ΔIoU (with - without)", "ΔIoU Histogram", "iou")

plot_paired_lines(df_merge, "hd95_new_px_with", "hd95_new_px_without", "HD95 (px)", "Paired HD95 Comparison", "hd95")
plot_delta_histogram(df_merge, "delta_hd95", "ΔHD95 (with - without)", "ΔHD95 Histogram", "hd95")

# === Step 9. 打印统计摘要（可以写入论文） ===
def print_stats(col_with, col_without, name):
    print(f"\n📊 {name} Summary:")
    print(f"  With attention    : mean={df_merge[col_with].mean():.4f} | std={df_merge[col_with].std():.4f} | median={df_merge[col_with].median():.4f}")
    print(f"  Without attention : mean={df_merge[col_without].mean():.4f} | std={df_merge[col_without].std():.4f} | median={df_merge[col_without].median():.4f}")
    delta_col = f"delta_{col_with.split('_')[0]}"
    print(f"  Δ (with - without): mean={df_merge[delta_col].mean():+.4f} | min={df_merge[delta_col].min():+.4f} | max={df_merge[delta_col].max():+.4f}")

print_stats("dice_new_with", "dice_new_without", "Dice")
print_stats("iou_new_with", "iou_new_without", "IoU")
print_stats("hd95_new_px_with", "hd95_new_px_without", "HD95 (px)")
