from pathlib import Path
import cv2
import numpy as np

val_mask_dir = Path('val_png/masks')
mask_files = sorted(val_mask_dir.glob('*.png'))

empty_count = 0
non_empty_count = 0

for mask_path in mask_files:
    mask = cv2.imread(str(mask_path), 0)  # 读取灰度图
    if np.sum(mask > 0) == 0:
        empty_count += 1
    else:
        non_empty_count += 1

print(f"✅ 验证集总掩码数: {len(mask_files)}")
print(f"❌ 全为 0 的掩码数量: {empty_count}")
print(f"✅ 有前景的掩码数量: {non_empty_count}")
