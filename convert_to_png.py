import SimpleITK as sitk
import numpy as np
import imageio
import os
from pathlib import Path

def convert_mha_to_png_frames(mha_dir, out_img_dir, out_mask_dir):
    mha_dir = Path(mha_dir)
    out_img_dir = Path(out_img_dir)
    out_mask_dir = Path(out_mask_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    for file in sorted(mha_dir.glob('images/*.mha')):
        name = file.stem
        img = sitk.ReadImage(str(file))
        img_np = sitk.GetArrayFromImage(img)  # (D, H, W)

        mask = sitk.ReadImage(str(mha_dir / 'masks' / f'{name}.mha'))
        mask_np = sitk.GetArrayFromImage(mask)

        for i in range(img_np.shape[0]):
            max_val = img_np[i].max()
            if max_val == 0:
                frame = np.zeros_like(img_np[i], dtype=np.uint8)
            else:
                frame = (img_np[i] / max_val * 255).astype(np.uint8)

            msk = (mask_np[i] > 0).astype(np.uint8) * 255

            imageio.imwrite(str(out_img_dir / f"{name}_slice{i:03d}.png"), frame)
            imageio.imwrite(str(out_mask_dir / f"{name}_slice{i:03d}.png"), msk)

    print("✅ 所有 .mha 转换为 PNG 单帧图像完成")


# 示例调用
convert_mha_to_png_frames(
    mha_dir='./train',
    out_img_dir='./train_png/images',
    out_mask_dir='./train_png/masks'
)

convert_mha_to_png_frames(
    mha_dir='./val',
    out_img_dir='./val_png/images',
    out_mask_dir='./val_png/masks'
)
