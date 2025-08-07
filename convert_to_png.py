from pathlib import Path
import SimpleITK as sitk
import numpy as np
import imageio
import cv2
from tqdm import tqdm

def convert_one_split(mha_root: str, out_root: str, keep_empty_ratio: float = 0.2):
    mha_root = Path(mha_root)
    out_img = Path(out_root, 'images'); out_img.mkdir(parents=True, exist_ok=True)
    out_msk = Path(out_root, 'masks');  out_msk.mkdir(parents=True, exist_ok=True)

    image_files = list((mha_root / 'images').glob('*.mha'))

    for f_img in tqdm(image_files, desc=f"Converting {mha_root.name}"):
        name = f_img.stem
        f_msk = mha_root / 'masks' / f'{name}.mha'

        # 读取图像与掩码
        img3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_img)))  # shape: (D, H, W)
        msk3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_msk)))  # shape: (D, H, W)

        for i, (sl, msk) in enumerate(zip(img3d, msk3d)):
            has_mask = msk.sum() > 0
            keep = has_mask or (np.random.rand() < keep_empty_ratio)
            if not keep:
                continue

            # 图像归一化到 [0, 255]
            sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            msk_u8 = (msk > 0).astype(np.uint8) * 255  # 二值掩码

            # 保存 PNG
            filename = f'{name}_s{i:03d}.png'
            imageio.imwrite(out_img / filename, sl_u8)
            imageio.imwrite(out_msk / filename, msk_u8)

    print(f'✅ {mha_root} → {out_root} 完成')

# 示例调用
convert_one_split('train', 'train_png')
convert_one_split('val',   'val_png')
