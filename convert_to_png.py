from pathlib import Path
import SimpleITK as sitk
import numpy as np
import imageio
import cv2
from tqdm import tqdm   # ✅ 引入 tqdm

def convert_one_split(mha_root: str, out_root: str):
    mha_root = Path(mha_root)
    out_img  = Path(out_root, 'images');  out_img.mkdir(parents=True, exist_ok=True)
    out_msk  = Path(out_root, 'masks');   out_msk.mkdir(parents=True, exist_ok=True)

    image_files = list((mha_root / 'images').glob('*.mha'))

    for f_img in tqdm(image_files, desc=f"Converting {mha_root.name}"):  # ✅ 添加 tqdm 包裹循环
        name = f_img.stem
        f_msk = mha_root / 'masks' / f'{name}.mha'

        img3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_img)))   # (D,H,W)
        msk3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_msk)))   # (D,H,W)

        for i, (sl, msk) in enumerate(zip(img3d, msk3d)):
            if np.sum(msk) == 0:  # 跳过空掩码 slice
                continue
            sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            imageio.imwrite(out_img / f'{name}_s{i:03d}.png', sl_u8)
            imageio.imwrite(out_msk / f'{name}_s{i:03d}.png', (msk > 0).astype(np.uint8) * 255)

    print(f'✅ {mha_root} → {out_root} 完成')

# 示例调用
convert_one_split('train', 'train_png')
convert_one_split('val',   'val_png')
