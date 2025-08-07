from pathlib import Path
import SimpleITK as sitk
import numpy as np
import imageio
import cv2
import json
from tqdm import tqdm

def convert_best_frame_only(mha_root: str, out_root: str):
    mha_root = Path(mha_root)
    out_img = Path(out_root, 'images'); out_img.mkdir(parents=True, exist_ok=True)
    out_msk = Path(out_root, 'masks');  out_msk.mkdir(parents=True, exist_ok=True)

    image_files = list((mha_root / 'images').glob('*.mha'))
    frame_dict = {}  # ✅ 存储最佳帧编号：{样本名: 帧编号}

    for f_img in tqdm(image_files, desc=f"Converting {mha_root.name} [Best Frame Only]"):
        name = f_img.stem
        f_msk = mha_root / 'masks' / f'{name}.mha'

        if not f_msk.exists():
            print(f"⚠️ 掩码不存在，跳过: {name}")
            continue

        # 读取图像和掩码 (Z, H, W)
        img3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_img)))
        msk3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_msk)))

        # 查找掩码最大的一帧
        areas = [msk.sum() for msk in msk3d]
        best_idx = int(np.argmax(areas))
        if areas[best_idx] == 0:
            print(f"⚠️ 无任何掩码，跳过: {name}")
            continue

        # 提取该帧图像和掩码
        sl = img3d[best_idx]
        msk = msk3d[best_idx]

        # 图像归一化到 [0,255]，掩码转为二值图
        sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        msk_u8 = (msk > 0).astype(np.uint8) * 255

        filename = f'{name}_s{best_idx:03d}.png'
        imageio.imwrite(out_img / filename, sl_u8)
        imageio.imwrite(out_msk / filename, msk_u8)

        frame_dict[name] = best_idx  # ✅ 记录最佳帧编号

    # 保存 JSON 文件
    json_path = out_msk / "best_frame_indices.json"
    with open(json_path, "w") as f:
        json.dump(frame_dict, f, indent=2)

    print(f'✅ {mha_root} → {out_root}（仅最佳帧）完成，共处理 {len(frame_dict)} 个样本')

# 示例调用
if __name__ == "__main__":
    convert_best_frame_only('train', 'train_png_best')
    convert_best_frame_only('val',   'val_png_best')
