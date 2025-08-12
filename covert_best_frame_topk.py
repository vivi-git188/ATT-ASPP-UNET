from pathlib import Path
import SimpleITK as sitk
import numpy as np
import imageio
import cv2
import json
from tqdm import tqdm


# 推理时固定 topk=1, neighbor_pad=0
def convert_best_frame_only(mha_root: str, out_root: str, topk: int = 3, neighbor_pad: int = 0):
    mha_root = Path(mha_root)
    out_img = Path(out_root, 'images'); out_img.mkdir(parents=True, exist_ok=True)
    out_msk = Path(out_root, 'masks');  out_msk.mkdir(parents=True, exist_ok=True)

    image_files = list((mha_root / 'images').glob('*.mha'))
    frame_dict = {}

    for f_img in tqdm(image_files, desc=f"Converting {mha_root.name} [TopK={topk}, pad={neighbor_pad}]"):
        name = f_img.stem
        f_msk = mha_root / 'masks' / f'{name}.mha'
        if not f_msk.exists():
            print(f"⚠️ 掩码不存在，跳过: {name}"); continue

        img3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_img)))  # (Z,H,W)
        msk3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_msk)))

        areas = np.array([msk.sum() for msk in msk3d])
        if areas.max() == 0:
            print(f"⚠️ 无任何掩码，跳过: {name}"); continue

        # 选 Top-K 面积帧
        idxs = areas.argsort()[::-1][:max(1, min(topk, len(areas)))]
        # 可选：在每个 Top 帧周围加入 ±neighbor_pad 的邻帧
        if neighbor_pad > 0:
            extra = []
            Z = len(areas)
            for i in idxs:
                for d in range(1, neighbor_pad+1):
                    if i-d >= 0: extra.append(i-d)
                    if i+d < Z: extra.append(i+d)
            idxs = np.unique(np.concatenate([idxs, extra]))

        saved = []
        for idx in idxs:
            sl = img3d[idx]
            msk = msk3d[idx]
            if msk.sum() == 0:  # 跳过纯空帧
                continue
            sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            msk_u8 = (msk > 0).astype(np.uint8) * 255
            filename = f'{name}_s{int(idx):03d}.png'
            imageio.imwrite(out_img / filename, sl_u8)
            imageio.imwrite(out_msk / filename, msk_u8)
            saved.append(int(idx))

        if saved:
            # 保存该病例导出的帧索引列表
            frame_dict[name] = saved

    with open(out_msk / "best_frame_indices.json", "w") as f:
        json.dump(frame_dict, f, indent=2)

    print(f'✅ {mha_root} → {out_root} 完成，共处理 {len(frame_dict)} 个样本')


# 示例调用
if __name__ == "__main__":
    convert_best_frame_only('train', 'train_png_best', topk=3, neighbor_pad=1)
    convert_best_frame_only('val', 'val_png_best', topk=3, neighbor_pad=1)
