from pathlib import Path
import SimpleITK as sitk
import numpy as np
import imageio
import cv2
import json
from tqdm import tqdm
import csv
from time import perf_counter
from datetime import datetime, timezone
### NEW ↓
import argparse
# ------------------------------------------------------------


def convert_best_frame_only(mha_root: str, out_root: str, split: str = "train"):  ### MOD
    """
    参数
    ----
    mha_root : 数据集根目录（含 images / masks 子文件夹）
    out_root : PNG 输出目录
    split    : 'train' | 'val' | 'test' —— 仅 val / test 会把 spacing 写入 JSON
    """
    t0 = perf_counter()
    start_wall = datetime.now(timezone.utc)

    mha_root = Path(mha_root)
    out_img = Path(out_root, 'images'); out_img.mkdir(parents=True, exist_ok=True)
    out_msk = Path(out_root, 'masks');  out_msk.mkdir(parents=True, exist_ok=True)

    image_files = list((mha_root / 'images').glob('*.mha'))
    ### MOD ↓—— dict 现在同时存 frame_idx 与 spacing
    frame_dict = {}        # {样本名: {"frame_idx": idx, "spacing": [sx,sy]}}

    for f_img in tqdm(image_files, desc=f"Converting {mha_root.name} [Best Frame Only]"):
        name = f_img.stem
        f_msk = mha_root / 'masks' / f'{name}.mha'

        if not f_msk.exists():
            print(f"⚠️ 掩码不存在，跳过: {name}")
            continue

        # (Z, H, W)
        img3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_img)))
        msk3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_msk)))

        # 找最佳帧（掩码像素和最大）
        areas = [msk.sum() for msk in msk3d]
        best_idx = int(np.argmax(areas))
        if areas[best_idx] == 0:
            print(f"⚠️ 无任何掩码，跳过: {name}")
            continue

        # 取该帧并写 PNG
        sl = img3d[best_idx]
        msk = msk3d[best_idx]
        sl_u8  = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        msk_u8 = (msk > 0).astype(np.uint8) * 255

        filename = f'{name}_s{best_idx:03d}.png'
        imageio.imwrite(out_img / filename, sl_u8)
        imageio.imwrite(out_msk / filename, msk_u8)

        ### MOD ↓—— 读取 spacing 并一并保存
        sx, sy, _ = sitk.ReadImage(str(f_img)).GetSpacing()
        frame_dict[name] = {"frame_idx": best_idx, "spacing": [float(sx), float(sy)]}

    # 保存 JSON（train 也保存无妨，推理阶段可选加载）
    json_path = out_msk / "best_frame_indices.json"
    with open(json_path, "w") as f:
        json.dump(frame_dict, f, indent=2)

    # 旧版 CSV 仍只保留 frame_idx，方便人眼查看
    csv_path = Path(out_root) / "mapping.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "best_frame_idx"])
        for case_id, info in sorted(frame_dict.items()):
            w.writerow([case_id, info["frame_idx"]])

    elapsed = perf_counter() - t0
    end_wall = datetime.now(timezone.utc)

    processed = len(frame_dict)
    print(f'{mha_root} → {out_root}（仅最佳帧）完成，处理 {processed} 个样本，用时 {elapsed:.3f}s')
    print(f'已写出映射: {csv_path}')
    return {
        "dataset": mha_root.name,
        "processed": processed,
        "total_s": elapsed,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "start_utc": start_wall.isoformat(),
        "end_utc": end_wall.isoformat(),
    }


# ------- CLI：任选 train / val / test ---------  ### NEW
if __name__ == "__main__":
    convert_best_frame_only('val', 'val_png_best_ac', 'val')
    # stat_train = convert_best_frame_only('train', 'train_png_best')
    # stat_val = convert_best_frame_only('val', 'val_png_best')

