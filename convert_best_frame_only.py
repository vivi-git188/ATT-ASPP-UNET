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

def convert_best_frame_only(mha_root: str, out_root: str):
    t0 = perf_counter()
    start_wall = datetime.now(timezone.utc)

    mha_root = Path(mha_root)
    out_img = Path(out_root, 'images'); out_img.mkdir(parents=True, exist_ok=True)
    out_msk = Path(out_root, 'masks');  out_msk.mkdir(parents=True, exist_ok=True)

    image_files = list((mha_root / 'images').glob('*.mha'))
    frame_dict = {}  # {样本名: 最佳帧编号}

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

        frame_dict[name] = best_idx

    # 保存 JSON
    json_path = out_msk / "best_frame_indices.json"
    with open(json_path, "w") as f:
        json.dump(frame_dict, f, indent=2)

    # 保存 CSV（case_id,best_frame_idx）
    csv_path = Path(out_root) / "mapping.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "best_frame_idx"])
        for case_id, idx in sorted(frame_dict.items()):
            w.writerow([case_id, idx])

    elapsed = perf_counter() - t0
    end_wall = datetime.now(timezone.utc)

    processed = len(frame_dict)
    print(f'{mha_root} → {out_root}（仅最佳帧）完成，处理 {processed} 个样本，用时 {elapsed:.3f}s（平均 {elapsed/processed:.3f}s/样本）' if processed else
          f'{mha_root} → {out_root}（仅最佳帧）完成，但没有可用样本。总用时 {elapsed:.3f}s')
    print(f'已写出映射: {csv_path}')
    # 返回统计供主程序汇总
    return {
        "dataset": mha_root.name,
        "processed": processed,
        "total_s": elapsed,
        "avg_s_per_sample": (elapsed/processed) if processed else 0.0,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "start_utc": start_wall.isoformat(),
        "end_utc": end_wall.isoformat(),
    }

# 示例调用：同时打印“总处理时间”
if __name__ == "__main__":
    t_all = perf_counter()
    stat_train = convert_best_frame_only('train', 'train_png_best')
    stat_val   = convert_best_frame_only('val',   'val_png_best')
    total_all = perf_counter() - t_all

    print("\n=== 汇总 ===")
    print(f"train: {stat_train['processed']} 个样本, {stat_train['total_s']:.3f}s")
    print(f"val  : {stat_val['processed']} 个样本, {stat_val['total_s']:.3f}s")
    print(f"总处理时间: {total_all:.3f}s")
