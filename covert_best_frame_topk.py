from pathlib import Path
import SimpleITK as sitk
import numpy as np
import imageio, cv2, json
from tqdm import tqdm

def convert_best_frame_only(
        mha_root: str,
        out_root: str,
        topk: int = 3,
        neighbor_pad: int = 0,
        neg_ratio: int = 2,      # 负样本 : 正样本   ← 可调
        neg_cap: int = 20,       # 每病例负样本上限 ← 可调
        seed: int = 2025
):
    """把 .mha 转 png，正样本=Top-K(+邻帧)；负样本=随机抽样空帧
       n_neg = min(neg_ratio * n_pos, neg_cap)"""
    rng = np.random.default_rng(seed)

    mha_root = Path(mha_root)
    out_img = Path(out_root, 'images'); out_img.mkdir(parents=True, exist_ok=True)
    out_msk = Path(out_root, 'masks');  out_msk.mkdir(parents=True, exist_ok=True)

    image_files = list((mha_root / 'images').glob('*.mha'))
    frame_dict = {}

    for f_img in tqdm(image_files, desc=f"{mha_root.name} [TopK={topk}, pad={neighbor_pad}]"):
        name = f_img.stem
        f_msk = mha_root / 'masks' / f'{name}.mha'
        if not f_msk.exists():
            print(f"⚠️ 掩码不存在，跳过: {name}"); continue

        img3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_img)))  # (Z,H,W)
        msk3d = sitk.GetArrayFromImage(sitk.ReadImage(str(f_msk)))  # (Z,H,W)
        Z = len(img3d)

        areas = msk3d.reshape(Z, -1).sum(axis=1)                   # 每帧前景像素
        pos_topk = areas.argsort()[::-1][:max(1, min(topk, Z))]

        # —— 邻帧扩充正样本 —— #
        if neighbor_pad > 0:
            extra = []
            for i in pos_topk:
                for d in range(1, neighbor_pad + 1):
                    if i - d >= 0: extra.append(i - d)
                    if i + d <  Z: extra.append(i + d)
            pos_idxs = np.unique(np.concatenate([pos_topk, extra]))
        else:
            pos_idxs = pos_topk
        pos_idxs = np.array([i for i in pos_idxs if areas[i] > 0], dtype=int)
        n_pos = len(pos_idxs)

        saved = []

        # —— 保存正样本帧 —— #
        for idx in pos_idxs:
            sl = img3d[idx]; msk = msk3d[idx]
            sl_u8  = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            msk_u8 = (msk > 0).astype(np.uint8) * 255
            fname  = f'{name}_s{idx:03d}.png'
            imageio.imwrite(out_img / fname, sl_u8)
            imageio.imwrite(out_msk / fname, msk_u8)
            saved.append(int(idx))

        # —— 采样负样本帧 —— #
        neg_pool = np.where(areas == 0)[0]
        if n_pos > 0 and len(neg_pool) > 0:
            n_neg = int(min(neg_ratio * n_pos, neg_cap, len(neg_pool)))
            if n_neg > 0:
                neg_idxs = rng.choice(neg_pool, n_neg, replace=False)
                for idx in neg_idxs:
                    sl = img3d[idx]
                    sl_u8  = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    msk_u8 = np.zeros_like(sl_u8, dtype=np.uint8)
                    fname  = f'{name}_s{idx:03d}.png'
                    imageio.imwrite(out_img / fname, sl_u8)
                    imageio.imwrite(out_msk / fname, msk_u8)
                    saved.append(int(idx))

        # —— 若全负病例，随机取 ≤neg_cap 帧 —— #
        if n_pos == 0 and len(neg_pool) > 0:
            n_neg = min(neg_cap, 5, len(neg_pool))
            neg_idxs = rng.choice(neg_pool, n_neg, replace=False)
            for idx in neg_idxs:
                sl = img3d[idx]
                sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                msk_u8 = np.zeros_like(sl_u8, dtype=np.uint8)
                fname  = f'{name}_s{idx:03d}.png'
                imageio.imwrite(out_img / fname, sl_u8)
                imageio.imwrite(out_msk / fname, msk_u8)
                saved.append(int(idx))

        if saved:
            frame_dict[name] = sorted(saved)

    # 保存索引
    (out_msk / "best_frame_indices.json").write_text(json.dumps(frame_dict, indent=2, ensure_ascii=False))
    print(f'✅ {mha_root} → {out_root} 完成，处理 {len(frame_dict)} 个病例')

if __name__ == "__main__":
    convert_best_frame_only('train', 'train_png_best',
                            topk=3, neighbor_pad=1,
                            neg_ratio=2, neg_cap=20)
    # convert_best_frame_only('val', 'val_png_best',
    #                         topk=3, neighbor_pad=1,
    #                         neg_ratio=2, neg_cap=20)
