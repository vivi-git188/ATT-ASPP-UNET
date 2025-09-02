#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert 3D .mha volumes (images + masks) into per-slice PNG pairs **(with per-frame path index)**.
"""
import csv
from pathlib import Path
import argparse, json, cv2, imageio, numpy as np, SimpleITK as sitk
from tqdm import tqdm

def get_xy_spacing_mm(img_sitk):
    """
    Read the pixel spacing (sx, sy) of the SimpleITK image, with units in mm. If the spacing information is missing or abnormal, return None.
    """
    try:
        sx, sy, *_ = img_sitk.GetSpacing()          # SITK 顺序 (sx, sy, sz)
        return (float(sx), float(sy)) if sx > 0 and sy > 0 else None
    except Exception:
        return None


def normalize_slice_to_u8(sl: np.ndarray):
    sl = sl.astype(np.float32)
    p1, p99 = np.percentile(sl, (1, 99))
    if p99 - p1 < 1e-5:                            # 全黑/全白
        return np.zeros_like(sl, np.uint8)
    sl = np.clip(sl, p1, p99)
    sl = (sl - p1) / (p99 - p1 + 1e-5)
    return (sl * 255).round().astype(np.uint8)


def decide_threshold_px(min_area_mm2, min_area_px, sx_sy_mm):
    """
   Calculate the number of pixels of the "prospective area threshold".
    """
    thr_px = int(max(1, min_area_px))
    if min_area_mm2 and sx_sy_mm:
        sx, sy = sx_sy_mm
        thr_from_mm = int(np.ceil(min_area_mm2 / (sx * sy)))
        thr_px = max(thr_px, thr_from_mm)
    return thr_px


def convert_frames_with_negatives(
    mha_root: str,
    out_root: str,
    topk: int = 3,
    neighbor_pad: int = 0,
    min_area_mm2: float | None = 80.0,
    min_area_px: int = 100,
    neg_strategy: str = "random",
    neg_ratio: float = 0.0,
    neg_cap: int = 5,
    neg_stride: int = 5,
    seed: int = 2025,
    export_neg_only: bool = False,
    neg_total_cap: int = 0,
):
    assert neg_strategy in ("all", "random", "stride")
    rng = np.random.default_rng(seed)

    mha_root = Path(mha_root)
    out_img = Path(out_root, "images"); out_img.mkdir(parents=True, exist_ok=True)
    out_msk = Path(out_root, "masks");  out_msk.mkdir(parents=True, exist_ok=True)

    image_files = list((mha_root / "images").glob("*.mha"))
    image_files = list(rng.permutation(image_files))  # 打乱顺序方便全局 cap

    index_dict: dict[str, dict] = {}
    neg_total_saved = 0

    for f_img in tqdm(image_files, desc=f"{mha_root.name} TopK={topk}, neg={neg_strategy}"):
        name = f_img.stem
        f_msk = mha_root / "masks" / f"{name}.mha"
        if not f_msk.exists():
            print(f"loss code---skip")
            continue

        img_itk = sitk.ReadImage(str(f_img))
        msk_itk = sitk.ReadImage(str(f_msk))
        img3d = sitk.GetArrayFromImage(img_itk)  # (Z,H,W)
        msk3d = sitk.GetArrayFromImage(msk_itk)
        Z = img3d.shape[0]

        sx_sy = get_xy_spacing_mm(img_itk) or get_xy_spacing_mm(msk_itk)
        if sx_sy is None:
            sx_sy = (1.0, 1.0)

        thr_px = decide_threshold_px(min_area_mm2, min_area_px, sx_sy)

        if thr_px is None:
            thr_px = int(min_area_px if min_area_px is not None else 0)

        areas_px = (msk3d > 0).reshape(Z, -1).sum(1)
        is_pos = areas_px >= thr_px
        pos_pool = np.where(is_pos)[0]
        neg_pool = np.where(~is_pos)[0]

        pos_idxs = np.array([], dtype=int)
        if not export_neg_only and pos_pool.size and topk > 0:
            order = np.argsort(areas_px[pos_pool])[::-1]
            top = pos_pool[order[:min(topk, len(pos_pool))]]
            if neighbor_pad > 0:
                extra = []
                for i in top:
                    extra += [j for j in range(i-neighbor_pad, i+neighbor_pad+1) if 0<=j<Z]
                pos_idxs = np.unique(np.concatenate([top, extra]))
                pos_idxs = pos_idxs[is_pos[pos_idxs]]
            else:
                pos_idxs = np.unique(top)

        neg_idxs = np.array([], dtype=int)
        if neg_pool.size:
            if neg_strategy == "all":
                neg_idxs = neg_pool
            elif neg_strategy == "random":
                n_neg = (len(neg_pool) if export_neg_only or not pos_idxs.size
                         else int(min(np.ceil(neg_ratio*len(pos_idxs)), neg_cap)))
                if n_neg > 0:
                    neg_idxs = rng.choice(neg_pool, n_neg, replace=False)
            elif neg_strategy == "stride":
                neg_idxs = neg_pool[::max(1, neg_stride)][:neg_cap]

        if neg_total_cap > 0:
            remain = neg_total_cap - neg_total_saved
            if remain <= 0:
                neg_idxs = np.array([], dtype=int)
            elif len(neg_idxs) > remain:
                neg_idxs = rng.choice(neg_idxs, remain, replace=False)

        saved_pos, saved_neg, saved_frames = [], [], []        # ← NEW

        for idx in pos_idxs:
            sl   = img3d[idx]; msk = (msk3d[idx] > 0).astype(np.uint8)
            sl_u8 = normalize_slice_to_u8(sl); msk_u8 = (msk*255).astype(np.uint8)
            fname = f"{name}_s{idx:03d}.png"
            imageio.imwrite(out_img / fname, sl_u8)
            imageio.imwrite(out_msk / fname, msk_u8)
            saved_pos.append(int(idx))
            ### NEW: 记录逐帧路径 ###
            saved_frames.append({"idx": int(idx), "cls": "pos",
                                 "img": f"images/{fname}", "mask": f"masks/{fname}"})

        for idx in neg_idxs:
            sl_u8 = normalize_slice_to_u8(img3d[idx])
            fname = f"{name}_s{idx:03d}.png"
            imageio.imwrite(out_img / fname, sl_u8)
            imageio.imwrite(out_msk / fname, np.zeros_like(sl_u8, np.uint8))
            saved_neg.append(int(idx))
            ### NEW ###
            saved_frames.append({"idx": int(idx), "cls": "neg",
                                 "img": f"images/{fname}", "mask": f"masks/{fname}"})

        neg_total_saved += len(saved_neg)

        if saved_pos or saved_neg:
            index_dict[name] = {
                "pos": sorted(saved_pos),
                "neg": sorted(saved_neg),
                "frames": saved_frames,                   
                "_meta": {
                    "thr_px": int(thr_px),
                    "spacing_xy_mm": None if sx_sy is None else [float(sx_sy[0]), float(sx_sy[1])],
                    "min_area_mm2": None if min_area_mm2 is None else float(min_area_mm2)
                }
            }

    (out_msk / "frame_indices.json").write_text(
        json.dumps(index_dict, indent=2, ensure_ascii=False)
    )
    csv_path = Path(out_root) / "mapping.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "frame_idx"])
        for case_id, info in sorted(index_dict.items()):
            for fr in info.get("frames", []):
                w.writerow([case_id,
                            fr["idx"]])


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mha_root", type=str, required=True, help="输入 MHA 根目录（包含 images/ 和 masks/）")
    parser.add_argument("--out_root", type=str, required=True, help="输出 PNG 根目录")

    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--neighbor_pad", type=int, default=0)
    parser.add_argument("--min_area_mm2", type=float, default=80.0)
    parser.add_argument("--min_area_px", type=int, default=100)
    parser.add_argument("--neg_strategy", type=str, default="random", choices=["all", "random", "stride"])
    parser.add_argument("--neg_ratio", type=float, default=0.0)
    parser.add_argument("--neg_cap", type=int, default=5)
    parser.add_argument("--neg_stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--export_neg_only", action="store_true")
    parser.add_argument("--neg_total_cap", type=int, default=0)
    return parser

def apply_stage_presets(args):
    return args


if __name__ == "__main__":
    args = apply_stage_presets(build_argparser().parse_args())
    convert_frames_with_negatives(
        mha_root=args.mha_root,
        out_root=args.out_root,
        topk=args.topk,
        neighbor_pad=args.neighbor_pad,
        min_area_mm2=args.min_area_mm2,
        min_area_px=args.min_area_px,
        neg_strategy=args.neg_strategy,
        neg_ratio=args.neg_ratio,
        neg_cap=args.neg_cap,
        neg_stride=args.neg_stride,
        seed=args.seed,
        export_neg_only=args.export_neg_only,
        neg_total_cap=args.neg_total_cap,
    )
