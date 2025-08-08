# circle_postprocess.py
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

def load_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Cannot read: {path}")
    return (m > 0).astype(np.uint8)

def largest_component(bin_mask: np.ndarray) -> np.ndarray:
    if bin_mask.max() == 0:
        return bin_mask
    num, labels = cv2.connectedComponents(bin_mask)
    if num <= 1:
        return bin_mask
    mx = 1 + np.argmax([(labels == i).sum() for i in range(1, num)])
    return (labels == mx).astype(np.uint8)

def circle_from_area(bin_mask: np.ndarray) -> tuple[tuple[float,float], float]:
    """同面积圆：中心=质心，半径= sqrt(area/pi)"""
    ys, xs = np.nonzero(bin_mask)
    if len(xs) == 0:
        return (bin_mask.shape[1]/2, bin_mask.shape[0]/2), 0.0
    cx = xs.mean()
    cy = ys.mean()
    area = bin_mask.sum()
    r = np.sqrt(area / np.pi)
    return (cx, cy), r

def circle_min_enclosing(bin_mask: np.ndarray) -> tuple[tuple[float,float], float]:
    """最小包围圆：基于最大连通域轮廓"""
    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = bin_mask.shape
        return (w/2, h/2), 0.0
    (x, y), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return (x, y), r

def render_circle(h: int, w: int, center: tuple[float,float], r: float, label: int=255) -> np.ndarray:
    cx, cy = center
    r = max(0.0, float(r))
    out = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), color=label, thickness=-1)
    return out

def clamp_circle(h: int, w: int, center: tuple[float,float], r: float) -> tuple[tuple[float,float], float]:
    """把圆限制在图像范围内，避免越界导致被裁掉太多"""
    cx, cy = center
    r = min(r, cx, cy, w-1-cx, h-1-cy)
    return (cx, cy), max(0.0, r)

def postprocess_to_circle(mask: np.ndarray, mode: str="area", blend: float=0.0) -> np.ndarray:
    """
    mode:
      - 'area': 同面积圆（默认），中心=质心，半径= sqrt(area/pi)
      - 'enclose': 最小包围圆
    blend: 0~1，两个半径的加权融合（0=纯 area，1=纯 enclose）
    """
    h, w = mask.shape
    comp = largest_component(mask)

    if comp.sum() == 0:
        return np.zeros_like(mask)

    (cx_a, cy_a), r_a = circle_from_area(comp)
    (cx_e, cy_e), r_e = circle_min_enclosing(comp)

    # 中心用质心（更稳定），可按需切换为包围圆中心
    cx, cy = cx_a, cy_a
    r = (1 - blend) * r_a + blend * r_e

    (cx, cy), r = clamp_circle(h, w, (cx, cy), r)
    circ = render_circle(h, w, (cx, cy), r, label=255)
    return circ

def process_dir(in_dir: Path, out_dir: Path, mode: str, blend: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}])
    for p in tqdm(imgs, desc=f"Postprocess → {mode} (blend={blend})"):
        m = load_mask(p)
        circ = postprocess_to_circle(m, mode=mode, blend=blend)
        cv2.imwrite(str(out_dir / p.name), circ)

def main():
    ap = argparse.ArgumentParser("Circularize masks to match round GT")
    ap.add_argument("--in_dir", required=True, help="输入掩码文件夹（预测的二值图）")
    ap.add_argument("--out_dir", required=True, help="输出文件夹")
    ap.add_argument("--mode", choices=["area","enclose"], default="area",
                    help="拟圆策略：'area' 同面积圆（默认），'enclose' 最小包围圆")
    ap.add_argument("--blend", type=float, default=0.0,
                    help="两种半径的加权融合，0=纯area，1=纯enclose，默认0")
    args = ap.parse_args()
    process_dir(Path(args.in_dir), Path(args.out_dir), args.mode, args.blend)

if __name__ == "__main__":
    main()
