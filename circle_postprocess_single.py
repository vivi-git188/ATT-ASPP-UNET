# circle_postprocess_single.py
import argparse
import numpy as np
import cv2
from pathlib import Path

def largest_component(bin_mask: np.ndarray) -> np.ndarray:
    if bin_mask.max() == 0:
        return bin_mask
    num, labels = cv2.connectedComponents(bin_mask)
    if num <= 1:
        return bin_mask
    mx = 1 + np.argmax([(labels == i).sum() for i in range(1, num)])
    return (labels == mx).astype(np.uint8)

def circle_from_area(bin_mask: np.ndarray):
    ys, xs = np.nonzero(bin_mask)
    if len(xs) == 0:
        return (bin_mask.shape[1]/2, bin_mask.shape[0]/2), 0.0
    cx = xs.mean()
    cy = ys.mean()
    area = bin_mask.sum()
    r = np.sqrt(area / np.pi)
    return (cx, cy), r

def circle_min_enclosing(bin_mask: np.ndarray):
    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = bin_mask.shape
        return (w/2, h/2), 0.0
    (x, y), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return (x, y), r

def clamp_circle(h, w, center, r):
    cx, cy = center
    r = min(r, cx, cy, w-1-cx, h-1-cy)
    return (cx, cy), max(0.0, r)

def render_circle(h, w, center, r, label=255):
    cx, cy = center
    r = max(0.0, float(r))
    out = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), color=label, thickness=-1)
    return out

def postprocess_to_circle(mask: np.ndarray, mode="area", blend=0.0):
    h, w = mask.shape
    comp = largest_component(mask)
    if comp.sum() == 0:
        return np.zeros_like(mask)

    (cx_a, cy_a), r_a = circle_from_area(comp)
    (cx_e, cy_e), r_e = circle_min_enclosing(comp)

    cx, cy = cx_a, cy_a
    r = (1 - blend) * r_a + blend * r_e
    (cx, cy), r = clamp_circle(h, w, (cx, cy), r)
    return render_circle(h, w, (cx, cy), r, label=255)

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Postprocess single mask to circle")
    ap.add_argument("--mask", required=True, help="输入的 mask PNG")
    ap.add_argument("--out", required=True, help="输出拟合圆形的 PNG")
    ap.add_argument("--mode", choices=["area","enclose"], default="area", help="拟圆策略")
    ap.add_argument("--blend", type=float, default=0.0, help="0=纯area, 1=纯enclose")
    args = ap.parse_args()

    m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Cannot read: {args.mask}")
    bin_mask = (m > 0).astype(np.uint8)
    circ = postprocess_to_circle(bin_mask, mode=args.mode, blend=args.blend)
    cv2.imwrite(args.out, circ)
    print(f"[✓] Saved circle mask → {args.out}")
