import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import jaccard_score, f1_score

def compute_iou(pred, gt):
    return jaccard_score(gt.flatten(), pred.flatten())

def compute_dice(pred, gt):
    return f1_score(gt.flatten(), pred.flatten())

def binarize(img):
    return (img > 127).astype(np.uint8)

def evaluate(pred_dir, gt_dir):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    dices = []
    ious = []

    for pred_file in pred_dir.glob("*_mask.png"):
        case_id = pred_file.stem.replace("_mask", "")
        gt_file = gt_dir / f"{case_id}.png"
        if not gt_file.exists():
            print(f"[!] Ground truth not found: {gt_file}")
            continue

        pred = binarize(cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE))
        gt   = binarize(cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE))

        if pred.shape != gt.shape:
            print(f"[!] Shape mismatch: {pred_file.name}")
            continue

        dice = compute_dice(pred, gt)
        iou  = compute_iou(pred, gt)
        dices.append(dice)
        ious.append(iou)

        print(f"{case_id}: Dice = {dice:.4f} | IoU = {iou:.4f}")

    print("\n[Summary]")
    print(f"Average Dice: {np.mean(dices):.4f}")
    print(f"Average IoU : {np.mean(ious):.4f}")


if __name__ == "__main__":
    evaluate("./preds_fixed_frame", "./val_png_best/masks")
