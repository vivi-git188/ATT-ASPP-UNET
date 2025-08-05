# eval_compare.py
import glob, numpy as np, SimpleITK as sitk
from medpy.metric import binary as M

pred_dir_A = "outputs/att_aspp"     # Attention-ASPP-UNet
pred_dir_B = "outputs/baseline"     # Baseline
gt_dir     = "val/labels"           # 验证集真值

def dice(arr1, arr2):
    return M.dc(arr1, arr2)

def iou(arr1, arr2):
    return M.binary_jaccard(arr1, arr2)

def hd95(arr1, arr2):
    return M.hd95(arr1, arr2)

def to_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(bool)

def metrics(pred, gt):
    return dice(pred, gt), iou(pred, gt), hd95(pred, gt)

rows = []
for gt_path in glob.glob(f"{gt_dir}/*.mha"):
    case = gt_path.split("/")[-1].split(".")[0]
    pred_A = to_array(f"{pred_dir_A}/{case}.mha")
    pred_B = to_array(f"{pred_dir_B}/{case}.mha")
    gt     = to_array(gt_path)

    rows.append((
        case,
        *metrics(pred_A, gt),
        *metrics(pred_B, gt)
    ))

print(f"{'Case':^32} | A-Dice | A-IoU | A-HD95 | B-Dice | B-IoU | B-HD95")
for r in rows:
    print("{:32s} | {:.4f} | {:.4f} | {:6.2f} | {:.4f} | {:.4f} | {:6.2f}".format(*r))
