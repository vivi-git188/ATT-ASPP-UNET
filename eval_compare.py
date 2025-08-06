import glob, os, time
import numpy as np, SimpleITK as sitk
from medpy.metric import binary as M
from tqdm import tqdm

pred_dir_A = "outputs/att_aspp"
pred_dir_B = "outputs/baseline"
gt_dir     = "val/masks"

def dice(a, b):  return M.dc(a, b)
def iou(a, b):   return M.jc(a, b)
# def hd95(a, b):  return M.hd95(a, b)   # 先别算

def to_array(p):
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(bool)

def metrics(p, g):
    # return dice(p, g), iou(p, g), hd95(p, g)
    return dice(p, g), iou(p, g)

rows, gt_paths = [], sorted(glob.glob(f"{gt_dir}/*.mha"))
for gt_path in tqdm(gt_paths, ncols=100, desc="Evaluating"):
    case = os.path.basename(gt_path)[:-4]
    t0 = time.time()
    try:
        pred_A = to_array(f"{pred_dir_A}/{case}.mha")
        pred_B = to_array(f"{pred_dir_B}/{case}.mha")
        gt     = to_array(gt_path)
        rows.append((case, *metrics(pred_A, gt), *metrics(pred_B, gt)))
        tqdm.write(f"[{case}] ✔  {time.time()-t0:.1f}s")
    except Exception as e:
        tqdm.write(f"[{case}] ❌ {e}")

print(f"{'Case':^32} | A-Dice | A-IoU | B-Dice | B-IoU")
for r in rows:
    print("{:32s} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(*r))
