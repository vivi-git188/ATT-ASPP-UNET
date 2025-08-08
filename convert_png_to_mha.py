import cv2
import numpy as np
import SimpleITK as sitk
from pathlib import Path

def convert_one_png_mask_to_mha(
    mask_png_path: str,
    ref_mha_path: str,
    frame_index: int,
    output_path: str
):
    # === Step 1: 读取 .png 掩码图 ===
    mask = cv2.imread(mask_png_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read {mask_png_path}")
    binary_mask = (mask > 127).astype(np.uint8) * 2  # Label=2 → ITK-SNAP 绿色

    # === Step 2: 读取参考 .mha 图像，用于继承空间信息 ===
    ref_img = sitk.ReadImage(ref_mha_path)
    total_frames = ref_img.GetSize()[2]

    # === Step 3: 构造 3D mask（Z轴只有一帧非零） ===
    mask_3d = np.zeros((total_frames, mask.shape[0], mask.shape[1]), dtype=np.uint8)
    if not (0 <= frame_index < total_frames):
        raise ValueError(f"Frame index {frame_index} out of bounds (0-{total_frames - 1})")
    mask_3d[frame_index] = binary_mask

    # === Step 4: 转为 SimpleITK Image，复制空间信息 ===
    mask_img = sitk.GetImageFromArray(mask_3d)
    mask_img.CopyInformation(ref_img)

    # === Step 5: 保存 .mha 掩码图 ===
    sitk.WriteImage(mask_img, output_path)
    print(f"[✓] Saved: {output_path} (frame {frame_index})")


# === 用法示例 ===
if __name__ == "__main__":
    convert_one_png_mask_to_mha(
        mask_png_path="outputs/preds_fixed_frame_circle/0d7c4d8f-6e07-4f2b-aa76-8915ce15a38f_s304_mask.png",
        ref_mha_path="val/images/0d7c4d8f-6e07-4f2b-aa76-8915ce15a38f.mha",
        frame_index=304,
        output_path="converted_mask/a38f_s304_pred_new.mha"
    )
