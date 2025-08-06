"""
The following is a the inference code for running the baseline algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-preliminary-development-phase | gzip -c > example-algorithm-preliminary-development-phase.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK
import cv2

case_id = os.getenv("CASE_ID", "output")  # fallback 名为 output.mha

TAG = os.getenv("MODEL_TAG", "baseline")

if TAG == "att_aspp":
    from model_attention_aspp import (
        FetalAbdomenSegmentation,
        select_fetal_abdomen_mask_and_frame,
    )
else:  # baseline
    from model import (
        FetalAbdomenSegmentation,
        select_fetal_abdomen_mask_and_frame,
    )
# from model import FetalAbdomenSegmentation, select_fetal_abdomen_mask_and_frame
# ./test
INPUT_PATH = Path("./test/input")
OUTPUT_PATH = Path("./test/output")
RESOURCE_PATH = Path("resources")


def run(case_id=case_id):
    # Read the input
    stacked_fetal_ultrasound_path = get_image_file_path(
        location=INPUT_PATH / "images/stacked-fetal-ultrasound")

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # print contents of input folder
    print("input folder contents:")
    print_directory_contents(INPUT_PATH)

    # Instantiate the algorithm
    algorithm = FetalAbdomenSegmentation()

    # Forward pass
    print(f"calling predict() on image: {stacked_fetal_ultrasound_path}")
    try:
        fetal_abdomen_probability_map = algorithm.predict(
            stacked_fetal_ultrasound_path, save_probabilities=True)
        print('✅ predict finished')
    except Exception as e:
        print(f"❌ predict failed with error: {e}")
        raise
    # Postprocess the output
    fetal_abdomen_postprocessed = algorithm.postprocess(
        fetal_abdomen_probability_map)

    # Select the fetal abdomen mask and the corresponding frame number
    fetal_abdomen_segmentation, fetal_abdomen_frame_number = select_fetal_abdomen_mask_and_frame(
        fetal_abdomen_postprocessed)
    print(f"掩码值分布: {np.unique(fetal_abdomen_segmentation)}")
    print(f"预测像素总数: {(fetal_abdomen_segmentation > 0).sum()}")

    # -----------------------------------------------------------
    # ⬆︎⬆︎⬆︎  这里开始是『把 224×224 掩码放大回 744×562』的补丁  ⬆︎⬆︎⬆︎
    # -----------------------------------------------------------
    import cv2, SimpleITK

    # 1️⃣ 读取原始 3-D 超声序列，拿到真实的 (H,W)
    ref_img  = SimpleITK.ReadImage(stacked_fetal_ultrasound_path[0])
    n_frames = SimpleITK.GetArrayFromImage(ref_img).shape[0]  # ← 840
    _, ref_h, ref_w = SimpleITK.GetArrayFromImage(ref_img).shape   # (N,H,W)

    # 2️⃣ 若当前掩码尺寸不是原尺寸，就最近邻放大 / 缩小
    if fetal_abdomen_segmentation.shape != (ref_h, ref_w):
        fetal_abdomen_segmentation = cv2.resize(
            fetal_abdomen_segmentation.astype("uint8"),   # 先转 uint8 防止 cv2 出警告
            (ref_w, ref_h),                               # 注意 (W,H) 顺序
            interpolation=cv2.INTER_NEAREST               # 最近邻，不会引入 0/1 以外的值
        )

    # 3️⃣ （可选）确保仍然是二值
    fetal_abdomen_segmentation = (fetal_abdomen_segmentation > 0).astype("uint8")
    # -----------------------------------------------------------
    # ⬇︎⬇︎⬇︎  下面继续原来的 write_array_as_image_file 调用  ⬇︎⬇︎⬇︎

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/fetal-abdomen-segmentation",
        array=fetal_abdomen_segmentation,
        frame_number=fetal_abdomen_frame_number,  # 305
        number_of_frames=n_frames,  # ✅ 传 840 而不是默认 128
        filename = f"{case_id}.mha"  # 👈 传入你希望的输出文件名
    )
    write_json_file(
        location=OUTPUT_PATH / "fetal-abdomen-frame-number.json",
        content=fetal_abdomen_frame_number
    )

    # Print the output
    print("output folder contents:")
    print_directory_contents(OUTPUT_PATH)

    # Print shape and type of the output
    print("\nprinting output shape and type:")
    print(f"shape: {fetal_abdomen_segmentation.shape}")
    print(f"type: {type(fetal_abdomen_segmentation)}")
    print(f"dtype: {fetal_abdomen_segmentation.dtype}")
    print(f"unique values: {np.unique(fetal_abdomen_segmentation)}")
    print(f"frame number: {fetal_abdomen_frame_number}")
    print(type(fetal_abdomen_frame_number))

    return 0


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))

# gamma
def gamma_transform(img, gamma=1.5):
    img = img / 255.0
    img = np.power(img, gamma)
    return np.uint8(img * 255)

def load_image_file_as_array(*, location):
    """
    读取 3-D 超声序列，对 **每一帧** 做 CLAHE + 中值滤波，
    返回 (1, N, H, W) 的 float32 数组，取值 0-1
    """
    import SimpleITK, cv2, numpy as np
    from pathlib import Path

    location = Path(location)
    itk_img  = SimpleITK.ReadImage(str(location))
    array    = SimpleITK.GetArrayFromImage(itk_img)        # (N, H, W)
    print(f"[DEBUG] Original array shape: {array.shape}")

    if array.ndim != 3:
        raise ValueError(f"Expected 3-D image (frames, H, W), got {array.shape}")

    # 输出文件夹，用来存几张对比图（可选）
    out_dir = Path("test/output/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 预先创建 CLAHE 实例
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))#clipLimit = 0.8也不错

    enhanced_stack = []
    for i, sl in enumerate(array):             # 遍历原始每一帧
        # 归一化到 [0,255] → uint8
        sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # CLAHE 局部对比度增强
        clahe_img  = clahe.apply(sl_u8)
        # 中值滤波去噪
        filtered   = cv2.medianBlur(clahe_img, 3)
        enhanced_stack.append(filtered)

        # 可选择性地把几帧保存出来看看
        if i in (0, len(array)//2, len(array)-1):
            cv2.imwrite(str(out_dir / f"frame{i:03d}_orig.png"), sl_u8)
            cv2.imwrite(str(out_dir / f"frame{i:03d}_enh.png"), filtered)

    # 重新堆叠回 3-D (N, H, W) 并归一化
    stacked_array = np.stack(enhanced_stack, axis=0)
    array_float   = stacked_array.astype(np.float32) / 255.0    # 0-1

    # (1, N, H, W) ——nnUNet 2-D 网络的期望输入格式
    return array_float[np.newaxis, ...]

    # Convert it to a Numpy array
    # return SimpleITK.GetArrayFromImage(result)

# Get image file path from input folder


def get_image_file_path(*, location):
    input_files = glob(str(location / "*.tiff")) + \
        glob(str(location / "*.mha"))
    return input_files


import numpy as np
import SimpleITK
from pathlib import Path

def write_array_as_image_file(
        *, location: Path, array: np.ndarray,
        frame_number: int = None,
        number_of_frames: int = 128,   # ← 新增，默认仍兼容旧逻辑
        filename: str = "output.mha"  # 👈 默认仍为原名
):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"
    array = np.squeeze(array)  # 去掉多余维度
    # 1️⃣ 仅支持 2D mask（单帧）
    assert array.ndim == 2, f"Expected a 2D array, got {array.ndim}D."

    # 2️⃣ 保留 float 概率图用于调试或软 Dice 评估
    prob_map = array.astype(np.float32)

    # 3️⃣ 转成 3D：放入 128 帧堆栈中指定帧位
    array_3d = convert_2d_mask_to_3d(
        mask_2d=prob_map,
        frame_number=frame_number,
        number_of_frames=number_of_frames  # ← 这里用传进来的真实帧数
    )
    # 4️⃣ 确保最后输出是二值掩码 ∈ {0, 1}，且为 uint8
    array_3d = np.where(array_3d > 0.5, 1, 0).astype(np.uint8)

    # 5️⃣ （保险）检查是否确实只有 0 和 1
    unique_vals = np.unique(array_3d)
    print("DEBUG: Unique values in 3D mask:", unique_vals)
    assert set(unique_vals).issubset({0, 1}), f"Non-binary values detected: {unique_vals}"

    # 6️⃣ 写入 .mha，显式指定类型为 sitkUInt8
    image = SimpleITK.GetImageFromArray(array_3d)
    image = SimpleITK.Cast(image, SimpleITK.sitkUInt8)
    image.SetSpacing([0.28, 0.28, 0.28])
    SimpleITK.WriteImage(
        image,
        location / filename,
        useCompression=True,
    )

    # 7️⃣ 读取写入后的文件做最终验证
    check_img = SimpleITK.ReadImage(location / filename)

    arr_check = SimpleITK.GetArrayFromImage(check_img)
    print("✅ Saved output.mha info:")
    print("Shape:", arr_check.shape)
    print("Spacing:", check_img.GetSpacing())
    print("Unique values in saved file:", np.unique(arr_check))


def convert_2d_mask_to_3d(*, mask_2d, frame_number, number_of_frames):
    # 把 1 → 2，这样 ITK-SNAP 会显示为绿色
    mask_2d = np.where(mask_2d == 1, 2, 0).astype(np.uint8)
    # Convert a 2D mask to a 3D mask
    mask_3d = np.zeros((number_of_frames, *mask_2d.shape), dtype=np.uint8)
    # If frame_number == -1, return a 3D mask with all zeros
    if frame_number == -1:
        return mask_3d
    # If frame_number is within the valid range, set the corresponding frame to the 2D mask
    if frame_number is not None and 0 <= frame_number < number_of_frames:
        mask_3d[frame_number, :, :] = mask_2d
        return mask_3d
    # If frame_number is None or out of bounds, raise a ValueError
    else:
        raise ValueError(
            f"frame_number must be between -1 and {number_of_frames - 1}, got {frame_number}."
        )


def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print_directory_contents(child_path)
        else:
            print(child_path)


def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(
        f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(
            f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(
            f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())