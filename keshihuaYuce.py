# 对比预测与真值掩码（中间帧）
import SimpleITK as sitk
import matplotlib.pyplot as plt

# 路径根据实际替换
gt_path = "test/input/04a04f2e-840b-47f8-a907-abe7aeab3f41.mha"
pred_path = "output/images/fetal-abdomen-segmentation/output.mha"

gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))

frame_idx = 38  # 可选帧
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gt[frame_idx], cmap='gray')
plt.title("Ground Truth")

plt.subplot(1, 2, 2)
plt.imshow(pred[frame_idx], cmap='gray')
plt.title("Prediction")
plt.show()
