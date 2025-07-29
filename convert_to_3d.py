import SimpleITK as sitk
from pathlib import Path

OUTPUT_PATH = Path("/output")
# 1. 读入 2-D 掩码
mask2d = sitk.ReadImage(OUTPUT_PATH / "output.mha")          # 大小: [X, Y]

# 2. 在 z 方向加一层深度
mask3d = sitk.JoinSeries(mask2d)               # 大小: [X, Y, 1]

# 3. 可选——把 z 轴分辨率设成 1 mm，方便查看
spacing_xy = mask2d.GetSpacing()               # (sx, sy)
mask3d.SetSpacing(spacing_xy + (1.0,))         # (sx, sy, 1.0)

# 4. 保存
sitk.WriteImage(mask3d, "output_3d.mha")
print("Done: output_3d.mha")
