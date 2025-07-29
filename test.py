import SimpleITK as sitk

# 替换为你的输入文件路径
image_path = "test/input/images/stacked-fetal-ultrasound/1dab0e37-e5e4-41d4-b132-32103b385c78.mha"

# 读取图像
try:
    image = sitk.ReadImage(image_path)
    size = image.GetSize()  # 返回 (width, height, depth)
    spacing = image.GetSpacing()
    print(f"✅ File: {image_path}")
    print(f"Shape (W, H, D): {size}")
    print(f"Spacing: {spacing}")
    print(f"Pixel Type: {sitk.GetPixelIDValueAsString(image.GetPixelID())}")
except Exception as e:
    print(f"❌ Error reading image: {e}")
