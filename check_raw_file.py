import os

# 设置路径（根据你本地路径进行替换）
images_dir = "raw/images"
masks_dir = "raw/masks"

# 获取文件名列表（只保留文件名，不含路径）
image_files = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
mask_files = sorted([f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))])

# 转为集合以方便比较
image_set = set(image_files)
mask_set = set(mask_files)

# 输出总数
print(f"🖼️ 图像数量: {len(image_files)}")
print(f"🎭 掩码数量: {len(mask_files)}")

# 检查一一对应关系
if image_set == mask_set:
    print("✅ 文件名完全一致，一一对应")
else:
    print("⚠️ 文件名不一致")
    print(f"📁 仅在 images 中的文件: {image_set - mask_set}")
    print(f"📁 仅在 masks 中的文件: {mask_set - image_set}")
