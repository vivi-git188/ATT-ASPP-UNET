# import os
# import shutil
#
# train_images = "train/masks"
# val_images = "val/masks"
# duplicates_dir = "train/duplicates"
#
# # 创建 duplicates 文件夹
# os.makedirs(duplicates_dir, exist_ok=True)
#
# # 获取文件名集合
# train_files = set(os.listdir(train_images))
# val_files = set(os.listdir(val_images))
#
# # 找交集（重名文件）
# duplicates = train_files.intersection(val_files)
#
# print(f"发现 {len(duplicates)} 个重名文件")
#
# # 移动重名文件
# for fname in duplicates:
#     src_path = os.path.join(train_images, fname)
#     dst_path = os.path.join(duplicates_dir, fname)
#
#     if os.path.exists(src_path):
#         shutil.move(src_path, dst_path)
#
# print(f"已将 {len(duplicates)} 个重名文件移到 {duplicates_dir}，并从 {train_images} 删除。")


import os

train_root = "val_png_top3_pad1"

for subdir in os.listdir(train_root):
    subpath = os.path.join(train_root, subdir)
    if os.path.isdir(subpath):
        files = [f for f in os.listdir(subpath) if os.path.isfile(os.path.join(subpath, f))]
        print(f"{subdir}: {len(files)} 个文件")

