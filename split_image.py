import pathlib, random, shutil, os
from tqdm import tqdm

random.seed(42)

root = pathlib.Path('./raw')
img_dir = root / 'images'
mask_dir = root / 'masks'

images = sorted(img_dir.glob('*'))
random.shuffle(images)

split_idx = int(0.8 * len(images))        # 80%
splits = {'train': images[:split_idx],
          'val'  : images[split_idx:]}

root2 = pathlib.Path('.')

# 创建目标文件夹
for split_name in splits:
    for sub in ['images', 'masks']:
        (root2 / split_name / sub).mkdir(parents=True, exist_ok=True)

# 复制文件，显示进度条
for split_name, split_imgs in splits.items():
    print(f"\n📦 复制 {split_name} 集合中的图像与掩码...")
    for img_path in tqdm(split_imgs, desc=f"{split_name} copying", unit="file"):
        mask_path = mask_dir / img_path.name
        shutil.copy(img_path, root2 / split_name / 'images' / img_path.name)
        shutil.copy(mask_path, root2 / split_name / 'masks' / mask_path.name)

print('\n✅ Done!')
print('Train:', len(splits["train"]), 'Val:', len(splits["val"]))
