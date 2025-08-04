import pathlib, random, shutil, os
random.seed(42)

root = pathlib.Path('./raw')
img_dir = root/'images'
mask_dir = root/'masks'

images = sorted(img_dir.glob('*'))
random.shuffle(images)

split_idx = int(0.8 * len(images))        # 80%
splits = {'train': images[:split_idx],
          'val'  : images[split_idx:]}

for split_name, split_imgs in splits.items():
    for sub in ['images', 'masks']:
        (root/split_name/sub).mkdir(parents=True, exist_ok=True)

    for img_path in split_imgs:
        mask_path = mask_dir/img_path.name
        shutil.copy(img_path, root/split_name/'images'/img_path.name)
        shutil.copy(mask_path, root/split_name/'masks'/mask_path.name)
print('Done! Train:', len(splits["train"]), 'Val:', len(splits["val"]))
