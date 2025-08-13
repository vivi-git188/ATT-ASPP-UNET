# check_pairs.py
from pathlib import Path
import sys

base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train_best")
img_dir = base / "images"
msk_dir = base / "masks"

if not img_dir.is_dir() or not msk_dir.is_dir():
    raise SystemExit(f"目录不存在：{img_dir} 或 {msk_dir}")

# 仅统计文件（忽略子目录）
imgs = sorted(p for p in img_dir.iterdir() if p.is_file())
msks = sorted(p for p in msk_dir.iterdir() if p.is_file())

img_stems = {p.stem for p in imgs}
msk_stems = {p.stem for p in msks}

missing_masks = sorted(img_stems - msk_stems)   # images中有而masks中没有
extra_masks   = sorted(msk_stems - img_stems)   # masks中多出来的

print(f"[Counts]")
print(f"images: {len(imgs)} files in {img_dir}")
print(f"masks : {len(msks)} files in {msk_dir}\n")

print(f"[Check] images 中无对应 mask 的文件数：{len(missing_masks)}")
for s in missing_masks[:50]:
    print("  -", s)
if len(missing_masks) > 50:
    print(f"  ... 还有 {len(missing_masks)-50} 个未显示")

print(f"\n[Check] masks 中无对应 image 的文件数：{len(extra_masks)}")
for s in extra_masks[:50]:
    print("  -", s)
if len(extra_masks) > 50:
    print(f"  ... 还有 {len(extra_masks)-50} 个未显示")

# 保存清单，便于后续处理
(out1 := base / "_images_without_masks.txt").write_text("\n".join(missing_masks))
(out2 := base / "_masks_without_images.txt").write_text("\n".join(extra_masks))
print(f"\n清单已保存：\n - {out1}\n - {out2}")
