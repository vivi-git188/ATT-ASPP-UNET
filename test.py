#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 train/iamge 下的文件是否存在于 val/images 中。
- 默认比较“文件名 + 扩展名”
- 可加 --by-stem 仅按“文件名（忽略扩展名）”比较
- 可加 -r/--recursive 递归子目录
结果会保存到 check_results/ 下的 present.txt 与 missing_in_val.txt
"""

from pathlib import Path
import argparse

def collect_files(d: Path, recursive: bool, by_stem: bool):
    if not d.exists():
        return set()
    it = d.rglob("*") if recursive else d.iterdir()
    items = set()
    for p in it:
        if p.is_file():
            items.add(p.stem if by_stem else p.name)
    return items

def main():
    ap = argparse.ArgumentParser(description="检查 train/iamge 是否存在于 val/images 中")
    ap.add_argument("--train", default="train/images", help="训练图片目录（默认：train/iamge）")
    ap.add_argument("--val",   default="val/images",   help="验证图片目录（默认：val/images）")
    ap.add_argument("-r", "--recursive", action="store_true", help="递归遍历子目录")
    ap.add_argument("--by-stem", action="store_true", help="按文件名比较（忽略扩展名）")
    args = ap.parse_args()

    train_dir = Path(args.train)
    val_dir = Path(args.val)

    # 如果 train/iamge 不存在，尝试常见拼写
    if not train_dir.exists():
        for alt in [Path("train/images"), Path("train/image")]:
            if alt.exists():
                print(f"[提示] 未找到 {train_dir} ，改用 {alt}")
                train_dir = alt
                break

    train_set = collect_files(train_dir, args.recursive, args.by_stem)
    val_set = collect_files(val_dir, args.recursive, args.by_stem)

    if not train_set:
        print(f"⚠️ 训练目录无文件或不存在：{train_dir.resolve() if train_dir.exists() else train_dir}")
        return
    if not val_set:
        print(f"⚠️ 验证目录无文件或不存在：{val_dir.resolve() if val_dir.exists() else val_dir}")

    missing_in_val = sorted(train_set - val_set)
    present_in_val = sorted(train_set & val_set)

    print(f"训练集文件数：{len(train_set)}")
    print(f"验证集中已存在：{len(present_in_val)}")
    print(f"验证集中缺失：{len(missing_in_val)}")

    if present_in_val:
        print("\n✅ 已存在（前20）：")
        for name in present_in_val[:20]:
            print(name)
    if missing_in_val:
        print("\n❌ 缺失（前20）：")
        for name in missing_in_val[:20]:
            print(name)

    out_dir = Path("check_results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "present.txt").write_text("\n".join(present_in_val), encoding="utf-8")
    (out_dir / "missing_in_val.txt").write_text("\n".join(missing_in_val), encoding="utf-8")
    print("\n结果已保存到：check_results/present.txt 与 check_results/missing_in_val.txt")

if __name__ == "__main__":
    main()
