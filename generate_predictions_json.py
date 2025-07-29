import json
from pathlib import Path

# 项目根目录下执行此脚本
ROOT = Path("test")
INPUT_IMAGES_DIR = ROOT / "input" / "images" / "stacked-fetal-ultrasound"
OUTPUT_SEG_DIR = Path("output") / "images" / "fetal-abdomen-segmentation"
FRAME_JSON = Path("output") / "fetal-abdomen-frame-number.json"
PRED_JSON = Path("output") / "predictions.json"

# 找输入文件名
input_files = list(INPUT_IMAGES_DIR.glob("*.mha"))
if not input_files:
    raise RuntimeError("找不到输入 .mha 文件，请确认 test/input/images/stacked-fetal-ultrasound 下是否有文件。")
input_name = input_files[0].name

# 检查分割输出文件是否存在
seg_path = OUTPUT_SEG_DIR / input_name+".mha"
print("seg_path------"+seg_path)
print("input_name"+input_name)
if not seg_path.exists():
    raise RuntimeError(f"找不到分割文件: {seg_path}")

# 检查 frame json 是否存在
if not FRAME_JSON.exists():
    raise RuntimeError(f"找不到帧号 JSON 文件: {FRAME_JSON}")

# 生成 predictions.json
predictions = [
    {
        "pk": "example-case",
        "inputs": [
            {
                "file": None,
                "image": {"name": input_name},
                "value": None,
                "interface": {
                    "slug": "stacked-fetal-ultrasound",
                    "kind": "Image",
                    "super_kind": "Image",
                    "relative_path": "images/stacked-fetal-ultrasound"
                }
            }
        ],
        "outputs": [
            {
                "file": None,
                "image": {"name": input_name+".mha"},
                "value": None,
                "interface": {
                    "slug": "fetal-abdomen-segmentation",
                    "kind": "Segmentation",
                    "super_kind": "Image",
                    "relative_path": "images/fetal-abdomen-segmentation"
                }
            },
            {
                "file": None,
                "image": None,
                "value": None,
                "interface": {
                    "slug": "fetal-abdomen-frame-number",
                    "kind": "Integer",
                    "super_kind": "Value",
                    "relative_path": "fetal-abdomen-frame-number.json"
                }
            }
        ],
        "status": "Succeeded",
        "started_at": "",
        "completed_at": ""
    }
]

with open(PRED_JSON, "w") as f:
    json.dump(predictions, f, indent=4)

print(f"已生成 predictions.json: {PRED_JSON.resolve()}")
