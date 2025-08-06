# batch_infer.py
import shutil, subprocess, os
from pathlib import Path
from glob import glob

# --------- 1. 配置区 ---------
VAL_IMG_DIR   = Path("val/images")
OUTPUT_ROOT   = Path("outputs")
MODEL_TAG     = os.environ.get("MODEL_TAG", "baseline")  # 环境变量决定跑谁
# baseline  -> 导入 model.py
# att_aspp  -> 导入 model_attention_aspp.py
# --------------------------------

def run_single(img_path: Path):
    # 1️⃣ 提取 case_id（去除后缀，只保留 UUID）
    case_id = img_path.stem.split("_")[0]

    # 2️⃣ 清空 test/input 并复制当前图像进去
    shutil.rmtree("test/input", ignore_errors=True)
    (Path("test/input/images/stacked-fetal-ultrasound")).mkdir(
        parents=True, exist_ok=True
    )
    dst = Path("test/input/images/stacked-fetal-ultrasound") / img_path.name
    shutil.copy(img_path, dst)

    # 3️⃣ 设置环境变量：模型名 + 当前 case_id
    env = os.environ.copy()
    env["MODEL_TAG"] = MODEL_TAG
    env["CASE_ID"]   = case_id

    # 4️⃣ 调用推理脚本
    subprocess.run(["python", "inference.py"], check=True, env=env)

    # 5️⃣ 移动保存结果文件（output.mha → outputs/baseline/xxx.mha）
    (OUTPUT_ROOT / MODEL_TAG).mkdir(parents=True, exist_ok=True)
    shutil.move(
        "test/output/images/fetal-abdomen-segmentation/{}.mha".format(case_id),
        OUTPUT_ROOT / MODEL_TAG / f"{case_id}.mha"
    )

if __name__ == "__main__":
    imgs = sorted(glob(str(VAL_IMG_DIR / "*.mha")))
    print(f"Found {len(imgs)} validation images.")
    for p in imgs:
        print(f"\n=== {Path(p).name} ===")
        run_single(Path(p))
