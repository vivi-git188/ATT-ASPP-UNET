# batch_infer.py
import shutil, subprocess, os
from pathlib import Path
from glob import glob

# --------- 1. 配置区 ---------
VAL_IMG_DIR   = Path("val/images/stacked-fetal-ultrasound")
OUTPUT_ROOT   = Path("outputs")
MODEL_TAG     = os.environ.get("MODEL_TAG", "baseline")  # 环境变量决定跑谁
# baseline  -> 导入 model.py
# att_aspp  -> 导入 model_attention_aspp.py
# --------------------------------

def run_single(img_path: Path):
    # 1) 把 test/input 下的旧文件清空
    shutil.rmtree("test/input", ignore_errors=True)
    (Path("test/input/images/stacked-fetal-ultrasound")).mkdir(
        parents=True, exist_ok=True
    )

    # 2) 复制当前 image 到 test/input
    dst = Path("test/input/images/stacked-fetal-ultrasound") / img_path.name
    shutil.copy(img_path, dst)

    # 3) 调用 inference.py（MODEL_TAG 通过 env 变量传递给 import）
    env = os.environ.copy()
    env["MODEL_TAG"] = MODEL_TAG
    subprocess.run(["python", "inference.py"], check=True, env=env)

    # 4) 把生成的 output.mha rename → outputs/{MODEL_TAG}/{case_id}.mha
    case_id = img_path.name.split("_")[0]            # '04a04f2e-...' 部分
    (OUTPUT_ROOT / MODEL_TAG).mkdir(parents=True, exist_ok=True)
    shutil.move(
        "test/output/images/fetal-abdomen-segmentation/output.mha",
        OUTPUT_ROOT / MODEL_TAG / f"{case_id}.mha",
    )

if __name__ == "__main__":
    imgs = sorted(glob(str(VAL_IMG_DIR / "*_0000.mha")))
    print(f"Found {len(imgs)} validation images.")
    for p in imgs:
        print(f"\n=== {Path(p).name} ===")
        run_single(Path(p))
