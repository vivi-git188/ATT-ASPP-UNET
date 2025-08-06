# ==============================================
#  model_attention_aspp.py  (推理封装，无循环依赖)
#  封装训练好的 Attention‑ASPP‑UNet 供 inference.py 调用
# ==============================================

from pathlib import Path
import numpy as np, torch, cv2, SimpleITK

from attention_aspp_unet import AttentionASPPUNet  # 网络结构
from aspp_postprocess_probability_maps import postprocess_single_probability_map

__all__ = [
    "FetalAbdomenSegmentation",
    "select_fetal_abdomen_mask_and_frame",
]

# -------------------------------------------------
# 独立的图像读取 + 预处理，避免与 inference.py 互相 import
# -------------------------------------------------

def load_image_file_as_array(*, location: Path):
    """读取 3‑D 超声图像，逐帧做 CLAHE+中值滤波 → (1,N,H,W) 0‑1 float32"""
    itk_img = SimpleITK.ReadImage(str(location))
    arr = SimpleITK.GetArrayFromImage(itk_img)  # (N,H,W)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = []
    for sl in arr:
        sl_u8 = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cla = clahe.apply(sl_u8)
        flt = cv2.medianBlur(cla, 3)
        enhanced.append(flt)
    stack = np.stack(enhanced).astype(np.float32) / 255.0  # (N,H,W)
    return stack[np.newaxis, ...]

# -------------------------------------------------
# 推理包装类
# -------------------------------------------------
class FetalAbdomenSegmentation:
    """Attention‑ASPP‑UNet 推理包装（接口与 baseline 保持一致）"""

    def __init__(
        self,
        checkpoint_path: str | Path = "checkpoints/best_model.pth",
        device: str | None = None,
        img_size: int = 224,
    ):
        self.device = torch.device(device) if device else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.img_size = img_size

        # ---------- 构建网络并加载权重 ----------
        # ⚠ base 通道数必须与训练阶段一致（你训练时用 base=32）
        self.net = AttentionASPPUNet(in_ch=1, num_classes=1, base=16).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        missing, unexpected = self.net.load_state_dict(ckpt, strict=False)
        print(
            f"[DEBUG] load_state_dict — missing: {len(missing)}, unexpected: {len(unexpected)}"
        )
        self.net.eval()
        print(f"✅ Loaded Attention‑ASPP‑UNet weights from {checkpoint_path}")

    # -------------------------------------------------
    # 前向推理：返回 (N,H,W) 概率图
    # -------------------------------------------------
    @torch.no_grad()
    def predict(self, input_img_path: list[str | Path], save_probabilities: bool = False):
        stack_np = load_image_file_as_array(location=Path(input_img_path[0]))  # (1,N,H,W)

        # resize → tensor (N,1,H,W)
        frames = [
            cv2.resize(sl, (self.img_size, self.img_size), cv2.INTER_AREA)
            for sl in stack_np[0]
        ]
        tensor = torch.from_numpy(np.stack(frames)).unsqueeze(1).to(self.device)

        # batch 前向
        BATCH = 8
        probs = []
        for i in range(0, len(tensor), BATCH):
            logit = self.net(tensor[i : i + BATCH])  # (b,1,H,W)
            probs.append(torch.sigmoid(logit).squeeze(1))
        prob_3d = torch.cat(probs, 0).cpu().numpy()  # (N,H,W)

        if save_probabilities:
            out_dir = Path("output/probabilities"); out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "probs.npy", prob_3d)
            print("📦 probs.npy saved to output/probabilities/")

        return prob_3d.astype(np.float32)

    # -------------------------------------------------
    # 后处理
    # -------------------------------------------------
    def postprocess(self, probability_map: np.ndarray):
        cfg = {"soft_threshold": 0.25}
        return postprocess_single_probability_map(probability_map, cfg)


# -------------------------------------------------
# 选择最大腹围帧 → 二值 2‑D mask
# -------------------------------------------------

def select_fetal_abdomen_mask_and_frame(mask_3d: np.ndarray):
    if mask_3d.ndim == 2:
        return (mask_3d > 0).astype(np.uint8), 0
    if mask_3d.ndim != 3:
        raise ValueError(f"Expect (N,H,W) mask, got {mask_3d.shape}")
    areas = (mask_3d > 0).sum(axis=(1, 2))
    idx = int(areas.argmax())
    if areas[idx] == 0:
        return np.zeros(mask_3d.shape[1:], np.uint8), -1
    return (mask_3d[idx] > 0).astype(np.uint8), idx
