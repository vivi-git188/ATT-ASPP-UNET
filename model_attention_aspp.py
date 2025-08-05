from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from attention_aspp_unet import AttentionASPPUNet          # ← 你的网络代码
from aspp_postprocess_probability_maps import postprocess_single_probability_map


# ----------------------------
# FetalAbdomenSegmentation
# ----------------------------
class FetalAbdomenSegmentation:
    """
    推理类 —— 使用 Attention-ASPP-UNet 替代 nnUNet
    """
    def __init__(self,
                 checkpoint_path: str = "checkpoints/best_model.pth",
                 device: str | None = None):
        # 1. 设备
        self.device = torch.device(device) if device else (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # 2. 构造网络并加载权重
        self.model = AttentionASPPUNet(in_ch=1, num_classes=1)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        print(f"✅ Attention-ASPP-UNet 权重加载完毕 ({checkpoint_path})")

    # ----------------------------
    # 推理 predict
    # ----------------------------
    @torch.no_grad()
    def predict(self,
                input_img_path: str | Path,
                save_probabilities: bool = False):
        from inference import load_image_file_as_array
        input_np = load_image_file_as_array(location=Path(input_img_path[0]))  # (1, N, H, W)

        # ---------- build a clean (N,1,224,224) tensor ----------
        resized = [cv2.resize(sl, (224, 224), cv2.INTER_AREA)  # sl already 0-1 float
                   for sl in input_np[0]]  # (N,H,W)
        input_torch = torch.from_numpy(np.stack(resized))[..., None]  # (N,224,224,1)
        input_torch = input_torch.permute(0, 3, 1, 2).to(self.device)  # (N,1,224,224)

        # ---------- no extra div(255.) ----------
        input_torch = input_torch.float()  # values stay in 0-1

        probs_list = []
        with torch.no_grad():
            for slice_t in input_torch:  # (1,224,224)
                logits = self.model(slice_t.unsqueeze(0))  # (1,1,224,224)
                probs = torch.sigmoid(logits)[0, 0].cpu()  # (224,224)
                probs_list.append(probs.numpy())
        probs_3d = np.stack(probs_list).astype(np.float32)  # (N,224,224)

        if save_probabilities:
            out = Path("output/probabilities")
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "probs.npy", probs_3d)
            print("📦 Saved raw probabilities to output/probabilities/probs.npy")

        return probs_3d

    # ----------------------------
    # 后处理 postprocess
    # ----------------------------
    def postprocess(self, probability_map: np.ndarray):
        """
        将概率图后处理为二值 Mask (3-D)
        """
        configs = {"soft_threshold": 0.5}
        mask = postprocess_single_probability_map(probability_map, configs)
        return mask


# -------------------------------------------------
# util: 选择最佳帧 + 输出二值 2-D 腹围 Mask
# -------------------------------------------------
def select_fetal_abdomen_mask_and_frame(mask_3d: np.ndarray):
    if mask_3d.ndim == 2:
        return (mask_3d > 0).astype(np.uint8), 0  # frame_idx=0 或 -1 均可
    if mask_3d.ndim != 3:
        raise ValueError(f"Expect (N,H,W) mask, got {mask_3d.shape}")

    areas = (mask_3d > 0).sum(axis=(1,2))        # (N,)
    idx   = int(areas.argmax())
    if areas[idx] == 0:
        print("⚠️  all-zero 3D mask, return empty slice")
        return np.zeros(mask_3d.shape[1:], np.uint8), -1

    return (mask_3d[idx] > 0).astype(np.uint8), idx     # → (H,W), int


