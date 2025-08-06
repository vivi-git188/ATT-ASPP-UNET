# ==============================================
#  model_attention_aspp.py  (推理封装，带 ROI 裁剪回贴)
# ==============================================
from pathlib import Path
import numpy as np, torch, cv2, SimpleITK, json, scipy.ndimage as ndi
from attention_aspp_unet import AttentionASPPUNet

__all__ = ["FetalAbdomenSegmentation", "select_fetal_abdomen_mask_and_frame"]

# ---------- 读取 + 预处理 ----------

def load_image_file_as_array(*, location: Path):
    img = SimpleITK.ReadImage(str(location))
    arr = SimpleITK.GetArrayFromImage(img)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    stack = [cv2.medianBlur(clahe.apply(cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX)
              .astype(np.uint8)), 3) for sl in arr]
    return (np.stack(stack).astype(np.float32) / 255.0)[None]

# ---------- ROI 裁剪 ----------

def crop_roi_224(img):
    h, w = img.shape; thr = img.mean() * 1.2
    ys, xs = np.where(img > thr)
    cx, cy = (w//2, h//2) if len(xs) == 0 else (int(xs.mean()), int(ys.mean()))
    x0, y0 = max(0, cx-112), max(0, cy-112)
    x0, y0 = min(x0, w-224), min(y0, h-224)
    patch = img[y0:y0+224, x0:x0+224]
    if patch.shape != (224, 224):
        patch = cv2.copyMakeBorder(patch, 0, 224-patch.shape[0], 0,
                                   224-patch.shape[1], cv2.BORDER_CONSTANT, value=0)
    return patch, (x0, y0)

# ---------- 推理包装 ----------

class FetalAbdomenSegmentation:
    def __init__(self, checkpoint_path="checkpoints/best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AttentionASPPUNet(in_ch=1, num_classes=1, base=16).to(self.device)
        miss, unexp = self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)
        print(f"[DEBUG] load_state — missing:{len(miss)} unexpected:{len(unexp)}")
        self.net.eval(); print("✅ Weights loaded")

    @torch.no_grad()
    def predict(self, input_img_path, save_probabilities=False):
        self.case_id = Path(input_img_path[0]).stem         # 供 postprocess()
        vol = load_image_file_as_array(location=Path(input_img_path[0]))
        idxs = np.linspace(0, vol.shape[1]-1, 128).astype(int)
        vol = vol[:, idxs]; N, H, W = vol.shape[1:]

        patches, coords = [], []
        for sl in vol[0]:
            p, (x0, y0) = crop_roi_224(sl)
            patches.append(p); coords.append((x0, y0))
        tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(self.device)

        outs = [torch.sigmoid(self.net(tensor[i:i+8])).squeeze(1) for i in range(0, N, 8)]
        prob_roi = torch.cat(outs).cpu().numpy()

        prob_full = np.zeros((N, H, W), np.float32)
        for i, (x0, y0) in enumerate(coords):
            h_roi, w_roi = min(224, H-y0), min(224, W-x0)
            prob_full[i, y0:y0+h_roi, x0:x0+w_roi] = cv2.resize(prob_roi[i], (w_roi, h_roi))

        if save_probabilities:
            Path("output/probabilities").mkdir(parents=True, exist_ok=True)
            np.save(f"output/probabilities/{self.case_id}_prob.npy", prob_full)
        return prob_full

    # ---------- 后处理 ----------
        # ---------- 后处理 ----------
    def postprocess(self, probability_map):
        """对齐标注帧；若 json 为单 int 直接用，否则 dict lookup; 若失败回退面积峰值"""
        frame_idx = -1
        json_path = Path("test/output/fetal-abdomen-frame-number.json")
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                # Grand‑Challenge 导出的 json 既可能是 {case:idx} 也可能直接是 int
                frame_idx = int(data if isinstance(data, int) else data.get(self.case_id, -1))
            except Exception as e:
                print(f"[WARN] frame json parse fail: {e}")

        bin_ = (probability_map > 0.05).astype(np.uint8)
        if frame_idx < 0 or frame_idx >= bin_.shape[0] or bin_[frame_idx].sum() == 0:
            frame_idx = int(bin_.sum((1, 2)).argmax())  # fallback

        frame = bin_[frame_idx]
        if frame.ndim != 2 or frame.size == 0:          # 安全检查
            mask = np.zeros_like(bin_, np.uint8); mask[frame_idx] = frame.astype(np.uint8)
            return mask

        # 膨胀 + 最大连通域（显式 3×3 结构，避免 SciPy 维度报错）
        structure = np.ones((3, 3), dtype=np.uint8)
        frame = ndi.binary_dilation(frame, structure=structure, iterations=1)
        labeled, n = ndi.label(frame, structure=structure)
        if n:
            sizes = ndi.sum(frame, labeled, index=range(1, n + 1))
            frame = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
        mask = np.zeros_like(bin_, np.uint8); mask[frame_idx] = frame
        return mask

# ---------- 提取最大腹围帧 ----------

def select_fetal_abdomen_mask_and_frame(mask_3d):
    if mask_3d.ndim == 2:
        return (mask_3d > 0).astype(np.uint8), 0
    areas = mask_3d.sum((1, 2)); idx = int(areas.argmax())
    if areas[idx] == 0:
        return np.zeros(mask_3d.shape[1:], np.uint8), -1
    return (mask_3d[idx] > 0).astype(np.uint8), idx
