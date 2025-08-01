import cv2
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("output/images")
original_path = output_dir / "original_input.png"
enhanced_path = output_dir / "enhanced_input.png"
original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
enhanced = cv2.imread(str(enhanced_path), cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Enhanced (CLAHE + Median)")
plt.imshow(enhanced, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
