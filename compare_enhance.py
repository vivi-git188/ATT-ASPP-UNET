import cv2
import matplotlib.pyplot as plt

original = cv2.imread("original_input.png", cv2.IMREAD_GRAYSCALE)
enhanced = cv2.imread("enhanced_input.png", cv2.IMREAD_GRAYSCALE)

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
