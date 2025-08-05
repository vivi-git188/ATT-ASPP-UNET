import numpy as np
import matplotlib.pyplot as plt

probs = np.load("output/probabilities/probs.npy")  # shape: (128, 224, 224)
areas = (probs > 0.5).sum(axis=(1, 2))

plt.plot(range(len(areas)), areas)
plt.xlabel("Frame Index")
plt.ylabel("Predicted Mask Area")
plt.title("Predicted Area per Frame")
plt.grid()
plt.show()
