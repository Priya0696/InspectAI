from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

dataset = Path("data/raw/mvtec/bottle")
category = "broken_large"

image_dir = dataset / "test" / category
mask_dir = dataset / "ground_truth" / category

image_path = sorted(image_dir.glob("*.png"))[0]
mask_path = mask_dir / f"{image_path.stem}_mask.png"

if not mask_path.exists():
    raise FileNotFoundError(f"Mask not found: {mask_path}")

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise ValueError("Could not read the ground-truth mask")

if mask.shape != image.shape[:2]:
    raise ValueError("Image and mask dimensions do not match")

# Create a transparent red overlay over defective pixels
overlay = image.copy()
overlay[mask > 0] = [255, 0, 0]

blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Original Image")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Ground-Truth Mask")

axes[2].imshow(blended)
axes[2].set_title("Defect Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output = Path("reports/bottle_defect_overlay.png")
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=150)
plt.show()

print(f"Saved visualization: {output}")
print(f"Defective pixels: {np.count_nonzero(mask)}")