from pathlib import Path

import cv2
import matplotlib.pyplot as plt

dataset = Path("data/raw/mvtec/bottle")

categories = [
    ("Normal", dataset / "train/good"),
    ("Broken Large", dataset / "test/broken_large"),
    ("Broken Small", dataset / "test/broken_small"),
    ("Contamination", dataset / "test/contamination"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, (label, folder) in zip(axes, categories):
    image_path = sorted(folder.glob("*.png"))[0]

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.imshow(image)
    ax.set_title(label)
    ax.axis("off")

plt.tight_layout()

output = Path("reports/bottle_samples.png")
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=150)
print(f"Visualization saved to: {output}")

plt.show()