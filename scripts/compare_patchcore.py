
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Paths
root = Path("data/raw/mvtec/bottle")
image_path = root / "test/broken_large/000.png"
mask_path = root / "ground_truth/broken_large/000_mask.png"

checkpoint = Path(
    "reports/patchcore/Patchcore/MVTecAD/bottle/"
    "v0/weights/lightning/model.ckpt"
)

output = Path("reports/patchcore_comparison.png")

# Check files
for path in (image_path, mask_path, checkpoint):
    if not path.is_file():
        raise FileNotFoundError(path)

# Recreate model configuration
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.01,
    pre_processor=Patchcore.configure_pre_processor(
        image_size=(256, 256),
    ),
)

dataset = PredictDataset(
    path=image_path,
    image_size=(256, 256),
)

engine = Engine(accelerator="cpu", devices=1)

print("Running PatchCore inference...")

predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=str(checkpoint.resolve()),
)

if not predictions:
    raise RuntimeError("No predictions returned")

prediction = predictions[0]

# Convert predicted map to NumPy
anomaly_map = prediction.anomaly_map
if isinstance(anomaly_map, torch.Tensor):
    anomaly_map = anomaly_map.detach().cpu().numpy()

anomaly_map = np.squeeze(np.asarray(anomaly_map))

# Load original image and ground truth
image = np.asarray(Image.open(image_path).convert("RGB"))
mask = np.asarray(Image.open(mask_path).convert("L"))

# Resize the predicted map to original image dimensions
height, width = image.shape[:2]
anomaly_map = np.asarray(
    Image.fromarray(anomaly_map.astype(np.float32)).resize(
        (width, height),
        resample=Image.Resampling.BILINEAR,
    )
)

# Create visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(image)
axes[0].set_title("Original Bottle")

axes[1].imshow(anomaly_map, cmap="jet")
axes[1].set_title("AI Anomaly Heatmap")

axes[2].imshow(mask, cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Ground-Truth Mask")

axes[3].imshow(image)
axes[3].imshow(anomaly_map, cmap="jet", alpha=0.45)
axes[3].set_title("AI Heatmap Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=150)
plt.close()

print(f"Comparison saved: {output}")
