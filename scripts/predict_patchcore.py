
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# Paths
checkpoint = Path(
    "reports/patchcore/Patchcore/MVTecAD/bottle/"
    "v0/weights/lightning/model.ckpt"
).resolve()

image_path = Path(
    "data/raw/mvtec/bottle/test/broken_large/000.png"
).resolve()

output = Path("reports/patchcore_first_prediction.png")

if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

if not image_path.is_file():
    raise FileNotFoundError(image_path)


# Use the same model architecture as training
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.01,
    pre_processor=Patchcore.configure_pre_processor(
        image_size=(256, 256),
    ),
)

# Dataset containing our test image
dataset = PredictDataset(
    path=image_path,
    image_size=(256, 256),
)

# Run inference using the saved Lightning checkpoint
engine = Engine(
    accelerator="cpu",
    devices=1,
    default_root_dir="reports/patchcore",
)

print("Loading checkpoint and predicting...")

predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=str(checkpoint),
)

if not predictions:
    raise RuntimeError("No predictions returned.")

prediction = predictions[0]

# Extract anomaly map
anomaly_map = prediction.anomaly_map

if isinstance(anomaly_map, torch.Tensor):
    anomaly_map = anomaly_map.detach().cpu().numpy()

anomaly_map = np.squeeze(np.asarray(anomaly_map))

# Load original image
image = Image.open(image_path).convert("RGB")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image)
axes[0].set_title("Original Bottle")

axes[1].imshow(anomaly_map, cmap="jet")
axes[1].set_title("PatchCore Anomaly Heatmap")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=150)
plt.close()

print(f"Prediction saved: {output}")
print(f"Anomaly score: {prediction.pred_score}")
