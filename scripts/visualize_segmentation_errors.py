
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import seed_everything

from anomalib.data import MVTecAD
from anomalib.data.utils import ValSplitMode, TestSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore


# 1. Reproduce the independent validation/test split
seed_everything(42, workers=True)

datamodule = MVTecAD(
    root=Path("data/raw/mvtec").resolve(),
    category="bottle",
    train_batch_size=4,
    eval_batch_size=4,
    num_workers=0,
    val_split_mode=ValSplitMode.FROM_TEST,
    val_split_ratio=0.5,
    test_split_mode=TestSplitMode.FROM_DIR,
)

datamodule.setup("fit")
datamodule.setup("test")

target = Path(
    "data/raw/mvtec/bottle/test/broken_large/004.png"
).resolve()

test_paths = {
    Path(path).resolve()
    for path in datamodule.test_data.samples["image_path"]
}

assert target in test_paths, (
    "Selected image is not in the held-out test set!"
)

# 2. Recreate PatchCore architecture
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.01,
    pre_processor=Patchcore.configure_pre_processor(
        image_size=(256, 256),
    ),
)

checkpoint = Path(
    "reports/patchcore_clean/Patchcore/MVTecAD/"
    "bottle/v0/weights/lightning/model.ckpt"
).resolve()

assert checkpoint.exists(), "Clean checkpoint not found!"

engine = Engine(
    accelerator="cpu",
    devices=1,
    default_root_dir="reports/segmentation_visualization",
)

# 3. Predict on the same held-out test subset
predictions = engine.predict(
    model=model,
    dataloaders=datamodule.test_dataloader(),
    ckpt_path=str(checkpoint),
    return_predictions=True,
)

# 4. Find the selected image
selected = None

for batch in predictions:
    paths = batch.image_path

    for index, path in enumerate(paths):
        if Path(path).resolve() == target:
            selected = {
                "image": batch.image[index].detach().cpu(),
                "gt_mask": batch.gt_mask[index].detach().cpu(),
                "pred_mask": batch.pred_mask[index].detach().cpu(),
            }
            break

    if selected is not None:
        break

if selected is None:
    raise RuntimeError("Selected image was not found in predictions!")

# 5. Convert tensors to NumPy arrays
image = selected["image"].numpy()
ground_truth = selected["gt_mask"].squeeze().numpy() > 0
predicted = selected["pred_mask"].squeeze().numpy() > 0

if image.ndim == 3 and image.shape[0] in (1, 3):
    image = np.transpose(image, (1, 2, 0))

image = np.clip(image, 0, 1)

if ground_truth.shape != predicted.shape:
    raise ValueError(
        f"Mask shape mismatch: {ground_truth.shape} "
        f"vs {predicted.shape}"
    )

# 6. Calculate segmentation errors
tp = predicted & ground_truth
fp = predicted & ~ground_truth
fn = ~predicted & ground_truth

tp_count = int(tp.sum())
fp_count = int(fp.sum())
fn_count = int(fn.sum())

precision = (
    tp_count / (tp_count + fp_count)
    if tp_count + fp_count else 0
)
recall = (
    tp_count / (tp_count + fn_count)
    if tp_count + fn_count else 0
)
f1 = (
    2 * tp_count / (2 * tp_count + fp_count + fn_count)
    if 2 * tp_count + fp_count + fn_count else 0
)

print("\n=== SELECTED IMAGE ===")
print(target.name)
print(f"True positives:  {tp_count:,}")
print(f"False positives: {fp_count:,}")
print(f"False negatives: {fn_count:,}")
print(f"Precision:       {precision:.4f}")
print(f"Recall:          {recall:.4f}")
print(f"Pixel F1:        {f1:.4f}")

# 7. Build a color-coded error map
error_map = np.zeros(
    (*ground_truth.shape, 3),
    dtype=np.float32,
)

error_map[tp] = [0.0, 0.8, 0.0]   # Green
error_map[fp] = [1.0, 0.0, 0.0]   # Red
error_map[fn] = [1.0, 0.6, 0.0]   # Orange

# 8. Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(image)
axes[0].set_title("Original bottle")

axes[1].imshow(ground_truth, cmap="gray")
axes[1].set_title("Ground truth")

axes[2].imshow(predicted, cmap="gray")
axes[2].set_title("PatchCore prediction")

axes[3].imshow(error_map)
axes[3].set_title("TP green | FP red | FN orange")

for ax in axes:
    ax.axis("off")

fig.suptitle(
    f"PatchCore Segmentation Analysis | Pixel F1: {f1:.3f}"
)

plt.tight_layout()

output = Path(
    "reports/patchcore_segmentation_errors.png"
)
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"\nVisualization saved: {output}")
