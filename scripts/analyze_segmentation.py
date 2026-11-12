
from pathlib import Path

import torch
from lightning.pytorch import seed_everything

from anomalib.data import MVTecAD
from anomalib.data.utils import ValSplitMode, TestSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore


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

assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"

engine = Engine(
    accelerator="cpu",
    devices=1,
    default_root_dir="reports/segmentation_analysis",
)

# Restore the trained model and its validation-calibrated thresholds.
# Predict on the held-out test dataloader, not the full test folder.
datamodule.setup("fit")
datamodule.setup("test")

print("Held-out test images:", len(datamodule.test_data))

predictions = engine.predict(
    model=model,
    dataloaders=datamodule.test_dataloader(),
    ckpt_path=str(checkpoint),
    return_predictions=True,
)

tp = fp = fn = 0

for batch in predictions:
    if batch.pred_mask is None or batch.gt_mask is None:
        raise RuntimeError("Prediction or ground-truth mask missing")

    predicted = batch.pred_mask.bool()
    actual = batch.gt_mask.bool()

    # Ignore singleton channel dimensions where present.
    predicted = predicted.squeeze()
    actual = actual.squeeze()

    if predicted.shape != actual.shape:
        raise ValueError(
            f"Mask shape mismatch: {predicted.shape} vs {actual.shape}"
        )

    tp += (predicted & actual).sum().item()
    fp += (predicted & ~actual).sum().item()
    fn += (~predicted & actual).sum().item()

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

print("\n=== PIXEL-LEVEL ERROR ANALYSIS ===")
print(f"True positives:  {tp:,}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")
print(f"Precision:       {precision:.4f}")
print(f"Recall:          {recall:.4f}")
print(f"Pixel F1:        {f1:.4f}")
print(f"Pixel IoU:       {iou:.4f}")

print("\nAnalysis complete.")
