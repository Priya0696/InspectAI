
from pathlib import Path
import csv
import json

import torch
from lightning.pytorch import seed_everything

from anomalib.data import MVTecAD
from anomalib.data.utils import ValSplitMode, TestSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore


# Reproduce the clean experiment
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

print("Validation images:", len(datamodule.val_data))

# Recreate the trained model architecture
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

assert checkpoint.exists(), "Clean checkpoint not found"

engine = Engine(
    accelerator="cpu",
    devices=1,
    default_root_dir="reports/threshold_optimization",
)

print("\n=== RUNNING VALIDATION PREDICTIONS ===")

predictions = engine.predict(
    model=model,
    dataloaders=datamodule.val_dataloader(),
    ckpt_path=str(checkpoint),
    return_predictions=True,
)

# Store normalized anomaly maps and ground-truth masks.
# No test images are used.
validation_data = []

for batch in predictions:
    if batch.anomaly_map is None or batch.gt_mask is None:
        raise RuntimeError("Missing anomaly map or ground truth")

    anomaly_map = batch.anomaly_map.detach().cpu()
    ground_truth = batch.gt_mask.detach().cpu().bool()

    if anomaly_map.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: {anomaly_map.shape} "
            f"vs {ground_truth.shape}"
        )

    validation_data.append((anomaly_map, ground_truth))

# Sweep normalized pixel thresholds
thresholds = [round(x / 100, 2) for x in range(30, 81, 5)]

results = []

print("\n=== VALIDATION THRESHOLD SWEEP ===")
print("Threshold | Precision | Recall | F1 | IoU")

for threshold in thresholds:
    tp = fp = fn = 0

    for anomaly_map, ground_truth in validation_data:
        predicted = anomaly_map > threshold

        tp += (predicted & ground_truth).sum().item()
        fp += (predicted & ~ground_truth).sum().item()
        fn += (~predicted & ground_truth).sum().item()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0

    iou_denominator = tp + fp + fn
    iou = tp / iou_denominator if iou_denominator else 0.0

    result = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }

    results.append(result)

    print(
        f"{threshold:9.2f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:.4f} | "
        f"{iou:.4f}"
    )

best = max(results, key=lambda item: item["f1"])

print("\n=== BEST VALIDATION THRESHOLD ===")
print(f"Threshold: {best['threshold']:.2f}")
print(f"Precision: {best['precision']:.4f}")
print(f"Recall:    {best['recall']:.4f}")
print(f"Pixel F1:  {best['f1']:.4f}")
print(f"IoU:       {best['iou']:.4f}")

# Save experiment results
output_dir = Path("reports/metrics")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "pixel_threshold_sweep.csv", "w") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["threshold", "precision", "recall", "f1", "iou"],
    )
    writer.writeheader()
    writer.writerows(results)

with open(output_dir / "best_pixel_threshold.json", "w") as file:
    json.dump(
        {
            "selected_on": "validation",
            "validation_images": len(datamodule.val_data),
            "threshold": best["threshold"],
            "validation_metrics": best,
        },
        file,
        indent=2,
    )

print("\nResults saved to reports/metrics/")
