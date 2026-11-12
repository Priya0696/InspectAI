
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


IMAGE_PATH = Path(
    "data/raw/mvtec/bottle/test/broken_large/004.png"
)

MASK_PATH = Path(
    "data/raw/mvtec/bottle/ground_truth/"
    "broken_large/004_mask.png"
)

CHECKPOINTS = {
    "256x256": Path(
        "reports/patchcore_clean/Patchcore/MVTecAD/"
        "bottle/v0/weights/lightning/model.ckpt"
    ),
    "384x384": Path(
        "reports/patchcore_384/Patchcore/MVTecAD/"
        "bottle/v0/weights/lightning/model.ckpt"
    ),
}

OUTPUT = Path("reports/patchcore_resolution_comparison.png")


def predict_mask(checkpoint_path):
    model = Patchcore.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    model.eval()

    engine = Engine(
    accelerator="cpu",
    devices=1,
    logger=False,
)

    image_size = (384, 384) if "384" in str(checkpoint_path) else (256, 256)

    dataset = PredictDataset(
    path=IMAGE_PATH,
    image_size=image_size,
)

    predictions = engine.predict(
        model=model,
        dataset=dataset,
    )

    batch = predictions[0]

    mask = batch.pred_mask[0].cpu().numpy()
    return mask.astype(bool)


def calculate_metrics(predicted, actual):
    tp = np.logical_and(predicted, actual).sum()
    fp = np.logical_and(predicted, ~actual).sum()
    fn = np.logical_and(~predicted, actual).sum()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    return precision, recall, f1


original = np.array(Image.open(IMAGE_PATH).convert("RGB"))
ground_truth = np.array(
    Image.open(MASK_PATH).convert("L")
) > 0

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(original)
axes[0].set_title("Original")

axes[1].imshow(ground_truth, cmap="gray")
axes[1].set_title("Ground Truth")

for index, (name, checkpoint) in enumerate(
    CHECKPOINTS.items(), start=2
):
    prediction = predict_mask(checkpoint)

    # Resize binary prediction to original mask dimensions.
    prediction = np.array(
        Image.fromarray(
            prediction.astype(np.uint8) * 255
        ).resize(
            (ground_truth.shape[1], ground_truth.shape[0]),
            resample=Image.Resampling.NEAREST,
        )
    ) > 0

    precision, recall, f1 = calculate_metrics(
        prediction, ground_truth
    )

    axes[index].imshow(prediction, cmap="gray")
    axes[index].set_title(
        f"{name}\n"
        f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
    )

    print(
        f"{name}: Precision={precision:.4f}, "
        f"Recall={recall:.4f}, F1={f1:.4f}"
    )

for ax in axes:
    ax.axis("off")

plt.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=160, bbox_inches="tight")
plt.close()

print(f"\nSaved comparison: {OUTPUT}")
