
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_prediction_visualization(
    image_path: str,
    predicted_mask: np.ndarray,
    output_path: str,
) -> Path:
    """Save original image, predicted mask, and defect overlay."""

    image = Image.open(image_path).convert("RGB")
    original = np.asarray(image)

    # Resize the predicted mask to the original image dimensions.
    mask_image = Image.fromarray(
        (predicted_mask > 0).astype(np.uint8) * 255
    )

    mask_image = mask_image.resize(
        image.size,
        resample=Image.Resampling.NEAREST,
    )

    mask = np.asarray(mask_image) > 0

    # Create a red overlay on detected defect pixels.
    overlay = original.astype(np.float32).copy()
    overlay[mask] = (
        0.55 * overlay[mask]
        + 0.45 * np.array([255, 0, 0])
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Predicted Defect Mask")

    axes[2].imshow(overlay)
    axes[2].set_title("Defect Overlay")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output
