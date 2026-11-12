
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def save_prediction_json(
    prediction: dict,
    visualization_path: str,
    output_path: str,
) -> Path:
    """Export an inspection result as JSON."""

    mask = np.asarray(prediction["predicted_mask"])
    defective_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": prediction["image_path"],
        "model": "PatchCore",
        "classification": prediction["classification"],
        "is_anomalous": bool(prediction["is_anomalous"]),
        "anomaly_score": float(prediction["anomaly_score"]),
        "mask_width": int(mask.shape[1]),
        "mask_height": int(mask.shape[0]),
        "defective_pixels": defective_pixels,
        "defect_area_percentage": round(
            100 * defective_pixels / total_pixels, 2
        ),
        "visualization_path": str(visualization_path),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(json.dumps(result, indent=2))

    return output
