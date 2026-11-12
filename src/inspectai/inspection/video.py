
import math
import tempfile
from pathlib import Path

import cv2


def inspect_video(
    video_path,
    predictor,
    sample_interval_seconds=1.0,
    max_frames=100,
):
    """
    Inspect sampled video frames using a trained PatchCore model.

    Args:
        video_path: Path to the video.
        predictor: Existing InspectAIPredictor instance.
        sample_interval_seconds: Sampling interval in seconds.
        max_frames: Maximum number of frames to inspect.

    Returns:
        A dictionary containing predictions and inspection summary.
    """

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    if predictor is None:
        raise ValueError("Predictor is not initialized")

    if (
        not math.isfinite(sample_interval_seconds)
        or sample_interval_seconds <= 0
    ):
        raise ValueError(
            "Sampling interval must be positive and finite"
        )

    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("Cannot open video")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Invalid video FPS")

    sample_every = max(
        1,
        round(fps * sample_interval_seconds),
    )

    results = []
    frame_index = 0
    limit_reached = False

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            while True:
                success, frame = capture.read()

                if not success:
                    break

                if frame_index % sample_every == 0:
                    if len(results) >= max_frames:
                        limit_reached = True
                        break

                    timestamp = frame_index / fps

                    frame_path = (
                        temp_dir / f"frame_{frame_index:06d}.png"
                    )

                    if not cv2.imwrite(str(frame_path), frame):
                        raise RuntimeError(
                            "Could not save video frame"
                        )

                    prediction = predictor.predict(
                        str(frame_path)
                    )

                    results.append(
                        {
                            "frame": frame_index,
                            "timestamp_seconds": round(
                                timestamp, 2
                            ),
                            "classification": prediction[
                                "classification"
                            ],
                            "anomaly_score": round(
                                float(
                                    prediction["anomaly_score"]
                                ),
                                4,
                            ),
                        }
                    )

                frame_index += 1

    finally:
        capture.release()

    if not results:
        raise ValueError("No video frames inspected")

    defective_frames = sum(
        row["classification"] == "DEFECTIVE"
        for row in results
    )

    decision = "FAIL" if defective_frames else "PASS"

    return {
        "inspection_type": "Video anomaly inspection",
        "video_file": video_path.name,
        "fps": round(fps, 2),
        "processed_video_frames": frame_index,
        "inspected_frames": len(results),
        "defective_frames": defective_frames,
        "decision": decision,
        "sampling_interval_seconds": sample_interval_seconds,
        "inspection_truncated": limit_reached,
        "predictions": results,
    }
