
import csv
from pathlib import Path

from inspectai.inference.predictor import InspectAIPredictor
from inspectai.inspection.video import inspect_video


VIDEO_PATH = Path(
    "data/processed/videos/bottle_inspection.mp4"
)

CHECKPOINT_PATH = Path(
    "reports/patchcore_384/Patchcore/MVTecAD/"
    "bottle/v0/weights/lightning/model.ckpt"
)

OUTPUT_DIR = Path("reports/video")
OUTPUT_CSV = OUTPUT_DIR / "video_predictions.csv"

SAMPLE_INTERVAL_SECONDS = 1.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(VIDEO_PATH)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    print("Loading PatchCore model...")

    predictor = InspectAIPredictor(
        checkpoint_path=str(CHECKPOINT_PATH)
    )

    print("Inspecting video...")

    report = inspect_video(
        video_path=VIDEO_PATH,
        predictor=predictor,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
        max_frames=100,
    )

    # Save frame-level predictions as CSV.
    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "timestamp_seconds",
                "classification",
                "anomaly_score",
            ],
        )

        writer.writeheader()
        writer.writerows(report["predictions"])

    # Display frame-level predictions.
    for row in report["predictions"]:
        print(
            f"{row['timestamp_seconds']:5.1f}s | "
            f"{row['classification']:10s} | "
            f"Score: {row['anomaly_score']:.4f}"
        )

    print("\n=== INSPECTAI VIDEO INSPECTION ===")
    print(
        f"Processed video frames: "
        f"{report['processed_video_frames']}"
    )
    print(
        f"Inspected frames: "
        f"{report['inspected_frames']}"
    )
    print(
        f"Defective sampled frames: "
        f"{report['defective_frames']}"
    )
    print(f"Video decision: {report['decision']}")
    print(
        f"Inspection truncated: "
        f"{report['inspection_truncated']}"
    )
    print(f"Results saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
