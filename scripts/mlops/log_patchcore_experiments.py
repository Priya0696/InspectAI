
from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("InspectAI-PatchCore")

EXPERIMENTS = [
    {
        "name": "PatchCore-256",
        "resolution": 256,
        "pixel_auroc": 0.975268,
        "pixel_f1": 0.646274,
        "image_auroc": 1.0,
        "image_f1": 1.0,
        "checkpoint": (
            "reports/patchcore_clean/Patchcore/"
            "MVTecAD/bottle/v0/weights/lightning/model.ckpt"
        ),
    },
    {
        "name": "PatchCore-384",
        "resolution": 384,
        "pixel_auroc": 0.985478,
        "pixel_f1": 0.750378,
        "image_auroc": 1.0,
        "image_f1": 1.0,
        "checkpoint": (
            "reports/patchcore_384/Patchcore/"
            "MVTecAD/bottle/v0/weights/lightning/model.ckpt"
        ),
    },
]


def main():
    for experiment in EXPERIMENTS:
        checkpoint_path = PROJECT_ROOT / experiment["checkpoint"]

        with mlflow.start_run(run_name=experiment["name"]):
            mlflow.log_params({
                "model": "PatchCore",
                "dataset": "MVTec AD bottle",
                "backbone": "resnet18",
                "resolution": experiment["resolution"],
                "coreset_ratio": 0.01,
                "random_seed": 42,
                "checkpoint_path": experiment["checkpoint"],
            })

            mlflow.log_metrics({
                "pixel_auroc": experiment["pixel_auroc"],
                "pixel_f1": experiment["pixel_f1"],
                "image_auroc": experiment["image_auroc"],
                "image_f1": experiment["image_f1"],
            })

            mlflow.set_tags({
                "project": "InspectAI",
                "phase": "2D anomaly detection",
                "experiment_type": "historical_import",
                "evaluation_note": (
                    "Previously measured held-out split; "
                    "model selection used test metrics."
                ),
                "checkpoint_exists": str(
                    checkpoint_path.is_file()
                ).lower(),
            })

        print(
            f"Logged {experiment['name']} | "
            f"Pixel AUROC: {experiment['pixel_auroc']:.4f} | "
            f"Pixel F1: {experiment['pixel_f1']:.4f}"
        )

    print("\nAll experiments logged successfully.")
    print(f"Tracking database: {DB_PATH}")


if __name__ == "__main__":
    main()
