
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "mlflow.db"

VERSION_PATH = (
    PROJECT_ROOT / "data/processed/dataset_version.json"
)

EXPERIMENT_NAME = "InspectAI-PatchCore"


def main():
    if not VERSION_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset version not found: {VERSION_PATH}"
        )

    with VERSION_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: {EXPERIMENT_NAME}"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=100,
    )

    target_names = {"PatchCore-256", "PatchCore-384"}
    linked_count = 0

    for run in runs:
        run_name = run.data.tags.get("mlflow.runName")

        if run_name not in target_names:
            continue

        # Historical association: confirm this dataset
        # matches the data used in the original runs
        # before treating it as verified provenance.
        client.set_tag(
            run.info.run_id,
            "dataset_version",
            dataset["dataset_version"],
        )

        client.set_tag(
            run.info.run_id,
            "dataset_name",
            dataset["dataset_name"],
        )

        client.set_tag(
            run.info.run_id,
            "dataset_lineage_status",
            "historical_association_unverified",
        )

        print(f"Linked: {run_name}")
        linked_count += 1

    print("\n=== DATASET LINEAGE ===")
    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Version: {dataset['dataset_version']}")
    print(f"Runs linked: {linked_count}")


if __name__ == "__main__":
    main()
