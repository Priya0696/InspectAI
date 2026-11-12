
import argparse
from pathlib import Path

from inspectai.ingestion.versioning import create_dataset_version


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/dataset_version.json"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    result = create_dataset_version(
        manifest_path=args.manifest,
        output_path=args.output,
        dataset_root=args.dataset_root,
    )

    print("\n=== INSPECTAI DATASET VERSION ===")
    print("Dataset:", result["dataset_name"])
    print("Version:", result["dataset_version"])
    print("Total images:", result["total_images"])
    print("Valid images:", result["valid_images"])
    print("Invalid images:", result["invalid_images"])
    print("Duplicates:", result["duplicate_images"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
