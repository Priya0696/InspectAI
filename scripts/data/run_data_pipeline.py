
import argparse
from pathlib import Path

from inspectai.ingestion.pipeline import ingest_images
from inspectai.ingestion.versioning import create_dataset_version


def main():
    parser = argparse.ArgumentParser(
        description="InspectAI automated dataset pipeline"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/mvtec/bottle"),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/bottle_manifest.csv"),
    )

    parser.add_argument(
        "--version-output",
        type=Path,
        default=Path("data/processed/dataset_version.json"),
    )

    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"Dataset directory not found: {args.input}")

    print("\n=== INSPECTAI DATA PIPELINE ===")
    print(f"Dataset: {args.input}")

    # Step 1: Discover, validate, and checksum images.
    records = ingest_images(
        input_dir=args.input,
        output_file=args.manifest,
    )

    if not records:
        raise RuntimeError("No images found in dataset")

    print(f"Discovered files: {len(records)}")
    print(f"Manifest saved: {args.manifest}")

    # Step 2: Generate version from the fresh manifest.
    version = create_dataset_version(
        manifest_path=args.manifest,
        output_path=args.version_output,
        dataset_name="mvtec-bottle",
    )

    print("\n=== DATA QUALITY ===")
    print(f"Valid: {version['valid_images']}")
    print(f"Invalid: {version['invalid_images']}")
    print(f"Duplicates: {version['duplicate_images']}")

    print("\n=== DATASET VERSION ===")
    print(version["dataset_version"])
    print(f"Metadata saved: {args.version_output}")

    if version["invalid_images"] > 0:
        raise RuntimeError(
            "Data quality check failed: invalid images found"
        )

    print("\nPipeline status: SUCCESS")


if __name__ == "__main__":
    main()
