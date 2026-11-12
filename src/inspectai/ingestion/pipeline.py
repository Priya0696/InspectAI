import csv
from pathlib import Path

from inspectai.ingestion.discover import discover_images
from inspectai.ingestion.validate import validate_image
from inspectai.ingestion.checksums import calculate_sha256


def ingest_images(
    input_dir: str | Path,
    output_file: str | Path,
) -> list[dict]:
    """Discover, validate and catalog images."""

    images = discover_images(input_dir)
    records = []
    seen_hashes = set()

    for image_path in images:
        metadata = validate_image(image_path)

        if metadata["valid"]:
            checksum = calculate_sha256(image_path)
            is_duplicate = checksum in seen_hashes
            seen_hashes.add(checksum)
        else:
            checksum = None
            is_duplicate = False

        records.append({
            **metadata,
            "sha256": checksum,
            "is_duplicate": is_duplicate,
        })

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "path", "valid", "width", "height",
        "format", "error", "sha256", "is_duplicate",
    ]

    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    return records