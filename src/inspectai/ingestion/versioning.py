
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def create_dataset_version(
    manifest_path: str | Path,
    output_path: str | Path,
    dataset_name: str = "mvtec-bottle",
    dataset_root: str | Path | None = None,
) -> dict:
    """Generate a reproducible fingerprint from an image manifest."""

    manifest_path = Path(manifest_path)
    output_path = Path(output_path)

    with manifest_path.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        records = list(csv.DictReader(file))

    if not records:
        raise ValueError("Dataset manifest is empty")

    root = Path(dataset_root).resolve() if dataset_root else None

    def normalized_path(value: str) -> str:
        path = Path(value)

        if root is not None:
            absolute_path = path.resolve()

            try:
                return absolute_path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"Image path is outside dataset root: {path}"
                ) from exc

        if path.is_absolute():
            raise ValueError(
                "Absolute image paths require --dataset-root "
                "for portable dataset fingerprints"
            )

        return path.as_posix()

    entries = []
    valid_count = 0
    duplicate_count = 0

    for record in records:
        is_valid = str(record["valid"]).lower() == "true"
        is_duplicate = (
            str(record["is_duplicate"]).lower() == "true"
        )

        if is_valid:
            valid_count += 1

        if is_duplicate:
            duplicate_count += 1

        entries.append({
            "path": normalized_path(record["path"]),
            "valid": is_valid,
            "sha256": record["sha256"] or None,
        })

    entries.sort(key=lambda item: item["path"])

    canonical_data = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
    )

    fingerprint = hashlib.sha256(
        canonical_data.encode("utf-8")
    ).hexdigest()

    version = {
        "dataset_name": dataset_name,
        "dataset_version": f"sha256:{fingerprint}",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_manifest": str(manifest_path),
        "total_images": len(records),
        "valid_images": valid_count,
        "invalid_images": len(records) - valid_count,
        "duplicate_images": duplicate_count,
        "fingerprint_method": (
            "SHA256 of sorted image paths, validity "
            "and content checksums"
        ),
    }

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with output_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(version, file, indent=2)

    return version
