
import csv

import pytest

from inspectai.ingestion.versioning import create_dataset_version


FIELDS = ["path", "valid", "sha256", "is_duplicate"]


def write_manifest(path, records):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


@pytest.fixture
def sample_records():
    return [
        {
            "path": "images/a.png",
            "valid": True,
            "sha256": "abc123",
            "is_duplicate": False,
        },
        {
            "path": "images/b.png",
            "valid": True,
            "sha256": "def456",
            "is_duplicate": False,
        },
    ]


def test_fingerprint_independent_of_row_order(
    tmp_path, sample_records
):
    manifest = tmp_path / "manifest.csv"

    write_manifest(manifest, sample_records)
    first = create_dataset_version(
        manifest, tmp_path / "version1.json"
    )

    write_manifest(manifest, list(reversed(sample_records)))
    second = create_dataset_version(
        manifest, tmp_path / "version2.json"
    )

    assert first["dataset_version"] == second["dataset_version"]


def test_fingerprint_changes_with_content(
    tmp_path, sample_records
):
    manifest = tmp_path / "manifest.csv"

    write_manifest(manifest, sample_records)
    first = create_dataset_version(
        manifest, tmp_path / "version1.json"
    )

    changed_records = [dict(row) for row in sample_records]
    changed_records[0]["sha256"] = "different_checksum"

    write_manifest(manifest, changed_records)
    second = create_dataset_version(
        manifest, tmp_path / "version2.json"
    )

    assert first["dataset_version"] != second["dataset_version"]


def test_empty_manifest_rejected(tmp_path):
    manifest = tmp_path / "empty.csv"
    write_manifest(manifest, [])

    with pytest.raises(ValueError, match="empty"):
        create_dataset_version(
            manifest, tmp_path / "version.json"
        )
