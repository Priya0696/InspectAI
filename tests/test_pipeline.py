import csv

from PIL import Image

from inspectai.ingestion.pipeline import ingest_images


def test_ingestion_pipeline(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    image1 = dataset / "image1.png"
    image2 = dataset / "image2.png"

    Image.new("RGB", (32, 32), "red").save(image1)
    image2.write_bytes(image1.read_bytes())

    output = tmp_path / "manifests" / "images.csv"

    records = ingest_images(dataset, output)

    assert len(records) == 2
    assert all(record["valid"] for record in records)
    assert sum(record["is_duplicate"] for record in records) == 1

    assert output.exists()

    with output.open(newline="") as file:
        saved_records = list(csv.DictReader(file))

    assert len(saved_records) == 2
    assert saved_records[0]["sha256"] == saved_records[1]["sha256"]