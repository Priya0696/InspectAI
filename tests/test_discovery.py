from PIL import Image

from inspectai.ingestion.discover import discover_images


def test_discover_images(tmp_path):
    images_dir = tmp_path / "dataset"
    images_dir.mkdir()

    Image.new("RGB", (32, 32)).save(images_dir / "part1.png")
    Image.new("RGB", (32, 32)).save(images_dir / "part2.jpg")
    (images_dir / "notes.txt").write_text("not an image")

    found = discover_images(images_dir)

    assert len(found) == 2
    assert {p.name for p in found} == {"part1.png", "part2.jpg"}


def test_nested_folders(tmp_path):
    nested = tmp_path / "dataset" / "good"
    nested.mkdir(parents=True)

    Image.new("RGB", (32, 32)).save(nested / "sample.png")

    found = discover_images(tmp_path / "dataset")

    assert len(found) == 1
    assert found[0].name == "sample.png"


def test_missing_directory(tmp_path):
    try:
        discover_images(tmp_path / "missing")
        assert False, "Expected NotADirectoryError"
    except NotADirectoryError:
        pass