from PIL import Image

from inspectai.ingestion.validate import validate_image


def test_valid_image(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (64, 48)).save(path)

    result = validate_image(path)

    assert result["valid"] is True
    assert result["width"] == 64
    assert result["height"] == 48


def test_missing_image(tmp_path):
    result = validate_image(tmp_path / "missing.png")

    assert result["valid"] is False
    assert result["error"] == "file_not_found"


def test_unsupported_file(tmp_path):
    result = validate_image(tmp_path / "document.txt")

    assert result["valid"] is False
    assert result["error"] == "unsupported_extension"


def test_corrupted_image(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")

    result = validate_image(path)

    assert result["valid"] is False
    assert result["error"] == "invalid_image"