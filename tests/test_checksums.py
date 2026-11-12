from inspectai.ingestion.checksums import calculate_sha256


def test_same_content_same_hash(tmp_path):
    first = tmp_path / "image1.png"
    second = tmp_path / "image2.png"

    first.write_bytes(b"identical image data")
    second.write_bytes(b"identical image data")

    assert calculate_sha256(first) == calculate_sha256(second)


def test_different_content_different_hash(tmp_path):
    first = tmp_path / "image1.png"
    second = tmp_path / "image2.png"

    first.write_bytes(b"image A")
    second.write_bytes(b"image B")

    assert calculate_sha256(first) != calculate_sha256(second)


def test_sha256_length(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"test data")

    checksum = calculate_sha256(image)

    assert len(checksum) == 64