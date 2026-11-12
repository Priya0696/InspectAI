import hashlib
from pathlib import Path


def calculate_sha256(path: str | Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    path = Path(path)
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()