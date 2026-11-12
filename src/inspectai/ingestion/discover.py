from pathlib import Path

from inspectai.ingestion.validate import SUPPORTED_EXTENSIONS


def discover_images(directory: str | Path) -> list[Path]:
    """Find supported image files in a directory and its subfolders."""

    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Dataset directory not found: {directory}"
        )

    images = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(images)