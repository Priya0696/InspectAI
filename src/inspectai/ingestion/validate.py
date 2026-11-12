from pathlib import Path
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def validate_image(path: str | Path) -> dict:
    """Validate an image and return its metadata."""
    path = Path(path)

    result = {
        "path": str(path),
        "valid": False,
        "width": None,
        "height": None,
        "format": None,
        "error": None,
    }

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result["error"] = "unsupported_extension"
        return result

    if not path.is_file():
        result["error"] = "file_not_found"
        return result

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            result["width"], result["height"] = image.size
            result["format"] = image.format

        result["valid"] = True

    except (OSError, UnidentifiedImageError, ValueError):
        result["error"] = "invalid_image"

    return result