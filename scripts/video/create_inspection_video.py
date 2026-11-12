
from pathlib import Path

import cv2


DATA_DIR = Path("data/raw/mvtec/bottle/test")
OUTPUT_DIR = Path("data/processed/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "bottle_inspection.mp4"

FRAME_SIZE = (640, 480)
FPS = 10
FRAMES_PER_IMAGE = 20

# Use known normal and defective examples.
image_paths = [
    DATA_DIR / "good/000.png",
    DATA_DIR / "good/001.png",
    DATA_DIR / "broken_large/004.png",
    DATA_DIR / "good/002.png",
]

for path in image_paths:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    FPS,
    FRAME_SIZE,
)

if not writer.isOpened():
    raise RuntimeError("Could not create video file")

try:
    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Cannot read: {image_path}")

        image = cv2.resize(image, FRAME_SIZE)

        # Repeat each image for two seconds.
        for _ in range(FRAMES_PER_IMAGE):
            writer.write(image)

        print(f"Added: {image_path.name}")
finally:
    writer.release()

capture = cv2.VideoCapture(str(OUTPUT_PATH))

if not capture.isOpened():
    raise RuntimeError("Generated video cannot be opened")

frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = capture.get(cv2.CAP_PROP_FPS)
capture.release()

print("\n=== VIDEO CREATED ===")
print(f"File: {OUTPUT_PATH}")
print(f"Frames: {frame_count}")
print(f"FPS: {fps:.1f}")
print(f"Duration: {frame_count / fps:.1f} seconds")
