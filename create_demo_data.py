from pathlib import Path
from PIL import Image, ImageDraw

output = Path("data/raw/demo")
output.mkdir(parents=True, exist_ok=True)

for i in range(5):
    image = Image.new("RGB", (256, 256), "lightgray")
    draw = ImageDraw.Draw(image)

    # Simulated industrial component
    draw.rectangle((60, 60, 196, 196), fill="steelblue")

    # Simulated defect in one sample
    if i == 4:
        draw.line((90, 90, 170, 170), fill="red", width=5)

    image.save(output / f"component_{i:03d}.png")

# Create an exact duplicate
duplicate = output / "component_duplicate.png"
duplicate.write_bytes((output / "component_000.png").read_bytes())

print("Created 6 demo images")