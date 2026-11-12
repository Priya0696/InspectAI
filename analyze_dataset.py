from pathlib import Path
from collections import Counter

dataset = Path("data/raw/mvtec/bottle")

train_images = list((dataset / "train").rglob("*.png"))
test_images = list((dataset / "test").rglob("*.png"))
masks = list((dataset / "ground_truth").rglob("*.png"))

print("=== InspectAI Dataset Analysis ===")
print(f"Training images: {len(train_images)}")
print(f"Test images: {len(test_images)}")
print(f"Ground-truth masks: {len(masks)}")

print("\nTest categories:")
categories = Counter(path.parent.name for path in test_images)

for category, count in sorted(categories.items()):
    print(f"{category}: {count}")

print("\nTraining categories:")
for category, count in sorted(
    Counter(path.parent.name for path in train_images).items()
):
    print(f"{category}: {count}")