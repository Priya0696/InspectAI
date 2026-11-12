
from pathlib import Path

from inspectai.inference.predictor import InspectAIPredictor


CHECKPOINT = (
    "reports/patchcore_384/Patchcore/MVTecAD/"
    "bottle/v0/weights/lightning/model.ckpt"
)

TEST_IMAGES = {
    "NORMAL": Path(
        "data/raw/mvtec/bottle/test/good/000.png"
    ),
    "DEFECTIVE": Path(
        "data/raw/mvtec/bottle/test/broken_large/004.png"
    ),
}


def main():
    predictor = InspectAIPredictor(CHECKPOINT)

    print("\n=== INSPECTAI INFERENCE TEST ===")

    for expected, image_path in TEST_IMAGES.items():
        if not image_path.is_file():
            print(f"Missing image: {image_path}")
            continue

        result = predictor.predict(str(image_path))

        predicted = result["classification"]

        print(f"\nImage: {image_path.name}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Anomaly score: {result['anomaly_score']:.4f}")
        print(
            "Defective pixels:",
            int(result["predicted_mask"].sum())
        )

        print(
            "Result:",
            "PASS" if predicted == expected else "FAIL"
        )


if __name__ == "__main__":
    main()
