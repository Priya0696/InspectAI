
from pathlib import Path
import json

from lightning.pytorch import seed_everything

from anomalib.data import MVTecAD
from anomalib.data.utils import ValSplitMode, TestSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore


# Reproduce the same data split as the 256x256 baseline
seed_everything(42, workers=True)

datamodule = MVTecAD(
    root=Path("data/raw/mvtec").resolve(),
    category="bottle",
    train_batch_size=2,
    eval_batch_size=2,
    num_workers=0,
    val_split_mode=ValSplitMode.FROM_TEST,
    val_split_ratio=0.5,
    test_split_mode=TestSplitMode.FROM_DIR,
)

datamodule.setup("fit")
datamodule.setup("test")

print("\n=== DATASET ===")
print("Training:", len(datamodule.train_data))
print("Validation:", len(datamodule.val_data))
print("Testing:", len(datamodule.test_data))

# Higher-resolution PatchCore
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.01,
    pre_processor=Patchcore.configure_pre_processor(
        image_size=(384, 384),
    ),
)

engine = Engine(
    accelerator="cpu",
    devices=1,
    max_epochs=1,
    default_root_dir="reports/patchcore_384",
)

print("\n=== TRAINING PATCHCORE 384x384 ===")
engine.fit(model=model, datamodule=datamodule)

print("\n=== EVALUATING HELD-OUT TEST SET ===")
results = engine.test(
    model=model,
    datamodule=datamodule,
)

print("\n=== 384x384 RESULTS ===")
metrics = {}

for result in results:
    for name, value in result.items():
        metrics[name] = float(value)
        print(f"{name}: {value}")

output = Path("reports/metrics/patchcore_384.json")
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w") as file:
    json.dump(
        {
            "experiment": "patchcore_384",
            "image_size": [384, 384],
            "train_images": len(datamodule.train_data),
            "validation_images": len(datamodule.val_data),
            "test_images": len(datamodule.test_data),
            "random_seed": 42,
            "metrics": metrics,
        },
        file,
        indent=2,
    )

print(f"\nResults saved to {output}")
