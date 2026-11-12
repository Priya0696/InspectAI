
from pathlib import Path

from lightning.pytorch import seed_everything

from anomalib.data import MVTecAD
from anomalib.data.utils import ValSplitMode, TestSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore


# 1. Reproducibility
seed_everything(42, workers=True)

# 2. Independent data splits
datamodule = MVTecAD(
    root=Path("data/raw/mvtec").resolve(),
    category="bottle",
    train_batch_size=4,
    eval_batch_size=4,
    num_workers=0,
    val_split_mode=ValSplitMode.FROM_TEST,
    val_split_ratio=0.5,
    test_split_mode=TestSplitMode.FROM_DIR,
)

# 3. PatchCore configuration
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.01,
    pre_processor=Patchcore.configure_pre_processor(
        image_size=(256, 256),
    ),
)

# 4. Separate experiment directory
engine = Engine(
    accelerator="cpu",
    devices=1,
    max_epochs=1,
    default_root_dir="reports/patchcore_clean",
)

print("\n=== TRAINING WITH INDEPENDENT SPLITS ===")
engine.fit(model=model, datamodule=datamodule)

print("\n=== EVALUATING ON HELD-OUT TEST SET ===")
results = engine.test(
    model=model,
    datamodule=datamodule,
)

print("\n=== CLEAN EVALUATION RESULTS ===")
for result in results:
    for metric, value in result.items():
        print(f"{metric}: {value}")

print("\nExperiment completed!")
