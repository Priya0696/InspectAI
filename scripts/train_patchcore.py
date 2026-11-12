
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# 1. Point to our existing dataset
dataset_root = Path("data/raw/mvtec").resolve()

# 2. Load the Bottle dataset
datamodule = MVTecAD(
    root=dataset_root,
    category="bottle",
    train_batch_size=4,
    eval_batch_size=4,
    num_workers=0,
)

# 3. Create the PatchCore model
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.01,
    pre_processor=Patchcore.configure_pre_processor(
        image_size=(256, 256),
    ),
)

# 4. Configure training
engine = Engine(
    accelerator="cpu",
    devices=1,
    max_epochs=1,
    default_root_dir="reports/patchcore",
)

# 5. Build the memory bank from normal images
print("Starting PatchCore training...")
engine.fit(model=model, datamodule=datamodule)

print("PatchCore training completed!")
