
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Dataset
datamodule = MVTecAD(
    root=Path("data/raw/mvtec").resolve(),
    category="bottle",
    train_batch_size=4,
    eval_batch_size=4,
    num_workers=0,
)

# Recreate the original model configuration
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.01,
    pre_processor=Patchcore.configure_pre_processor(
        image_size=(256, 256),
    ),
)

# Use our verified checkpoint
checkpoint = Path(
    "reports/patchcore/Patchcore/MVTecAD/bottle/"
    "v0/weights/lightning/model.ckpt"
).resolve()

if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

# Evaluation
engine = Engine(
    accelerator="cpu",
    devices=1,
    default_root_dir="reports/patchcore_evaluation",
)

print("Evaluating PatchCore on MVTec Bottle test images...")

results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(checkpoint),
)

print("\nEvaluation results:")
for result in results:
    for metric, value in result.items():
        print(f"{metric}: {value}")
