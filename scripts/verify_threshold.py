
from pathlib import Path

from lightning.pytorch import seed_everything
from anomalib.data import MVTecAD
from anomalib.data.utils import ValSplitMode, TestSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore

seed_everything(42, workers=True)

data = MVTecAD(
    root=Path("data/raw/mvtec").resolve(),
    category="bottle",
    train_batch_size=4,
    eval_batch_size=4,
    num_workers=0,
    val_split_mode=ValSplitMode.FROM_TEST,
    val_split_ratio=0.5,
    test_split_mode=TestSplitMode.FROM_DIR,
)

data.setup("fit")

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.01,
    pre_processor=Patchcore.configure_pre_processor(
        image_size=(256, 256),
    ),
)

checkpoint = Path(
    "reports/patchcore_clean/Patchcore/MVTecAD/"
    "bottle/v0/weights/lightning/model.ckpt"
).resolve()

engine = Engine(
    accelerator="cpu",
    devices=1,
    default_root_dir="reports/threshold_verification",
)

predictions = engine.predict(
    model=model,
    dataloaders=data.val_dataloader(),
    ckpt_path=str(checkpoint),
    return_predictions=True,
)

different_pixels = 0
total_pixels = 0

for batch in predictions:
    expected = batch.anomaly_map > 0.5
    actual = batch.pred_mask.bool()

    different_pixels += (expected != actual).sum().item()
    total_pixels += actual.numel()

print("\n=== THRESHOLD VERIFICATION ===")
print("Different pixels:", different_pixels)
print("Total pixels:", total_pixels)
print("Exact match:", different_pixels == 0)
