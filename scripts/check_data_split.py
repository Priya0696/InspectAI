
from pathlib import Path

from lightning.pytorch import seed_everything
from anomalib.data import MVTecAD
from anomalib.data.utils import ValSplitMode, TestSplitMode

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

print("\n=== INDEPENDENT DATA SPLITS ===")
print("Training:", len(data.train_data))
print("Validation:", len(data.val_data))
print("Testing:", len(data.test_data))

def get_paths(dataset):
    samples = dataset.samples
    return set(str(Path(path).resolve()) for path in samples["image_path"])

train_paths = get_paths(data.train_data)
val_paths = get_paths(data.val_data)
test_paths = get_paths(data.test_data)

print("\n=== OVERLAP CHECK ===")
print("Train / Validation:", len(train_paths & val_paths))
print("Train / Test:", len(train_paths & test_paths))
print("Validation / Test:", len(val_paths & test_paths))

assert not (train_paths & val_paths)
assert not (train_paths & test_paths)
assert not (val_paths & test_paths)

print("\nSUCCESS: All three datasets are independent!")
