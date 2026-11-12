
from pathlib import Path

import torch

checkpoint_path = Path(
    "reports/patchcore/Patchcore/MVTecAD/bottle/"
    "v0/weights/lightning/model.ckpt"
)

if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

print("\n=== CHECKPOINT INFORMATION ===")
print("Epoch:", checkpoint.get("epoch"))
print("Global step:", checkpoint.get("global_step"))

print("\n=== THRESHOLD-RELATED STATE ===")
state = checkpoint.get("state_dict", {})

matches = [
    (name, value)
    for name, value in state.items()
    if any(
        keyword in name.lower()
        for keyword in (
            "threshold",
            "normalization",
            "min_max",
        )
    )
]

if matches:
    for name, value in matches:
        if isinstance(value, torch.Tensor):
            print(f"{name}: {value.detach().cpu().tolist()}")
        else:
            print(f"{name}: {value}")
else:
    print("No threshold-related entries found in state_dict.")

print("\n=== CALLBACKS ===")
callbacks = checkpoint.get("callbacks", {})

for name in callbacks:
    print(name)

print("\n=== HYPERPARAMETERS ===")
hparams = checkpoint.get("hyper_parameters", {})

if isinstance(hparams, dict):
    for name, value in hparams.items():
        print(f"{name}: {value}")
else:
    print(hparams)
