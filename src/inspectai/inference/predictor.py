
from pathlib import Path

import torch
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


class InspectAIPredictor:
    """Run industrial anomaly detection using trained PatchCore."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        self.device = torch.device("cpu")

        self.model = Patchcore.load_from_checkpoint(
            str(self.checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )
        self.model.eval()

        self.engine = Engine(
            accelerator="cpu",
            devices=1,
            logger=False,
        )

        print("InspectAI predictor initialized")

    def get_model_info(self) -> dict:
        memory_bank = self.model.model.memory_bank

        return {
            "model": "PatchCore",
            "device": str(self.device),
            "memory_bank_size": int(memory_bank.shape[0]),
            "feature_dimensions": int(memory_bank.shape[1]),
            "checkpoint": str(self.checkpoint_path),
        }

    def predict(self, image_path: str) -> dict:
        """Predict whether an industrial image contains a defect."""

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        dataset = PredictDataset(
            path=path,
            image_size=(384, 384),
        )

        with torch.inference_mode():
            predictions = self.engine.predict(
                model=self.model,
                dataset=dataset,
            )

        if not predictions:
            raise RuntimeError("Model returned no predictions")

        batch = predictions[0]

        score = float(batch.pred_score[0].cpu().item())
        label = int(batch.pred_label[0].cpu().item())

        mask = batch.pred_mask[0].cpu().numpy()

        return {
            "image_path": str(path),
            "anomaly_score": score,
            "is_anomalous": bool(label),
            "classification": (
                "DEFECTIVE" if label else "NORMAL"
            ),
            "predicted_mask": mask,
        }
