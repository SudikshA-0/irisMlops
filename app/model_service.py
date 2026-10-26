"""Loads the trained model and serves predictions."""
import json

import joblib
import pandas as pd

from app.schemas import IrisInput
from src import config
from src.logger import get_logger
from src.train import train_model

logger = get_logger("pipeline.service")


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.metadata: dict = {}

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not config.MODEL_PATH.exists():
            logger.warning("Model not found, training a new one...")
            train_model()
        self.model = joblib.load(config.MODEL_PATH)
        self.metadata = json.loads(config.METADATA_PATH.read_text())
        logger.info("Model loaded (accuracy=%.4f)", self.metadata["accuracy"])

    def predict(self, payload: IrisInput) -> dict:
        features = pd.DataFrame(
            [[
                payload.sepal_length,
                payload.sepal_width,
                payload.petal_length,
                payload.petal_width,
            ]],
            columns=self.metadata["features"],
        )
        proba = self.model.predict_proba(features)[0]
        idx = int(proba.argmax())
        return {
            "prediction": idx,
            "flower_name": self.metadata["classes"][idx],
            "confidence": round(float(proba[idx]), 4),
        }


model_service = ModelService()
