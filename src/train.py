"""Training pipeline: load data -> train -> evaluate -> save model + metadata.

Run:  python -m src.train
"""
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src import config
from src.logger import get_logger

logger = get_logger("pipeline.train")


def load_data():
    logger.info("Loading Iris dataset...")
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y, list(data.target_names)


def train_model() -> dict:
    X, y, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    logger.info("Train size=%d, Test size=%d", len(X_train), len(X_test))

    logger.info("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    logger.info("Model accuracy: %.4f", accuracy)
    logger.info(
        "\n%s", classification_report(y_test, predictions, target_names=target_names)
    )

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    metadata = {
        "model": "RandomForestClassifier",
        "accuracy": accuracy,
        "n_estimators": config.N_ESTIMATORS,
        "features": list(X.columns),
        "classes": target_names,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("Model saved to %s", config.MODEL_PATH)
    return metadata


if __name__ == "__main__":
    train_model()
