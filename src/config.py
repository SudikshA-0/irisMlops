"""Central configuration for paths and training parameters."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

MODEL_PATH = MODEL_DIR / "iris_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
LOG_FILE = LOG_DIR / "pipeline.log"

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
