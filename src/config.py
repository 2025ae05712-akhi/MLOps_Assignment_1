"""Project-wide paths, feature definitions and settings."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "heart_raw.csv"
CLEAN_CSV = DATA_DIR / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT_NAME = "heart-disease-risk"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5

TARGET = "target"
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_model_path() -> Path:
    """Return the serialized model path, overridable with the MODEL_PATH env var."""
    return Path(os.getenv("MODEL_PATH", MODEL_DIR / "model.joblib"))
