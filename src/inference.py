"""Load the trained model and score single patient records."""

from pathlib import Path

import joblib
import pandas as pd

from src.config import DECISION_THRESHOLD, FEATURES, get_model_path


def load_model(path: Path | None = None):
    """Load the serialized scikit-learn pipeline."""
    return joblib.load(path or get_model_path())


def predict_record(model, record: dict) -> dict:
    """Return the predicted label and the model's confidence for one patient."""
    # Cast to float so int and float inputs hit the same encoder categories.
    X = pd.DataFrame([record], columns=FEATURES).astype(float)
    probability = float(model.predict_proba(X)[0, 1])
    label = int(probability >= DECISION_THRESHOLD)
    confidence = probability if label == 1 else 1.0 - probability
    return {
        "prediction": label,
        "label": "heart_disease" if label else "no_heart_disease",
        "probability_heart_disease": round(probability, 4),
        "confidence": round(confidence, 4),
    }
