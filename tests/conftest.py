"""Shared fixtures: a synthetic dataset and a small trained model file."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import FEATURES, TARGET
from src.models import build_pipeline


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small dataset with the production schema, some missing `ca` values and a learnable signal."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "age": rng.normal(55, 9, n),
            "trestbps": rng.normal(130, 15, n),
            "chol": rng.normal(240, 40, n),
            "thalach": rng.normal(150, 20, n),
            "oldpeak": rng.uniform(0, 4, n),
            "sex": rng.integers(0, 2, n).astype(float),
            "cp": rng.integers(1, 5, n).astype(float),
            "fbs": rng.integers(0, 2, n).astype(float),
            "restecg": rng.integers(0, 3, n).astype(float),
            "exang": rng.integers(0, 2, n).astype(float),
            "slope": rng.integers(1, 4, n).astype(float),
            "ca": rng.integers(0, 4, n).astype(float),
            "thal": rng.choice([3.0, 6.0, 7.0], n),
        }
    )[FEATURES]
    X.loc[::17, "ca"] = np.nan
    risk = 0.03 * (X["age"] - 55) + 0.8 * (X["cp"] == 4) + rng.normal(0, 0.5, n)
    y = pd.Series((risk > 0).astype(int), name=TARGET)
    return X, y


@pytest.fixture
def fitted_model_path(tmp_path, synthetic_data):
    """Serialize a pipeline trained on the synthetic data and return its path."""
    X, y = synthetic_data
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    return path
