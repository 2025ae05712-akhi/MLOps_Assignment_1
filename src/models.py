"""Candidate model definitions and hyperparameter grids."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def build_pipeline(estimator) -> Pipeline:
    """Combine the shared preprocessing with an estimator into one pipeline."""
    return Pipeline([("preprocess", build_preprocessor()), ("clf", estimator)])


def candidate_models() -> dict[str, tuple[Pipeline, dict]]:
    """Return each candidate pipeline with its tuning grid."""
    return {
        "logistic_regression": (
            build_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 4, 8],
                "clf__min_samples_leaf": [1, 3, 5],
            },
        ),
    }
