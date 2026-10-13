"""Unit tests for preprocessing, model pipelines and inference."""

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.config import FEATURES, RAW_CSV
from src.data import load_dataset, split_features
from src.inference import load_model, predict_record
from src.models import build_pipeline, candidate_models
from src.preprocessing import build_preprocessor


def _to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def test_preprocessor_removes_missing_values(synthetic_data):
    X, y = synthetic_data
    assert X.isna().any().any()

    transformed = _to_dense(build_preprocessor().fit_transform(X, y))

    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(X)


def test_preprocessor_ignores_unseen_category(synthetic_data):
    X, y = synthetic_data
    preprocessor = build_preprocessor().fit(X, y)
    unseen = X.iloc[[0]].copy()
    unseen["cp"] = 99.0

    transformed = _to_dense(preprocessor.transform(unseen))

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_candidate_models_produce_valid_probabilities(synthetic_data):
    X, y = synthetic_data

    models = candidate_models()

    assert len(models) >= 2
    for name, (pipeline, grid) in models.items():
        assert grid, f"{name} has no tuning grid"
        proba = pipeline.fit(X, y).predict_proba(X)
        assert proba.shape == (len(X), 2), name
        assert np.all((proba >= 0) & (proba <= 1)), name
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, err_msg=name)


def test_saved_model_round_trip_gives_identical_predictions(tmp_path, synthetic_data):
    X, y = synthetic_data
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)

    loaded = joblib.load(path)

    np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X))


def test_predict_record_returns_label_and_confidence(fitted_model_path):
    model = load_model(fitted_model_path)
    record = {feature: 1.0 for feature in FEATURES}
    record["age"], record["trestbps"], record["chol"] = 60.0, 140.0, 250.0

    result = predict_record(model, record)

    assert result["prediction"] in (0, 1)
    assert result["label"] in ("heart_disease", "no_heart_disease")
    assert 0.5 <= result["confidence"] <= 1.0
    assert 0.0 <= result["probability_heart_disease"] <= 1.0


def test_predict_record_accepts_int_and_float_inputs(fitted_model_path):
    model = load_model(fitted_model_path)
    as_float = {feature: 1.0 for feature in FEATURES}
    as_int = {feature: 1 for feature in FEATURES}

    assert predict_record(model, as_int) == predict_record(model, as_float)


@pytest.mark.skipif(not RAW_CSV.exists(), reason="run data/download_data.py first")
def test_logistic_regression_cv_roc_auc_on_real_data():
    X, y = split_features(load_dataset())
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, C=0.1))

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")

    assert scores.mean() > 0.85
