"""API tests: endpoints, validation, logging and metrics."""

import pytest
from fastapi.testclient import TestClient

import src.api as api

VALID_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}  # fmt: skip


@pytest.fixture
def client(fitted_model_path, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(fitted_model_path))
    monkeypatch.setattr(api, "LOG_DIR", tmp_path / "logs")
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_reports_model_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_prediction_and_confidence(client):
    response = client.post("/predict", json=VALID_PATIENT)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["label"] == ("heart_disease" if body["prediction"] else "no_heart_disease")


def test_predict_accepts_missing_optional_fields(client):
    patient = {k: v for k, v in VALID_PATIENT.items() if k not in ("ca", "thal")}

    response = client.post("/predict", json=patient)

    assert response.status_code == 200


@pytest.mark.parametrize(
    "bad_patient",
    [
        {**VALID_PATIENT, "age": -5},
        {**VALID_PATIENT, "sex": 3},
        {k: v for k, v in VALID_PATIENT.items() if k != "chol"},
    ],
)
def test_predict_rejects_invalid_input(client, bad_patient):
    response = client.post("/predict", json=bad_patient)

    assert response.status_code == 422


def test_metrics_endpoint_exposes_request_and_prediction_counters(client):
    client.post("/predict", json=VALID_PATIENT)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert "api_predictions_total" in response.text


def test_requests_are_written_to_log_file(client, tmp_path):
    client.post("/predict", json=VALID_PATIENT)

    log_text = (tmp_path / "logs" / "api.log").read_text()

    assert "method=POST path=/predict status=200" in log_text
