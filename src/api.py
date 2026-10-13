"""FastAPI service that serves heart disease risk predictions."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from src.config import LOG_DIR, get_model_path
from src.inference import load_model, predict_record

logger = logging.getLogger("heart_api")

REQUEST_COUNT = Counter(
    "api_requests_total", "HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "HTTP request latency", ["method", "endpoint"]
)
PREDICTION_COUNT = Counter("api_predictions_total", "Predictions by label", ["label"])
KNOWN_ENDPOINTS = {"/predict", "/health", "/metrics"}


def configure_logging() -> None:
    """Send API logs to stdout and to LOG_DIR/api.log (handlers are reset on each call)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(LOG_DIR / "api.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.model = load_model(get_model_path())
    logger.info("Loaded model from %s", get_model_path())
    yield


class HeartFeatures(BaseModel):
    """Patient measurements. Field meanings follow the UCI Heart Disease dataset."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }
        }
    )  # fmt: skip

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: float = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: float = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=30, le=250, description="Maximum heart rate achieved")
    exang: float = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: float = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: float | None = Field(None, ge=0, le=3, description="Major vessels coloured (optional)")
    thal: float | None = Field(None, description="3 normal, 6 fixed, 7 reversible (optional)")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_heart_disease: float
    confidence: float


app = FastAPI(
    title="Heart Disease Risk API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path if request.url.path in KNOWN_ENDPOINTS else "other"
    REQUEST_COUNT.labels(request.method, endpoint, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
    logger.info(
        "method=%s path=%s status=%s latency_ms=%.2f client=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed * 1000,
        request.client.host if request.client else "unknown",
    )
    return response


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model")}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HeartFeatures, request: Request) -> dict:
    result = predict_record(request.app.state.model, features.model_dump())
    PREDICTION_COUNT.labels(label=result["label"]).inc()
    logger.info("prediction=%s confidence=%.4f", result["label"], result["confidence"])
    return result


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
