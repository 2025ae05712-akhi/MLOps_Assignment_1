# Heart Disease Risk: MLOps Pipeline

End-to-end MLOps project for predicting heart disease risk from the UCI Heart Disease (Cleveland) dataset. It covers data acquisition and EDA, model training with MLflow tracking, a FastAPI inference service, unit tests, CI/CD with GitHub Actions, a Docker image, Kubernetes manifests, and Prometheus monitoring.

## Project structure

```
.
├── data/download_data.py      # Downloads the UCI dataset to data/heart_raw.csv
├── scripts/eda.py             # EDA summary and figures
├── src/
│   ├── config.py              # Paths, constants, feature lists
│   ├── data.py                # Loading and cleaning
│   ├── preprocessing.py       # ColumnTransformer (impute, scale, one-hot)
│   ├── models.py              # Candidate models and hyperparameter grids
│   ├── train.py               # CV tuning, test evaluation, MLflow logging, model selection
│   ├── inference.py           # Model loading and single-record scoring
│   └── api.py                 # FastAPI service with logging and Prometheus metrics
├── tests/                     # pytest unit and API tests
├── k8s/deployment.yaml        # Deployment and LoadBalancer Service
├── monitoring/prometheus.yml  # Prometheus scrape config
├── .github/workflows/ci.yml   # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml         # API + Prometheus + Grafana
├── models/                    # Saved model and metrics.json
└── reports/                   # EDA summary and figures
```

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Reproduce the pipeline

```bash
python data/download_data.py     # fetch dataset (303 rows, 14 columns)
python -m scripts.eda            # EDA summary and figures in reports/
python -m src.train              # tune, evaluate, log to MLflow, save models/model.joblib
pytest                           # run unit and API tests
ruff check .                     # lint
```

View MLflow runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Results

Model selection uses 5-fold stratified cross-validation ROC-AUC on the 80% training split. The 20% test split is used only for reporting.

| Model | CV ROC-AUC (mean ± std) | Test accuracy | Test precision | Test recall | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic regression (selected) | 0.903 ± 0.017 | 0.885 | 0.839 | 0.929 | 0.965 |
| Random forest | 0.902 ± 0.033 | 0.852 | 0.806 | 0.893 | 0.948 |

Selected hyperparameters: `C=0.1`, `class_weight=balanced`. Full metrics are in [models/metrics.json](models/metrics.json).

## Serving the model

Run locally:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness and model status |
| POST | `/predict` | Risk prediction for one patient |
| GET | `/metrics` | Prometheus metrics |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

Response fields: `prediction` (0 or 1), `label`, `probability_heart_disease`, `confidence`. The fields `ca` and `thal` are optional.

Requests are logged to stdout and to `logs/api.log` (override the directory with `LOG_DIR`). The model path can be set with `MODEL_PATH`.

## Docker

The image needs `models/model.joblib`, so run the training step first.

```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest
```

Or with monitoring:

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default login admin/admin; add Prometheus at `http://prometheus:9090` as a data source)

## Kubernetes deployment

Local cluster with Minikube:

```bash
minikube start
minikube image load heart-disease-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl get pods,svc
minikube tunnel                       # in a separate terminal, to expose the LoadBalancer
kubectl get svc heart-api             # use EXTERNAL-IP, port 80
```

Cloud cluster: push the image to a registry (the CI publish job pushes to `ghcr.io/<owner>/<repo>`), change the `image:` field in `k8s/deployment.yaml`, then apply the manifest. The `EXTERNAL-IP` of the `heart-api` service is the public URL.

## CI/CD

The workflow in [.github/workflows/ci.yml](.github/workflows/ci.yml) runs on pushes and pull requests:

1. **lint-and-test**: `ruff check`, dataset download, `pytest` with a JUnit report.
2. **train**: EDA and training, uploads the model and MLflow data as artifacts.
3. **docker**: builds the image, runs the container, and smoke-tests `/health`, `/predict`, and `/metrics`.
4. **publish** (main branch only): pushes the image to GitHub Container Registry.

## Monitoring

The API exports these Prometheus metrics at `/metrics`:

- `api_requests_total{method,endpoint,status}`
- `api_request_latency_seconds` (histogram)
- `api_predictions_total{label}`

[monitoring/prometheus.yml](monitoring/prometheus.yml) scrapes the API every 15 seconds.

## Screenshots

Place screenshots in `screenshots/`:

- EDA figures (`reports/figures/`)
- MLflow UI showing experiments and runs
- Passing GitHub Actions run
- Docker container running and `/health` response
- `kubectl get pods,svc` and the external URL response
- Prometheus targets page and Grafana dashboard

## Notes

- Dataset: UCI Heart Disease (Cleveland), 303 rows, binary target (`num > 0` means heart disease).
- The `ca` and `thal` columns contain a few missing values that are imputed in the preprocessing pipeline.
