# Heart Disease Risk Prediction: MLOps Assignment 01 (AIMLCZG523)

> Draft prepared with an assistant. Replace the placeholders in square brackets with your own analysis, screenshots, and reflections before submitting.

## 1. Data acquisition and EDA (5 marks)

- Source: UCI Machine Learning Repository, Heart Disease dataset (Cleveland subset), downloaded by [data/download_data.py](data/download_data.py).
- Shape: 303 rows, 13 features plus the target. Missing values: `ca` (4), `thal` (2), encoded as `?` in the source.
- Target: `num > 0` is mapped to 1 (heart disease). Classes: 164 no disease, 139 disease (45.9% positive).
- No duplicate rows.
- Figures: [reports/figures/histograms_by_target.png](reports/figures/histograms_by_target.png), [reports/figures/class_balance.png](reports/figures/class_balance.png), [reports/figures/correlation_heatmap.png](reports/figures/correlation_heatmap.png).
- Summary statistics: [reports/eda_summary.txt](reports/eda_summary.txt).

[Add your observations here, e.g. higher `oldpeak` and lower `thalach` are associated with disease, `cp` and `thal` are strongly categorical, and so on.]

## 2. Feature engineering and model development (8 marks)

- Preprocessing ([src/preprocessing.py](src/preprocessing.py)): numeric features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) use median imputation and standard scaling. Categorical features (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) use most-frequent imputation and one-hot encoding, with unseen categories ignored at inference.
- Candidate models ([src/models.py](src/models.py)): logistic regression and random forest, each in a scikit-learn Pipeline with a tuning grid.
- Tuning: 5-fold stratified cross-validation with `GridSearchCV` on the training split (80%). Selection metric: ROC-AUC.

| Model | CV ROC-AUC | Test ROC-AUC | Test recall | Test precision |
|---|---|---|---|---|
| Logistic regression (selected) | 0.903 ± 0.017 | 0.965 | 0.929 | 0.839 |
| Random forest | 0.902 ± 0.033 | 0.948 | 0.893 | 0.806 |

Justification for the choice: [explain in your own words why logistic regression is preferred, e.g. comparable CV score, lower variance, interpretability, faster inference].

## 3. Experiment tracking with MLflow (5 marks)

- Tracking backend: SQLite at `mlflow.db` (see [src/config.py](src/config.py)).
- Experiment: `heart-disease-risk`. Each model has a parent run with nested runs for each tuning candidate.
- Logged: parameters, CV metrics (mean and std), test metrics, confusion matrix and ROC figures, cross-validation results CSV, and the fitted model.
- Screenshot: [screenshots/mlflow_runs.png] [add screenshot].

## 4. Model packaging and reproducibility (7 marks)

- Model saved as `models/model.joblib` and logged as an MLflow model (cloudpickle serialization).
- Dependencies pinned in [requirements.txt](requirements.txt) and [requirements-dev.txt](requirements-dev.txt).
- Fixed random seed (`RANDOM_STATE=42`) and stratified splits.
- Configuration centralised in [src/config.py](src/config.py).
- Reproduce with the commands in [README.md](README.md).

## 5. CI/CD and automated testing (8 marks)

- Tests in [tests/](tests/): data loading and cleaning, preprocessing behaviour, model validity, save and load round trip, inference, and API endpoints (validation, optional fields, metrics, and logging). 20 tests pass locally.
- Linting with ruff.
- GitHub Actions workflow [.github/workflows/ci.yml](.github/workflows/ci.yml) with four jobs: lint and test, train, Docker build and smoke test, and publish to GHCR.
- Screenshot of a passing run: [screenshots/ci_pass.png] [add screenshot].

## 6. Model containerization with Docker (5 marks)

- [Dockerfile](Dockerfile): Python 3.11 slim base, pinned requirements, non-root user, health check, `uvicorn` entry point.
- [docker-compose.yml](docker-compose.yml) runs the API together with Prometheus and Grafana.
- Screenshot: [screenshots/docker_running.png] [add screenshot].

## 7. Production deployment (7 marks)

- Kubernetes manifest [k8s/deployment.yaml](k8s/deployment.yaml): Deployment with 2 replicas, readiness and liveness probes on `/health`, resource requests and limits, and a LoadBalancer Service on port 80.
- Target: [Minikube or cloud provider, e.g. AKS, GKE, EKS] [state your choice].
- Public URL or access instructions: [add URL or steps].
- Screenshots: [screenshots/k8s_pods.png], [screenshots/service_url.png] [add screenshots].

## 8. Monitoring and logging (3 marks)

- Structured request logs to stdout and `logs/api.log`: method, path, status, and latency.
- Prometheus metrics at `/metrics`: request counter, latency histogram, prediction counter by label.
- Prometheus configuration: [monitoring/prometheus.yml](monitoring/prometheus.yml). Grafana is available through docker-compose.
- Screenshot: [screenshots/prometheus_targets.png] [add screenshot].

## 9. Documentation and reflection (2 marks)

- README with setup, reproduction, serving, Docker, Kubernetes, CI, and monitoring instructions: [README.md](README.md).
- Limitations: small dataset (303 rows), single-site data, class imbalance is modest, no drift detection in production.
- Future work: [e.g. model drift monitoring, automated retraining, A/B testing, authentication on the API].

## References

- Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. UCI Machine Learning Repository, Heart Disease dataset.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
- MLflow documentation: https://mlflow.org/docs/latest/
- FastAPI documentation: https://fastapi.tiangolo.com/
