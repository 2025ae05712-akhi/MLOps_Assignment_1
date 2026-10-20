Subject: MLOps Assignment 1 - Lab Deployment Instructions

Dear Team,

Please find below the complete step-by-step instructions to deploy the MLOps Assignment 1 (Heart Disease Prediction Pipeline) on your BITS Pilani lab environment.

---

## MLOPS ASSIGNMENT 1 - LAB DEPLOYMENT GUIDE

### Project Overview
This is an end-to-end MLOps project for predicting heart disease risk using the UCI Heart Disease (Cleveland) dataset. It includes data processing, model training with MLflow tracking, a FastAPI inference service, unit tests, and monitoring.

**Repository**: https://github.com/2025ae05712-akhi/MLOps_Assignment_1

---

### QUICK START (3 Steps)

#### Step 1: Clone Repository
```bash
cd ~ && git clone https://github.com/2025ae05712-akhi/MLOps_Assignment_1.git && cd MLOps_Assignment_1
```

#### Step 2: Run Automated Deployment
```bash
bash lab.sh
```

This automatically:
- Creates Python 3.11 virtual environment
- Installs all dependencies
- Downloads the UCI Heart Disease dataset (303 rows)
- Runs exploratory data analysis (EDA)
- Trains model with hyperparameter tuning using MLflow
- Validates with unit tests

**Estimated Time**: 5-10 minutes

#### Step 3: Start API Service (after Step 2 completes)
```bash
.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000
```

API endpoint: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Prediction: `POST http://localhost:8000/predict`

---

### OPTIONAL: View MLflow Dashboard (in separate terminal)
```bash
cd ~/MLOps_Assignment_1 && .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Access at: `http://localhost:5000`

---

### DETAILED MANUAL STEPS (if lab.sh fails)

#### 1. Clone & Navigate
```bash
cd ~
git clone https://github.com/2025ae05712-akhi/MLOps_Assignment_1.git
cd MLOps_Assignment_1
```

#### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

#### 3. Upgrade Package Manager
```bash
pip install --upgrade pip setuptools wheel
```

#### 4. Install Dependencies
```bash
pip install -r requirements-dev.txt
```

#### 5. Download Dataset
```bash
python data/download_data.py
```

#### 6. Run Exploratory Data Analysis
```bash
python -m scripts.eda
```

Output: `reports/eda_summary.txt` and visualizations in `reports/figures/`

#### 7. Train Model with Hyperparameter Tuning
```bash
python -m src.train
```

This performs:
- 5-fold stratified cross-validation on 80% training set
- Hyperparameter tuning for Logistic Regression and Random Forest
- ROC-AUC optimization
- Model evaluation on 20% test set
- Logging to MLflow

Selected model: **Logistic Regression** with `C=0.1, class_weight=balanced`
- Test Accuracy: 88.5%
- Test ROC-AUC: 0.965

#### 8. Run Unit Tests
```bash
pytest -v
```

All 20 tests should pass.

#### 9. Start API Service
```bash
.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000
```

#### 10. Example API Request (in new terminal)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6
  }'
```

Response format:
```json
{
  "prediction": 1,
  "label": "Heart Disease Present",
  "probability_heart_disease": 0.92,
  "confidence": 0.92
}
```

---

### PROJECT STRUCTURE
```
MLOps_Assignment_1/
├── data/
│   ├── download_data.py          # Downloads UCI Heart Disease dataset
│   ├── heart_raw.csv             # Raw data (303 rows, 14 features)
│   └── heart_clean.csv           # Cleaned data
├── scripts/
│   └── eda.py                    # Exploratory Data Analysis
├── src/
│   ├── config.py                 # Paths, constants, feature names
│   ├── data.py                   # Dataset loading and cleaning
│   ├── preprocessing.py          # Data pipeline (imputation, scaling, one-hot)
│   ├── models.py                 # ML models and hyperparameter grids
│   ├── train.py                  # Training with MLflow logging
│   ├── inference.py              # Model loading and prediction
│   └── api.py                    # FastAPI service with Prometheus metrics
├── tests/
│   ├── test_data.py              # Data pipeline tests
│   ├── test_model.py             # Model tests
│   └── test_api.py               # API endpoint tests
├── models/
│   ├── model.joblib              # Trained model (serialized)
│   └── metrics.json              # Performance metrics
├── reports/
│   ├── eda_summary.txt           # EDA statistics
│   └── figures/                  # Visualizations
├── k8s/
│   └── deployment.yaml           # Kubernetes manifests
├── monitoring/
│   └── prometheus.yml            # Prometheus scrape config
├── docker-compose.yml            # Docker Compose for full stack
├── Dockerfile                    # Container image definition
├── lab.sh                        # Lab deployment script (USE THIS)
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
└── README.md                     # Full documentation
```

---

### TROUBLESHOOTING

| Issue | Solution |
|-------|----------|
| `bash: lab.sh: command not found` | Add execute permission: `chmod +x lab.sh` |
| `Python 3.11 not found` | The script uses `uv` to install Python 3.11. If `uv` fails, use manual steps with Python 3.10+ |
| `Port 8000 already in use` | Use different port: `uvicorn src.api:app --port 8001` |
| `pip install fails` | Try: `pip install --upgrade pip` then retry |
| `mlflow command not found` | Ensure venv is activated: `source .venv/bin/activate` |

---

### ADDITIONAL FEATURES

**Environment Variables** (optional):
```bash
export MODEL_PATH=/path/to/custom/model.joblib
export LOG_DIR=/path/to/logs
```

**Docker Deployment** (requires Docker):
```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest
```

**Full Stack with Monitoring**:
```bash
docker compose up --build
```
- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

---

### SUPPORT & DOCUMENTATION

- Full README: See repository README.md
- Lab Setup Guide: See SETUP_LAB.md
- Test Results: `reports/junit.xml`
- Model Metrics: `models/metrics.json`

---

**Need Help?**
1. Check SETUP_LAB.md in the repository
2. Review README.md for detailed documentation
3. Check test results: `pytest -v`
4. Check logs: `cat logs/api.log`

**Enjoy your MLOps lab experience!**

---

Best Regards,
MLOps Team
