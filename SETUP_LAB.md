# Lab Setup Instructions

Follow these steps in your BITS Pilani lab terminal to deploy the MLOps project.

## Quick Start (Copy-Paste These Commands)

### Step 1: Clone the repository
```bash
cd ~ && git clone https://github.com/2025ae05712-akhi/MLOps_Assignment_1.git && cd MLOps_Assignment_1
```

### Step 2: Run the lab setup script
```bash
bash lab.sh
```

This will automatically:
- Install Python 3.11 via `uv`
- Install all dependencies
- Download the dataset
- Run EDA
- Train the model with MLflow
- Run all tests

### Step 3: Start the API service (after Step 2 completes)
```bash
.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Step 4: View MLflow UI (in a new terminal)
```bash
cd ~/MLOps_Assignment_1 && .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

MLflow will be available at `http://localhost:5000`

---

## Manual Steps (if lab.sh fails)

### 1. Clone & Enter Directory
```bash
cd ~ && git clone https://github.com/2025ae05712-akhi/MLOps_Assignment_1.git
cd MLOps_Assignment_1
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
```

### 4. Download Data
```bash
python data/download_data.py
```

### 5. Run EDA
```bash
python -m scripts.eda
```

### 6. Train Model
```bash
python -m src.train
```

### 7. Run Tests
```bash
pytest -v
```

### 8. Start API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## Troubleshooting

- **Python version error**: Ensure Python 3.10+ is available
- **Permission denied**: Add `chmod +x lab.sh` before running
- **Port already in use**: Change port: `uvicorn src.api:app --port 8001`

For more details, see [README.md](README.md)
