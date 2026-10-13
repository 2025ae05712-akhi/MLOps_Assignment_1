"""Download the UCI Heart Disease (Cleveland) dataset and save it as a headed CSV.

Usage: python data/download_data.py [--force]
"""

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
OUTPUT = Path(__file__).resolve().parent / "heart_raw.csv"
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    if OUTPUT.exists() and not args.force:
        print(f"{OUTPUT} already exists, skipping download")
        return

    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), header=None, names=RAW_COLUMNS)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} rows and {df.shape[1]} columns to {OUTPUT}")


if __name__ == "__main__":
    main()
