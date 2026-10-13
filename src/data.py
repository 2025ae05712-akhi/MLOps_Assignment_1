"""Load and clean the UCI Heart Disease dataset."""

from pathlib import Path

import pandas as pd

from src.config import CLEAN_CSV, FEATURES, RAW_CSV, TARGET


def load_dataset(path: Path = RAW_CSV) -> pd.DataFrame:
    """Return the dataset with '?' as NaN, numeric dtypes and a binary target.

    The original `num` column (0 = no disease, 1-4 = disease severity) is
    binarised into `target` (0 = no disease, 1 = disease).
    """
    df = pd.read_csv(path, na_values="?")
    df = df.apply(pd.to_numeric, errors="coerce")
    df[TARGET] = (df.pop("num") > 0).astype(int)
    return df


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into the model feature matrix and the target vector."""
    return df[FEATURES].astype(float), df[TARGET]


def save_clean(df: pd.DataFrame, path: Path = CLEAN_CSV) -> Path:
    """Write the cleaned dataset (missing values kept, imputed later in the pipeline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
