"""Unit tests for data loading and cleaning."""

import pandas as pd
import pytest

from src.config import CLEAN_CSV, FEATURES, RAW_CSV, TARGET
from src.data import load_dataset, save_clean, split_features

RAW_HEADER = "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num\n"


def write_raw(tmp_path, body: str):
    path = tmp_path / "raw.csv"
    path.write_text(RAW_HEADER + body)
    return path


def test_load_dataset_replaces_question_marks_and_binarises_target(tmp_path):
    path = write_raw(
        tmp_path,
        "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0\n67,1,4,160,286,0,2,108,1,1.5,2,?,7,2\n",
    )

    df = load_dataset(path)

    assert "num" not in df.columns
    assert df[TARGET].tolist() == [0, 1]
    assert pd.isna(df.loc[1, "ca"])
    assert pd.api.types.is_numeric_dtype(df["thal"])


def test_split_features_returns_expected_columns(tmp_path):
    path = write_raw(tmp_path, "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0\n")

    X, y = split_features(load_dataset(path))

    assert list(X.columns) == FEATURES
    assert X.dtypes.eq(float).all()
    assert len(X) == len(y) == 1


def test_save_clean_writes_csv_with_same_columns(tmp_path):
    df = load_dataset(write_raw(tmp_path, "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0\n"))

    out = save_clean(df, tmp_path / "clean.csv")

    assert pd.read_csv(out).columns.tolist() == df.columns.tolist()
    assert out.exists()


@pytest.mark.skipif(not RAW_CSV.exists(), reason="run data/download_data.py first")
def test_full_dataset_has_expected_shape_and_classes():
    df = load_dataset()

    assert df.shape == (303, 14)
    assert set(df[TARGET].unique()) == {0, 1}


@pytest.mark.skipif(not CLEAN_CSV.exists(), reason="run scripts/eda.py first")
def test_clean_csv_has_target_column():
    assert TARGET in pd.read_csv(CLEAN_CSV).columns
