"""Tune, evaluate and track candidate models with MLflow, then save the best one."""

import json

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from src.config import (
    CV_FOLDS,
    DECISION_THRESHOLD,
    EXPERIMENT_NAME,
    FEATURES,
    METRICS_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    get_model_path,
)
from src.data import load_dataset, save_clean, split_features
from src.models import candidate_models

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


def evaluate_on_test(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute held-out test metrics at the decision threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_recall": recall_score(y_test, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def log_tuning_candidates(search: GridSearchCV) -> None:
    """Log every hyperparameter combination as a nested MLflow run."""
    for i, params in enumerate(search.cv_results_["params"]):
        with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(
                {
                    "cv_roc_auc_mean": float(search.cv_results_["mean_test_score"][i]),
                    "cv_roc_auc_std": float(search.cv_results_["std_test_score"][i]),
                }
            )


def log_plots(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Log confusion matrix and ROC curve figures to the active run."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title("Confusion matrix (test set)")
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC curve (test set)")
    mlflow.log_figure(fig, "plots/roc_curve.png")
    plt.close(fig)


def train_one(name, pipeline, grid, cv, X_train, y_train, X_test, y_test) -> dict:
    """Run grid search, cross-validation and test evaluation for one model family."""
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("model_family", name)
        mlflow.log_params(
            {
                "cv_folds": cv.n_splits,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "decision_threshold": DECISION_THRESHOLD,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "features": ",".join(FEATURES),
            }
        )

        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        log_tuning_candidates(search)
        mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})

        tuning_table = pd.DataFrame(search.cv_results_).drop(columns=["params"])
        mlflow.log_text(tuning_table.to_csv(index=False), "tuning/cv_results.csv")

        best = search.best_estimator_
        cv_scores = cross_validate(best, X_train, y_train, cv=cv, scoring=CV_SCORING)
        cv_metrics = {}
        for metric in CV_SCORING:
            values = cv_scores[f"test_{metric}"]
            cv_metrics[f"cv_{metric}_mean"] = float(values.mean())
            cv_metrics[f"cv_{metric}_std"] = float(values.std())
        mlflow.log_metrics(cv_metrics)

        test_metrics = evaluate_on_test(best, X_test, y_test)
        mlflow.log_metrics(test_metrics)
        log_plots(best, X_test, y_test)

        signature = infer_signature(X_test, best.predict_proba(X_test))
        mlflow.sklearn.log_model(
            best,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_test.head(3),
        )

        return {
            "name": name,
            "run_id": run.info.run_id,
            "best_params": search.best_params_,
            "cv": cv_metrics,
            "test": test_metrics,
            "model": best,
        }


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_dataset()
    save_clean(df)
    X, y = split_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = [
        train_one(name, pipeline, grid, cv, X_train, y_train, X_test, y_test)
        for name, (pipeline, grid) in candidate_models().items()
    ]

    # Select on cross-validated ROC-AUC so the test set is only used for reporting.
    best = max(results, key=lambda r: r["cv"]["cv_roc_auc_mean"])
    MlflowClient().set_tag(best["run_id"], "selected_for_serving", "true")

    model_path = get_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], model_path)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "selected_model": best["name"],
        "selected_params": best["best_params"],
        "features": FEATURES,
        "models": [{k: v for k, v in r.items() if k != "model"} for r in results],
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2, default=str))

    print(f"Selected model: {best['name']} (CV ROC-AUC {best['cv']['cv_roc_auc_mean']:.4f})")
    print(f"Saved model to {model_path}")
    print(f"Test ROC-AUC: {best['test']['test_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
