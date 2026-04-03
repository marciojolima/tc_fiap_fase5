from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common.config_loader import load_config
from common.logger import get_logger
from common.seed import set_global_seed
from models.baseline import build_baseline_model, build_challenger_model

logger = get_logger(__name__)


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc": roc_auc_score(y_test, y_pred_proba),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }
    return metrics


def run_training() -> None:
    config = load_config()

    set_global_seed(config["seed"])

    target_col = config["data"]["target_col"]

    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Baseline
    baseline_cfg = config["models"]["baseline"]
    baseline_name = baseline_cfg["name"]
    baseline_params = {k: v for k, v in baseline_cfg.items() if k != "name"}

    with mlflow.start_run(run_name=baseline_name) as run:
        mlflow.log_params(baseline_params)
        mlflow.log_param("test_size", config["split"]["test_size"])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples", X_train.shape[0])

        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("owner", config["mlflow"]["owner"])
        mlflow.set_tag("phase", config["mlflow"]["phase"])
        mlflow.set_tag("dataset", config["mlflow"]["dataset_name"])
        mlflow.set_tag("role", "baseline")

        logger.info("Iniciando treino baseline — %s", baseline_params)

        model = build_baseline_model(baseline_params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        dump(model, artifacts_dir / "baseline_model.pkl")

        logger.info("Run baseline registrada — ID: %s", run.info.run_id)
        logger.info("Métricas baseline: %s", metrics)

    # Challenger
    challenger_cfg = config["models"]["challenger"]
    challenger_name = challenger_cfg["name"]
    challenger_params = {k: v for k, v in challenger_cfg.items() if k != "name"}

    with mlflow.start_run(run_name=challenger_name) as run:
        mlflow.log_params(challenger_params)
        mlflow.log_param("test_size", config["split"]["test_size"])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples", X_train.shape[0])

        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("owner", config["mlflow"]["owner"])
        mlflow.set_tag("phase", config["mlflow"]["phase"])
        mlflow.set_tag("dataset", config["mlflow"]["dataset_name"])
        mlflow.set_tag("role", "challenger")

        logger.info("Iniciando treino challenger — %s", challenger_params)

        model_v2 = build_challenger_model(challenger_params)
        model_v2.fit(X_train, y_train)

        metrics_v2 = evaluate_model(model_v2, X_test, y_test)
        mlflow.log_metrics(metrics_v2)
        mlflow.sklearn.log_model(model_v2, "model")

        dump(model_v2, artifacts_dir / "challenger_model.pkl")

        logger.info("Run challenger registrada — ID: %s", run.info.run_id)
        logger.info("Métricas challenger: %s", metrics_v2)

    # Comparação
    comparison = {
        "baseline": metrics,
        "challenger": metrics_v2,
        "delta": {m: metrics_v2[m] - metrics[m] for m in metrics.keys()},
    }

    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    logger.info("Comparação final salva em artifacts/metrics.json")


if __name__ == "__main__":
    run_training()
