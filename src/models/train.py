"""Treinamento dos modelos baseline e challenger com rastreio em MLflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, NamedTuple

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


class TrainingConfig(NamedTuple):
    """Configuração necessária para a etapa de treinamento."""

    seed: int
    target_col: str
    test_size: float
    baseline_cfg: dict[str, Any]
    challenger_cfg: dict[str, Any]
    mlflow_cfg: dict[str, Any]


class DatasetSplits(NamedTuple):
    """Conjuntos de treino e teste já separados em atributos e alvo."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


class ModelSpec(NamedTuple):
    """Especificação de um experimento de modelo."""

    name: str
    params: dict[str, Any]
    role: str
    builder: Callable[[dict[str, Any]], Any]
    output_path: Path


def load_training_config() -> TrainingConfig:
    """Carrega a configuração necessária para o treinamento."""

    config = load_config()
    return TrainingConfig(
        seed=config["seed"],
        target_col=config["data"]["target_col"],
        test_size=config["split"]["test_size"],
        baseline_cfg=config["models"]["baseline"],
        challenger_cfg=config["models"]["challenger"],
        mlflow_cfg=config["mlflow"],
    )


def load_training_data(target_col: str) -> DatasetSplits:
    """Carrega os dados processados e separa atributos e variável alvo."""

    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    logger.info(
        "Dados processados carregados — Train: %s | Test: %s",
        train_df.shape,
        test_df.shape,
    )
    logger.info(
        "Treino configurado com %d features e %d amostras",
        X_train.shape[1],
        X_train.shape[0],
    )

    return DatasetSplits(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def configure_mlflow(mlflow_cfg: dict[str, Any]) -> None:
    """Configura o tracking e o experimento do MLflow."""

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    logger.info(
        "MLflow configurado — tracking_uri=%s | experiment=%s",
        mlflow_cfg["tracking_uri"],
        mlflow_cfg["experiment_name"],
    )


def build_model_spec(
    model_cfg: dict[str, Any],
    role: str,
    builder: Callable[[dict[str, Any]], Any],
    output_path: Path,
) -> ModelSpec:
    """Monta a especificação de treino de um modelo."""

    return ModelSpec(
        name=model_cfg["name"],
        params={k: v for k, v in model_cfg.items() if k != "name"},
        role=role,
        builder=builder,
        output_path=output_path,
    )


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calcula métricas de classificação no conjunto de teste."""

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        "auc": roc_auc_score(y_test, y_pred_proba),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }


def log_run_metadata(
    params: dict[str, Any],
    role: str,
    cfg: TrainingConfig,
    datasets: DatasetSplits,
) -> None:
    """Registra parâmetros e metadados padrão no MLflow."""

    mlflow.log_params(params)
    mlflow.log_param("test_size", cfg.test_size)
    mlflow.log_param("n_features", datasets.X_train.shape[1])
    mlflow.log_param("n_samples", datasets.X_train.shape[0])

    mlflow.set_tag("model_type", "classification")
    mlflow.set_tag("framework", "sklearn")
    mlflow.set_tag("owner", cfg.mlflow_cfg["owner"])
    mlflow.set_tag("phase", cfg.mlflow_cfg["phase"])
    mlflow.set_tag("dataset", cfg.mlflow_cfg["dataset_name"])
    mlflow.set_tag("role", role)


def train_and_log_model(
    spec: ModelSpec,
    cfg: TrainingConfig,
    datasets: DatasetSplits,
) -> dict[str, float]:
    """Treina um modelo, registra no MLflow e salva o artefato serializado."""

    with mlflow.start_run(run_name=spec.name) as run:
        log_run_metadata(spec.params, spec.role, cfg, datasets)

        logger.info("Iniciando treino %s — %s", spec.role, spec.params)

        model = spec.builder(spec.params)
        model.fit(datasets.X_train, datasets.y_train)

        metrics = evaluate_model(model, datasets.X_test, datasets.y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        dump(model, spec.output_path)

        logger.info("Run %s registrada — ID: %s", spec.role, run.info.run_id)
        logger.info("Métricas %s: %s", spec.role, metrics)

    return metrics


def build_comparison(
    baseline_metrics: dict[str, float],
    challenger_metrics: dict[str, float],
) -> dict[str, Any]:
    """Monta a comparação consolidada entre baseline e challenger."""

    return {
        "baseline": baseline_metrics,
        "challenger": challenger_metrics,
        "delta": {
            metric: challenger_metrics[metric] - baseline_metrics[metric]
            for metric in baseline_metrics
        },
    }


def save_metrics_comparison(comparison: dict[str, Any], output_path: Path) -> None:
    """Persiste em disco a comparação consolidada de métricas."""

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    logger.info("Comparação final salva em %s", output_path)


def run_training() -> None:
    """Orquestra o treinamento de baseline e challenger."""

    cfg = load_training_config()
    set_global_seed(cfg.seed)

    datasets = load_training_data(cfg.target_col)
    configure_mlflow(cfg.mlflow_cfg)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    baseline_spec = build_model_spec(
        model_cfg=cfg.baseline_cfg,
        role="baseline",
        builder=build_baseline_model,
        output_path=artifacts_dir / "baseline_model.pkl",
    )
    baseline_metrics = train_and_log_model(
        spec=baseline_spec,
        cfg=cfg,
        datasets=datasets,
    )

    challenger_spec = build_model_spec(
        model_cfg=cfg.challenger_cfg,
        role="challenger",
        builder=build_challenger_model,
        output_path=artifacts_dir / "challenger_model.pkl",
    )
    challenger_metrics = train_and_log_model(
        spec=challenger_spec,
        cfg=cfg,
        datasets=datasets,
    )

    comparison = build_comparison(baseline_metrics, challenger_metrics)
    save_metrics_comparison(comparison, artifacts_dir / "metrics.json")


if __name__ == "__main__":
    run_training()
