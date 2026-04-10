"""Executor de treino para um experimento por vez com rastreio em MLflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, NamedTuple

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

from common.config_loader import (
    DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
    load_global_config,
    load_training_experiment_config,
)
from common.logger import get_logger
from common.seed import set_global_seed
from models.catalog import build_model

logger = get_logger("models.train")


class ExperimentTrainingConfig(NamedTuple):
    """Contrato carregado para um experimento individual de treino."""

    seed: int
    target_col: str
    test_size: float
    algorithm: str
    flavor: str
    experiment_name: str
    run_name: str
    model_version: str
    model_params: dict[str, Any]
    threshold: float
    feature_set: str
    model_path: Path
    training_data_version: str
    git_sha: str
    git_tag: str
    git_nearest_tag: str
    risk_level: str
    fairness_checked: bool
    mlflow_cfg: dict[str, Any]
    registry_cfg: dict[str, Any]


class DatasetSplits(NamedTuple):
    """Conjuntos de treino e teste já separados em atributos e alvo."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


class ModelSpec(NamedTuple):
    """Especificação resolvida para treinar um experimento único."""

    name: str
    run_name: str
    algorithm: str
    params: dict[str, Any]
    output_path: Path


def build_metadata_output_path(model_output_path: Path) -> Path:
    """Resolve o caminho do sidecar JSON com metadados do modelo salvo."""

    return model_output_path.with_name(f"{model_output_path.stem}_metadata.json")


def resolve_git_sha() -> str:
    """Obtém o SHA atual do Git para rastreabilidade do experimento."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

    return result.stdout.strip()


def resolve_git_tag() -> str:
    """Obtém a tag Git associada exatamente ao commit atual, se existir."""

    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "post_release_commits"

    return result.stdout.strip()


def resolve_git_nearest_tag() -> str:
    """Obtém a tag Git mais próxima no histórico do commit atual, se existir."""

    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

    return result.stdout.strip()


def compute_training_data_version(
    train_path: Path = Path("data/processed/train.parquet"),
    test_path: Path = Path("data/processed/test.parquet"),
) -> str:
    """Calcula um hash estável dos dados processados usados no treino."""

    hasher = hashlib.md5()
    for path in (train_path, test_path):
        with open(path, "rb") as file_obj:
            while chunk := file_obj.read(8192):
                hasher.update(chunk)

    return hasher.hexdigest()


def load_experiment_training_config(
    experiment_config_path: str = DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
) -> ExperimentTrainingConfig:
    """Carrega um experimento individual combinando config global e local."""

    global_cfg = load_global_config()
    experiment_cfg = load_training_experiment_config(experiment_config_path)
    governance_cfg = experiment_cfg.get("governance", {})

    mlflow_tags = {
        "owner": global_cfg["mlflow"]["owner"],
        "phase": global_cfg["mlflow"]["phase"],
        "dataset_name": global_cfg["mlflow"]["dataset_name"],
    }
    mlflow_tags.update(experiment_cfg["mlflow"].get("tags", {}))

    return ExperimentTrainingConfig(
        seed=global_cfg["seed"],
        target_col=experiment_cfg["dataset"].get(
            "target_col",
            global_cfg["data"]["target_col"],
        ),
        test_size=global_cfg["split"]["test_size"],
        algorithm=experiment_cfg["experiment"]["algorithm"],
        flavor=experiment_cfg["experiment"]["flavor"],
        experiment_name=experiment_cfg["experiment"]["name"],
        run_name=experiment_cfg["experiment"]["run_name"],
        model_version=experiment_cfg["experiment"].get(
            "version",
            experiment_cfg["experiment"]["name"],
        ),
        model_params=experiment_cfg["training"]["params"],
        threshold=experiment_cfg["inference"]["threshold"],
        feature_set=experiment_cfg["dataset"]["feature_set"],
        model_path=Path(experiment_cfg["artifacts"]["model_path"]),
        training_data_version=compute_training_data_version(),
        git_sha=resolve_git_sha(),
        git_tag=resolve_git_tag(),
        git_nearest_tag=resolve_git_nearest_tag(),
        risk_level=governance_cfg.get("risk_level", "medium"),
        fairness_checked=governance_cfg.get("fairness_checked", False),
        mlflow_cfg={
            "tracking_uri": global_cfg["mlflow"]["tracking_uri"],
            "experiment_name": experiment_cfg["mlflow"].get(
                "experiment_name",
                global_cfg["mlflow"]["experiment_name"],
            ),
            "tags": mlflow_tags,
        },
        registry_cfg=experiment_cfg["registry"],
    )


def build_model_spec(cfg: ExperimentTrainingConfig) -> ModelSpec:
    """Converte a configuração carregada em uma especificação executável."""

    return ModelSpec(
        name=cfg.experiment_name,
        run_name=cfg.run_name,
        algorithm=cfg.algorithm,
        params=cfg.model_params,
        output_path=cfg.model_path,
    )


def resolve_runtime_model_params(
    model_params: dict[str, Any],
    y_train: pd.Series,
) -> dict[str, Any]:
    """Resolve parâmetros dependentes dos dados de treino."""

    resolved_params = dict(model_params)
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    for key, value in resolved_params.items():
        if value == "__neg_pos_ratio__":
            resolved_params[key] = negative_count / max(positive_count, 1)

    return resolved_params


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


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Calcula métricas de classificação no conjunto de teste."""

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "auc": roc_auc_score(y_test, y_pred_proba),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }


def log_run_metadata(
    params: dict[str, Any],
    cfg: ExperimentTrainingConfig,
    datasets: DatasetSplits,
) -> None:
    """Registra parâmetros e metadados padrão no MLflow."""

    mlflow.log_params(params)
    mlflow.log_param("test_size", cfg.test_size)
    mlflow.log_param("n_features", datasets.X_train.shape[1])
    mlflow.log_param("n_samples", datasets.X_train.shape[0])
    mlflow.log_param("threshold", cfg.threshold)
    mlflow.log_param("feature_set", cfg.feature_set)
    mlflow.log_param("fairness_checked", cfg.fairness_checked)

    mlflow.set_tag("model_type", "classification")
    mlflow.set_tag("framework", cfg.flavor)
    mlflow.set_tag("algorithm", cfg.algorithm)
    mlflow.set_tag("model_name", cfg.experiment_name)
    mlflow.set_tag("model_version", cfg.model_version)
    mlflow.set_tag("training_data_version", cfg.training_data_version)
    mlflow.set_tag("git_sha", cfg.git_sha)
    mlflow.set_tag("git_tag", cfg.git_tag)
    mlflow.set_tag("git_nearest_tag", cfg.git_nearest_tag)
    mlflow.set_tag("risk_level", cfg.risk_level)
    mlflow.set_tag("fairness_checked", str(cfg.fairness_checked).lower())
    for key, value in cfg.mlflow_cfg["tags"].items():
        mlflow.set_tag(key, value)


def train_and_log_model(
    spec: ModelSpec,
    cfg: ExperimentTrainingConfig,
    datasets: DatasetSplits,
) -> dict[str, float]:
    """Treina um modelo, registra no MLflow e salva o artefato serializado."""

    with mlflow.start_run(run_name=spec.run_name) as run:
        log_run_metadata(spec.params, cfg, datasets)

        logger.info("Iniciando treino %s — %s", spec.algorithm, spec.params)

        model = build_model(spec.algorithm, spec.params)
        model.fit(datasets.X_train, datasets.y_train)

        metrics = evaluate_model(
            model,
            datasets.X_test,
            datasets.y_test,
            threshold=cfg.threshold,
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        dump(model, spec.output_path)
        metadata_output_path = build_metadata_output_path(spec.output_path)
        metadata_output_path.write_text(
            json.dumps(
                {
                    "experiment_name": cfg.experiment_name,
                    "run_name": cfg.run_name,
                    "model_version": cfg.model_version,
                    "algorithm": cfg.algorithm,
                    "flavor": cfg.flavor,
                    "model_path": str(spec.output_path),
                    "threshold": cfg.threshold,
                    "feature_set": cfg.feature_set,
                    "training_data_version": cfg.training_data_version,
                    "git_sha": cfg.git_sha,
                    "git_tag": cfg.git_tag,
                    "git_nearest_tag": cfg.git_nearest_tag,
                    "risk_level": cfg.risk_level,
                    "fairness_checked": cfg.fairness_checked,
                    "mlflow_experiment_name": cfg.mlflow_cfg["experiment_name"],
                    "metrics": metrics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info("Run registrada — ID: %s", run.info.run_id)
        logger.info("Métricas %s: %s", spec.name, metrics)
        logger.info("Modelo salvo em %s", spec.output_path)
        logger.info("Metadados do modelo salvos em %s", metadata_output_path)

    return metrics


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando para o executor de treino."""

    parser = argparse.ArgumentParser(
        description="Executa o treino de um único experimento de churn.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
        help="Caminho relativo para o YAML do experimento de treino.",
    )
    return parser.parse_args()


def run_training(
    experiment_config_path: str = DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
) -> dict[str, float]:
    """Executa o treino atômico de um experimento individual."""

    cfg = load_experiment_training_config(experiment_config_path)
    set_global_seed(cfg.seed)

    datasets = load_training_data(cfg.target_col)
    configure_mlflow(cfg.mlflow_cfg)

    spec = build_model_spec(cfg)._replace(
        params=resolve_runtime_model_params(cfg.model_params, datasets.y_train)
    )
    return train_and_log_model(
        spec=spec,
        cfg=cfg,
        datasets=datasets,
    )


def main() -> None:
    """Ponto de entrada para execução local do treino."""

    args = parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
