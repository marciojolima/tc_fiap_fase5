"""Executor de treino para um experimento por vez com rastreio em MLflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, NamedTuple
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from joblib import dump
from mlflow.tracking import MlflowClient
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
from model_lifecycle.business_metrics import (
    BusinessMetricsEvaluator,
    PrecisionAtTopK,
    RecallAtTopK,
)
from model_lifecycle.catalog import build_model

logger = get_logger("model_lifecycle.train")

DEFAULT_RECALL_TOP_K = 0.20
DEFAULT_RECALL_TARGET = 0.70
DEFAULT_PRECISION_TOP_K = 0.20
DEFAULT_PRECISION_TARGET = 0.35


class BusinessMetricsConfig(NamedTuple):
    """Configuração centralizada das métricas de negócio do treino."""

    recall_top_k: float
    recall_target: float
    precision_top_k: float
    precision_target: float


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
    feature_service_name: str
    model_path: Path
    training_data_version: str
    git_sha: str
    git_tag: str
    git_nearest_tag: str
    risk_level: str
    fairness_checked: bool
    business_metrics: BusinessMetricsConfig
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


class RetrainingMlflowContext(NamedTuple):
    """Contexto opcional para etiquetar runs de treino disparadas por retreino."""

    request_id: str
    reason: str
    trigger_mode: str
    promotion_policy: str
    drift_status: str
    max_feature_psi: float
    prediction_psi: float | None
    drifted_features: list[str]
    reference_row_count: int
    current_row_count: int


def build_experiment_training_config(
    experiment_cfg: Mapping[str, Any],
    *,
    global_cfg: Mapping[str, Any] | None = None,
) -> ExperimentTrainingConfig:
    """Converte o contrato bruto do experimento em configuração tipada."""

    active_global_cfg = global_cfg or load_global_config()
    governance_cfg = experiment_cfg.get("governance", {})

    mlflow_tags = {
        "owner": active_global_cfg["mlflow"]["owner"],
        "phase": active_global_cfg["mlflow"]["phase"],
        "dataset_name": active_global_cfg["mlflow"]["dataset_name"],
    }
    business_metrics_cfg = active_global_cfg.get("business_metrics", {})
    mlflow_tags.update(experiment_cfg["mlflow"].get("tags", {}))

    return ExperimentTrainingConfig(
        seed=active_global_cfg["seed"],
        target_col=experiment_cfg["dataset"].get(
            "target_col",
            active_global_cfg["data"]["target_col"],
        ),
        test_size=active_global_cfg["split"]["test_size"],
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
        feature_service_name=experiment_cfg["feast"]["feature_service_name"],
        model_path=Path(experiment_cfg["artifacts"]["model_path"]),
        training_data_version=compute_training_data_version(),
        git_sha=resolve_git_sha(),
        git_tag=resolve_git_tag(),
        git_nearest_tag=resolve_git_nearest_tag(),
        risk_level=governance_cfg.get("risk_level", "medium"),
        fairness_checked=governance_cfg.get("fairness_checked", False),
        business_metrics=BusinessMetricsConfig(
            recall_top_k=float(
                business_metrics_cfg.get("recall_top_k", DEFAULT_RECALL_TOP_K)
            ),
            recall_target=float(
                business_metrics_cfg.get("recall_target", DEFAULT_RECALL_TARGET)
            ),
            precision_top_k=float(
                business_metrics_cfg.get(
                    "precision_top_k",
                    DEFAULT_PRECISION_TOP_K,
                )
            ),
            precision_target=float(
                business_metrics_cfg.get(
                    "precision_target",
                    DEFAULT_PRECISION_TARGET,
                )
            ),
        ),
        mlflow_cfg={
            "tracking_uri": active_global_cfg["mlflow"]["tracking_uri"],
            "experiment_name": experiment_cfg["mlflow"].get(
                "experiment_name",
                active_global_cfg["mlflow"]["experiment_name"],
            ),
            "tags": mlflow_tags,
        },
        registry_cfg=experiment_cfg["registry"],
    )


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

    experiment_cfg = load_training_experiment_config(experiment_config_path)
    return build_experiment_training_config(
        experiment_cfg,
        global_cfg=load_global_config(),
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
    ensure_mlflow_experiment(mlflow_cfg)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    logger.info(
        "MLflow configurado — tracking_uri=%s | experiment=%s",
        mlflow_cfg["tracking_uri"],
        mlflow_cfg["experiment_name"],
    )


def resolve_sqlite_tracking_db_path(tracking_uri: str) -> Path | None:
    """Extrai o caminho local do backend SQLite usado pelo MLflow."""

    parsed_uri = urlparse(tracking_uri)
    if parsed_uri.scheme != "sqlite":
        return None

    db_path = parsed_uri.path
    if not db_path:
        return None

    return Path(db_path).resolve()


def build_mlflow_experiment_artifact_root(tracking_uri: str) -> Path | None:
    """Resolve o diretório-base de artefatos para backends SQLite locais."""

    db_path = resolve_sqlite_tracking_db_path(tracking_uri)
    if db_path is None:
        return None

    return db_path.parent


def build_mlflow_experiment_artifact_location(
    tracking_uri: str,
    experiment_ref: str,
) -> str | None:
    """Monta o artifact_location canônico para um experimento local."""

    artifact_root = build_mlflow_experiment_artifact_root(tracking_uri)
    if artifact_root is None:
        return None

    return str((artifact_root / experiment_ref).resolve())


def update_mlflow_experiment_artifact_location(
    tracking_uri: str,
    experiment_id: str,
    artifact_location: str,
) -> None:
    """Atualiza o artifact_location do experimento no backend SQLite local."""

    db_path = resolve_sqlite_tracking_db_path(tracking_uri)
    if db_path is None:
        raise ValueError(
            "Atualização de artifact_location só é suportada para tracking URI SQLite."
        )

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
            (artifact_location, int(experiment_id)),
        )
        connection.commit()


def ensure_mlflow_experiment(mlflow_cfg: dict[str, Any]) -> None:
    """Garante experimento MLflow compatível com o ambiente atual."""

    tracking_uri = mlflow_cfg["tracking_uri"]
    experiment_name = mlflow_cfg["experiment_name"]
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        artifact_location = build_mlflow_experiment_artifact_location(
            tracking_uri,
            experiment_name,
        )
        client.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
        logger.info(
            "Experimento MLflow criado | experiment=%s | artifact_location=%s",
            experiment_name,
            artifact_location,
        )
        return

    desired_artifact_location = build_mlflow_experiment_artifact_location(
        tracking_uri,
        str(experiment.experiment_id),
    )
    if desired_artifact_location is None:
        return
    if experiment.artifact_location == desired_artifact_location:
        return

    update_mlflow_experiment_artifact_location(
        tracking_uri,
        str(experiment.experiment_id),
        desired_artifact_location,
    )
    logger.info(
        "Experimento MLflow ajustado | experiment=%s | de=%s | para=%s",
        experiment_name,
        experiment.artifact_location,
        desired_artifact_location,
    )


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    business_metrics_evaluator: BusinessMetricsEvaluator | None = None,
) -> dict[str, float]:
    """Calcula métricas de classificação no conjunto de teste."""

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, y_pred_proba),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }
    if business_metrics_evaluator is None:
        return metrics

    return metrics | business_metrics_evaluator.evaluate(
        np.asarray(y_test),
        y_pred_proba,
    )


def build_business_metrics_evaluator(
    cfg: BusinessMetricsConfig,
) -> BusinessMetricsEvaluator:
    """Cria o agregador das métricas de negócio configuradas."""

    return BusinessMetricsEvaluator(
        metrics=(
            RecallAtTopK(
                top_k=cfg.recall_top_k,
                target=cfg.recall_target,
            ),
            PrecisionAtTopK(
                top_k=cfg.precision_top_k,
                target=cfg.precision_target,
            ),
        )
    )


def log_run_metadata(
    params: dict[str, Any],
    cfg: ExperimentTrainingConfig,
    datasets: DatasetSplits,
    retraining_context: RetrainingMlflowContext | None = None,
) -> None:
    """Registra parâmetros e metadados padrão no MLflow."""

    mlflow.log_params(params)
    mlflow.log_param("test_size", cfg.test_size)
    mlflow.log_param("n_features", datasets.X_train.shape[1])
    mlflow.log_param("n_samples", datasets.X_train.shape[0])
    mlflow.log_param("threshold", cfg.threshold)
    mlflow.log_param("feature_set", cfg.feature_set)
    mlflow.log_param("feature_service_name", cfg.feature_service_name)
    mlflow.log_param("fairness_checked", cfg.fairness_checked)
    mlflow.log_param("recall_top_k", cfg.business_metrics.recall_top_k)
    mlflow.log_param("recall_target", cfg.business_metrics.recall_target)
    mlflow.log_param("precision_top_k", cfg.business_metrics.precision_top_k)
    mlflow.log_param("precision_target", cfg.business_metrics.precision_target)

    mlflow.set_tag("model_type", "classification")
    mlflow.set_tag("framework", cfg.flavor)
    mlflow.set_tag("algorithm", cfg.algorithm)
    mlflow.set_tag("model_name", cfg.experiment_name)
    mlflow.set_tag("model_version", cfg.model_version)
    mlflow.set_tag("feature_service_name", cfg.feature_service_name)
    mlflow.set_tag("training_data_version", cfg.training_data_version)
    mlflow.set_tag("git_sha", cfg.git_sha)
    mlflow.set_tag("git_tag", cfg.git_tag)
    mlflow.set_tag("git_nearest_tag", cfg.git_nearest_tag)
    mlflow.set_tag("risk_level", cfg.risk_level)
    mlflow.set_tag("fairness_checked", str(cfg.fairness_checked).lower())
    for key, value in cfg.mlflow_cfg["tags"].items():
        mlflow.set_tag(key, value)
    if retraining_context is not None:
        mlflow.set_tag("retrain", "true")
        mlflow.set_tag("training_trigger", "drift_monitoring")
        mlflow.set_tag("retraining_request_id", retraining_context.request_id)
        mlflow.set_tag("drift_status", retraining_context.drift_status)
        mlflow.log_param("drift_max_feature_psi", retraining_context.max_feature_psi)
        if retraining_context.prediction_psi is not None:
            mlflow.log_param(
                "drift_prediction_psi",
                retraining_context.prediction_psi,
            )
        mlflow.log_param(
            "drifted_feature_count",
            len(retraining_context.drifted_features),
        )
        mlflow.log_param(
            "drift_reference_row_count",
            retraining_context.reference_row_count,
        )
        mlflow.log_param(
            "drift_current_row_count",
            retraining_context.current_row_count,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            context_path = Path(tmp_dir) / "retraining_context.json"
            context_path.write_text(
                json.dumps(retraining_context._asdict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            mlflow.log_artifact(
                str(context_path),
                artifact_path="retraining",
            )
    else:
        mlflow.set_tag("retrain", "false")


def train_and_log_model(
    spec: ModelSpec,
    cfg: ExperimentTrainingConfig,
    datasets: DatasetSplits,
    retraining_context: RetrainingMlflowContext | None = None,
) -> dict[str, float]:
    """Treina um modelo, registra no MLflow e salva o artefato serializado."""

    run_name = (
        f"{spec.run_name}_retrain"
        if retraining_context is not None
        else spec.run_name
    )
    with mlflow.start_run(run_name=run_name) as run:
        log_run_metadata(
            spec.params,
            cfg,
            datasets,
            retraining_context=retraining_context,
        )

        logger.info("Iniciando treino %s — %s", spec.algorithm, spec.params)

        model = build_model(spec.algorithm, spec.params)
        model.fit(datasets.X_train, datasets.y_train)

        metrics = evaluate_model(
            model,
            datasets.X_test,
            datasets.y_test,
            threshold=cfg.threshold,
            business_metrics_evaluator=build_business_metrics_evaluator(
                cfg.business_metrics
            ),
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
                    "feature_service_name": cfg.feature_service_name,
                    "training_data_version": cfg.training_data_version,
                    "git_sha": cfg.git_sha,
                    "git_tag": cfg.git_tag,
                    "git_nearest_tag": cfg.git_nearest_tag,
                    "risk_level": cfg.risk_level,
                    "fairness_checked": cfg.fairness_checked,
                    "business_metrics": {
                        "recall_top_k": cfg.business_metrics.recall_top_k,
                        "recall_target": cfg.business_metrics.recall_target,
                        "precision_top_k": cfg.business_metrics.precision_top_k,
                        "precision_target": cfg.business_metrics.precision_target,
                    },
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
    retraining_context: RetrainingMlflowContext | None = None,
    experiment_config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Executa o treino atômico de um experimento individual."""

    cfg = (
        build_experiment_training_config(experiment_config)
        if experiment_config is not None
        else load_experiment_training_config(experiment_config_path)
    )
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
        retraining_context=retraining_context,
    )


def main() -> None:
    """Ponto de entrada para execução local do treino."""

    args = parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
