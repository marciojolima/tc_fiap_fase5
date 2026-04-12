"""Detecção batch de drift com Evidently e gatilho auditável de retreino."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from joblib import load

from common.config_loader import DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH, load_config
from common.logger import get_logger
from models.retraining import run_retraining_request

DEFAULT_MONITORING_CONFIG_PATH = "configs/monitoring_config.yaml"
TARGET_COLUMN = "Exited"
PREDICTION_PROBABILITY_COLUMN = "churn_probability"
PREDICTION_CLASS_COLUMN = "churn_prediction"
MONITORING_METADATA_COLUMNS = {
    "timestamp",
    "model_name",
    "threshold",
    PREDICTION_PROBABILITY_COLUMN,
    PREDICTION_CLASS_COLUMN,
}
MIN_BIN_EDGE_COUNT = 2
MIN_CURRENT_SAMPLE_SIZE_FOR_REPORT = 30
DEFAULT_MIN_CURRENT_SAMPLE_SIZE_FOR_DECISION = 30

logger = get_logger("monitoring.drift")


@dataclass(frozen=True)
class DriftDecision:
    """Resultado consolidado para governança de monitoramento."""

    status: str
    max_feature_psi: float
    mean_feature_psi: float
    drifted_feature_share: float
    prediction_psi: float | None
    drifted_features: list[str]

    @property
    def retraining_recommended(self) -> bool:
        return self.status == "critical"


@dataclass(frozen=True)
class MonitoringPreparedInputs:
    """Conjuntos e metadados já resolvidos para uma execução de monitoramento."""

    current_path: str
    reference_dataset: pd.DataFrame
    current_dataset: pd.DataFrame
    reference_features: pd.DataFrame
    current_features: pd.DataFrame
    reference_row_count: int
    current_row_count: int


@dataclass(frozen=True)
class MonitoringExecutionContext:
    """Contexto resumido da execução para persistência dos artefatos."""

    created_at: str
    reference_row_count: int
    current_row_count: int
    minimum_current_sample_size_for_decision: int


@dataclass(frozen=True)
class RetrainingRequestContext:
    """Metadados extras usados para abrir uma solicitação de retreino."""

    model_path: str
    trigger_mode: str
    training_config_path: str
    reference_row_count: int
    current_row_count: int


def load_monitoring_config(
    config_path: str = DEFAULT_MONITORING_CONFIG_PATH,
) -> dict[str, Any]:
    """Carrega as configurações de drift."""

    return load_config(config_path)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carrega parquet, csv ou jsonl para uma rotina de monitoramento."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset de monitoramento não encontrado: {dataset_path}. "
            "Gere inferências via /predict ou execute `poetry run task mldriftdemo` "
            "para uma comparação demonstrativa."
        )
    if dataset_path.stat().st_size == 0:
        raise ValueError(
            f"Dataset de monitoramento está vazio: {dataset_path}. "
            "Gere inferências via /predict antes de executar o drift real."
        )

    if dataset_path.suffix == ".parquet":
        return pd.read_parquet(dataset_path)
    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)
    if dataset_path.suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(str(dataset_path), lines=True)

    raise ValueError(f"Formato de dataset não suportado para drift: {dataset_path}")


def load_feature_columns(path: str | Path) -> list[str]:
    """Carrega a lista de features transformadas esperadas pelo modelo."""

    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def prepare_feature_matrix(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    feature_pipeline_path: str | Path,
) -> pd.DataFrame:
    """Resolve uma matriz transformada a partir de dados brutos ou já processados."""

    if set(feature_columns).issubset(dataset.columns):
        return dataset[feature_columns].copy()

    raw_features = dataset.drop(
        columns=[
            TARGET_COLUMN,
            *MONITORING_METADATA_COLUMNS,
        ],
        errors="ignore",
    )
    feature_pipeline = load(feature_pipeline_path)
    transformed_features = feature_pipeline.transform(raw_features)
    return transformed_features[feature_columns].copy()


def build_reference_predictions(
    reference_features: pd.DataFrame,
    reference_dataset: pd.DataFrame,
    model_path: str | Path,
) -> pd.Series | None:
    """Obtém probabilidades de referência para comparar prediction drift."""

    if PREDICTION_PROBABILITY_COLUMN in reference_dataset.columns:
        return reference_dataset[PREDICTION_PROBABILITY_COLUMN].astype(float)

    model = load(model_path)
    if not hasattr(model, "predict_proba"):
        return None

    return pd.Series(
        model.predict_proba(reference_features)[:, 1],
        name=PREDICTION_PROBABILITY_COLUMN,
    )


def prepare_monitoring_inputs(
    drift_config: dict[str, Any],
    current_data_path: str | None,
) -> MonitoringPreparedInputs:
    """Carrega datasets e matrizes de features necessárias para o drift."""

    current_path = current_data_path or drift_config["current_data_path"]
    reference_dataset = load_dataset(drift_config["reference_data_path"])
    current_dataset = load_dataset(current_path)
    feature_columns = load_feature_columns(drift_config["feature_columns_path"])

    reference_features = prepare_feature_matrix(
        dataset=reference_dataset,
        feature_columns=feature_columns,
        feature_pipeline_path=drift_config["feature_pipeline_path"],
    )
    current_features = prepare_feature_matrix(
        dataset=current_dataset,
        feature_columns=feature_columns,
        feature_pipeline_path=drift_config["feature_pipeline_path"],
    )

    return MonitoringPreparedInputs(
        current_path=current_path,
        reference_dataset=reference_dataset,
        current_dataset=current_dataset,
        reference_features=reference_features,
        current_features=current_features,
        reference_row_count=len(reference_dataset),
        current_row_count=len(current_dataset),
    )


def calculate_numeric_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calcula PSI para uma coluna numérica."""

    reference_values = pd.to_numeric(reference, errors="coerce").dropna()
    current_values = pd.to_numeric(current, errors="coerce").dropna()
    if reference_values.empty or current_values.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(reference_values.quantile(quantiles).to_numpy())
    if len(bin_edges) < MIN_BIN_EDGE_COUNT:
        min_value = min(reference_values.min(), current_values.min())
        max_value = max(reference_values.max(), current_values.max())
        if min_value == max_value:
            return 0.0
        bin_edges = np.linspace(min_value, max_value, bins + 1)

    reference_counts = pd.cut(
        reference_values,
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    ).value_counts(sort=False)
    current_counts = pd.cut(
        current_values,
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    ).value_counts(sort=False)

    return calculate_psi_from_distributions(reference_counts, current_counts)


def calculate_categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """Calcula PSI para uma coluna categórica ou discreta."""

    reference_counts = reference.fillna("__missing__").astype(str).value_counts()
    current_counts = current.fillna("__missing__").astype(str).value_counts()
    all_categories = reference_counts.index.union(current_counts.index)
    return calculate_psi_from_distributions(
        reference_counts.reindex(all_categories, fill_value=0),
        current_counts.reindex(all_categories, fill_value=0),
    )


def calculate_psi_from_distributions(
    reference_counts: pd.Series,
    current_counts: pd.Series,
) -> float:
    """Calcula PSI a partir de duas distribuições de contagem."""

    epsilon = 1e-6
    reference_share = reference_counts / max(reference_counts.sum(), 1)
    current_share = current_counts / max(current_counts.sum(), 1)
    reference_share = reference_share.clip(lower=epsilon)
    current_share = current_share.clip(lower=epsilon)

    psi_values = (current_share - reference_share) * np.log(
        current_share / reference_share
    )
    return float(psi_values.sum())


def calculate_feature_psi(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
) -> dict[str, float]:
    """Calcula PSI por feature para decisão de alerta."""

    psi_by_feature: dict[str, float] = {}
    for column in reference_features.columns:
        if pd.api.types.is_numeric_dtype(reference_features[column]):
            psi = calculate_numeric_psi(
                reference_features[column],
                current_features[column],
            )
        else:
            psi = calculate_categorical_psi(
                reference_features[column],
                current_features[column],
            )
        psi_by_feature[column] = psi

    return psi_by_feature


def decide_drift_status(
    psi_by_feature: dict[str, float],
    warning_threshold: float,
    critical_threshold: float,
    prediction_psi: float | None = None,
) -> DriftDecision:
    """Consolida PSI de features e predição em um status único."""

    drifted_features = [
        feature_name
        for feature_name, psi in psi_by_feature.items()
        if psi >= warning_threshold
    ]
    max_feature_psi = max(psi_by_feature.values(), default=0.0)
    mean_feature_psi = (
        float(np.mean(list(psi_by_feature.values()))) if psi_by_feature else 0.0
    )
    drifted_feature_share = (
        len(drifted_features) / len(psi_by_feature) if psi_by_feature else 0.0
    )

    critical_detected = max_feature_psi >= critical_threshold or (
        prediction_psi is not None and prediction_psi >= critical_threshold
    )
    warning_detected = max_feature_psi >= warning_threshold or (
        prediction_psi is not None and prediction_psi >= warning_threshold
    )
    if critical_detected:
        status = "critical"
    elif warning_detected:
        status = "warning"
    else:
        status = "ok"

    return DriftDecision(
        status=status,
        max_feature_psi=max_feature_psi,
        mean_feature_psi=mean_feature_psi,
        drifted_feature_share=drifted_feature_share,
        prediction_psi=prediction_psi,
        drifted_features=drifted_features,
    )


def apply_minimum_sample_size_policy(
    decision: DriftDecision,
    *,
    current_row_count: int,
    minimum_current_sample_size_for_decision: int,
) -> DriftDecision:
    """Bloqueia decisão operacional quando a amostra current é insuficiente."""

    if current_row_count >= minimum_current_sample_size_for_decision:
        return decision

    logger.warning(
        "Amostra current insuficiente para decisão operacional de drift "
        "(%d linhas). Mínimo configurado: %d. O PSI será registrado apenas "
        "para observabilidade, sem abrir retreino.",
        current_row_count,
        minimum_current_sample_size_for_decision,
    )
    return DriftDecision(
        status="insufficient_data",
        max_feature_psi=decision.max_feature_psi,
        mean_feature_psi=decision.mean_feature_psi,
        drifted_feature_share=decision.drifted_feature_share,
        prediction_psi=decision.prediction_psi,
        drifted_features=decision.drifted_features,
    )


def build_evidently_report(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Gera o relatório HTML de drift usando Evidently."""

    if len(current_features) < MIN_CURRENT_SAMPLE_SIZE_FOR_REPORT:
        logger.warning(
            "Amostra current pequena para relatório de drift (%d linhas). "
            "O Evidently pode emitir warnings estatísticos; ideal >= %d linhas.",
            len(current_features),
            MIN_CURRENT_SAMPLE_SIZE_FOR_REPORT,
        )

    report = Report([DataDriftPreset()])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Degrees of freedom <= 0 for slice",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in divide",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in multiply",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        snapshot = report.run(
            reference_data=reference_features,
            current_data=current_features,
        )
    report_output_path = Path(output_path)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(report_output_path))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Persiste um dicionário como JSON legível."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Acrescenta uma linha JSON em um histórico orientado a eventos."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_metrics_payload(
    decision: DriftDecision,
    psi_by_feature: dict[str, float],
    context: MonitoringExecutionContext,
) -> dict[str, Any]:
    """Monta o payload consolidado de métricas da execução de drift."""

    return {
        "created_at": context.created_at,
        "status": decision.status,
        "retraining_recommended": decision.retraining_recommended,
        "max_feature_psi": decision.max_feature_psi,
        "mean_feature_psi": decision.mean_feature_psi,
        "drifted_feature_share": decision.drifted_feature_share,
        "prediction_psi": decision.prediction_psi,
        "drifted_features": decision.drifted_features,
        "feature_psi": psi_by_feature,
        "reference_row_count": context.reference_row_count,
        "current_row_count": context.current_row_count,
        "minimum_current_sample_size_for_decision": (
            context.minimum_current_sample_size_for_decision
        ),
        "decision_eligible": (
            context.current_row_count
            >= context.minimum_current_sample_size_for_decision
        ),
    }


def write_retraining_placeholder(
    path: str | Path,
    decision: DriftDecision,
    context: RetrainingRequestContext,
) -> dict[str, Any]:
    """Registra uma solicitação auditável de retreino."""

    payload = {
        "request_id": str(uuid4()),
        "status": "requested",
        "reason": "critical_data_or_prediction_drift",
        "model_path": context.model_path,
        "training_config_path": context.training_config_path,
        "created_at": datetime.now(UTC).isoformat(),
        "trigger_mode": context.trigger_mode,
        "promotion_policy": "manual_approval_required",
        "drift_status": decision.status,
        "max_feature_psi": decision.max_feature_psi,
        "prediction_psi": decision.prediction_psi,
        "drifted_features": decision.drifted_features,
        "reference_row_count": context.reference_row_count,
        "current_row_count": context.current_row_count,
    }
    write_json(path, payload)
    return payload


def append_drift_run_history(path: str | Path, payload: dict[str, Any]) -> None:
    """Acrescenta uma linha com o resumo operacional de uma execução de drift."""

    append_jsonl(path, payload)


def maybe_trigger_retraining(
    decision: DriftDecision,
    *,
    retraining_config: dict[str, Any],
    model_path: str,
    reference_row_count: int,
    current_row_count: int,
) -> dict[str, Any] | None:
    """Executa o fluxo de retreino quando a política atual assim exigir."""

    if not (
        decision.retraining_recommended
        and retraining_config.get("enabled", False)
    ):
        return None

    retraining_request_payload = write_retraining_placeholder(
        path=retraining_config["request_path"],
        decision=decision,
        context=RetrainingRequestContext(
            model_path=model_path,
            trigger_mode=retraining_config.get("trigger_mode", "manual"),
            training_config_path=retraining_config.get(
                "training_config_path",
                DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
            ),
            reference_row_count=reference_row_count,
            current_row_count=current_row_count,
        ),
    )
    trigger_mode = retraining_config.get("trigger_mode", "manual")
    if trigger_mode != "manual":
        run_retraining_request(
            request_path=retraining_config["request_path"],
            output_path=retraining_config.get("run_path"),
        )

    return retraining_request_payload


def run_drift_monitoring(
    config_path: str = DEFAULT_MONITORING_CONFIG_PATH,
    current_data_path: str | None = None,
) -> DriftDecision:
    """Executa a comparação reference/current e salva os artefatos de monitoramento."""

    config = load_monitoring_config(config_path)
    drift_config = config["drift"]
    if not drift_config.get("enabled", False):
        logger.info("Monitoramento de drift desabilitado")
        return DriftDecision("ok", 0.0, 0.0, 0.0, None, [])

    prepared_inputs = prepare_monitoring_inputs(drift_config, current_data_path)

    build_evidently_report(
        reference_features=prepared_inputs.reference_features,
        current_features=prepared_inputs.current_features,
        output_path=drift_config["report_html_path"],
    )

    psi_by_feature = calculate_feature_psi(
        prepared_inputs.reference_features,
        prepared_inputs.current_features,
    )
    prediction_psi = None
    if drift_config.get("prediction_drift", {}).get("enabled", False):
        reference_predictions = build_reference_predictions(
            reference_features=prepared_inputs.reference_features,
            reference_dataset=prepared_inputs.reference_dataset,
            model_path=drift_config["model_path"],
        )
        if (
            reference_predictions is not None
            and PREDICTION_PROBABILITY_COLUMN
            in prepared_inputs.current_dataset.columns
        ):
            prediction_psi = calculate_numeric_psi(
                reference_predictions,
                prepared_inputs.current_dataset[PREDICTION_PROBABILITY_COLUMN],
            )

    decision = decide_drift_status(
        psi_by_feature=psi_by_feature,
        warning_threshold=drift_config["data_drift"]["warning_threshold"],
        critical_threshold=drift_config["data_drift"]["critical_threshold"],
        prediction_psi=prediction_psi,
    )
    minimum_current_sample_size_for_decision = drift_config.get(
        "minimum_current_sample_size_for_decision",
        DEFAULT_MIN_CURRENT_SAMPLE_SIZE_FOR_DECISION,
    )
    decision = apply_minimum_sample_size_policy(
        decision,
        current_row_count=prepared_inputs.current_row_count,
        minimum_current_sample_size_for_decision=(
            minimum_current_sample_size_for_decision
        ),
    )
    execution_context = MonitoringExecutionContext(
        created_at=datetime.now(UTC).isoformat(),
        reference_row_count=prepared_inputs.reference_row_count,
        current_row_count=prepared_inputs.current_row_count,
        minimum_current_sample_size_for_decision=(
            minimum_current_sample_size_for_decision
        ),
    )
    metrics_payload = build_metrics_payload(
        decision,
        psi_by_feature,
        execution_context,
    )
    write_json(drift_config["metrics_json_path"], metrics_payload)
    write_json(drift_config["status_path"], metrics_payload)

    retraining_request_payload = maybe_trigger_retraining(
        decision,
        retraining_config=drift_config.get("retraining", {}),
        model_path=drift_config["model_path"],
        reference_row_count=prepared_inputs.reference_row_count,
        current_row_count=prepared_inputs.current_row_count,
    )

    append_drift_run_history(
        drift_config["runs_history_path"],
        {
            "created_at": execution_context.created_at,
            "status": decision.status,
            "reference_data_path": drift_config["reference_data_path"],
            "current_data_path": prepared_inputs.current_path,
            "reference_row_count": prepared_inputs.reference_row_count,
            "current_row_count": prepared_inputs.current_row_count,
            "max_feature_psi": decision.max_feature_psi,
            "mean_feature_psi": decision.mean_feature_psi,
            "drifted_feature_share": decision.drifted_feature_share,
            "prediction_psi": decision.prediction_psi,
            "drifted_features": decision.drifted_features,
            "retraining_recommended": decision.retraining_recommended,
            "decision_eligible": (
                prepared_inputs.current_row_count
                >= minimum_current_sample_size_for_decision
            ),
            "minimum_current_sample_size_for_decision": (
                minimum_current_sample_size_for_decision
            ),
            "retraining_request_id": (
                retraining_request_payload["request_id"]
                if retraining_request_payload is not None
                else None
            ),
        },
    )

    logger.info("Drift monitoring finalizado com status=%s", decision.status)
    return decision


def parse_args() -> argparse.Namespace:
    """Lê argumentos da rotina batch de monitoramento."""

    parser = argparse.ArgumentParser(
        description="Executa detecção batch de drift com Evidently.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_MONITORING_CONFIG_PATH,
        help="Caminho relativo para o YAML de monitoramento.",
    )
    parser.add_argument(
        "--current",
        default=None,
        help="Dataset current alternativo para comparar contra a referência.",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada para execução local."""

    args = parse_args()
    try:
        run_drift_monitoring(config_path=args.config, current_data_path=args.current)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
