"""Detecção batch de drift com Evidently e gatilho auditável de retreino."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from joblib import load

from common.config_loader import load_config
from common.logger import get_logger

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

logger = get_logger(__name__)


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


def load_monitoring_config(
    config_path: str = DEFAULT_MONITORING_CONFIG_PATH,
) -> dict[str, Any]:
    """Carrega as configurações de drift."""

    return load_config(config_path)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carrega parquet, csv ou jsonl para uma rotina de monitoramento."""

    dataset_path = Path(path)
    if dataset_path.suffix == ".parquet":
        return pd.read_parquet(dataset_path)
    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)
    if dataset_path.suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(dataset_path, lines=True)

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


def build_evidently_report(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Gera o relatório HTML de drift usando Evidently."""

    report = Report([DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference_features,
        current_data=current_features,
    )
    report_output_path = Path(output_path)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(report_output_path)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Persiste um dicionário como JSON legível."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_retraining_placeholder(
    path: str | Path,
    decision: DriftDecision,
    model_path: str | Path,
) -> None:
    """Registra uma solicitação de retreino sem executar promoção automática."""

    write_json(
        path,
        {
            "status": "requested",
            "reason": "critical_data_or_prediction_drift",
            "model_path": str(model_path),
            "created_at": datetime.now(UTC).isoformat(),
            "action": "manual_approval_required",
            "drift_status": decision.status,
            "max_feature_psi": decision.max_feature_psi,
            "prediction_psi": decision.prediction_psi,
        },
    )


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

    build_evidently_report(
        reference_features=reference_features,
        current_features=current_features,
        output_path=drift_config["report_html_path"],
    )

    psi_by_feature = calculate_feature_psi(reference_features, current_features)
    prediction_psi = None
    if drift_config.get("prediction_drift", {}).get("enabled", False):
        reference_predictions = build_reference_predictions(
            reference_features=reference_features,
            reference_dataset=reference_dataset,
            model_path=drift_config["model_path"],
        )
        if (
            reference_predictions is not None
            and PREDICTION_PROBABILITY_COLUMN in current_dataset.columns
        ):
            prediction_psi = calculate_numeric_psi(
                reference_predictions,
                current_dataset[PREDICTION_PROBABILITY_COLUMN],
            )

    decision = decide_drift_status(
        psi_by_feature=psi_by_feature,
        warning_threshold=drift_config["data_drift"]["warning_threshold"],
        critical_threshold=drift_config["data_drift"]["critical_threshold"],
        prediction_psi=prediction_psi,
    )
    metrics_payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": decision.status,
        "retraining_recommended": decision.retraining_recommended,
        "max_feature_psi": decision.max_feature_psi,
        "mean_feature_psi": decision.mean_feature_psi,
        "drifted_feature_share": decision.drifted_feature_share,
        "prediction_psi": decision.prediction_psi,
        "drifted_features": decision.drifted_features,
        "feature_psi": psi_by_feature,
    }
    write_json(drift_config["metrics_json_path"], metrics_payload)
    write_json(drift_config["status_path"], metrics_payload)

    retraining_config = drift_config.get("retraining", {})
    if decision.retraining_recommended and retraining_config.get("enabled", False):
        write_retraining_placeholder(
            path=retraining_config["request_path"],
            decision=decision,
            model_path=drift_config["model_path"],
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
    run_drift_monitoring(config_path=args.config, current_data_path=args.current)


if __name__ == "__main__":
    main()
