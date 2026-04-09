"""Registro leve de inferências para monitoramento batch de drift."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.config_loader import load_config
from common.logger import get_logger
from serving.schemas import ChurnPredictionRequest

DEFAULT_MONITORING_CONFIG_PATH = "configs/monitoring_config.yaml"

logger = get_logger(__name__)


@dataclass(frozen=True)
class PredictionLogContext:
    """Metadados da predição necessários para drift de produção."""

    probability: float
    prediction: int
    model_name: str
    threshold: float


def load_monitoring_config(
    config_path: str = DEFAULT_MONITORING_CONFIG_PATH,
) -> dict[str, Any]:
    """Carrega a configuração de monitoramento do projeto."""

    return load_config(config_path)


def is_inference_logging_enabled(config: dict[str, Any]) -> bool:
    """Indica se o logging de inferências deve ocorrer durante o serving."""

    drift_config = config.get("drift", {})
    return bool(
        drift_config.get("enabled", False) and drift_config.get("current_data_path")
    )


def build_inference_log_record(
    payload: ChurnPredictionRequest,
    probability: float,
    prediction: int,
    model_name: str,
    threshold: float,
) -> dict[str, Any]:
    """Monta uma linha auditável com entrada, predição e metadados mínimos."""

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "threshold": threshold,
        "churn_probability": probability,
        "churn_prediction": prediction,
        **payload.model_dump(by_alias=True),
    }


def append_inference_log(
    record: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Acrescenta uma inferência em formato JSON Lines."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_prediction_for_monitoring(
    payload: ChurnPredictionRequest,
    prediction_context: PredictionLogContext,
    config_path: str = DEFAULT_MONITORING_CONFIG_PATH,
) -> None:
    """Registra a inferência atual sem interromper a resposta da API."""

    try:
        monitoring_config = load_monitoring_config(config_path)
        if not is_inference_logging_enabled(monitoring_config):
            return

        record = build_inference_log_record(
            payload=payload,
            probability=prediction_context.probability,
            prediction=prediction_context.prediction,
            model_name=prediction_context.model_name,
            threshold=prediction_context.threshold,
        )
        append_inference_log(
            record=record,
            output_path=monitoring_config["drift"]["current_data_path"],
        )
    except OSError:
        logger.exception("Falha ao registrar inferência para monitoramento de drift")
