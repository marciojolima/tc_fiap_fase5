"""Gera predições sintéticas em JSONL para experimentos de drift.

Objetivo:
- produzir arquivos compatíveis com o contrato de monitoramento atual;
- permitir geração controlada de lotes com ou sem drift intencional;
- facilitar testes locais do pipeline batch de drift sem depender da API.

Saída padrão:
- artifacts/monitoring/drift/experiments/predictions/synthetic_predictions_v1.jsonl

Exemplos:
python -m scripts.generate_synthetic_predictions \
  --num-predictions 50 \
  --drift no_drift
python -m scripts.generate_synthetic_predictions \
  --num-predictions 80 \
  --drift with_drift
python -m scripts.generate_synthetic_predictions \
  --num-predictions 120 \
  --drift with_drift \
  --output artifacts/monitoring/drift/experiments/predictions/\
synthetic_predictions_v1_with_drift_120.jsonl \
  --metadata-output artifacts/monitoring/drift/experiments/predictions/\
synthetic_predictions_v1_with_drift_120.metadata.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.common.seed import set_global_seed
from src.monitoring.inference_log import append_inference_log
from src.scenario_analysis.synthetic_drifts import (
    INPUT_COLUMNS,
    build_baseline_like_batch,
    build_mixed_extreme_drift_batch,
)
from src.serving.pipeline import (
    build_serving_config,
    load_feature_pipeline,
    load_prediction_model,
)

DEFAULT_INPUT_PATH = Path("data/raw/Customer-Churn-Records.csv")
DEFAULT_OUTPUT_PATH = Path(
    "artifacts/monitoring/drift/experiments/predictions/"
    "synthetic_predictions_v1.jsonl"
)
DEFAULT_EXPERIMENT_CONFIG_PATH = "configs/training/model_current.yaml"
DEFAULT_SEED = 42
_DRIFT_MODE_NO_DRIFT = "no_drift"
_DRIFT_MODE_WITH_DRIFT = "with_drift"

logger = get_logger("scripts.generate_synthetic_predictions")


@dataclass(frozen=True)
class SyntheticPredictionGenerationConfig:
    """Configuração da geração das predições sintéticas."""

    num_predictions: int
    drift_mode: Literal["no_drift", "with_drift"]
    input_path: Path = DEFAULT_INPUT_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    metadata_output_path: Path | None = None
    experiment_config_path: str = DEFAULT_EXPERIMENT_CONFIG_PATH
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class PredictionOutputContext:
    """Metadados do modelo usados para montar o JSONL de saída."""

    model_name: str
    model_version: str
    threshold: float


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da CLI."""

    parser = argparse.ArgumentParser(
        description="Gera predições sintéticas compatíveis com o monitoramento."
    )
    parser.add_argument(
        "--num-predictions",
        type=int,
        required=True,
        help="Quantidade de predições sintéticas a gerar.",
    )
    parser.add_argument(
        "--drift",
        type=str,
        choices=[_DRIFT_MODE_NO_DRIFT, _DRIFT_MODE_WITH_DRIFT],
        required=True,
        help="Define se o lote deve sair sem drift ou com drift intencional.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV base usado para preservar o domínio das features.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Arquivo JSONL de saída com as predições sintéticas.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Caminho opcional do JSON com resumo da geração.",
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        default=DEFAULT_EXPERIMENT_CONFIG_PATH,
        help="Configuração do modelo atual usada para score do lote.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed global para reprodutibilidade.",
    )
    return parser.parse_args()


def validate_generation_config(
    config: SyntheticPredictionGenerationConfig,
) -> None:
    """Valida parâmetros básicos da geração."""

    if config.num_predictions <= 0:
        raise ValueError("--num-predictions deve ser maior que zero.")


def load_generation_base_dataframe(input_path: Path) -> pd.DataFrame:
    """Carrega o CSV base e restringe às colunas aceitas pelo serving."""

    dataset = pd.read_csv(input_path)
    missing_columns = [
        column_name
        for column_name in INPUT_COLUMNS
        if column_name not in dataset.columns
    ]
    if missing_columns:
        raise KeyError(
            "CSV base não contém todas as colunas exigidas pelo serving: "
            f"{missing_columns}"
        )
    return dataset[INPUT_COLUMNS].copy()


def generate_input_batch(
    base_dataframe: pd.DataFrame,
    num_predictions: int,
    drift_mode: Literal["no_drift", "with_drift"],
    seed: int,
) -> pd.DataFrame:
    """Gera lote de entrada preservando o contrato do serving."""

    if drift_mode == _DRIFT_MODE_NO_DRIFT:
        batch = build_baseline_like_batch(base_dataframe, num_predictions, seed)
    elif drift_mode == _DRIFT_MODE_WITH_DRIFT:
        batch = build_mixed_extreme_drift_batch(base_dataframe, num_predictions, seed)
    else:
        raise ValueError(f"Modo de drift inválido: {drift_mode}")

    return batch[INPUT_COLUMNS].copy()


def score_prediction_batch(
    batch_dataframe: pd.DataFrame,
    experiment_config_path: str,
) -> tuple[np.ndarray, np.ndarray, PredictionOutputContext]:
    """Aplica o modelo atual ao lote sintético e retorna metadados."""

    serving_config = build_serving_config(experiment_config_path)
    feature_pipeline = load_feature_pipeline(serving_config.feature_pipeline_path)
    model = load_prediction_model(serving_config.model_path)

    transformed_features = feature_pipeline.transform(batch_dataframe)
    probabilities = model.predict_proba(transformed_features)[:, 1]
    predictions = (probabilities >= serving_config.threshold).astype(int)

    return (
        probabilities,
        predictions,
        PredictionOutputContext(
            model_name=serving_config.model_name,
            model_version=serving_config.model_version,
            threshold=serving_config.threshold,
        ),
    )


def build_prediction_records(
    batch_dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    context: PredictionOutputContext,
) -> list[dict[str, Any]]:
    """Converte o lote em registros no formato do log de inferência."""

    created_at = datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []
    for row, probability, prediction in zip(
        batch_dataframe.to_dict(orient="records"),
        probabilities,
        predictions,
        strict=True,
    ):
        normalized_row = row.copy()
        for numeric_column in ("Balance", "EstimatedSalary"):
            normalized_row[numeric_column] = round(
                float(normalized_row[numeric_column]),
                2,
            )

        records.append(
            {
                "timestamp": created_at,
                "model_name": context.model_name,
                "model_version": context.model_version,
                "threshold": float(context.threshold),
                "churn_probability": float(probability),
                "churn_prediction": int(prediction),
                **normalized_row,
            }
        )
    return records


def save_prediction_records(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Salva os registros em JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    for record in records:
        append_inference_log(record=record, output_path=output_path)


def build_generation_metadata(
    config: SyntheticPredictionGenerationConfig,
    records: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Monta metadados sintéticos para auditoria local."""

    probabilities = [record["churn_probability"] for record in records]
    predictions = [record["churn_prediction"] for record in records]

    return {
        "stage": "synthetic_prediction_generation",
        "created_at": datetime.now(UTC).isoformat(),
        "output_path": str(output_path),
        "num_predictions": len(records),
        "drift_mode": config.drift_mode,
        "positive_rate": float(np.mean(predictions)),
        "mean_probability": float(np.mean(probabilities)),
        "generation_config": {
            **asdict(config),
            "input_path": str(config.input_path),
            "output_path": str(config.output_path),
            "metadata_output_path": (
                str(config.metadata_output_path)
                if config.metadata_output_path is not None
                else None
            ),
        },
    }


def save_metadata(metadata: dict[str, Any], metadata_output_path: Path) -> None:
    """Persiste os metadados em JSON."""

    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_output_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=False)


def run_generation(
    config: SyntheticPredictionGenerationConfig,
) -> tuple[Path, dict[str, Any]]:
    """Executa a geração ponta a ponta."""

    validate_generation_config(config)
    set_global_seed(config.seed)

    base_dataframe = load_generation_base_dataframe(config.input_path)
    batch_dataframe = generate_input_batch(
        base_dataframe=base_dataframe,
        num_predictions=config.num_predictions,
        drift_mode=config.drift_mode,
        seed=config.seed,
    )
    probabilities, predictions, prediction_context = score_prediction_batch(
        batch_dataframe=batch_dataframe,
        experiment_config_path=config.experiment_config_path,
    )
    records = build_prediction_records(
        batch_dataframe=batch_dataframe,
        probabilities=probabilities,
        predictions=predictions,
        context=prediction_context,
    )
    save_prediction_records(records, config.output_path)
    metadata = build_generation_metadata(
        config=config,
        records=records,
        output_path=config.output_path,
    )
    if config.metadata_output_path is not None:
        save_metadata(metadata, config.metadata_output_path)
    return config.output_path, metadata


def main() -> None:
    """Ponto de entrada da CLI."""

    args = parse_args()
    config = SyntheticPredictionGenerationConfig(
        num_predictions=args.num_predictions,
        drift_mode=args.drift,
        input_path=args.input_csv,
        output_path=args.output,
        metadata_output_path=args.metadata_output,
        experiment_config_path=args.experiment_config,
        seed=args.seed,
    )
    output_path, metadata = run_generation(config)
    logger.info(
        "Predições sintéticas geradas em %s com drift_mode=%s e rows=%s",
        output_path,
        config.drift_mode,
        metadata["num_predictions"],
    )


if __name__ == "__main__":
    main()
