"""Geração reprodutível de lotes sintéticos para validar drift com Evidently."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from common.config_loader import (
    DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
    load_global_config,
)
from common.data_loader import load_raw_data
from common.logger import get_logger
from common.seed import set_global_seed
from serving.pipeline import (
    build_serving_config,
    load_feature_pipeline,
    load_prediction_model,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/scenario_analysis/drift")
DEFAULT_BATCH_SIZE = 60
INPUT_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Card Type",
    "Point Earned",
]
SCENARIO_NAMES = (
    "baseline_like",
    "age_drift",
    "wealth_drift",
    "high_risk_prediction_drift",
    "mixed_extreme_drift",
)

logger = get_logger("scenario_analysis.synthetic_drifts")


@dataclass(frozen=True)
class SyntheticBatchConfig:
    """Configuração da geração reprodutível de lotes sintéticos."""

    seed: int
    tracking_uri: str
    mlflow_experiment_name: str
    output_dir: Path
    batch_size: int
    experiment_config_path: str


@dataclass(frozen=True)
class SyntheticBatchResult:
    """Resumo da geração de um lote sintético."""

    scenario_name: str
    output_path: Path
    row_count: int
    mean_probability: float
    positive_rate: float


def load_synthetic_batch_config(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    experiment_config_path: str = DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
) -> SyntheticBatchConfig:
    """Carrega parâmetros globais para geração dos lotes sintéticos."""

    global_config = load_global_config()
    return SyntheticBatchConfig(
        seed=global_config["seed"],
        tracking_uri=global_config["mlflow"]["tracking_uri"],
        mlflow_experiment_name=global_config["mlflow"].get(
            "monitoring_drift_experiment_name",
            "datathon-churn-monitoring-drift",
        ),
        output_dir=output_dir,
        batch_size=batch_size,
        experiment_config_path=experiment_config_path,
    )


def load_generation_base_dataframe() -> pd.DataFrame:
    """Carrega a base bruta e mantém apenas as colunas aceitas pelo serving."""

    raw_dataset = load_raw_data()
    return raw_dataset[INPUT_COLUMNS].copy()


def _sample_base_dataframe(
    base_dataframe: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """Amostra registros de referência preservando o domínio original."""

    return (
        base_dataframe.sample(n=batch_size, replace=True, random_state=seed)
        .reset_index(drop=True)
        .copy()
    )


def _clip_int(values: np.ndarray, minimum: int, maximum: int) -> np.ndarray:
    """Aplica clipping inteiro preservando os limites de validação da API."""

    return np.clip(np.rint(values), minimum, maximum).astype(int)


def _clip_float(
    values: np.ndarray,
    minimum: float,
    maximum: float | None = None,
) -> np.ndarray:
    """Aplica clipping float para manter consistência com o schema de serving."""

    clipped = np.maximum(values, minimum)
    if maximum is not None:
        clipped = np.minimum(clipped, maximum)
    return clipped.astype(float)


def build_baseline_like_batch(
    base_dataframe: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """Gera um lote parecido com a base de referência."""

    return _sample_base_dataframe(base_dataframe, batch_size, seed)


def build_age_drift_batch(
    base_dataframe: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """Gera drift de idade e leve mudança de geografia."""

    rng = np.random.default_rng(seed)
    batch = _sample_base_dataframe(base_dataframe, batch_size, seed)
    batch["Age"] = _clip_int(
        rng.normal(loc=72, scale=8, size=batch_size),
        minimum=58,
        maximum=92,
    )
    batch["Tenure"] = _clip_int(
        rng.integers(0, 4, size=batch_size),
        minimum=0,
        maximum=10,
    )
    batch["Geography"] = rng.choice(
        ["Germany", "Germany", "Spain"],
        size=batch_size,
    )
    return batch


def build_wealth_drift_batch(
    base_dataframe: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """Gera drift de renda, saldo e perfil premium."""

    rng = np.random.default_rng(seed)
    batch = _sample_base_dataframe(base_dataframe, batch_size, seed)
    batch["Balance"] = _clip_float(
        rng.normal(loc=135000, scale=25000, size=batch_size),
        minimum=20000.0,
    )
    batch["EstimatedSalary"] = _clip_float(
        rng.normal(loc=165000, scale=30000, size=batch_size),
        minimum=30000.0,
    )
    batch["Card Type"] = rng.choice(
        ["GOLD", "PLATINUM", "DIAMOND"],
        size=batch_size,
    )
    batch["Point Earned"] = _clip_int(
        rng.normal(loc=780, scale=120, size=batch_size),
        minimum=300,
        maximum=1000,
    )
    return batch


def build_high_risk_prediction_drift_batch(
    base_dataframe: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """Gera perfis que tendem a concentrar previsões de churn mais altas."""

    rng = np.random.default_rng(seed)
    batch = _sample_base_dataframe(base_dataframe, batch_size, seed)
    batch["CreditScore"] = _clip_int(
        rng.normal(loc=390, scale=45, size=batch_size),
        minimum=300,
        maximum=560,
    )
    batch["Age"] = _clip_int(
        rng.normal(loc=67, scale=9, size=batch_size),
        minimum=45,
        maximum=92,
    )
    batch["Tenure"] = _clip_int(
        rng.integers(0, 3, size=batch_size),
        minimum=0,
        maximum=10,
    )
    batch["Balance"] = _clip_float(
        rng.normal(loc=12000, scale=9000, size=batch_size),
        minimum=0.0,
    )
    batch["NumOfProducts"] = rng.choice([1, 4], size=batch_size, p=[0.7, 0.3])
    batch["IsActiveMember"] = 0
    batch["EstimatedSalary"] = _clip_float(
        rng.normal(loc=32000, scale=12000, size=batch_size),
        minimum=1000.0,
    )
    batch["Card Type"] = rng.choice(["SILVER", "GOLD"], size=batch_size)
    batch["Point Earned"] = _clip_int(
        rng.normal(loc=180, scale=55, size=batch_size),
        minimum=0,
        maximum=400,
    )
    return batch


def build_mixed_extreme_drift_batch(
    base_dataframe: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """Combina mudanças fortes de distribuição para disparar drift crítico."""

    rng = np.random.default_rng(seed)
    batch = _sample_base_dataframe(base_dataframe, batch_size, seed)
    batch["CreditScore"] = _clip_int(
        rng.normal(loc=430, scale=110, size=batch_size),
        minimum=300,
        maximum=850,
    )
    batch["Age"] = _clip_int(
        rng.normal(loc=74, scale=11, size=batch_size),
        minimum=30,
        maximum=95,
    )
    batch["Balance"] = _clip_float(
        rng.choice([0.0, 180000.0], size=batch_size, p=[0.55, 0.45])
        + rng.normal(loc=0, scale=8000, size=batch_size),
        minimum=0.0,
    )
    batch["NumOfProducts"] = rng.choice([1, 3, 4], size=batch_size, p=[0.2, 0.35, 0.45])
    batch["IsActiveMember"] = rng.choice([0, 1], size=batch_size, p=[0.8, 0.2])
    batch["EstimatedSalary"] = _clip_float(
        rng.choice([18000.0, 190000.0], size=batch_size, p=[0.6, 0.4])
        + rng.normal(loc=0, scale=9000, size=batch_size),
        minimum=1000.0,
    )
    batch["Geography"] = rng.choice(["Germany", "Spain"], size=batch_size, p=[0.7, 0.3])
    batch["Card Type"] = rng.choice(
        ["SILVER", "DIAMOND"],
        size=batch_size,
        p=[0.6, 0.4],
    )
    batch["Point Earned"] = _clip_int(
        rng.choice([90, 920], size=batch_size, p=[0.65, 0.35])
        + rng.normal(loc=0, scale=30, size=batch_size),
        minimum=0,
        maximum=1000,
    )
    return batch


def generate_synthetic_batch(
    scenario_name: str,
    base_dataframe: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """Despacha a geração de um cenário sintético nomeado."""

    builders = {
        "baseline_like": build_baseline_like_batch,
        "age_drift": build_age_drift_batch,
        "wealth_drift": build_wealth_drift_batch,
        "high_risk_prediction_drift": build_high_risk_prediction_drift_batch,
        "mixed_extreme_drift": build_mixed_extreme_drift_batch,
    }
    if scenario_name not in builders:
        raise ValueError(
            f"Cenário sintético inválido: {scenario_name}. "
            f"Disponíveis: {list(builders)}"
        )

    return builders[scenario_name](base_dataframe, batch_size, seed)


def score_batch_predictions(
    batch_dataframe: pd.DataFrame,
    experiment_config_path: str,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Gera probabilidades e classes previstas com o modelo atual."""

    serving_config = build_serving_config(experiment_config_path)
    feature_pipeline = load_feature_pipeline(serving_config.feature_pipeline_path)
    model = load_prediction_model(serving_config.model_path)

    transformed_features = feature_pipeline.transform(batch_dataframe)
    probabilities = model.predict_proba(transformed_features)[:, 1]
    predictions = (probabilities >= serving_config.threshold).astype(int)
    return (
        probabilities,
        predictions,
        serving_config.model_name,
        serving_config.threshold,
    )


def build_prediction_records(
    batch_dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Converte o lote em registros compatíveis com o monitoramento current."""

    created_at = datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []
    for row, probability, prediction in zip(
        batch_dataframe.to_dict(orient="records"),
        probabilities,
        predictions,
        strict=True,
    ):
        records.append(
            {
                "timestamp": created_at,
                "model_name": model_name,
                "threshold": threshold,
                "churn_probability": float(probability),
                "churn_prediction": int(prediction),
                **row,
            }
        )

    return records


def save_jsonl_records(records: list[dict[str, Any]], output_path: Path) -> None:
    """Persiste os registros em JSONL para alimentar o Evidently."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_batch_manifest(
    scenario_name: str,
    output_path: Path,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> Path:
    """Cria um resumo JSON do lote gerado."""

    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    manifest_payload = {
        "scenario_name": scenario_name,
        "output_path": str(output_path),
        "row_count": int(len(probabilities)),
        "mean_probability": float(probabilities.mean()),
        "positive_rate": float(predictions.mean()),
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def log_batch_generation_run(
    scenario_name: str,
    output_path: Path,
    manifest_path: Path,
    result: SyntheticBatchResult,
    cfg: SyntheticBatchConfig,
) -> None:
    """Registra a geração do lote no MLflow para auditoria e demo."""

    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    run_name = f"scenario_analysis_drift::{scenario_name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("scenario_name", scenario_name)
        mlflow.log_param("batch_size", result.row_count)
        mlflow.log_param("seed", cfg.seed)
        mlflow.log_param("experiment_config_path", cfg.experiment_config_path)
        mlflow.log_metric("mean_probability", result.mean_probability)
        mlflow.log_metric("positive_rate", result.positive_rate)
        mlflow.set_tag("flow", "scenario_analysis_drift_generation")
        mlflow.set_tag("artifact_type", "synthetic_drift_scenario")
        mlflow.log_artifact(
            str(output_path),
            artifact_path=f"scenario_analysis/drift/{scenario_name}",
        )
        mlflow.log_artifact(
            str(manifest_path),
            artifact_path=f"scenario_analysis/drift/{scenario_name}",
        )


def generate_and_log_synthetic_batch(
    scenario_name: str,
    cfg: SyntheticBatchConfig,
    base_dataframe: pd.DataFrame | None = None,
) -> SyntheticBatchResult:
    """Gera um lote sintético, persiste em JSONL e registra no MLflow."""

    generation_base_dataframe = base_dataframe
    if generation_base_dataframe is None:
        generation_base_dataframe = load_generation_base_dataframe()
    scenario_seed = cfg.seed + SCENARIO_NAMES.index(scenario_name)
    batch_dataframe = generate_synthetic_batch(
        scenario_name=scenario_name,
        base_dataframe=generation_base_dataframe,
        batch_size=cfg.batch_size,
        seed=scenario_seed,
    )
    probabilities, predictions, model_name, threshold = score_batch_predictions(
        batch_dataframe=batch_dataframe,
        experiment_config_path=cfg.experiment_config_path,
    )
    records = build_prediction_records(
        batch_dataframe=batch_dataframe,
        probabilities=probabilities,
        predictions=predictions,
        model_name=model_name,
        threshold=threshold,
    )
    output_path = cfg.output_dir / f"{scenario_name}.jsonl"
    save_jsonl_records(records, output_path)
    manifest_path = write_batch_manifest(
        scenario_name=scenario_name,
        output_path=output_path,
        probabilities=probabilities,
        predictions=predictions,
    )
    result = SyntheticBatchResult(
        scenario_name=scenario_name,
        output_path=output_path,
        row_count=len(records),
        mean_probability=float(probabilities.mean()),
        positive_rate=float(predictions.mean()),
    )
    log_batch_generation_run(
        scenario_name=scenario_name,
        output_path=output_path,
        manifest_path=manifest_path,
        result=result,
        cfg=cfg,
    )
    logger.info(
        "Lote sintético gerado — cenário=%s | linhas=%d | prob_media=%.4f | "
        "taxa_positiva=%.4f | saída=%s",
        result.scenario_name,
        result.row_count,
        result.mean_probability,
        result.positive_rate,
        result.output_path,
    )
    return result


def generate_synthetic_batch_suite(
    scenario_names: list[str],
    cfg: SyntheticBatchConfig,
) -> list[SyntheticBatchResult]:
    """Gera um conjunto de lotes sintéticos usando a mesma configuração."""

    set_global_seed(cfg.seed)
    base_dataframe = load_generation_base_dataframe()
    return [
        generate_and_log_synthetic_batch(
            scenario_name=scenario_name,
            cfg=cfg,
            base_dataframe=base_dataframe,
        )
        for scenario_name in scenario_names
    ]


def parse_args() -> argparse.Namespace:
    """Lê argumentos CLI para geração de lotes sintéticos de drift."""

    parser = argparse.ArgumentParser(
        description="Gera lotes sintéticos reprodutíveis para validar drift.",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_NAMES,
        help="Gera apenas um cenário sintético.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Gera todos os cenários sintéticos disponíveis.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Quantidade de registros por lote sintético.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Diretório de saída dos lotes gerados.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
        help="Configuração do experimento usada para scoring do lote.",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada da rotina de geração reprodutível."""

    args = parse_args()
    if args.all or not args.scenario:
        scenario_names = list(SCENARIO_NAMES)
    else:
        scenario_names = [args.scenario]
    cfg = load_synthetic_batch_config(
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        experiment_config_path=args.config,
    )
    results = generate_synthetic_batch_suite(scenario_names=scenario_names, cfg=cfg)
    print(
        json.dumps(
            [
                {
                    "scenario_name": result.scenario_name,
                    "output_path": str(result.output_path),
                    "row_count": result.row_count,
                    "mean_probability": result.mean_probability,
                    "positive_rate": result.positive_rate,
                }
                for result in results
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
