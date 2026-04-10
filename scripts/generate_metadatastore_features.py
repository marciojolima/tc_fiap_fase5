"""Gera colunas temporais e operacionais para simulação de feature store.

Objetivo:
- preparar o dataset enriquecido para uso em uma feature store local;
- adicionar colunas de referência temporal e materialização.

Colunas geradas:
- snapshot_date: data sintética de referência temporal do registro
- materialized_at: data sintética de materialização/publicação na store

Regras:
- snapshot_date é distribuída deterministicamente a partir do RowNumber;
- materialized_at é inicialmente igual a snapshot_date.

Observação:
- este script opera sobre um dataset já enriquecido na camada de negócio.

Exemplo:
python -m scripts.generate_metadatastore_features \
  --input data/raw/customer_churn_business_v1.csv \
  --output data/raw/customer_churn_store_v2.csv \
  --metadata-output data/raw/customer_churn_store_v2.metadata.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.common.logger import get_logger

logger = get_logger("scripts.generate_metadatastore_features")


@dataclass(frozen=True)
class StoreFeatureGenerationConfig:
    """Configuração da geração das colunas da feature store."""

    base_snapshot_date: str = "2024-01-01"
    cycle_days: int = 365
    row_number_column: str = "RowNumber"
    snapshot_column: str = "snapshot_date"
    materialized_at_column: str = "materialized_at"


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando."""

    parser = argparse.ArgumentParser(
        description="Gera colunas para simulação de feature store."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Caminho do CSV de entrada.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Caminho do CSV enriquecido de saída.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Caminho opcional do JSON com metadados da geração.",
    )
    parser.add_argument(
        "--base-snapshot-date",
        type=str,
        default="2024-01-01",
        help="Data base para distribuição determinística de snapshot_date.",
    )
    parser.add_argument(
        "--cycle-days",
        type=int,
        default=365,
        help="Quantidade de dias usada no ciclo da distribuição.",
    )
    return parser.parse_args()


def validate_required_columns(
    dataset: pd.DataFrame,
    row_number_column: str,
) -> None:
    """Valida a presença das colunas obrigatórias."""

    if row_number_column not in dataset.columns:
        raise KeyError(
            f"Coluna obrigatória ausente para geração da store: '{row_number_column}'"
        )


def build_snapshot_dates(
    dataset: pd.DataFrame,
    base_snapshot_date: str,
    cycle_days: int,
    row_number_column: str,
) -> pd.Series:
    """Cria snapshot_date de forma determinística a partir do RowNumber.

    Regra:
    snapshot_date = base_snapshot_date + ((RowNumber - 1) % cycle_days) dias
    """

    base_date = pd.Timestamp(base_snapshot_date)
    offsets = (dataset[row_number_column].astype(int) - 1) % cycle_days
    snapshot_dates = base_date + pd.to_timedelta(offsets, unit="D")
    return pd.Series(pd.to_datetime(snapshot_dates), index=dataset.index).dt.normalize()


def enrich_with_store_features(
    dataset: pd.DataFrame,
    config: StoreFeatureGenerationConfig,
) -> pd.DataFrame:
    """Retorna cópia do dataset com colunas da simulação de feature store."""

    validate_required_columns(
        dataset=dataset,
        row_number_column=config.row_number_column,
    )

    enriched_dataset = dataset.copy()
    snapshot_dates = build_snapshot_dates(
        dataset=enriched_dataset,
        base_snapshot_date=config.base_snapshot_date,
        cycle_days=config.cycle_days,
        row_number_column=config.row_number_column,
    )

    enriched_dataset[config.snapshot_column] = snapshot_dates.dt.strftime("%Y-%m-%d")
    enriched_dataset[config.materialized_at_column] = enriched_dataset[
        config.snapshot_column
    ]

    return enriched_dataset


def build_generation_metadata(
    input_path: Path,
    output_path: Path,
    dataset_before: pd.DataFrame,
    dataset_after: pd.DataFrame,
    config: StoreFeatureGenerationConfig,
) -> dict:
    """Monta metadados básicos da geração para rastreabilidade."""

    added_columns = [
        column_name
        for column_name in dataset_after.columns
        if column_name not in dataset_before.columns
    ]

    return {
        "stage": "store_feature_generation",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_before": int(len(dataset_before)),
        "rows_after": int(len(dataset_after)),
        "columns_before": int(len(dataset_before.columns)),
        "columns_after": int(len(dataset_after.columns)),
        "added_columns": added_columns,
        "generation_config": asdict(config),
        "snapshot_date_min": dataset_after[config.snapshot_column].min(),
        "snapshot_date_max": dataset_after[config.snapshot_column].max(),
        "materialized_at_min": dataset_after[config.materialized_at_column].min(),
        "materialized_at_max": dataset_after[config.materialized_at_column].max(),
    }


def save_metadata(metadata: dict, metadata_output_path: Path) -> None:
    """Salva os metadados em JSON."""

    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_output_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=False)


def main() -> None:
    """Ponto de entrada para a geração das colunas da feature store."""

    args = parse_args()

    config = StoreFeatureGenerationConfig(
        base_snapshot_date=args.base_snapshot_date,
        cycle_days=args.cycle_days,
    )

    dataset = pd.read_csv(args.input)
    logger.info("Dataset bruto carregado de %s com shape=%s", args.input, dataset.shape)
    enriched_dataset = enrich_with_store_features(dataset=dataset, config=config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched_dataset.to_csv(args.output, index=False)

    logger.info("Dataset preparado para feature store salvo em %s", args.output)
    logger.info(
        "Intervalo de %s: %s até %s",
        config.snapshot_column,
        enriched_dataset[config.snapshot_column].min(),
        enriched_dataset[config.snapshot_column].max(),
    )

    if args.metadata_output is not None:
        metadata = build_generation_metadata(
            input_path=args.input,
            output_path=args.output,
            dataset_before=dataset,
            dataset_after=enriched_dataset,
            config=config,
        )
        save_metadata(metadata, args.metadata_output)
        logger.info("Metadados salvos em %s", args.metadata_output)


if __name__ == "__main__":
    main()
