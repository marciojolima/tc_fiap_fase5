"""Gera features sintéticas de negócio para o dataset de churn.

Objetivo:
- enriquecer o dataset com colunas de negócio reproduzíveis;
- preparar uma nova versão do dado antes da etapa de feature store.

Colunas geradas:
- customer_since_date: data sintética aproximada de início do relacionamento

Regras:
- customer_since_date é derivada de uma data âncora menos o Tenure em anos;
- um deslocamento determinístico em dias é aplicado para evitar padrão artificial.

Observação:
- este script não cria metadados operacionais da feature store;
- ele representa apenas enriquecimento de negócio.

python -m scripts.generate_business_features \
  --input data/raw/Customer-Churn-Records.csv \
  --output data/raw/customer_churn_business_v1.csv \
  --metadata-output data/raw/customer_churn_business_v1.metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.common.logger import get_logger

logger = get_logger("scripts.generate_business_features")


@dataclass(frozen=True)
class BusinessFeatureGenerationConfig:
    """Configuração da geração das features de negócio."""

    anchor_date: str = "2024-12-31"
    tenure_column: str = "Tenure"
    stable_id_column: str = "CustomerId"
    customer_since_column: str = "customer_since_date"
    jitter_days_range: int = 365


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando."""

    parser = argparse.ArgumentParser(
        description="Gera features sintéticas de negócio para o dataset de churn."
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
        "--anchor-date",
        type=str,
        default="2024-12-31",
        help="Data âncora usada para derivar customer_since_date.",
    )
    parser.add_argument(
        "--stable-id-column",
        type=str,
        default="CustomerId",
        help="Coluna usada para gerar deslocamento determinístico.",
    )
    parser.add_argument(
        "--jitter-days-range",
        type=int,
        default=365,
        help="Faixa de dias usada na variação determinística.",
    )
    return parser.parse_args()


def validate_required_columns(
    dataset: pd.DataFrame,
    tenure_column: str,
    stable_id_column: str,
) -> None:
    """Valida a presença das colunas obrigatórias."""

    missing_columns = [
        column_name
        for column_name in (tenure_column, stable_id_column)
        if column_name not in dataset.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Colunas obrigatórias ausentes para geração de negócio: {missing_columns}"
        )


def stable_hash_to_int(value: str) -> int:
    """Converte string em inteiro estável usando MD5."""

    return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)


def build_deterministic_day_offsets(
    dataset: pd.DataFrame,
    stable_id_column: str,
    jitter_days_range: int,
) -> pd.Series:
    """Gera deslocamentos determinísticos em dias para cada registro."""

    if jitter_days_range <= 0:
        return pd.Series([0] * len(dataset), index=dataset.index)

    offsets = (
        dataset[stable_id_column]
        .astype(str)
        .apply(lambda value: stable_hash_to_int(value) % jitter_days_range)
    )
    return offsets.astype(int)


def build_customer_since_dates(
    dataset: pd.DataFrame,
    anchor_date: str,
    tenure_column: str,
    stable_id_column: str,
    jitter_days_range: int,
) -> pd.Series:
    """Cria customer_since_date com variação determinística e reprodutível.

    Regra:
    customer_since_date ≈ anchor_date - Tenure anos + deslocamento_determinístico
    """

    reference_date = pd.Timestamp(anchor_date)
    normalized_tenure = dataset[tenure_column].fillna(0).astype(int)
    day_offsets = build_deterministic_day_offsets(
        dataset=dataset,
        stable_id_column=stable_id_column,
        jitter_days_range=jitter_days_range,
    )

    customer_since_dates = [
        reference_date - pd.DateOffset(years=tenure) + pd.Timedelta(days=offset)
        for tenure, offset in zip(normalized_tenure, day_offsets, strict=False)
    ]

    return pd.Series(
        pd.to_datetime(customer_since_dates), index=dataset.index
    ).dt.normalize()


def enrich_with_business_features(
    dataset: pd.DataFrame,
    config: BusinessFeatureGenerationConfig,
) -> pd.DataFrame:
    """Retorna cópia do dataset com features sintéticas de negócio."""

    validate_required_columns(
        dataset=dataset,
        tenure_column=config.tenure_column,
        stable_id_column=config.stable_id_column,
    )

    enriched_dataset = dataset.copy()
    customer_since_dates = build_customer_since_dates(
        dataset=enriched_dataset,
        anchor_date=config.anchor_date,
        tenure_column=config.tenure_column,
        stable_id_column=config.stable_id_column,
        jitter_days_range=config.jitter_days_range,
    )

    enriched_dataset[config.customer_since_column] = customer_since_dates.dt.strftime(
        "%Y-%m-%d"
    )
    return enriched_dataset


def build_generation_metadata(
    input_path: Path,
    output_path: Path,
    dataset_before: pd.DataFrame,
    dataset_after: pd.DataFrame,
    config: BusinessFeatureGenerationConfig,
) -> dict:
    """Monta metadados básicos da geração para rastreabilidade."""

    added_columns = [
        column_name
        for column_name in dataset_after.columns
        if column_name not in dataset_before.columns
    ]

    return {
        "stage": "business_feature_generation",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_before": int(len(dataset_before)),
        "rows_after": int(len(dataset_after)),
        "columns_before": int(len(dataset_before.columns)),
        "columns_after": int(len(dataset_after.columns)),
        "added_columns": added_columns,
        "generation_config": asdict(config),
        "customer_since_date_min": dataset_after[config.customer_since_column].min(),
        "customer_since_date_max": dataset_after[config.customer_since_column].max(),
    }


def save_metadata(metadata: dict, metadata_output_path: Path) -> None:
    """Salva os metadados em JSON."""

    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_output_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=False)


def main() -> None:
    """Ponto de entrada para a geração das features sintéticas de negócio."""

    args = parse_args()

    config = BusinessFeatureGenerationConfig(
        anchor_date=args.anchor_date,
        stable_id_column=args.stable_id_column,
        jitter_days_range=args.jitter_days_range,
    )

    dataset = pd.read_csv(args.input)
    logger.info("Dataset bruto carregado de %s com shape=%s", args.input, dataset.shape)

    enriched_dataset = enrich_with_business_features(dataset=dataset, config=config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched_dataset.to_csv(args.output, index=False)

    logger.info("Dataset com feature de negócio salvo em %s", args.output)
    logger.info(
        "Intervalo de %s: %s até %s",
        config.customer_since_column,
        enriched_dataset[config.customer_since_column].min(),
        enriched_dataset[config.customer_since_column].max(),
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
