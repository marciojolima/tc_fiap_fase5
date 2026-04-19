"""Exporta features prontas do pipeline atual para consumo pelo Feast."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from common.config_loader import load_config
from common.data_loader import load_raw_data
from common.logger import get_logger
from features.feature_engineering import (
    clean_interim_data,
    load_feature_engineering_config,
    remove_direct_identifier_columns,
)
from serving.pipeline import load_feature_pipeline

from .config import (
    FEATURE_ENTITY_JOIN_KEY,
    FEATURE_STORE_EXPORT_METADATA_PATH,
    FEATURE_STORE_EXPORT_PATH,
    ONLINE_FEATURE_COLUMNS,
)

logger = get_logger("feast_ops.export")

DEFAULT_EVENT_TIMESTAMP_START = pd.Timestamp("2024-01-01T00:00:00Z")


@dataclass(frozen=True)
class FeatureStoreExportSummary:
    """Resumo serializável da exportação usada pela feature store."""

    output_path: str
    row_count: int
    feature_count: int
    entity_column: str
    event_timestamp_column: str
    created_timestamp_column: str
    export_generated_at: str
    event_timestamp_start: str
    event_timestamp_end: str


def build_feature_store_export_dataframe(
    raw_dataset: pd.DataFrame,
    feature_pipeline,
) -> pd.DataFrame:
    """Gera um dataset Feast-ready reaproveitando o pipeline oficial de features."""

    stage_config = load_feature_engineering_config()
    global_config = load_config()
    target_column = global_config["data"]["target_col"]

    cleaned_source = clean_interim_data(raw_dataset)
    minimized_dataset = remove_direct_identifier_columns(
        raw_dataset=cleaned_source,
        direct_identifier_columns=stage_config.direct_identifier_columns,
    )
    feature_input = minimized_dataset.drop(columns=[target_column], errors="ignore")
    transformed_features = feature_pipeline.transform(feature_input)

    selected_features = transformed_features.loc[:, ONLINE_FEATURE_COLUMNS].copy()
    event_timestamps = DEFAULT_EVENT_TIMESTAMP_START + pd.to_timedelta(
        range(len(cleaned_source)),
        unit="m",
    )

    export_dataframe = pd.DataFrame(
        {
            FEATURE_ENTITY_JOIN_KEY: cleaned_source["CustomerId"].astype("int64"),
            "event_timestamp": event_timestamps,
            "created_timestamp": event_timestamps,
        }
    )

    return pd.concat(
        [
            export_dataframe.reset_index(drop=True),
            selected_features.reset_index(drop=True),
        ],
        axis=1,
    )


def export_features_for_feast(
    output_path: Path = FEATURE_STORE_EXPORT_PATH,
    metadata_path: Path = FEATURE_STORE_EXPORT_METADATA_PATH,
) -> FeatureStoreExportSummary:
    """Executa a exportação das features prontas para a camada offline do Feast."""

    raw_dataset = load_raw_data()
    feature_pipeline = load_feature_pipeline()
    feast_ready_dataframe = build_feature_store_export_dataframe(
        raw_dataset=raw_dataset,
        feature_pipeline=feature_pipeline,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feast_ready_dataframe.to_parquet(output_path, index=False)

    summary = FeatureStoreExportSummary(
        output_path=str(output_path),
        row_count=len(feast_ready_dataframe),
        feature_count=len(ONLINE_FEATURE_COLUMNS),
        entity_column=FEATURE_ENTITY_JOIN_KEY,
        event_timestamp_column="event_timestamp",
        created_timestamp_column="created_timestamp",
        export_generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
        event_timestamp_start=feast_ready_dataframe["event_timestamp"].min().isoformat(),
        event_timestamp_end=feast_ready_dataframe["event_timestamp"].max().isoformat(),
    )

    metadata_path.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "Dataset offline do Feast exportado em %s com %d linhas e %d features",
        output_path,
        summary.row_count,
        summary.feature_count,
    )
    return summary


def main() -> None:
    """Ponto de entrada de CLI para exportação da feature store."""

    export_features_for_feast()


if __name__ == "__main__":
    main()
