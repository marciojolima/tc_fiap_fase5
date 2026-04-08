"""Testes unitários para scripts de geração de dados sintéticos auxiliares."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_business_features import (
    BusinessFeatureGenerationConfig,
    build_customer_since_dates,
    build_deterministic_day_offsets,
    enrich_with_business_features,
)
from scripts.generate_business_features import (
    build_generation_metadata as build_business_generation_metadata,
)
from scripts.generate_business_features import (
    save_metadata as save_business_metadata,
)
from scripts.generate_business_features import (
    validate_required_columns as validate_business_required_columns,
)
from scripts.generate_metadatastore_features import (
    StoreFeatureGenerationConfig,
    build_snapshot_dates,
    enrich_with_store_features,
)
from scripts.generate_metadatastore_features import (
    build_generation_metadata as build_store_generation_metadata,
)
from scripts.generate_metadatastore_features import (
    save_metadata as save_store_metadata,
)
from scripts.generate_metadatastore_features import (
    validate_required_columns as validate_store_required_columns,
)


@pytest.fixture
def business_input_dataframe() -> pd.DataFrame:
    """Dataset mínimo para testar enriquecimento de negócio."""

    return pd.DataFrame(
        {
            "CustomerId": [101, 202, 303],
            "Tenure": [2, 0, 5],
            "RowNumber": [1, 2, 3],
            "Exited": [0, 1, 0],
        }
    )


def test_validate_business_required_columns_raises_for_missing_columns() -> None:
    dataset = pd.DataFrame({"CustomerId": [1, 2]})

    with pytest.raises(KeyError, match="Colunas obrigatórias ausentes"):
        validate_business_required_columns(
            dataset=dataset,
            tenure_column="Tenure",
            stable_id_column="CustomerId",
        )


def test_build_deterministic_day_offsets_returns_zero_when_range_is_non_positive(
    business_input_dataframe: pd.DataFrame,
) -> None:
    offsets = build_deterministic_day_offsets(
        dataset=business_input_dataframe,
        stable_id_column="CustomerId",
        jitter_days_range=0,
    )

    assert offsets.tolist() == [0, 0, 0]


def test_build_customer_since_dates_respects_anchor_and_tenure_without_jitter(
    business_input_dataframe: pd.DataFrame,
) -> None:
    customer_since_dates = build_customer_since_dates(
        dataset=business_input_dataframe,
        anchor_date="2024-12-31",
        tenure_column="Tenure",
        stable_id_column="CustomerId",
        jitter_days_range=0,
    )

    assert customer_since_dates.dt.strftime("%Y-%m-%d").tolist() == [
        "2022-12-31",
        "2024-12-31",
        "2019-12-31",
    ]


def test_enrich_with_business_features_adds_customer_since_date(
    business_input_dataframe: pd.DataFrame,
) -> None:
    config = BusinessFeatureGenerationConfig(
        anchor_date="2024-12-31", jitter_days_range=0
    )

    enriched_dataset = enrich_with_business_features(
        dataset=business_input_dataframe,
        config=config,
    )

    assert len(enriched_dataset) == len(business_input_dataframe)
    assert config.customer_since_column in enriched_dataset.columns
    assert enriched_dataset[config.customer_since_column].tolist() == [
        "2022-12-31",
        "2024-12-31",
        "2019-12-31",
    ]


def test_build_business_generation_metadata_reports_added_columns(
    business_input_dataframe: pd.DataFrame,
) -> None:
    config = BusinessFeatureGenerationConfig(jitter_days_range=0)
    enriched_dataset = enrich_with_business_features(
        dataset=business_input_dataframe,
        config=config,
    )

    metadata = build_business_generation_metadata(
        input_path=Path("data/raw/input.csv"),
        output_path=Path("data/raw/output.csv"),
        dataset_before=business_input_dataframe,
        dataset_after=enriched_dataset,
        config=config,
    )

    assert metadata["stage"] == "business_feature_generation"
    assert metadata["rows_before"] == 3  # noqa: PLR2004
    assert metadata["rows_after"] == 3  # noqa: PLR2004
    assert metadata["added_columns"] == ["customer_since_date"]


def test_save_business_metadata_persists_json(tmp_path: Path) -> None:
    metadata_output_path = tmp_path / "business_metadata.json"
    metadata = {"stage": "business_feature_generation", "rows_after": 3}

    save_business_metadata(metadata, metadata_output_path)

    assert json.loads(metadata_output_path.read_text(encoding="utf-8")) == metadata


def test_validate_store_required_columns_raises_for_missing_row_number() -> None:
    dataset = pd.DataFrame({"CustomerId": [1, 2]})

    with pytest.raises(
        KeyError, match="Coluna obrigatória ausente para geração da store"
    ):
        validate_store_required_columns(dataset=dataset, row_number_column="RowNumber")


def test_build_snapshot_dates_is_deterministic_from_row_number(
    business_input_dataframe: pd.DataFrame,
) -> None:
    snapshot_dates = build_snapshot_dates(
        dataset=business_input_dataframe,
        base_snapshot_date="2024-01-01",
        cycle_days=2,
        row_number_column="RowNumber",
    )

    assert snapshot_dates.dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-01",
    ]


def test_enrich_with_store_features_adds_snapshot_and_materialized_at(
    business_input_dataframe: pd.DataFrame,
) -> None:
    config = StoreFeatureGenerationConfig(base_snapshot_date="2024-01-01", cycle_days=2)

    enriched_dataset = enrich_with_store_features(
        dataset=business_input_dataframe,
        config=config,
    )

    assert config.snapshot_column in enriched_dataset.columns
    assert config.materialized_at_column in enriched_dataset.columns
    assert enriched_dataset[config.snapshot_column].tolist() == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-01",
    ]
    assert (
        enriched_dataset[config.snapshot_column]
        == enriched_dataset[config.materialized_at_column]
    ).all()


def test_build_store_generation_metadata_reports_added_columns(
    business_input_dataframe: pd.DataFrame,
) -> None:
    config = StoreFeatureGenerationConfig(base_snapshot_date="2024-01-01", cycle_days=2)
    enriched_dataset = enrich_with_store_features(
        dataset=business_input_dataframe,
        config=config,
    )

    metadata = build_store_generation_metadata(
        input_path=Path("data/raw/business.csv"),
        output_path=Path("data/raw/store.csv"),
        dataset_before=business_input_dataframe,
        dataset_after=enriched_dataset,
        config=config,
    )

    assert metadata["stage"] == "store_feature_generation"
    assert metadata["added_columns"] == ["snapshot_date", "materialized_at"]
    assert metadata["snapshot_date_min"] == "2024-01-01"
    assert metadata["snapshot_date_max"] == "2024-01-02"


def test_save_store_metadata_persists_json(tmp_path: Path) -> None:
    metadata_output_path = tmp_path / "store_metadata.json"
    metadata = {"stage": "store_feature_generation", "rows_after": 3}

    save_store_metadata(metadata, metadata_output_path)

    assert json.loads(metadata_output_path.read_text(encoding="utf-8")) == metadata
