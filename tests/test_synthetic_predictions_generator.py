from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_synthetic_predictions import (
    PredictionOutputContext,
    SyntheticPredictionGenerationConfig,
    build_generation_metadata,
    build_prediction_records,
    generate_input_batch,
    load_generation_base_dataframe,
    run_generation,
    save_metadata,
    validate_generation_config,
)

EXPECTED_MONITORED_BALANCE = 10000.12345
EXPECTED_MONITORED_ESTIMATED_SALARY = 50000.98765


def build_base_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "CreditScore": 620,
                "Geography": "France",
                "Gender": "Female",
                "Age": 40,
                "Tenure": 5,
                "Balance": 10000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0,
                "Card Type": "GOLD",
                "Point Earned": 250,
            },
            {
                "CreditScore": 710,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 33,
                "Tenure": 8,
                "Balance": 82000.0,
                "NumOfProducts": 1,
                "HasCrCard": 0,
                "IsActiveMember": 1,
                "EstimatedSalary": 92000.0,
                "Card Type": "DIAMOND",
                "Point Earned": 700,
            },
        ]
    )


def test_validate_generation_config_rejects_non_positive_batch_size() -> None:
    config = SyntheticPredictionGenerationConfig(
        num_predictions=0,
        drift_mode="no_drift",
    )

    with pytest.raises(ValueError, match="maior que zero"):
        validate_generation_config(config)


def test_load_generation_base_dataframe_keeps_serving_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    dataframe = build_base_dataframe().assign(Exited=[0, 1])
    dataframe.to_csv(input_path, index=False)

    loaded = load_generation_base_dataframe(input_path)

    assert list(loaded.columns) == list(build_base_dataframe().columns)


def test_generate_input_batch_supports_both_modes() -> None:
    base_dataframe = build_base_dataframe()

    no_drift_batch = generate_input_batch(base_dataframe, 5, "no_drift", 42)
    with_drift_batch = generate_input_batch(base_dataframe, 5, "with_drift", 42)

    assert len(no_drift_batch) == 5  # noqa: PLR2004
    assert len(with_drift_batch) == 5  # noqa: PLR2004
    assert with_drift_batch["Age"].mean() > no_drift_batch["Age"].mean()


def test_build_prediction_records_includes_full_monitoring_contract() -> None:
    monitoring_features_dataframe = pd.DataFrame(
        [
            {
                "Card Type": 1.0,
                "Gender": 0.0,
                "Geo_Germany": 0.0,
                "Geo_Spain": 0.0,
                "CreditScore": 0.25,
                "Age": 0.5,
                "Tenure": 0.2,
                "Balance": 10000.12345,
                "NumOfProducts": 0.3,
                "HasCrCard": 1.0,
                "IsActiveMember": 1.0,
                "EstimatedSalary": 50000.98765,
                "Point Earned": 0.4,
                "BalancePerProduct": 5000.0,
                "PointsPerSalary": 0.005,
            },
            {
                "Card Type": 3.0,
                "Gender": 1.0,
                "Geo_Germany": 1.0,
                "Geo_Spain": 0.0,
                "CreditScore": 1.5,
                "Age": 1.2,
                "Tenure": 0.7,
                "Balance": 82000.0,
                "NumOfProducts": -1.0,
                "HasCrCard": 0.0,
                "IsActiveMember": 1.0,
                "EstimatedSalary": 92000.0,
                "Point Earned": 2.4,
                "BalancePerProduct": 82000.0,
                "PointsPerSalary": 0.0076,
            },
        ]
    )

    records = build_prediction_records(
        monitoring_features_dataframe=monitoring_features_dataframe,
        probabilities=np.array([0.2, 0.8]),
        predictions=np.array([0, 1]),
        context=PredictionOutputContext(
            model_name="random_forest_current",
            model_version="0.2.0",
            threshold=0.5,
        ),
    )

    assert len(records) == 2  # noqa: PLR2004
    assert records[0]["monitoring_contract"] == "transformed_features_v1"
    assert records[0]["feature_source"] == "synthetic_transformed_batch"
    assert records[0]["model_version"] == "0.2.0"
    assert records[1]["churn_prediction"] == 1
    assert records[0]["Balance"] == EXPECTED_MONITORED_BALANCE
    assert records[0]["EstimatedSalary"] == EXPECTED_MONITORED_ESTIMATED_SALARY
    assert "Geography" not in records[0]
    assert "Geo_Germany" in records[0]
    assert "timestamp" in records[0]
    assert datetime.fromisoformat(records[0]["timestamp"]).utcoffset() == timedelta(
        hours=-3
    )


def test_build_generation_metadata_reports_summary() -> None:
    config = SyntheticPredictionGenerationConfig(
        num_predictions=2,
        drift_mode="with_drift",
        output_path=Path("artifacts/out.jsonl"),
    )
    monitoring_features_dataframe = pd.DataFrame(
        [
            {
                "Card Type": 1.0,
                "Gender": 0.0,
                "Geo_Germany": 0.0,
                "Geo_Spain": 0.0,
                "CreditScore": 0.25,
                "Age": 0.5,
                "Tenure": 0.2,
                "Balance": 10000.12345,
                "NumOfProducts": 0.3,
                "HasCrCard": 1.0,
                "IsActiveMember": 1.0,
                "EstimatedSalary": 50000.98765,
                "Point Earned": 0.4,
                "BalancePerProduct": 5000.0,
                "PointsPerSalary": 0.005,
            },
            {
                "Card Type": 3.0,
                "Gender": 1.0,
                "Geo_Germany": 1.0,
                "Geo_Spain": 0.0,
                "CreditScore": 1.5,
                "Age": 1.2,
                "Tenure": 0.7,
                "Balance": 82000.0,
                "NumOfProducts": -1.0,
                "HasCrCard": 0.0,
                "IsActiveMember": 1.0,
                "EstimatedSalary": 92000.0,
                "Point Earned": 2.4,
                "BalancePerProduct": 82000.0,
                "PointsPerSalary": 0.0076,
            },
        ]
    )
    records = build_prediction_records(
        monitoring_features_dataframe=monitoring_features_dataframe,
        probabilities=np.array([0.1, 0.9]),
        predictions=np.array([0, 1]),
        context=PredictionOutputContext(
            model_name="random_forest_current",
            model_version="0.2.0",
            threshold=0.5,
        ),
    )

    metadata = build_generation_metadata(config, records, config.output_path)

    assert metadata["drift_mode"] == "with_drift"
    assert metadata["num_predictions"] == 2  # noqa: PLR2004
    assert metadata["positive_rate"] == 0.5  # noqa: PLR2004


def test_save_metadata_persists_json(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = {"stage": "synthetic_prediction_generation", "num_predictions": 3}

    save_metadata(metadata, metadata_path)

    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata


def test_run_generation_writes_jsonl_and_optional_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "synthetic_predictions.jsonl"
    metadata_path = tmp_path / "synthetic_predictions.metadata.json"
    config = SyntheticPredictionGenerationConfig(
        num_predictions=4,
        drift_mode="no_drift",
        input_path=tmp_path / "ignored.csv",
        output_path=output_path,
        metadata_output_path=metadata_path,
    )

    monkeypatch.setattr(
        "scripts.generate_synthetic_predictions.load_generation_base_dataframe",
        lambda input_path: build_base_dataframe(),
    )
    monkeypatch.setattr(
        "scripts.generate_synthetic_predictions.score_prediction_batch",
        lambda batch_dataframe, experiment_config_path: (
            pd.DataFrame(
                [
                    {
                        "Card Type": 1.0,
                        "Gender": 0.0,
                        "Geo_Germany": 0.0,
                        "Geo_Spain": 1.0,
                        "CreditScore": 0.1,
                        "Age": 0.2,
                        "Tenure": 0.3,
                        "Balance": 0.4,
                        "NumOfProducts": 0.5,
                        "HasCrCard": 1.0,
                        "IsActiveMember": 1.0,
                        "EstimatedSalary": 0.6,
                        "Point Earned": 0.7,
                        "BalancePerProduct": 0.8,
                        "PointsPerSalary": 0.9,
                    }
                ]
                * 4
            ),
            np.array([0.1, 0.2, 0.7, 0.9]),
            np.array([0, 0, 1, 1]),
            PredictionOutputContext(
                model_name="random_forest_current",
                model_version="0.2.0",
                threshold=0.5,
            ),
        ),
    )

    generated_output_path, metadata = run_generation(config)

    assert generated_output_path == output_path
    assert output_path.exists()
    assert metadata_path.exists()
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 4  # noqa: PLR2004
    assert metadata["mean_probability"] == 0.475  # noqa: PLR2004
