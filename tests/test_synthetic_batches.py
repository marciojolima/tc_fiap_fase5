from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from scenario_analysis.synthetic_drifts import (
    INPUT_COLUMNS,
    SyntheticBatchConfig,
    build_high_risk_prediction_drift_batch,
    build_prediction_records,
    generate_and_log_synthetic_batch,
    generate_synthetic_batch,
)

EXPECTED_BATCH_LINES = 4


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


def build_batch_config(tmp_path: Path) -> SyntheticBatchConfig:
    return SyntheticBatchConfig(
        seed=42,
        tracking_uri="file:./mlruns",
        mlflow_experiment_name="monitoring-drift-test",
        output_dir=tmp_path,
        batch_size=4,
        experiment_config_path="configs/training/model_current.yaml",
    )


def test_generate_synthetic_batch_returns_input_contract() -> None:
    batch = generate_synthetic_batch(
        scenario_name="baseline_like",
        base_dataframe=build_base_dataframe(),
        batch_size=6,
        seed=42,
    )

    assert list(batch.columns) == INPUT_COLUMNS
    assert len(batch) == 6  # noqa: PLR2004


def test_high_risk_prediction_drift_batch_applies_expected_constraints() -> None:
    batch = build_high_risk_prediction_drift_batch(
        base_dataframe=build_base_dataframe(),
        batch_size=8,
        seed=42,
    )

    assert batch["IsActiveMember"].eq(0).all()
    assert batch["CreditScore"].between(300, 560).all()
    assert batch["Card Type"].isin(["SILVER", "GOLD"]).all()


def test_build_prediction_records_includes_monitoring_metadata() -> None:
    batch = build_base_dataframe()
    records = build_prediction_records(
        batch_dataframe=batch,
        probabilities=np.array([0.2, 0.8]),
        predictions=np.array([0, 1]),
        model_name="random_forest_current",
        threshold=0.5,
    )

    assert len(records) == 2  # noqa: PLR2004
    assert records[0]["model_name"] == "random_forest_current"
    assert records[1]["churn_prediction"] == 1
    assert "timestamp" in records[0]
    assert datetime.fromisoformat(records[0]["timestamp"]).utcoffset() == timedelta(
        hours=-3
    )


def test_generate_and_log_synthetic_batch_writes_jsonl_and_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = build_batch_config(tmp_path)
    logged_artifacts: list[tuple[str, str]] = []

    def write_dummy_report(
        scenario_output_path: Path,
        report_output_path: Path,
        monitoring_config_path: str,
    ) -> Path:
        report_output_path.write_text("<html></html>", encoding="utf-8")
        return report_output_path

    monkeypatch.setattr(
        "scenario_analysis.synthetic_drifts.load_generation_base_dataframe",
        build_base_dataframe,
    )
    monkeypatch.setattr(
        "scenario_analysis.synthetic_drifts.score_batch_predictions",
        lambda batch_dataframe, experiment_config_path: (
            np.array([0.1, 0.2, 0.7, 0.9]),
            np.array([0, 0, 1, 1]),
            "random_forest_current",
            0.5,
        ),
    )
    monkeypatch.setattr(
        "scenario_analysis.synthetic_drifts.build_drift_report_for_scenario",
        write_dummy_report,
    )
    monkeypatch.setattr(
        "scenario_analysis.synthetic_drifts.mlflow.set_tracking_uri",
        lambda uri: None,
    )
    monkeypatch.setattr(
        "scenario_analysis.synthetic_drifts.mlflow.set_experiment",
        lambda name: None,
    )

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(
        "scenario_analysis.synthetic_drifts.mlflow.start_run",
        lambda run_name: DummyRun(),
    )
    monkeypatch.setattr(
        "scenario_analysis.synthetic_drifts.mlflow.log_param",
        lambda key, value: None,
    )
    monkeypatch.setattr(
        "scenario_analysis.synthetic_drifts.mlflow.log_metric",
        lambda key, value: None,
    )
    monkeypatch.setattr(
        "scenario_analysis.synthetic_drifts.mlflow.set_tag",
        lambda key, value: None,
    )
    monkeypatch.setattr(
        "scenario_analysis.synthetic_drifts.mlflow.log_artifact",
        lambda path, artifact_path: logged_artifacts.append((path, artifact_path)),
    )

    result = generate_and_log_synthetic_batch("baseline_like", config)

    assert result.row_count == EXPECTED_BATCH_LINES
    assert result.output_path.exists()
    manifest_path = result.output_path.with_name(
        f"{result.output_path.stem}_manifest.json"
    )
    report_path = result.output_path.with_name(f"{result.output_path.stem}_report.html")
    assert manifest_path.exists()
    assert report_path.exists()
    assert (
        len(result.output_path.read_text(encoding="utf-8").splitlines())
        == EXPECTED_BATCH_LINES
    )
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["scenario_name"] == "baseline_like"
    assert manifest_payload["report_path"] == str(report_path)
    assert logged_artifacts
