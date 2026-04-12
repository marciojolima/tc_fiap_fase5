from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from monitoring.drift import (
    calculate_numeric_psi,
    decide_drift_status,
    load_dataset,
    run_drift_monitoring,
)
from monitoring.inference_log import (
    append_inference_log,
    build_inference_log_record,
)
from serving.schemas import ChurnPredictionRequest


def test_build_inference_log_record_preserves_payload_aliases() -> None:
    payload = ChurnPredictionRequest(
        CreditScore=650,
        Geography="France",
        Gender="Female",
        Age=35,
        Tenure=5,
        Balance=2000.0,
        NumOfProducts=2,
        HasCrCard=1,
        IsActiveMember=1,
        EstimatedSalary=60000.0,
        **{"Card Type": "GOLD", "Point Earned": 200},
    )

    record = build_inference_log_record(
        payload=payload,
        probability=0.82,
        prediction=1,
        model_name="random_forest_current",
        model_version="0.2.0",
        threshold=0.5,
    )

    assert record["churn_probability"] == 0.82  # noqa: PLR2004
    assert record["churn_prediction"] == 1
    assert record["model_version"] == "0.2.0"
    assert record["Card Type"] == "GOLD"
    assert record["Point Earned"] == 200  # noqa: PLR2004
    assert "timestamp" in record


def test_append_inference_log_writes_json_lines(tmp_path) -> None:
    log_path = tmp_path / "predictions.jsonl"

    append_inference_log({"CreditScore": 650}, log_path)
    append_inference_log({"CreditScore": 700}, log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["CreditScore"] for line in lines] == [650, 700]


def test_decide_drift_status_marks_critical_for_high_feature_psi() -> None:
    decision = decide_drift_status(
        psi_by_feature={"Age": 0.25, "Balance": 0.03},
        warning_threshold=0.10,
        critical_threshold=0.20,
    )

    assert decision.status == "critical"
    assert decision.retraining_recommended is True
    assert decision.drifted_features == ["Age"]


def test_calculate_numeric_psi_detects_distribution_change() -> None:
    reference = pd.Series([30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
    current = pd.Series([70, 71, 72, 73, 74, 75, 76, 77, 78, 79])

    assert calculate_numeric_psi(reference, current) > 0.20  # noqa: PLR2004


def test_load_dataset_raises_clear_error_for_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "predictions.jsonl"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset(missing_path)

    assert "Dataset de monitoramento não encontrado" in str(exc_info.value)
    assert "mldriftdemo" in str(exc_info.value)


def test_run_drift_monitoring_writes_metrics_and_retraining_placeholder(
    tmp_path,
    monkeypatch,
) -> None:
    feature_columns = ["Age", "Balance"]
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    feature_columns_path = tmp_path / "feature_columns.json"
    report_path = tmp_path / "reports" / "drift_report.html"
    metrics_path = tmp_path / "reports" / "drift_metrics.json"
    status_path = tmp_path / "artifacts" / "monitoring" / "drift_status.json"
    retrain_path = tmp_path / "artifacts" / "retraining" / "retrain_request.json"
    retrain_run_path = tmp_path / "artifacts" / "retraining" / "retrain_run.json"
    config_path = tmp_path / "monitoring_config.yaml"

    pd.DataFrame(
        {
            "Age": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            "Balance": [100.0] * 10,
            "Exited": [0, 1] * 5,
        }
    ).to_parquet(reference_path, index=False)
    pd.DataFrame(
        {
            "Age": [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            "Balance": [100.0] * 10,
        }
    ).to_parquet(current_path, index=False)
    feature_columns_path.write_text(
        json.dumps(feature_columns),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "drift": {
                    "enabled": True,
                    "reference_data_path": str(reference_path),
                    "current_data_path": str(current_path),
                    "feature_columns_path": str(feature_columns_path),
                    "feature_pipeline_path": str(tmp_path / "feature_pipeline.joblib"),
                    "model_path": str(tmp_path / "model.pkl"),
                    "report_html_path": str(report_path),
                    "metrics_json_path": str(metrics_path),
                    "status_path": str(status_path),
                    "data_drift": {
                        "enabled": True,
                        "warning_threshold": 0.10,
                        "critical_threshold": 0.20,
                    },
                    "prediction_drift": {"enabled": False},
                    "retraining": {
                        "enabled": True,
                        "trigger_mode": "manual",
                        "training_config_path": "configs/training/model_current.yaml",
                        "request_path": str(retrain_path),
                        "run_path": str(retrain_run_path),
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    def write_dummy_report(reference_features, current_features, output_path) -> None:
        assert list(reference_features.columns) == feature_columns
        assert list(current_features.columns) == feature_columns
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(
        "monitoring.drift.build_evidently_report",
        write_dummy_report,
    )

    decision = run_drift_monitoring(config_path=str(config_path))

    assert decision.status == "critical"
    assert report_path.exists()
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["status"] == "critical"
    retrain_payload = json.loads(retrain_path.read_text(encoding="utf-8"))
    assert retrain_payload["status"] == "requested"
    assert retrain_payload["trigger_mode"] == "manual"
    assert (
        retrain_payload["training_config_path"]
        == "configs/training/model_current.yaml"
    )
    assert retrain_payload["promotion_policy"] == "manual_approval_required"


def test_run_drift_monitoring_executes_retraining_for_auto_mode(
    tmp_path,
    monkeypatch,
) -> None:
    feature_columns = ["Age", "Balance"]
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    feature_columns_path = tmp_path / "feature_columns.json"
    report_path = tmp_path / "reports" / "drift_report.html"
    metrics_path = tmp_path / "reports" / "drift_metrics.json"
    status_path = tmp_path / "artifacts" / "monitoring" / "drift_status.json"
    retrain_path = tmp_path / "artifacts" / "retraining" / "retrain_request.json"
    retrain_run_path = tmp_path / "artifacts" / "retraining" / "retrain_run.json"
    config_path = tmp_path / "monitoring_config.yaml"
    retraining_calls: list[tuple[str, str]] = []

    pd.DataFrame(
        {
            "Age": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            "Balance": [100.0] * 10,
            "Exited": [0, 1] * 5,
        }
    ).to_parquet(reference_path, index=False)
    pd.DataFrame(
        {
            "Age": [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            "Balance": [100.0] * 10,
        }
    ).to_parquet(current_path, index=False)
    feature_columns_path.write_text(
        json.dumps(feature_columns),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "drift": {
                    "enabled": True,
                    "reference_data_path": str(reference_path),
                    "current_data_path": str(current_path),
                    "feature_columns_path": str(feature_columns_path),
                    "feature_pipeline_path": str(tmp_path / "feature_pipeline.joblib"),
                    "model_path": str(tmp_path / "model.pkl"),
                    "report_html_path": str(report_path),
                    "metrics_json_path": str(metrics_path),
                    "status_path": str(status_path),
                    "data_drift": {
                        "enabled": True,
                        "warning_threshold": 0.10,
                        "critical_threshold": 0.20,
                    },
                    "prediction_drift": {"enabled": False},
                    "retraining": {
                        "enabled": True,
                        "trigger_mode": "auto_train_manual_promote",
                        "training_config_path": "configs/training/model_current.yaml",
                        "request_path": str(retrain_path),
                        "run_path": str(retrain_run_path),
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "monitoring.drift.build_evidently_report",
        lambda **_: report_path.parent.mkdir(parents=True, exist_ok=True)
        or report_path.write_text("<html></html>", encoding="utf-8"),
    )
    monkeypatch.setattr(
        "monitoring.drift.run_retraining_request",
        lambda request_path, output_path: retraining_calls.append(
            (request_path, output_path)
        ),
    )

    decision = run_drift_monitoring(config_path=str(config_path))

    assert decision.status == "critical"
    assert retraining_calls == [(str(retrain_path), str(retrain_run_path))]
