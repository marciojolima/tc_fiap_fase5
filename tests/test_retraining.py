from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.retraining import load_retraining_request, run_retraining_request
from models.train import ExperimentTrainingConfig

EXPECTED_RETRAIN_AUC = 0.91
EXPECTED_REFERENCE_ROW_COUNT = 8000
EXPECTED_CURRENT_ROW_COUNT = 4


def build_request_payload() -> dict[str, object]:
    return {
        "request_id": "req-123",
        "status": "requested",
        "reason": "critical_data_or_prediction_drift",
        "model_path": "artifacts/models/model_current.pkl",
        "training_config_path": "configs/training/model_current.yaml",
        "created_at": "2026-04-12T00:00:00+00:00",
        "trigger_mode": "auto_train_manual_promote",
        "promotion_policy": "manual_approval_required",
        "drift_status": "critical",
        "max_feature_psi": 0.25,
        "prediction_psi": 0.14,
        "drifted_features": ["Age"],
        "reference_row_count": EXPECTED_REFERENCE_ROW_COUNT,
        "current_row_count": EXPECTED_CURRENT_ROW_COUNT,
    }


def build_experiment_training_config(model_path: Path) -> ExperimentTrainingConfig:
    return ExperimentTrainingConfig(
        seed=42,
        target_col="Exited",
        test_size=0.2,
        algorithm="random_forest",
        flavor="sklearn",
        experiment_name="random_forest_current",
        run_name="random_forest_current",
        model_version="0.2.0",
        model_params={"n_estimators": 200},
        threshold=0.5,
        feature_set="processed_v1",
        model_path=model_path,
        training_data_version="data-hash-123",
        git_sha="abc123",
        git_tag="post_release_commits",
        git_nearest_tag="v0.2.0",
        risk_level="high",
        fairness_checked=False,
        mlflow_cfg={
            "tracking_uri": "file:./mlruns",
            "experiment_name": "datathon-churn-baseline",
            "tags": {"owner": "team"},
        },
        registry_cfg={"enabled": False},
    )


def test_load_retraining_request_parses_contract(tmp_path: Path) -> None:
    request_path = tmp_path / "retrain_request.json"
    request_path.write_text(
        json.dumps(build_request_payload()),
        encoding="utf-8",
    )

    request = load_retraining_request(request_path)

    assert request.request_id == "req-123"
    assert request.status == "requested"
    assert request.training_config_path == "configs/training/model_current.yaml"
    assert request.drifted_features == ["Age"]
    assert request.reference_row_count == EXPECTED_REFERENCE_ROW_COUNT
    assert request.current_row_count == EXPECTED_CURRENT_ROW_COUNT


def test_run_retraining_request_executes_training_and_writes_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "retrain_request.json"
    output_path = tmp_path / "retrain_run.json"
    request_path.write_text(
        json.dumps(build_request_payload()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "models.retraining.run_training",
        lambda config_path, retraining_context=None: {"auc": 0.91, "f1": 0.80},
    )
    monkeypatch.setattr(
        "models.retraining.load_experiment_training_config",
        lambda config_path: build_experiment_training_config(
            tmp_path / "artifacts" / "models" / "model_current.pkl"
        ),
    )

    result = run_retraining_request(
        request_path=request_path,
        output_path=output_path,
    )

    assert result["status"] == "completed"
    assert result["experiment_name"] == "random_forest_current"
    assert result["metrics"]["auc"] == EXPECTED_RETRAIN_AUC

    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert request_payload["status"] == "completed"
    assert "executed_at" in request_payload

    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_payload["request_id"] == "req-123"
    assert output_payload["status"] == "completed"


def test_run_retraining_request_marks_failure_when_training_breaks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "retrain_request.json"
    output_path = tmp_path / "retrain_run.json"
    request_path.write_text(
        json.dumps(build_request_payload()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "models.retraining.run_training",
        lambda config_path, retraining_context=None: (_ for _ in ()).throw(
            RuntimeError("falha no treino")
        ),
    )

    with pytest.raises(RuntimeError, match="falha no treino"):
        run_retraining_request(
            request_path=request_path,
            output_path=output_path,
        )

    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert request_payload["status"] == "failed"
    assert request_payload["failure_reason"] == "falha no treino"

    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_payload["status"] == "failed"
    assert output_payload["failure_reason"] == "falha no treino"
