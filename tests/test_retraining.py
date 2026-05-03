from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from model_lifecycle.retraining import (
    create_challenger_training_config,
    load_retraining_request,
    run_retraining_request,
)
from model_lifecycle.train import BusinessMetricsConfig, ExperimentTrainingConfig

EXPECTED_RETRAIN_AUC = 0.91
EXPECTED_REFERENCE_ROW_COUNT = 8000
EXPECTED_CURRENT_ROW_COUNT = 4


def build_request_payload() -> dict[str, object]:
    return {
        "request_id": "req-123",
        "status": "requested",
        "reason": "critical_data_or_prediction_drift",
        "model_path": "artifacts/models/current.pkl",
        "training_config_path": "configs/model_lifecycle/current.json",
        "created_at": "2026-04-12T00:00:00+00:00",
        "trigger_mode": "auto_train_manual_promote",
        "promotion_policy": "manual_approval_required",
        "promotion_decision_path": (
            "artifacts/evaluation/model/retraining/promotion_decision.json"
        ),
        "promotion_rules": {
            "criteria": "criteria_guardrails_plus_score",
            "primary_metric": "recall",
            "minimum_improvement": 0.005,
            "metric_weights": {
                "recall": 0.35,
                "precision": 0.25,
                "f1": 0.25,
                "auc": 0.15,
            },
            "metric_guardrails": {
                "recall": -0.02,
                "precision": -0.02,
            },
        },
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
        experiment_name="current",
        run_name="current",
        model_version="0.2.0",
        model_params={"n_estimators": 200},
        threshold=0.5,
        feature_set="processed_v1",
        feature_service_name="customer_churn_rf_v2",
        model_path=model_path,
        training_data_version="data-hash-123",
        git_sha="abc123",
        git_tag="post_release_commits",
        git_nearest_tag="v0.2.0",
        risk_level="high",
        fairness_checked=False,
        business_metrics=BusinessMetricsConfig(
            recall_top_k=0.2,
            recall_target=0.7,
            precision_top_k=0.2,
            precision_target=0.35,
        ),
        mlflow_cfg={
            "tracking_uri": "sqlite:///mlruns/mlflow.db",
            "experiment_name": "datathon-churn-baseline",
            "tags": {"owner": "team"},
        },
        registry_cfg={"enabled": False},
    )


def write_request_with_config_path(tmp_path: Path, config_path: str) -> Path:
    request_path = tmp_path / "retrain_request.json"
    payload = build_request_payload()
    payload["training_config_path"] = config_path
    request_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return request_path


def test_load_retraining_request_parses_contract(tmp_path: Path) -> None:
    request_path = tmp_path / "retrain_request.json"
    request_path.write_text(
        json.dumps(build_request_payload()),
        encoding="utf-8",
    )

    request = load_retraining_request(request_path)

    assert request.request_id == "req-123"
    assert request.status == "requested"
    assert request.training_config_path == "configs/model_lifecycle/current.json"
    assert request.drifted_features == ["Age"]
    assert request.promotion_rules["criteria"] == "criteria_guardrails_plus_score"
    assert request.promotion_rules["primary_metric"] == "recall"
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
    generated_config_path = tmp_path / "generated_retrain.yaml"
    generated_config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "current",
                    "run_name": "current_challenger_req123",
                    "version": "0.2.0-challenger-req123",
                    "algorithm": "random_forest",
                    "flavor": "sklearn",
                },
                "dataset": {
                    "target_col": "Exited",
                    "feature_set": "processed_v1",
                },
                "training": {"params": {"n_estimators": 200}},
                "inference": {"threshold": 0.5},
                "feast": {"feature_service_name": "customer_churn_rf_v2"},
                "artifacts": {
                    "model_path": str(
                        tmp_path
                        / "artifacts"
                        / "models"
                        / "challengers"
                        / "current_req123.pkl"
                    )
                },
                "mlflow": {"tags": {"candidate_type": "challenger"}},
                "registry": {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "model_lifecycle.retraining.run_training",
        lambda config_path, retraining_context=None: {"auc": 0.91, "f1": 0.80},
    )
    monkeypatch.setattr(
        "model_lifecycle.retraining.create_challenger_training_config",
        lambda request: str(generated_config_path),
    )
    monkeypatch.setattr(
        "model_lifecycle.retraining.load_experiment_training_config",
        lambda config_path: build_experiment_training_config(
            tmp_path / "artifacts" / "models" / "challengers" / "current_req123.pkl"
        ),
    )
    monkeypatch.setattr(
        "model_lifecycle.retraining.evaluate_challenger_promotion",
        lambda **_: {
            "status": "eligible",
            "eligible_for_promotion": True,
            "recommended_action": "manual_review_for_promotion",
        },
    )

    result = run_retraining_request(
        request_path=request_path,
        output_path=output_path,
    )

    assert result["status"] == "completed"
    assert result["experiment_name"] == "current"
    assert result["metrics"]["auc"] == EXPECTED_RETRAIN_AUC
    assert result["promotion_decision"]["status"] == "eligible"
    assert "challenger_training_config_path" in result

    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert request_payload["status"] == "completed"
    assert "executed_at" in request_payload

    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_payload["request_id"] == "req-123"
    assert output_payload["status"] == "completed"


def test_create_challenger_training_config_preserves_json_format(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "current.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment": {
                    "name": "current",
                    "run_name": "current",
                    "version": "0.2.0",
                    "algorithm": "random_forest",
                    "flavor": "sklearn",
                },
                "dataset": {
                    "target_col": "Exited",
                    "feature_set": "processed_v1",
                },
                "training": {
                    "params": {
                        "n_estimators": 200,
                    }
                },
                "inference": {
                    "threshold": 0.5,
                },
                "feast": {
                    "feature_service_name": "customer_churn_rf_v2",
                },
                "artifacts": {
                    "model_path": "artifacts/models/current.pkl",
                },
                "mlflow": {
                    "tags": {
                        "candidate_type": "current",
                    }
                },
                "registry": {
                    "enabled": False,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    generated_config_path = create_challenger_training_config(
        load_retraining_request(
            write_request_with_config_path(tmp_path, str(config_path))
        )
    )

    assert generated_config_path.endswith(".json")
    generated_payload = json.loads(
        Path(generated_config_path).read_text(encoding="utf-8")
    )
    assert generated_payload["mlflow"]["tags"]["candidate_type"] == "challenger"
    assert (
        "artifacts/models/challengers/"
        in generated_payload["artifacts"]["model_path"]
    )


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
        "model_lifecycle.retraining.run_training",
        lambda config_path, retraining_context=None: (_ for _ in ()).throw(
            RuntimeError("falha no treino")
        ),
    )
    monkeypatch.setattr(
        "model_lifecycle.retraining.create_challenger_training_config",
        lambda request: str(tmp_path / "generated_retrain.json"),
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
