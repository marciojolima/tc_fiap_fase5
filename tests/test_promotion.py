from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lifecycle.promotion import (
    build_promotion_decision_payload,
    evaluate_challenger_promotion,
    resolve_promotion_rule,
)

EXPECTED_AUC_DELTA = 0.01
EXPECTED_WEIGHTED_SCORE_MIN_DELTA = 0.01


def build_metadata(
    metric_values: dict[str, float],
    *,
    model_path: str,
    model_version: str,
) -> dict[str, object]:
    return {
        "experiment_name": "current",
        "model_version": model_version,
        "model_path": model_path,
        "metrics": {"accuracy": 0.85, **metric_values},
    }


def test_build_promotion_decision_payload_marks_eligible_when_auc_improves() -> None:
    payload = build_promotion_decision_payload(
        request_id="req-123",
        champion_metadata=build_metadata(
            {"auc": 0.87, "f1": 0.64, "precision": 0.63, "recall": 0.64},
            model_path="artifacts/models/current.pkl",
            model_version="0.2.0",
        ),
        challenger_metadata=build_metadata(
            {"auc": 0.88, "f1": 0.65, "precision": 0.63, "recall": 0.64},
            model_path="artifacts/models/challengers/current_req123.pkl",
            model_version="0.2.0-challenger-req123",
        ),
        rule=resolve_promotion_rule(
            {"primary_metric": "auc", "minimum_improvement": 0.005}
        ),
    )

    assert payload["status"] == "eligible"
    assert payload["eligible_for_promotion"] is True
    assert payload["metric_deltas"]["auc"] == pytest.approx(EXPECTED_AUC_DELTA)
    assert payload["promotion_rule"]["criteria"] == "criteria_best_single_metric"


def test_build_promotion_decision_payload_marks_eligible_when_weighted_score_improves(
) -> None:
    payload = build_promotion_decision_payload(
        request_id="req-234",
        champion_metadata=build_metadata(
            {"auc": 0.87, "f1": 0.64, "precision": 0.62, "recall": 0.61},
            model_path="artifacts/models/current.pkl",
            model_version="0.2.0",
        ),
        challenger_metadata=build_metadata(
            {"auc": 0.89, "f1": 0.69, "precision": 0.67, "recall": 0.70},
            model_path="artifacts/models/challengers/current_req234.pkl",
            model_version="0.2.0-challenger-req234",
        ),
        rule=resolve_promotion_rule(
            {
                "criteria": "criteria_best_general",
                "minimum_improvement": 0.01,
                "metric_weights": {
                    "recall": 0.4,
                    "precision": 0.2,
                    "f1": 0.2,
                    "auc": 0.2,
                },
            }
        ),
    )

    assert payload["status"] == "eligible"
    assert payload["eligible_for_promotion"] is True
    assert payload["reason"] == "challenger_weighted_score_delta_meets_threshold"
    assert (
        payload["evaluation_summary"]["weighted_score"]["delta"]
        > EXPECTED_WEIGHTED_SCORE_MIN_DELTA
    )


def test_build_promotion_decision_payload_rejects_guardrail_violation() -> None:
    payload = build_promotion_decision_payload(
        request_id="req-345",
        champion_metadata=build_metadata(
            {"auc": 0.87, "f1": 0.64, "precision": 0.63, "recall": 0.64},
            model_path="artifacts/models/current.pkl",
            model_version="0.2.0",
        ),
        challenger_metadata=build_metadata(
            {"auc": 0.90, "f1": 0.70, "precision": 0.65, "recall": 0.58},
            model_path="artifacts/models/challengers/current_req345.pkl",
            model_version="0.2.0-challenger-req345",
        ),
        rule=resolve_promotion_rule(
            {
                "criteria": "criteria_guardrails_plus_score",
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
            }
        ),
    )

    assert payload["status"] == "rejected"
    assert payload["eligible_for_promotion"] is False
    assert payload["reason"] == "challenger_failed_metric_guardrails"
    assert payload["evaluation_summary"]["guardrails_passed"] is False
    assert (
        payload["evaluation_summary"]["guardrail_results"]["recall"]["passed"] is False
    )


def test_evaluate_challenger_promotion_persists_rejection_decision(
    tmp_path: Path,
) -> None:
    champion_metadata_path = tmp_path / "champion_metadata.json"
    challenger_metadata_path = tmp_path / "challenger_metadata.json"
    output_path = tmp_path / "promotion_decision.json"

    champion_metadata_path.write_text(
        json.dumps(
            build_metadata(
                {"auc": 0.87, "f1": 0.64, "precision": 0.63, "recall": 0.64},
                model_path="artifacts/models/current.pkl",
                model_version="0.2.0",
            )
        ),
        encoding="utf-8",
    )
    challenger_metadata_path.write_text(
        json.dumps(
            build_metadata(
                {"auc": 0.872, "f1": 0.65, "precision": 0.63, "recall": 0.64},
                model_path="artifacts/models/challengers/current_req123.pkl",
                model_version="0.2.0-challenger-req123",
            )
        ),
        encoding="utf-8",
    )

    payload = evaluate_challenger_promotion(
        request_id="req-123",
        champion_metadata_path=champion_metadata_path,
        challenger_metadata_path=challenger_metadata_path,
        output_path=output_path,
        rules={"primary_metric": "auc", "minimum_improvement": 0.005},
    )

    assert payload["status"] == "rejected"
    assert payload["recommended_action"] == "retain_current_champion"
    persisted_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted_payload["request_id"] == "req-123"
