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


def build_metadata(
    *,
    model_path: str,
    model_version: str,
    auc: float,
    f1: float,
) -> dict[str, object]:
    return {
        "experiment_name": "current",
        "model_version": model_version,
        "model_path": model_path,
        "metrics": {
            "auc": auc,
            "f1": f1,
            "precision": 0.63,
            "recall": 0.64,
            "accuracy": 0.85,
        },
    }


def test_build_promotion_decision_payload_marks_eligible_when_auc_improves() -> None:
    payload = build_promotion_decision_payload(
        request_id="req-123",
        champion_metadata=build_metadata(
            model_path="artifacts/models/current.pkl",
            model_version="0.2.0",
            auc=0.87,
            f1=0.64,
        ),
        challenger_metadata=build_metadata(
            model_path="artifacts/models/challengers/current_req123.pkl",
            model_version="0.2.0-challenger-req123",
            auc=0.88,
            f1=0.65,
        ),
        rule=resolve_promotion_rule(
            {"primary_metric": "auc", "minimum_improvement": 0.005}
        ),
    )

    assert payload["status"] == "eligible"
    assert payload["eligible_for_promotion"] is True
    assert payload["metric_deltas"]["auc"] == pytest.approx(EXPECTED_AUC_DELTA)


def test_evaluate_challenger_promotion_persists_rejection_decision(
    tmp_path: Path,
) -> None:
    champion_metadata_path = tmp_path / "champion_metadata.json"
    challenger_metadata_path = tmp_path / "challenger_metadata.json"
    output_path = tmp_path / "promotion_decision.json"

    champion_metadata_path.write_text(
        json.dumps(
            build_metadata(
                model_path="artifacts/models/current.pkl",
                model_version="0.2.0",
                auc=0.87,
                f1=0.64,
            )
        ),
        encoding="utf-8",
    )
    challenger_metadata_path.write_text(
        json.dumps(
            build_metadata(
                model_path="artifacts/models/challengers/current_req123.pkl",
                model_version="0.2.0-challenger-req123",
                auc=0.872,
                f1=0.65,
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
