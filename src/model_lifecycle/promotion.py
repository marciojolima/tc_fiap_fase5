"""Comparação champion vs challenger para decisão auditável de promoção."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

DEFAULT_PROMOTION_DECISION_PATH = (
    "artifacts/evaluation/model/retraining/promotion_decision.json"
)


class PromotionRule(NamedTuple):
    """Regra mínima para avaliar um challenger contra o champion."""

    primary_metric: str
    minimum_improvement: float


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Persiste um payload JSON garantindo criação do diretório pai."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_model_metadata(path: str | Path) -> dict[str, Any]:
    """Carrega o sidecar JSON com metadados persistidos do modelo."""

    metadata_path = Path(path)
    with open(metadata_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def resolve_promotion_rule(
    rules: dict[str, Any] | None,
) -> PromotionRule:
    """Normaliza a política mínima de promoção do challenger."""

    promotion_rules = rules or {}
    return PromotionRule(
        primary_metric=str(promotion_rules.get("primary_metric", "auc")),
        minimum_improvement=float(
            promotion_rules.get("minimum_improvement", 0.005)
        ),
    )


def build_metric_deltas(
    champion_metrics: dict[str, float],
    challenger_metrics: dict[str, float],
) -> dict[str, float]:
    """Calcula os deltas entre challenger e champion para métricas conhecidas."""

    shared_metrics = champion_metrics.keys() & challenger_metrics.keys()
    return {
        metric_name: float(
            challenger_metrics[metric_name] - champion_metrics[metric_name]
        )
        for metric_name in sorted(shared_metrics)
    }


def build_promotion_decision_payload(
    *,
    request_id: str,
    champion_metadata: dict[str, Any],
    challenger_metadata: dict[str, Any],
    rule: PromotionRule,
) -> dict[str, Any]:
    """Monta uma decisão auditável de elegibilidade do challenger."""

    champion_metrics = champion_metadata["metrics"]
    challenger_metrics = challenger_metadata["metrics"]
    metric_deltas = build_metric_deltas(champion_metrics, challenger_metrics)
    metric_name = rule.primary_metric
    champion_metric = float(champion_metrics[metric_name])
    challenger_metric = float(challenger_metrics[metric_name])
    metric_delta = float(challenger_metric - champion_metric)
    eligible_for_promotion = metric_delta >= rule.minimum_improvement

    return {
        "request_id": request_id,
        "status": "eligible" if eligible_for_promotion else "rejected",
        "eligible_for_promotion": eligible_for_promotion,
        "recommended_action": (
            "manual_review_for_promotion"
            if eligible_for_promotion
            else "retain_current_champion"
        ),
        "reason": (
            f"challenger_{metric_name}_delta_meets_threshold"
            if eligible_for_promotion
            else f"challenger_{metric_name}_delta_below_threshold"
        ),
        "promotion_rule": {
            "primary_metric": metric_name,
            "minimum_improvement": rule.minimum_improvement,
        },
        "champion": {
            "experiment_name": champion_metadata["experiment_name"],
            "model_version": champion_metadata["model_version"],
            "model_path": champion_metadata["model_path"],
            "metrics": champion_metrics,
        },
        "challenger": {
            "experiment_name": challenger_metadata["experiment_name"],
            "model_version": challenger_metadata["model_version"],
            "model_path": challenger_metadata["model_path"],
            "metrics": challenger_metrics,
        },
        "metric_deltas": metric_deltas,
    }


def evaluate_challenger_promotion(
    *,
    request_id: str,
    champion_metadata_path: str | Path,
    challenger_metadata_path: str | Path,
    output_path: str | Path = DEFAULT_PROMOTION_DECISION_PATH,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compara champion e challenger e persiste a decisão de promoção."""

    payload = build_promotion_decision_payload(
        request_id=request_id,
        champion_metadata=load_model_metadata(champion_metadata_path),
        challenger_metadata=load_model_metadata(challenger_metadata_path),
        rule=resolve_promotion_rule(rules),
    )
    write_json(output_path, payload)
    return payload
