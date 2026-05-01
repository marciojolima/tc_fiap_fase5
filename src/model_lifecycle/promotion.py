"""Comparação champion vs challenger para decisão auditável de promoção."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

DEFAULT_PROMOTION_CRITERIA = "criteria_guardrails_plus_score"
DEFAULT_PRIMARY_METRIC = "auc"
DEFAULT_MINIMUM_IMPROVEMENT = 0.005
DEFAULT_METRIC_WEIGHTS = {
    "recall": 0.35,
    "precision": 0.25,
    "f1": 0.25,
    "auc": 0.15,
}
DEFAULT_METRIC_GUARDRAILS = {
    "recall": -0.02,
    "precision": -0.02,
}
SUPPORTED_PROMOTION_CRITERIA = {
    "criteria_best_single_metric",
    "criteria_best_general",
    "criteria_guardrails_plus_score",
}

DEFAULT_PROMOTION_DECISION_PATH = (
    "artifacts/evaluation/model/retraining/promotion_decision.json"
)


class PromotionRule(NamedTuple):
    """Regra mínima para avaliar um challenger contra o champion."""

    criteria: str
    primary_metric: str
    minimum_improvement: float
    metric_weights: dict[str, float]
    metric_guardrails: dict[str, float]


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
    default_criteria = (
        "criteria_best_single_metric"
        if "primary_metric" in promotion_rules
        and "metric_weights" not in promotion_rules
        and "metric_guardrails" not in promotion_rules
        else DEFAULT_PROMOTION_CRITERIA
    )
    criteria = str(promotion_rules.get("criteria", default_criteria))
    if criteria not in SUPPORTED_PROMOTION_CRITERIA:
        supported_criteria = ", ".join(sorted(SUPPORTED_PROMOTION_CRITERIA))
        raise ValueError(
            f"Critério de promoção '{criteria}' não suportado. "
            f"Disponíveis: {supported_criteria}"
        )

    return PromotionRule(
        criteria=criteria,
        primary_metric=str(
            promotion_rules.get("primary_metric", DEFAULT_PRIMARY_METRIC)
        ),
        minimum_improvement=float(
            promotion_rules.get("minimum_improvement", DEFAULT_MINIMUM_IMPROVEMENT)
        ),
        metric_weights=_normalize_metric_weights(
            promotion_rules.get("metric_weights")
        ),
        metric_guardrails=_normalize_metric_guardrails(
            promotion_rules.get("metric_guardrails")
        ),
    )


def _normalize_metric_weights(weights: Any) -> dict[str, float]:
    """Normaliza pesos de métricas para score composto."""

    raw_weights = weights or DEFAULT_METRIC_WEIGHTS
    normalized = {str(key): float(value) for key, value in dict(raw_weights).items()}
    if not normalized:
        raise ValueError("metric_weights não pode ser vazio.")

    invalid_weights = {
        metric_name: weight
        for metric_name, weight in normalized.items()
        if weight < 0
    }
    if invalid_weights:
        raise ValueError(
            "metric_weights deve conter apenas valores não negativos: "
            f"{invalid_weights}"
        )
    return normalized


def _normalize_metric_guardrails(guardrails: Any) -> dict[str, float]:
    """Normaliza limites mínimos de delta aceitos para métricas críticas."""

    raw_guardrails = (
        guardrails
        if guardrails is not None
        else DEFAULT_METRIC_GUARDRAILS
    )
    return {
        str(metric_name): float(minimum_delta)
        for metric_name, minimum_delta in dict(raw_guardrails).items()
    }


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


def _require_metric(metrics: dict[str, float], metric_name: str) -> float:
    """Retorna uma métrica obrigatória ou falha com mensagem amigável."""

    try:
        return float(metrics[metric_name])
    except KeyError as exc:
        available_metrics = ", ".join(sorted(metrics))
        raise KeyError(
            f"Métrica obrigatória '{metric_name}' ausente. "
            f"Disponíveis: {available_metrics}"
        ) from exc


def _compute_weighted_score(
    metrics: dict[str, float],
    metric_weights: dict[str, float],
) -> float:
    """Calcula score composto normalizado pelas métricas ponderadas."""

    weighted_metrics = {
        metric_name: weight
        for metric_name, weight in metric_weights.items()
        if weight > 0
    }
    if not weighted_metrics:
        raise ValueError("Ao menos um peso positivo é necessário para score composto.")

    total_weight = sum(weighted_metrics.values())
    score = sum(
        _require_metric(metrics, metric_name) * weight
        for metric_name, weight in weighted_metrics.items()
    )
    return float(score / total_weight)


def _evaluate_metric_guardrails(
    metric_deltas: dict[str, float],
    metric_guardrails: dict[str, float],
) -> tuple[bool, dict[str, dict[str, float | bool]]]:
    """Avalia se os deltas respeitam os limites mínimos definidos."""

    results: dict[str, dict[str, float | bool]] = {}
    for metric_name, minimum_delta in metric_guardrails.items():
        observed_delta = _require_metric(metric_deltas, metric_name)
        passed = observed_delta >= minimum_delta
        results[metric_name] = {
            "minimum_delta": float(minimum_delta),
            "observed_delta": float(observed_delta),
            "passed": passed,
        }
    return all(result["passed"] for result in results.values()), results


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
    champion_metric = _require_metric(champion_metrics, metric_name)
    challenger_metric = _require_metric(challenger_metrics, metric_name)
    metric_delta = float(challenger_metric - champion_metric)

    champion_weighted_score = _compute_weighted_score(
        champion_metrics,
        rule.metric_weights,
    )
    challenger_weighted_score = _compute_weighted_score(
        challenger_metrics,
        rule.metric_weights,
    )
    weighted_score_delta = float(challenger_weighted_score - champion_weighted_score)
    guardrails_passed, guardrail_results = _evaluate_metric_guardrails(
        metric_deltas,
        rule.metric_guardrails,
    )

    if rule.criteria == "criteria_best_single_metric":
        eligible_for_promotion = metric_delta >= rule.minimum_improvement
        reason = (
            f"challenger_{metric_name}_delta_meets_threshold"
            if eligible_for_promotion
            else f"challenger_{metric_name}_delta_below_threshold"
        )
    elif rule.criteria == "criteria_best_general":
        eligible_for_promotion = weighted_score_delta >= rule.minimum_improvement
        reason = (
            "challenger_weighted_score_delta_meets_threshold"
            if eligible_for_promotion
            else "challenger_weighted_score_delta_below_threshold"
        )
    else:
        eligible_for_promotion = (
            guardrails_passed and weighted_score_delta >= rule.minimum_improvement
        )
        if not guardrails_passed:
            reason = "challenger_failed_metric_guardrails"
        elif eligible_for_promotion:
            reason = "challenger_guardrails_passed_and_score_delta_meets_threshold"
        else:
            reason = "challenger_guardrails_passed_but_score_delta_below_threshold"

    return {
        "request_id": request_id,
        "status": "eligible" if eligible_for_promotion else "rejected",
        "eligible_for_promotion": eligible_for_promotion,
        "recommended_action": (
            "manual_review_for_promotion"
            if eligible_for_promotion
            else "retain_current_champion"
        ),
        "reason": reason,
        "promotion_rule": {
            "criteria": rule.criteria,
            "primary_metric": metric_name,
            "minimum_improvement": rule.minimum_improvement,
            "metric_weights": rule.metric_weights,
            "metric_guardrails": rule.metric_guardrails,
        },
        "evaluation_summary": {
            "primary_metric": {
                "name": metric_name,
                "champion_value": champion_metric,
                "challenger_value": challenger_metric,
                "delta": metric_delta,
            },
            "weighted_score": {
                "champion_value": champion_weighted_score,
                "challenger_value": challenger_weighted_score,
                "delta": weighted_score_delta,
            },
            "guardrails_passed": guardrails_passed,
            "guardrail_results": guardrail_results,
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
