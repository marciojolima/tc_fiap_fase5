import pytest

from model_lifecycle.business_metrics import (
    BusinessMetricsEvaluator,
    PrecisionAtTopK,
    RecallAtTopK,
)

EXPECTED_HALF_RECALL = 0.5
EXPECTED_TOP_K = 0.2
EXPECTED_SMALL_TOP_K = 0.01


def test_business_metrics_evaluator_returns_expected_metrics_for_top_20() -> None:
    evaluator = BusinessMetricsEvaluator(
        metrics=(
            RecallAtTopK(top_k=EXPECTED_TOP_K, target=0.7),
            PrecisionAtTopK(top_k=EXPECTED_TOP_K, target=0.35),
        )
    )

    metrics = evaluator.evaluate(
        y_true=[1, 0, 1, 0, 0],
        y_score=[0.95, 0.30, 0.90, 0.20, 0.10],
    )

    assert metrics["churn_recall_top"] == EXPECTED_HALF_RECALL
    assert metrics["churn_precision_top"] == 1.0
    assert metrics["retention_recall_k"] == EXPECTED_TOP_K
    assert metrics["retention_precision_k"] == EXPECTED_TOP_K


def test_recall_at_top_k_returns_zero_when_dataset_is_empty() -> None:
    metric = RecallAtTopK(top_k=EXPECTED_TOP_K, target=0.7)

    result = metric.evaluate(y_true=[], y_score=[])

    assert result["churn_recall_top"] == 0.0
    assert result["retention_recall_k"] == EXPECTED_TOP_K


def test_precision_at_top_k_returns_zero_when_dataset_is_empty() -> None:
    metric = PrecisionAtTopK(top_k=EXPECTED_TOP_K, target=0.35)

    result = metric.evaluate(y_true=[], y_score=[])

    assert result["churn_precision_top"] == 0.0
    assert result["retention_precision_k"] == EXPECTED_TOP_K


def test_recall_at_top_k_returns_zero_when_no_positive_exists() -> None:
    metric = RecallAtTopK(top_k=0.4, target=0.7)

    result = metric.evaluate(y_true=[0, 0, 0], y_score=[0.9, 0.2, 0.1])

    assert result["churn_recall_top"] == 0.0


def test_precision_at_top_k_handles_minimum_top_count_of_one() -> None:
    metric = PrecisionAtTopK(top_k=EXPECTED_SMALL_TOP_K, target=0.35)

    result = metric.evaluate(y_true=[1, 0, 0], y_score=[0.8, 0.6, 0.5])

    assert result["churn_precision_top"] == 1.0


def test_business_metric_raises_for_invalid_top_k() -> None:
    metric = RecallAtTopK(top_k=0.0, target=0.7)

    with pytest.raises(ValueError, match="top_k"):
        metric.evaluate(y_true=[1], y_score=[0.9])


def test_business_metric_raises_for_mismatched_input_sizes() -> None:
    metric = PrecisionAtTopK(top_k=EXPECTED_TOP_K, target=0.35)

    with pytest.raises(ValueError, match="mesmo tamanho"):
        metric.evaluate(y_true=[1, 0], y_score=[0.9])
