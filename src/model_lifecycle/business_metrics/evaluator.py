"""Agregador de métricas de negócio calculadas durante o treino."""

from __future__ import annotations

from collections.abc import Iterable

from numpy.typing import ArrayLike

from model_lifecycle.business_metrics.base import BusinessMetric


class BusinessMetricsEvaluator:
    """Executa um conjunto de métricas e agrega saída pronta para MLflow."""

    def __init__(self, metrics: Iterable[BusinessMetric]) -> None:
        self.metrics = tuple(metrics)

    def evaluate(
        self,
        y_true: ArrayLike,
        y_score: ArrayLike,
    ) -> dict[str, float]:
        aggregated_metrics: dict[str, float] = {}
        for metric in self.metrics:
            aggregated_metrics.update(metric.evaluate(y_true, y_score))
        return aggregated_metrics
