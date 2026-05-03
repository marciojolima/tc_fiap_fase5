"""Implementação de recall@top-k para o contexto de churn."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from model_lifecycle.business_metrics.utils import prepare_business_metric_inputs


class RecallAtTopK:
    """Mede cobertura de churners reais dentro do top-k por risco previsto."""

    metric_name = "churn_recall_top"

    def __init__(self, *, top_k: float, target: float) -> None:
        self.top_k = float(top_k)
        self.target = float(target)

    def evaluate(
        self,
        y_true: ArrayLike,
        y_score: ArrayLike,
    ) -> dict[str, float]:
        y_true_array, _, top_mask, _ = prepare_business_metric_inputs(
            y_true,
            y_score,
            top_k=self.top_k,
        )
        positive_count = int(np.count_nonzero(y_true_array))
        positives_in_top = int(np.count_nonzero(y_true_array & top_mask))
        metric_value = (
            float(positives_in_top / positive_count)
            if positive_count > 0
            else 0.0
        )
        return {self.metric_name: metric_value}
