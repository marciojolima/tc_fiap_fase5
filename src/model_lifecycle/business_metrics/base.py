"""Contratos base para métricas de negócio orientadas a churn."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class BusinessMetric(Protocol):
    """Contrato mínimo de uma métrica de negócio calculada por ranking."""

    metric_name: str
    top_k: float
    target: float

    def evaluate(
        self,
        y_true: NDArray[np.float64] | NDArray[np.int64],
        y_score: NDArray[np.float64],
    ) -> dict[str, float]:
        """Calcula a métrica e retorna payload pronto para MLflow."""
