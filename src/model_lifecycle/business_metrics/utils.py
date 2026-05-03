"""Utilitários compartilhados para métricas de negócio baseadas em ranking."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def prepare_business_metric_inputs(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    top_k: float,
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.bool_], int]:
    """Normaliza entradas e seleciona o recorte top-k por score."""

    if not 0 < top_k <= 1:
        raise ValueError("top_k deve estar no intervalo (0, 1].")

    y_true_array = np.asarray(y_true, dtype=bool).reshape(-1)
    y_score_array = np.asarray(y_score, dtype=float).reshape(-1)

    if y_true_array.size != y_score_array.size:
        raise ValueError("y_true e y_score devem possuir o mesmo tamanho.")

    sample_count = y_true_array.size
    if sample_count == 0:
        return y_true_array, y_score_array, np.zeros(0, dtype=bool), 0

    top_count = max(int(math.ceil(sample_count * top_k)), 1)
    top_indices = np.argpartition(y_score_array, -top_count)[-top_count:]
    top_mask = np.zeros(sample_count, dtype=bool)
    top_mask[top_indices] = True
    return y_true_array, y_score_array, top_mask, top_count
