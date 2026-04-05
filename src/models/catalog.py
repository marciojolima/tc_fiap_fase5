"""Catálogo central de algoritmos sklearn suportados pelo projeto."""

from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_CATALOG: dict[str, Callable[..., Any]] = {
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def build_model(algorithm: str, params: dict[str, Any]) -> Any:
    """Instancia um modelo sklearn a partir do nome lógico do algoritmo."""

    try:
        model_builder = MODEL_CATALOG[algorithm]
    except KeyError as exc:
        supported_algorithms = ", ".join(sorted(MODEL_CATALOG))
        raise ValueError(
            f"Algoritmo '{algorithm}' não suportado. "
            f"Disponíveis: {supported_algorithms}"
        ) from exc

    return model_builder(**params)
