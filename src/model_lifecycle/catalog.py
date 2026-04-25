"""Catálogo central de algoritmos suportados pelo projeto."""

from __future__ import annotations

import importlib
from typing import Any, Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_CATALOG: dict[str, Callable[..., Any]] = {
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def build_model(algorithm: str, params: dict[str, Any]) -> Any:
    """Instancia um modelo a partir do nome lógico do algoritmo."""

    if algorithm == "xgboost":
        try:
            xgboost_module = importlib.import_module("xgboost")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "O algoritmo 'xgboost' requer a dependência opcional "
                "'xgboost'. Execute `poetry install` após atualizar o ambiente."
            ) from exc

        return xgboost_module.XGBClassifier(**params)

    try:
        model_builder = MODEL_CATALOG[algorithm]
    except KeyError as exc:
        supported_algorithms = ", ".join(sorted(MODEL_CATALOG))
        raise ValueError(
            f"Algoritmo '{algorithm}' não suportado. "
            f"Disponíveis: {supported_algorithms}"
        ) from exc

    return model_builder(**params)
