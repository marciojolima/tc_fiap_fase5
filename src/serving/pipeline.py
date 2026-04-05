from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from joblib import load

from common.config_loader import load_config
from features.feature_engineering import create_features
from serving.schemas import ChurnPredictionRequest


class ServingConfig(NamedTuple):
    """Configuração necessária para inferência em produção."""

    target_col: str
    leakage_columns: list[str]
    model_path: Path
    preprocessor_path: Path
    threshold: float


def load_serving_config() -> ServingConfig:
    """Carrega a configuração usada pela API de inferência."""

    config = load_config()
    return ServingConfig(
        target_col=config["data"]["target_col"],
        leakage_columns=config["features"]["leakage_columns"],
        model_path=Path("artifacts/challenger_model.pkl"),
        preprocessor_path=Path("artifacts/preprocessor.joblib"),
        threshold=0.5,
    )


@lru_cache
def load_prediction_model() -> Any:
    """Carrega o modelo challenger persistido para serving."""

    return load(load_serving_config().model_path)


@lru_cache
def load_preprocessor() -> Any:
    """Carrega o pré-processador persistido para serving."""

    return load(load_serving_config().preprocessor_path)


def prepare_inference_dataframe(
    payload: ChurnPredictionRequest,
    cfg: ServingConfig,
) -> pd.DataFrame:
    """Prepara um único registro bruto para inferência no modelo."""

    df = pd.DataFrame([payload.model_dump(by_alias=True)])
    df_feat = create_features(df)

    leakage_in_features = [c for c in cfg.leakage_columns if c != cfg.target_col]
    existing_leakage = [c for c in leakage_in_features if c in df_feat.columns]
    if existing_leakage:
        df_feat = df_feat.drop(columns=existing_leakage)

    return df_feat


def predict_from_dataframe(df_feat: pd.DataFrame) -> tuple[float, int]:
    """Aplica pré-processamento e modelo para obter a predição final."""

    preprocessor = load_preprocessor()
    model = load_prediction_model()
    threshold = load_serving_config().threshold

    X = preprocessor.transform(df_feat)
    probability = float(model.predict_proba(X)[0][1])
    prediction = int(probability >= threshold)
    return probability, prediction
