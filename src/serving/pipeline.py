from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from joblib import load

from common.config_loader import (
    DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
    load_global_config,
    load_training_experiment_config,
)
from features.feature_engineering import create_features, normalize_feature_names
from serving.schemas import ChurnPredictionRequest


class ServingConfig(NamedTuple):
    """Configuração necessária para inferência em produção."""

    target_col: str
    leakage_columns: list[str]
    model_path: Path
    preprocessor_path: Path
    threshold: float
    model_name: str
    run_name: str


def build_serving_config(
    experiment_config_path: str = DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
) -> ServingConfig:
    """Carrega a configuração usada pela API de inferência."""

    global_config = load_global_config()
    experiment_config = load_training_experiment_config(experiment_config_path)

    return ServingConfig(
        target_col=global_config["data"]["target_col"],
        leakage_columns=global_config["features"]["leakage_columns"],
        model_path=Path(experiment_config["artifacts"]["model_path"]),
        preprocessor_path=Path("artifacts/preprocessor.joblib"),
        threshold=experiment_config["inference"]["threshold"],
        model_name=experiment_config["experiment"]["name"],
        run_name=experiment_config["experiment"]["run_name"],
    )


@lru_cache
def _load_artifact(path_str: str) -> Any:
    """Carrega um artefato serializado e reutiliza-o por caminho."""

    return load(Path(path_str))


def load_serving_config() -> ServingConfig:
    """Carrega a configuração padrão usada pela API de inferência."""

    return build_serving_config()


def load_prediction_model(model_path: Path | None = None) -> Any:
    """Carrega o modelo ativo persistido para serving."""

    path = model_path or load_serving_config().model_path
    return _load_artifact(str(path))


def load_preprocessor(preprocessor_path: Path | None = None) -> Any:
    """Carrega o pré-processador persistido para serving."""

    path = preprocessor_path or load_serving_config().preprocessor_path
    return _load_artifact(str(path))


# Compatibilidade com testes e chamadas legadas que limpam cache diretamente
# nas funções públicas de carregamento.
load_prediction_model.cache_clear = _load_artifact.cache_clear
load_preprocessor.cache_clear = _load_artifact.cache_clear


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

    cfg = load_serving_config()
    return predict_from_dataframe_with_config(df_feat, cfg)


def predict_from_dataframe_with_config(
    df_feat: pd.DataFrame,
    cfg: ServingConfig,
) -> tuple[float, int]:
    """Aplica pré-processamento e modelo para obter a predição final."""

    try:
        preprocessor = load_preprocessor(cfg.preprocessor_path)
    except TypeError:
        preprocessor = load_preprocessor()

    try:
        model = load_prediction_model(cfg.model_path)
    except TypeError:
        model = load_prediction_model()

    threshold = cfg.threshold

    X_array = preprocessor.transform(df_feat)
    feature_names = normalize_feature_names(
        preprocessor.get_feature_names_out().tolist()
    )
    X = pd.DataFrame(X_array, columns=feature_names, index=df_feat.index)

    probability = float(model.predict_proba(X)[0][1])
    prediction = int(probability >= threshold)
    return probability, prediction
