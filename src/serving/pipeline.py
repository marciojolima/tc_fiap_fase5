from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from common.config_loader import (
    DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
    load_global_config,
    load_training_experiment_config,
)
from common.logger import get_logger
from serving.schemas import ChurnPredictionRequest

logger = get_logger("serving.pipeline")


@dataclass(frozen=True)
class ServingConfig:
    """Configuração necessária para inferência em produção."""

    target_col: str
    leakage_columns: list[str]
    drop_columns: list[str]
    governed_columns: list[str]
    model_path: Path
    feature_pipeline_path: Path
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
        drop_columns=global_config["data"]["drop_columns"],
        governed_columns=global_config["features"].get("governed_columns", []),
        model_path=Path(experiment_config["artifacts"]["model_path"]),
        feature_pipeline_path=Path("artifacts/feature_pipeline.joblib"),
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


def load_feature_pipeline(feature_pipeline_path: Path | None = None) -> Pipeline:
    """Carrega o pipeline completo de transformação de features."""

    path = feature_pipeline_path or load_serving_config().feature_pipeline_path
    return _load_artifact(str(path))


def load_preprocessor(feature_pipeline_path: Path | None = None) -> Pipeline:
    """Compatibilidade retroativa com o antigo nome do carregador."""

    return load_feature_pipeline(feature_pipeline_path)


# Compatibilidade com testes e chamadas legadas que limpam cache diretamente
# nas funções públicas de carregamento.
load_prediction_model.cache_clear = _load_artifact.cache_clear
load_feature_pipeline.cache_clear = _load_artifact.cache_clear
load_preprocessor.cache_clear = _load_artifact.cache_clear


def build_inference_input_dataframe(payload: ChurnPredictionRequest) -> pd.DataFrame:
    """Converte o payload validado em um DataFrame bruto de inferência."""

    return pd.DataFrame([payload.model_dump(by_alias=True)])


def prepare_inference_dataframe(
    payload: ChurnPredictionRequest,
    cfg: ServingConfig,
) -> pd.DataFrame:
    """Prepara o registro de inferência usando o mesmo pipeline do treino."""

    inference_input_dataframe = build_inference_input_dataframe(payload)
    feature_pipeline = load_feature_pipeline(cfg.feature_pipeline_path)
    transformed_features = feature_pipeline.transform(inference_input_dataframe)

    logger.info(
        "LGPD: inferência preparada sem identificadores diretos; colunas vedadas por "
        "política: %s",
        cfg.drop_columns,
    )
    governed_columns_in_payload = [
        column_name
        for column_name in cfg.governed_columns
        if column_name in inference_input_dataframe.columns
    ]
    if governed_columns_in_payload:
        logger.info(
            "LGPD: colunas utilizadas sob governança para predição em produção: %s",
            governed_columns_in_payload,
        )

    return transformed_features


def predict_from_dataframe(transformed_features: pd.DataFrame) -> tuple[float, int]:
    """Aplica o modelo já carregado sobre features transformadas."""

    cfg = load_serving_config()
    return predict_from_dataframe_with_config(transformed_features, cfg)


def predict_from_dataframe_with_config(
    transformed_features: pd.DataFrame,
    cfg: ServingConfig,
) -> tuple[float, int]:
    """Aplica o modelo para obter a predição final a partir de features prontas."""

    try:
        model = load_prediction_model(cfg.model_path)
    except TypeError:
        model = load_prediction_model()

    threshold = cfg.threshold
    probability = float(model.predict_proba(transformed_features)[0][1])
    prediction = int(probability >= threshold)
    return probability, prediction
