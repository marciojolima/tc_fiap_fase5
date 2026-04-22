from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from feast import FeatureStore
from joblib import load
from sklearn.pipeline import Pipeline

from common.config_loader import (
    DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
    load_global_config,
    load_training_experiment_config,
)
from common.logger import get_logger
from feast_ops.config import (
    DEFAULT_FEATURE_SERVICE_NAME,
    FEATURE_ENTITY_JOIN_KEY,
    FEATURE_STORE_REPO_PATH,
    ONLINE_FEATURE_COLUMNS,
)
from monitoring.metrics import (
    finish_feast_lookup_for_monitor,
    finish_model_predict_for_monitor,
    start_step_timer_for_monitor,
)
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
    model_version: str
    run_name: str
    feast_repo_path: Path
    feast_entity_key: str
    feast_feature_service_name: str


@dataclass(frozen=True)
class PreparedInferencePayload:
    """Payload resolvido para predição e monitoramento."""

    transformed_features: pd.DataFrame
    monitoring_features: dict[str, Any]
    request_metadata: dict[str, Any]
    feature_source: str
    customer_id: int | None = None


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
        feature_pipeline_path=Path("artifacts/models/feature_pipeline.joblib"),
        threshold=experiment_config["inference"]["threshold"],
        model_name=experiment_config["experiment"]["name"],
        model_version=experiment_config["experiment"]["version"],
        run_name=experiment_config["experiment"]["run_name"],
        feast_repo_path=FEATURE_STORE_REPO_PATH,
        feast_entity_key=FEATURE_ENTITY_JOIN_KEY,
        feast_feature_service_name=experiment_config.get("feast", {}).get(
            "feature_service_name",
            DEFAULT_FEATURE_SERVICE_NAME,
        ),
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


@lru_cache
def load_feast_store(repo_path_str: str) -> FeatureStore:
    """Carrega e reutiliza a instância local do Feast."""

    logger.info("Carregando repositório Feast em %s", repo_path_str)
    return FeatureStore(repo_path=repo_path_str)


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


def build_monitoring_features_from_dataframe(
    transformed_features: pd.DataFrame,
) -> dict[str, Any]:
    """Converte a linha transformada em payload auditável para monitoramento."""

    return transformed_features.iloc[0].to_dict()


def prepare_request_inference_payload(
    payload: ChurnPredictionRequest,
    cfg: ServingConfig,
) -> PreparedInferencePayload:
    """Resolve inferência a partir do payload bruto enviado pelo cliente."""

    logger.info("Preparando inferência a partir do payload bruto da requisição")
    transformed_features = prepare_inference_dataframe(payload, cfg)
    logger.info(
        "Payload bruto transformado com %d features para inferência",
        transformed_features.shape[1],
    )
    return PreparedInferencePayload(
        transformed_features=transformed_features,
        monitoring_features=build_monitoring_features_from_dataframe(
            transformed_features
        ),
        request_metadata={"feature_source": "request_payload"},
        feature_source="request_payload",
    )


def fetch_online_features_from_feast(
    customer_id: int,
    cfg: ServingConfig,
) -> dict[str, list[Any]]:
    """Consulta a online store por meio do FeatureService configurado."""

    start_time = start_step_timer_for_monitor()
    logger.info(
        "Consultando Feast online store para customer_id=%s via feature service %s",
        customer_id,
        cfg.feast_feature_service_name,
    )
    try:
        store = load_feast_store(str(cfg.feast_repo_path))
        feature_service = store.get_feature_service(cfg.feast_feature_service_name)
        online_features = store.get_online_features(
            features=feature_service,
            entity_rows=[{cfg.feast_entity_key: customer_id}],
        ).to_dict()
        logger.info(
            "Consulta ao Feast concluída para customer_id=%s", customer_id
        )
        return online_features
    finally:
        finish_feast_lookup_for_monitor(start_time)


def prepare_online_inference_payload(
    customer_id: int,
    cfg: ServingConfig,
) -> PreparedInferencePayload:
    """Resolve inferência a partir das features materializadas no Feast."""

    online_features = fetch_online_features_from_feast(customer_id, cfg)
    features_only = {
        feature_name: online_features.get(feature_name, [None])[0]
        for feature_name in ONLINE_FEATURE_COLUMNS
    }
    available_feature_count = sum(
        feature_value is not None for feature_value in features_only.values()
    )
    logger.info(
        "Feast retornou %d/%d features preenchidas para customer_id=%s",
        available_feature_count,
        len(ONLINE_FEATURE_COLUMNS),
        customer_id,
    )
    if all(feature_value is None for feature_value in features_only.values()):
        logger.warning(
            "Nenhuma feature online encontrada no Feast para customer_id=%s",
            customer_id,
        )
        raise LookupError(
            "Nenhuma feature online encontrada para o customer_id "
            f"{customer_id} na Feature Store."
        )

    transformed_features = pd.DataFrame([features_only])
    logger.info(
        "Payload online preparado com %d features para customer_id=%s",
        transformed_features.shape[1],
        customer_id,
    )
    return PreparedInferencePayload(
        transformed_features=transformed_features,
        monitoring_features=features_only,
        request_metadata={
            "feature_source": "feast_online_store",
            "customer_id": customer_id,
        },
        feature_source="feast_online_store",
        customer_id=customer_id,
    )


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
    start_time = start_step_timer_for_monitor()
    probability = float(model.predict_proba(transformed_features)[0][1])
    finish_model_predict_for_monitor(start_time)
    prediction = int(probability >= threshold)
    logger.info(
        (
            "Predição concluída | modelo=%s | versao=%s | threshold=%.3f | "
            "prob=%.6f | pred=%d"
        ),
        cfg.model_name,
        cfg.model_version,
        threshold,
        probability,
        prediction,
    )
    return probability, prediction
