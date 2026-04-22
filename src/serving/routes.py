from __future__ import annotations

from fastapi import APIRouter, HTTPException

from common.logger import get_logger
from monitoring.inference_log import PredictionLogContext, log_prediction_for_monitoring
from monitoring.metrics import (
    finish_predict_request_for_monitor,
    start_predict_request_for_monitor,
)
from serving.pipeline import (
    load_serving_config,
    predict_from_dataframe,
    prepare_online_inference_payload,
    prepare_request_inference_payload,
)
from serving.schemas import (
    ChurnCustomerLookupRequest,
    ChurnPredictionRequest,
    ChurnPredictionResponse,
)

router = APIRouter()
logger = get_logger("serving.routes")

DATA_DICT_TABLE = """
### Dicionário de Dados de Entrada

| Campo | Tradução | Tipo | Restrições | Explicação |
|---|---|---|---|---|
| **CreditScore** | Pontuação de Crédito | int | 300-850 | Confiabilidade financeira |
| **Geography** | País | str | Germany, France, Spain | País de residência |
| **Gender** | Gênero | str | Male, Female | Gênero do cliente |
| **Age** | Idade | int | 18-100 | Idade em anos |
| **Tenure** | Tempo de Casa | int | 0-10 | Anos como cliente |
| **Balance** | Saldo | float | >= 0 | Valor em conta |
| **NumOfProducts** | Nº de Produtos | int | 1-4 | Serviços ativos |
| **HasCrCard** | Possui Cartão | int | 0 ou 1 | 1 = Sim |
| **IsActiveMember** | Membro Ativo | int | 0 ou 1 | 1 = Movimenta a conta |
| **EstimatedSalary** | Salário Estimado | float | > 0 | Rendimento anual |
| **Card Type** | Tipo de Cartão | str | DIAMOND, GOLD, etc | Categoria do cartão |
| **Point Earned** | Pontos | int | >= 0 | Pontos acumulados |
"""


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    description=(
        "Realiza a inferência de churn a partir do `customer_id`, consultando "
        "as features materializadas na online store do Feast."
    ),
)
def predict_churn(payload: ChurnCustomerLookupRequest) -> ChurnPredictionResponse:
    start_time = start_predict_request_for_monitor()
    status_code = "500"

    cfg = load_serving_config()
    try:
        logger.info(
            "Recebida requisição POST /predict para customer_id=%s",
            payload.customer_id,
        )
        prepared_payload = prepare_online_inference_payload(payload.customer_id, cfg)
        probability, prediction = predict_from_dataframe(
            prepared_payload.transformed_features
        )
        log_prediction_for_monitoring(
            feature_payload=prepared_payload.monitoring_features,
            prediction_context=PredictionLogContext(
                probability=probability,
                prediction=prediction,
                model_name=cfg.model_name,
                model_version=cfg.model_version,
                threshold=cfg.threshold,
            ),
            request_metadata=prepared_payload.request_metadata,
        )
        status_code = "200"
        logger.info(
            "Resposta /predict pronta | customer_id=%s | source=%s",
            payload.customer_id,
            prepared_payload.feature_source,
        )
        return ChurnPredictionResponse(
            churn_probability=probability,
            churn_prediction=prediction,
            model_name=cfg.model_name,
            threshold=cfg.threshold,
            feature_source=prepared_payload.feature_source,
            customer_id=prepared_payload.customer_id,
        )
    except LookupError as exc:
        logger.warning(
            "Falha em /predict | customer_id=%s | motivo=%s",
            payload.customer_id,
            exc,
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        finish_predict_request_for_monitor(
            start_time,
            method="POST",
            status_code=status_code,
        )


@router.post(
    "/predict/raw",
    response_model=ChurnPredictionResponse,
    description=(
        "Rota legada para inferência direta a partir do payload bruto.\n"
        f"{DATA_DICT_TABLE}"
    ),
)
def predict_churn_from_raw(
    payload: ChurnPredictionRequest,
) -> ChurnPredictionResponse:
    start_time = start_predict_request_for_monitor()
    status_code = "500"

    cfg = load_serving_config()
    try:
        logger.info("Recebida requisição POST /predict/raw")
        prepared_payload = prepare_request_inference_payload(payload, cfg)
        probability, prediction = predict_from_dataframe(
            prepared_payload.transformed_features
        )
        log_prediction_for_monitoring(
            feature_payload=prepared_payload.monitoring_features,
            prediction_context=PredictionLogContext(
                probability=probability,
                prediction=prediction,
                model_name=cfg.model_name,
                model_version=cfg.model_version,
                threshold=cfg.threshold,
            ),
            request_metadata=prepared_payload.request_metadata,
        )
        status_code = "200"
        logger.info(
            "Resposta /predict/raw pronta | source=%s",
            prepared_payload.feature_source,
        )
        return ChurnPredictionResponse(
            churn_probability=probability,
            churn_prediction=prediction,
            model_name=cfg.model_name,
            threshold=cfg.threshold,
            feature_source=prepared_payload.feature_source,
            customer_id=prepared_payload.customer_id,
        )
    finally:
        finish_predict_request_for_monitor(
            start_time,
            method="POST",
            status_code=status_code,
        )
