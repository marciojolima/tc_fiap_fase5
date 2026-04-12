from __future__ import annotations

from fastapi import APIRouter

from monitoring.inference_log import PredictionLogContext, log_prediction_for_monitoring
from monitoring.metrics import (
    finish_predict_request_for_monitor,
    start_predict_request_for_monitor,
)
from serving.pipeline import (
    load_serving_config,
    predict_from_dataframe,
    prepare_inference_dataframe,
)
from serving.schemas import ChurnPredictionRequest, ChurnPredictionResponse

router = APIRouter()

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
        f"Realiza a inferência de churn para um cliente individual.\n{DATA_DICT_TABLE}"
    ),
)
def predict_churn(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    start_time = start_predict_request_for_monitor()
    status_code = "500"

    cfg = load_serving_config()
    try:
        transformed_features = prepare_inference_dataframe(payload, cfg)
        probability, prediction = predict_from_dataframe(transformed_features)
        log_prediction_for_monitoring(
            payload=payload,
            prediction_context=PredictionLogContext(
                probability=probability,
                prediction=prediction,
                model_name=cfg.model_name,
                model_version=cfg.model_version,
                threshold=cfg.threshold,
            ),
        )
        status_code = "200"
        return ChurnPredictionResponse(
            churn_probability=probability,
            churn_prediction=prediction,
            model_name=cfg.model_name,
            threshold=cfg.threshold,
        )
    finally:
        finish_predict_request_for_monitor(
            start_time,
            method="POST",
            status_code=status_code,
        )
