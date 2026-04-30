from __future__ import annotations

from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, HTTPException

from common.logger import get_logger
from model_lifecycle.train import build_metadata_output_path, run_training
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
    TrainModelRequest,
    TrainModelResponse,
)
from src.evaluation.model.drift.prediction_logger import (
    PredictionLogContext,
    log_prediction_for_monitoring,
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

TRAIN_NOTES = """
### Notas Operacionais do Endpoint

- O endpoint executa treino síncrono de um único experimento por chamada.
- O payload segue o mesmo contrato lógico de
  `configs/model_lifecycle/current.json`.
- O endpoint valida o schema com Pydantic antes de iniciar o treino.
- O endpoint **não** promove automaticamente o modelo para o serving.
- `artifacts.model_path` deve apontar para um caminho de artefato
  challenger, diferente do champion ativo.
"""


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def _validate_train_output_path(payload: TrainModelRequest) -> None:
    """Bloqueia sobrescrita do modelo champion ativo pelo endpoint síncrono."""

    active_model_path = load_serving_config().model_path.resolve()
    requested_model_resolved = Path(payload.artifacts.model_path).resolve()

    if requested_model_resolved == active_model_path:
        raise HTTPException(
            status_code=409,
            detail=(
                "O endpoint /train não pode sobrescrever o modelo ativo do serving. "
                "Informe um artifacts.model_path de challenger."
            ),
        )


def _load_request_serving_config(model_name: str):
    """Carrega a configuração do modelo solicitado pelo cliente."""

    try:
        return load_serving_config(model_name=model_name)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Modelo '{model_name}' não encontrado. Use `current` ou um "
                "experimento existente em configs/model_lifecycle/experiments."
            ),
        ) from exc


@router.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    description=(
        "Realiza a inferência de churn a partir do `customer_id`, consultando "
        "as features materializadas na online store do Feast. Aceita seleção "
        "opcional do modelo via `model_name`."
    ),
)
def predict_churn(payload: ChurnCustomerLookupRequest) -> ChurnPredictionResponse:
    start_time = start_predict_request_for_monitor()
    status_code = "500"

    cfg = _load_request_serving_config(payload.model_name)
    try:
        logger.info(
            "Recebida requisição POST /predict para customer_id=%s | model=%s",
            payload.customer_id,
            payload.model_name,
        )
        prepared_payload = prepare_online_inference_payload(payload.customer_id, cfg)
        probability, prediction = predict_from_dataframe(
            prepared_payload.transformed_features,
            cfg,
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
        "Aceita seleção opcional do modelo via `model_name`.\n"
        f"{DATA_DICT_TABLE}"
    ),
)
def predict_churn_from_raw(
    payload: ChurnPredictionRequest,
) -> ChurnPredictionResponse:
    start_time = start_predict_request_for_monitor()
    status_code = "500"

    cfg = _load_request_serving_config(payload.model_name)
    try:
        logger.info(
            "Recebida requisição POST /predict/raw | model=%s",
            payload.model_name,
        )
        prepared_payload = prepare_request_inference_payload(payload, cfg)
        probability, prediction = predict_from_dataframe(
            prepared_payload.transformed_features,
            cfg,
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


@router.post(
    "/train",
    response_model=TrainModelResponse,
    description=(
        "Executa um treino síncrono de um único experimento a partir de um "
        "payload JSON validado.\n"
        f"{TRAIN_NOTES}"
    ),
)
def train_model(payload: TrainModelRequest) -> TrainModelResponse:
    _validate_train_output_path(payload)
    logger.info(
        "Recebida requisição POST /train | experiment=%s | model_path=%s",
        payload.experiment.name,
        payload.artifacts.model_path,
    )
    start_time = perf_counter()
    metrics = run_training(experiment_config=payload.model_dump())
    training_time_seconds = round(perf_counter() - start_time, 3)
    metadata_path = build_metadata_output_path(Path(payload.artifacts.model_path))
    logger.info(
        "Treino síncrono concluído via /train | experiment=%s | model_path=%s",
        payload.experiment.name,
        payload.artifacts.model_path,
    )
    return TrainModelResponse(
        status="completed",
        experiment_name=payload.experiment.name,
        run_name=payload.experiment.run_name,
        model_version=payload.experiment.version,
        model_path=payload.artifacts.model_path,
        metadata_path=str(metadata_path),
        metrics=metrics,
        training_time_seconds=training_time_seconds,
        promoted_to_serving=False,
        message=(
            "Treino concluído com sucesso. O modelo foi salvo como challenger e "
            "não foi promovido automaticamente para o serving."
        ),
    )
