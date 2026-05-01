from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, ValidationError

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
    LOOKUP_PREDICTION_BATCH_EXAMPLE,
    LOOKUP_PREDICTION_SINGLE_EXAMPLE,
    RAW_PREDICTION_BATCH_EXAMPLE,
    RAW_PREDICTION_SINGLE_EXAMPLE,
    ChurnCustomerLookupRequest,
    ChurnPredictionBatchItemResponse,
    ChurnPredictionBatchResponse,
    ChurnPredictionBatchSummary,
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
DATA_DICT_TABLE += (
    "| **model_name** | Modelo de serving | str | current, "
    "rf_v2_precision, rf_v3_recall | Modelo da predição; se omitido, "
    "usa `current`. |\n"
)

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

PREDICT_CONTRACT_NOTES = """
### Contrato de Entrada

- Se enviar **1 item**, envie um **objeto JSON** e a resposta será um
  **objeto JSON**.
- Se enviar **2 ou mais itens**, envie um **array JSON** e a resposta será um
  **objeto de lote** com `items` e `summary`.
- Em lote, cada item é processado isoladamente e erros de negócio ou validação
  retornam apenas no item afetado.
"""

PredictLookupBody = Annotated[
    dict[str, Any] | list[dict[str, Any]],
    Body(
        ...,
        openapi_examples={
            "single": {
                "summary": "Objeto único",
                "value": LOOKUP_PREDICTION_SINGLE_EXAMPLE,
            },
            "batch": {
                "summary": "Array com dois itens",
                "value": LOOKUP_PREDICTION_BATCH_EXAMPLE,
            },
        },
    ),
]

PredictRawBody = Annotated[
    dict[str, Any] | list[dict[str, Any]],
    Body(
        ...,
        openapi_examples={
            "single": {
                "summary": "Objeto único",
                "value": RAW_PREDICTION_SINGLE_EXAMPLE,
            },
            "batch": {
                "summary": "Array com dois itens",
                "value": RAW_PREDICTION_BATCH_EXAMPLE,
            },
        },
    ),
]


def _serialize_validation_error(exc: ValidationError) -> str:
    """Converte erros de validação do Pydantic em mensagem compacta por item."""

    parts = []
    for error in exc.errors():
        location = " -> ".join(str(item) for item in error.get("loc", ()))
        message = str(error.get("msg", "erro de validação"))
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts)


def _normalize_request_items(
    payload: Any,
    model_cls: type[BaseModel],
) -> tuple[list[tuple[int, BaseModel]], list[ChurnPredictionBatchItemResponse], bool]:
    """Normaliza payload único ou lista e valida cada item individualmente."""

    is_batch = isinstance(payload, list)
    raw_items = payload if is_batch else [payload]
    validated_items: list[tuple[int, BaseModel]] = []
    validation_errors: list[ChurnPredictionBatchItemResponse] = []

    for index, raw_item in enumerate(raw_items):
        try:
            item = (
                raw_item
                if isinstance(raw_item, model_cls)
                else model_cls.model_validate(raw_item)
            )
        except ValidationError as exc:
            if not is_batch:
                raise HTTPException(status_code=422, detail=exc.errors()) from exc
            validation_errors.append(
                ChurnPredictionBatchItemResponse(
                    index=index,
                    status="error",
                    error=_serialize_validation_error(exc),
                )
            )
            continue

        validated_items.append((index, item))

    return validated_items, validation_errors, is_batch


def _build_prediction_response(
    *,
    probability: float,
    prediction: int,
    cfg,
    feature_source: str,
    customer_id: int | None,
) -> ChurnPredictionResponse:
    """Padroniza a montagem da resposta unitária de inferência."""

    return ChurnPredictionResponse(
        churn_probability=probability,
        churn_prediction=prediction,
        model_name=cfg.model_name,
        threshold=cfg.threshold,
        feature_source=feature_source,
        customer_id=customer_id,
    )


def _build_prediction_batch_response(
    items: list[ChurnPredictionBatchItemResponse],
) -> ChurnPredictionBatchResponse:
    """Calcula o resumo de sucesso parcial para respostas em lote."""

    total = len(items)
    success = sum(item.status == "ok" for item in items)
    return ChurnPredictionBatchResponse(
        items=items,
        summary=ChurnPredictionBatchSummary(
            total=total,
            success=success,
            errors=total - success,
        ),
    )


def _predict_lookup_item(
    payload: ChurnCustomerLookupRequest,
) -> ChurnPredictionResponse:
    """Executa a inferência online para um único customer_id."""

    cfg = _load_request_serving_config(payload.model_name)
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
    logger.info(
        "Resposta /predict pronta | customer_id=%s | source=%s",
        payload.customer_id,
        prepared_payload.feature_source,
    )
    return _build_prediction_response(
        probability=probability,
        prediction=prediction,
        cfg=cfg,
        feature_source=prepared_payload.feature_source,
        customer_id=prepared_payload.customer_id,
    )


def _predict_raw_item(
    payload: ChurnPredictionRequest,
) -> ChurnPredictionResponse:
    """Executa a inferência raw para um único payload validado."""

    cfg = _load_request_serving_config(payload.model_name)
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
    logger.info(
        "Resposta /predict/raw pronta | source=%s",
        prepared_payload.feature_source,
    )
    return _build_prediction_response(
        probability=probability,
        prediction=prediction,
        cfg=cfg,
        feature_source=prepared_payload.feature_source,
        customer_id=prepared_payload.customer_id,
    )


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
    response_model=ChurnPredictionResponse | ChurnPredictionBatchResponse,
    description=(
        "Realiza a inferência de churn a partir do `customer_id`, consultando "
        "as features materializadas na online store do Feast. Aceita seleção "
        "opcional do modelo via `model_name`.\n"
        f"{PREDICT_CONTRACT_NOTES}"
    ),
)
def predict_churn(
    payload: PredictLookupBody,
) -> ChurnPredictionResponse | ChurnPredictionBatchResponse:
    start_time = start_predict_request_for_monitor()
    status_code = "500"

    try:
        validated_items, batch_errors, is_batch = _normalize_request_items(
            payload,
            ChurnCustomerLookupRequest,
        )
        if not is_batch:
            response = _predict_lookup_item(validated_items[0][1])
            status_code = "200"
            return response

        batch_items = list(batch_errors)
        for raw_index, validated_item in validated_items:
            try:
                response = _predict_lookup_item(validated_item)
            except (HTTPException, LookupError) as exc:
                error_message = (
                    exc.detail
                    if isinstance(exc, HTTPException)
                    else str(exc)
                )
                batch_items.append(
                    ChurnPredictionBatchItemResponse(
                        index=raw_index,
                        status="error",
                        error=error_message,
                    )
                )
                continue

            batch_items.append(
                ChurnPredictionBatchItemResponse(
                    index=raw_index,
                    status="ok",
                    result=response,
                )
            )

        status_code = "200"
        return _build_prediction_batch_response(
            sorted(batch_items, key=lambda item: item.index)
        )
    except LookupError as exc:
        logger.warning("Falha em /predict | motivo=%s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        finish_predict_request_for_monitor(
            start_time,
            method="POST",
            status_code=status_code,
        )


@router.post(
    "/predict/raw",
    response_model=ChurnPredictionResponse | ChurnPredictionBatchResponse,
    description=(
        "Rota para inferência direta a partir do payload bruto.\n"
        "Aceita seleção opcional do modelo via `model_name`.\n"
        f"{PREDICT_CONTRACT_NOTES}\n"
        f"{DATA_DICT_TABLE}"
    ),
)
def predict_churn_from_raw(
    payload: PredictRawBody,
) -> ChurnPredictionResponse | ChurnPredictionBatchResponse:
    start_time = start_predict_request_for_monitor()
    status_code = "500"

    try:
        validated_items, batch_errors, is_batch = _normalize_request_items(
            payload,
            ChurnPredictionRequest,
        )
        if not is_batch:
            response = _predict_raw_item(validated_items[0][1])
            status_code = "200"
            return response

        batch_items = list(batch_errors)
        logger.info(
            "Recebida requisição POST /predict/raw em lote | itens_validos=%d",
            len(validated_items),
        )
        for raw_index, validated_item in validated_items:
            try:
                response = _predict_raw_item(validated_item)
            except HTTPException as exc:
                batch_items.append(
                    ChurnPredictionBatchItemResponse(
                        index=raw_index,
                        status="error",
                        error=str(exc.detail),
                    )
                )
                continue

            batch_items.append(
                ChurnPredictionBatchItemResponse(
                    index=raw_index,
                    status="ok",
                    result=response,
                )
            )

        status_code = "200"
        return _build_prediction_batch_response(
            sorted(batch_items, key=lambda item: item.index)
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
