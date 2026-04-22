from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from common.logger import get_logger

logger = get_logger("monitoring.metrics")

PREDICT_REQUESTS_TOTAL = Counter(
    "churn_serving_predict_requests_total",
    "Total de requisicoes recebidas no endpoint de predicao.",
    labelnames=("method", "status_code"),
)
PREDICT_REQUEST_LATENCY_SECONDS = Histogram(
    "churn_serving_predict_latency_seconds",
    "Latencia das requisicoes do endpoint de predicao em segundos.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)
PREDICT_FEAST_LOOKUP_LATENCY_SECONDS = Histogram(
    "churn_serving_predict_feast_lookup_latency_seconds",
    "Latencia da consulta de features online no Feast em segundos.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)
PREDICT_MODEL_LATENCY_SECONDS = Histogram(
    "churn_serving_predict_model_latency_seconds",
    "Latencia da inferencia do modelo em segundos.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0),
)
PREDICT_REQUESTS_IN_PROGRESS = Gauge(
    "churn_serving_predict_requests_in_progress",
    "Quantidade de requisicoes de predicao em andamento.",
)
LLM_CHAT_REQUESTS_TOTAL = Counter(
    "churn_serving_llm_chat_requests_total",
    "Total de requisicoes recebidas no endpoint /llm/chat.",
    labelnames=("method", "status_code"),
)
LLM_CHAT_REQUEST_LATENCY_SECONDS = Histogram(
    "churn_serving_llm_chat_latency_seconds",
    "Latencia das requisicoes do endpoint /llm/chat em segundos.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0),
)
LLM_CHAT_OLLAMA_LATENCY_SECONDS = Histogram(
    "churn_serving_llm_chat_ollama_latency_seconds",
    "Latencia da chamada HTTP ao Ollama durante /llm/chat em segundos.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0),
)
LLM_CHAT_REQUESTS_IN_PROGRESS = Gauge(
    "churn_serving_llm_chat_requests_in_progress",
    "Quantidade de requisicoes /llm/chat em andamento.",
)


def register_prometheus_metrics(app: FastAPI) -> None:
    """Registra endpoint para exportacao de metricas Prometheus."""

    if getattr(app.state, "prometheus_metrics_registered", False):
        return

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.state.prometheus_metrics_registered = True
    logger.info(
        "Prometheus configurado no serving com endpoint /metrics "
        "e metricas de latencia para /predict e /llm/chat"
    )


def start_predict_request_for_monitor() -> float:
    """Marca o inicio de uma requisicao de predicao."""

    PREDICT_REQUESTS_IN_PROGRESS.inc()
    return perf_counter()


def start_step_timer_for_monitor() -> float:
    """Marca o inicio de uma etapa interna da inferencia."""

    return perf_counter()


def finish_predict_request_for_monitor(
    start_time: float,
    *,
    method: str,
    status_code: str,
) -> None:
    """Registra a latencia e o desfecho HTTP de uma requisicao de predicao."""

    elapsed_seconds = perf_counter() - start_time
    PREDICT_REQUEST_LATENCY_SECONDS.observe(elapsed_seconds)
    PREDICT_REQUESTS_TOTAL.labels(
        method=method,
        status_code=status_code,
    ).inc()
    PREDICT_REQUESTS_IN_PROGRESS.dec()


def finish_feast_lookup_for_monitor(start_time: float) -> None:
    """Registra a latencia da consulta online ao Feast."""

    PREDICT_FEAST_LOOKUP_LATENCY_SECONDS.observe(perf_counter() - start_time)


def finish_model_predict_for_monitor(start_time: float) -> None:
    """Registra a latencia da inferencia do modelo."""

    PREDICT_MODEL_LATENCY_SECONDS.observe(perf_counter() - start_time)


def start_llm_chat_request_for_monitor() -> float:
    """Marca o inicio de uma requisicao para /llm/chat."""

    LLM_CHAT_REQUESTS_IN_PROGRESS.inc()
    return perf_counter()


def finish_llm_chat_request_for_monitor(
    start_time: float,
    *,
    method: str,
    status_code: str,
) -> None:
    """Registra a latencia e o desfecho HTTP de uma requisicao /llm/chat."""

    elapsed_seconds = perf_counter() - start_time
    LLM_CHAT_REQUEST_LATENCY_SECONDS.observe(elapsed_seconds)
    LLM_CHAT_REQUESTS_TOTAL.labels(
        method=method,
        status_code=status_code,
    ).inc()
    LLM_CHAT_REQUESTS_IN_PROGRESS.dec()


def finish_llm_chat_ollama_call_for_monitor(start_time: float) -> None:
    """Registra a latencia da chamada ao Ollama durante /llm/chat."""

    LLM_CHAT_OLLAMA_LATENCY_SECONDS.observe(perf_counter() - start_time)
