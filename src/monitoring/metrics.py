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
PREDICT_REQUESTS_IN_PROGRESS = Gauge(
    "churn_serving_predict_requests_in_progress",
    "Quantidade de requisicoes de predicao em andamento.",
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
        "e metrica de latencia para /predict"
    )


def start_predict_request_for_monitor() -> float:
    """Marca o inicio de uma requisicao de predicao."""

    PREDICT_REQUESTS_IN_PROGRESS.inc()
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
