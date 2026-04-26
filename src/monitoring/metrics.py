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
LLM_CHAT_PROVIDER_LATENCY_SECONDS = Histogram(
    "churn_serving_llm_chat_provider_latency_seconds",
    "Latencia da chamada ao provider LLM durante /llm/chat em segundos.",
    labelnames=("provider",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0),
)
LLM_CHAT_REQUESTS_IN_PROGRESS = Gauge(
    "churn_serving_llm_chat_requests_in_progress",
    "Quantidade de requisicoes /llm/chat em andamento.",
)
RAG_INDEX_FILES_TOTAL = Gauge(
    "churn_serving_rag_index_files_total",
    "Quantidade de arquivos carregados no corpus do RAG.",
)
RAG_INDEX_CHUNKS_TOTAL = Gauge(
    "churn_serving_rag_index_chunks_total",
    "Quantidade de chunks indexados no RAG.",
)
RAG_INDEX_SOURCE_BYTES = Gauge(
    "churn_serving_rag_index_source_bytes",
    "Total de bytes lidos das fontes do RAG.",
)
RAG_INDEX_EMBEDDINGS_BYTES = Gauge(
    "churn_serving_rag_index_embeddings_bytes",
    "Bytes ocupados pela matriz de embeddings do RAG.",
)
RAG_INDEX_ESTIMATED_MEMORY_BYTES = Gauge(
    "churn_serving_rag_index_estimated_memory_bytes",
    "Estimativa de memoria em bytes usada pelo indice do RAG em memoria.",
)
RAG_INDEX_PROCESS_RSS_DELTA_BYTES = Gauge(
    "churn_serving_rag_index_process_rss_delta_bytes",
    "Delta do RSS do processo durante a inicializacao do RAG em bytes.",
)
RAG_INDEX_STARTUP_DURATION_SECONDS = Gauge(
    "churn_serving_rag_index_startup_duration_seconds",
    "Tempo total da inicializacao mais recente do RAG em segundos.",
)
RAG_INDEX_STAGE_DURATION_SECONDS = Gauge(
    "churn_serving_rag_index_stage_duration_seconds",
    "Tempo por etapa da inicializacao do RAG em segundos.",
    labelnames=("stage",),
)
RAG_INDEX_BUILD_TOTAL = Counter(
    "churn_serving_rag_index_build_total",
    "Quantidade de inicializacoes do RAG por origem do indice.",
    labelnames=("source",),
)
RAG_INDEX_LAST_CACHE_HIT = Gauge(
    "churn_serving_rag_index_last_cache_hit",
    "Indica se a ultima inicializacao do RAG reutilizou cache (1) ou nao (0).",
)
RAG_QUERY_LATENCY_SECONDS = Histogram(
    "churn_serving_rag_query_latency_seconds",
    "Latencia da busca vetorial do RAG em segundos.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)
RAG_QUERY_REQUESTS_TOTAL = Counter(
    "churn_serving_rag_query_requests_total",
    "Total de consultas executadas no indice do RAG.",
    labelnames=("top_k", "returned_contexts"),
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
        "e metricas de latencia para /predict, /llm/chat e o RAG"
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


def finish_llm_chat_provider_call_for_monitor(
    start_time: float,
    *,
    provider: str,
) -> None:
    """Registra a latencia da chamada ao provider LLM durante /llm/chat."""

    LLM_CHAT_PROVIDER_LATENCY_SECONDS.labels(provider=provider).observe(
        perf_counter() - start_time
    )


def report_rag_index_stats(stats: dict[str, object]) -> None:
    """Publica metricas do indice RAG carregado no processo."""

    RAG_INDEX_FILES_TOTAL.set(float(stats.get("file_count", 0)))
    RAG_INDEX_CHUNKS_TOTAL.set(float(stats.get("chunk_count", 0)))
    RAG_INDEX_SOURCE_BYTES.set(float(stats.get("source_bytes", 0)))
    RAG_INDEX_EMBEDDINGS_BYTES.set(float(stats.get("embeddings_bytes", 0)))
    RAG_INDEX_ESTIMATED_MEMORY_BYTES.set(
        float(stats.get("index_estimated_memory_bytes", 0))
    )
    RAG_INDEX_PROCESS_RSS_DELTA_BYTES.set(
        float(stats.get("process_rss_delta_bytes", 0))
    )
    RAG_INDEX_STARTUP_DURATION_SECONDS.set(
        float(stats.get("total_duration_seconds", 0))
    )
    RAG_INDEX_LAST_CACHE_HIT.set(1.0 if stats.get("cache_hit") else 0.0)
    source = str(stats.get("build_source", "unknown"))
    RAG_INDEX_BUILD_TOTAL.labels(source=source).inc()

    stage_durations = stats.get("stage_durations_seconds", {})
    if isinstance(stage_durations, dict):
        for stage, duration in stage_durations.items():
            RAG_INDEX_STAGE_DURATION_SECONDS.labels(stage=str(stage)).set(
                float(duration)
            )


def report_rag_query(
    *,
    duration_seconds: float,
    top_k: int,
    returned_contexts: int,
) -> None:
    """Publica metricas de consulta do indice do RAG."""

    RAG_QUERY_LATENCY_SECONDS.observe(duration_seconds)
    RAG_QUERY_REQUESTS_TOTAL.labels(
        top_k=str(top_k),
        returned_contexts=str(returned_contexts),
    ).inc()
