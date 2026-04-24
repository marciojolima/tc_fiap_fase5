from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.rag_pipeline import initialize_rag_index
from common.logger import get_logger
from monitoring.metrics import register_prometheus_metrics
from serving.llm_routes import router as llm_router
from serving.routes import router

logger = get_logger("serving.app")


def _format_bytes(value: object) -> str:
    try:
        bytes_value = float(value)
    except (TypeError, ValueError):
        return "n/a"

    return f"{bytes_value / (1024 * 1024):.2f} MB"


def _log_serving_ready_banner(rag_stats: dict[str, object]) -> None:
    cache_status = "HIT" if rag_stats.get("cache_hit") else "MISS"
    index_memory = _format_bytes(rag_stats.get("index_estimated_memory_bytes"))
    rss_delta = _format_bytes(rag_stats.get("process_rss_delta_bytes"))
    total_duration = float(rag_stats.get("total_duration_seconds", 0.0))

    logger.info(
        "\n"
        "=============================\n"
        "=============================\n"
        "======= SERVING READY =======\n"
        "=============================\n"
        "=============================\n"
        "API pronta para uso | RAG cache=%s | arquivos=%s | chunks=%s | "
        "indice_mem=%s | rss_delta=%s | startup_rag=%.3fs",
        cache_status,
        rag_stats.get("file_count", "n/a"),
        rag_stats.get("chunk_count", "n/a"),
        index_memory,
        rss_delta,
        total_duration,
    )


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    rag_stats = initialize_rag_index()
    _log_serving_ready_banner(rag_stats)
    yield


def create_app() -> FastAPI:
    """Cria a aplicação FastAPI e registra as rotas HTTP."""

    app = FastAPI(
        title="TC FIAP Fase 5 API - Datathon",
        version="0.1.0",
        description="API de serving para predição de churn bancário.",
        lifespan=app_lifespan,
    )
    app.include_router(router)
    app.include_router(llm_router)
    register_prometheus_metrics(app)
    return app


app = create_app()
