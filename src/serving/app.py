from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.rag_pipeline import initialize_rag_index
from monitoring.metrics import register_prometheus_metrics
from serving.llm_routes import router as llm_router
from serving.routes import router


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    initialize_rag_index()
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
