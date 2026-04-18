from __future__ import annotations

from fastapi import FastAPI

from monitoring.metrics import register_prometheus_metrics
from serving.llm_routes import router as llm_router
from serving.routes import router


def create_app() -> FastAPI:
    """Cria a aplicação FastAPI e registra as rotas HTTP."""

    app = FastAPI(
        title="TC FIAP Fase 5 API - Datathon",
        version="0.1.0",
        description="API de serving para predição de churn bancário.",
    )
    app.include_router(router)
    app.include_router(llm_router)
    register_prometheus_metrics(app)
    return app


app = create_app()
