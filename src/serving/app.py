from fastapi import FastAPI

app = FastAPI(
    title="TC FIAP Fase 5 API",
    version="0.1.0",
    description="API base para serving do projeto Datathon.",
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
