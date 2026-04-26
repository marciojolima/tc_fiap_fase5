FROM python:3.13-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock README.md ./

RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --only main --extras evaluation --no-root --no-interaction --no-ansi

COPY configs ./configs
COPY src ./src

RUN pip install --no-cache-dir --no-deps .

CMD ["python", "-m", "src.evaluation.llm_agent.ragas_eval"]
