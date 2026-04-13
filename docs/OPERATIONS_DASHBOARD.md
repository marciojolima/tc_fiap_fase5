# Dashboard Operacional

O projeto expõe métricas Prometheus no serving por meio do endpoint `GET /metrics`.

As métricas operacionais iniciais são:

- `churn_serving_predict_latency_seconds`: latência do endpoint `POST /predict`
- `churn_serving_predict_requests_total`: total de requisições por método e status HTTP
- `churn_serving_predict_requests_in_progress`: requisições de predição em andamento

## Como subir localmente

1. Suba a stack local:

```bash
cp .env.example .env
poetry run task observability
```

2. Gere tráfego no endpoint de predição:

Use o Swagger em `http://127.0.0.1:8000/docs` ou faça chamadas para `POST /predict`.

## Acessos

- Serving: `http://127.0.0.1:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- MLflow: `http://127.0.0.1:5000`
- Usuário padrão Grafana: `admin`
- Senha padrão Grafana: `admin`

O dashboard `Serving Operacional - Churn API` é provisionado automaticamente.

## Observação

O Prometheus foi configurado para fazer scrape do serviço `serving` diretamente pela rede interna do Docker Compose, usando o alvo `serving:8000`.
