# Dashboard Operacional

O projeto expõe métricas Prometheus no serving por meio do endpoint `GET /metrics`.

As métricas operacionais iniciais são:

- `churn_serving_predict_latency_seconds`: latência do endpoint `POST /predict`
- `churn_serving_predict_requests_total`: total de requisições por método e status HTTP
- `churn_serving_predict_requests_in_progress`: requisições de predição em andamento

## Como subir localmente

1. Inicie a API:

```bash
poetry run task serving
```

2. Em outro terminal, suba Prometheus e Grafana:

```bash
poetry run task observability
```

## Acessos

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Usuário padrão Grafana: `admin`
- Senha padrão Grafana: `admin`

O dashboard `Serving Operacional - Churn API` é provisionado automaticamente.

## Observação

O Prometheus foi configurado para fazer scrape da API local em `host.docker.internal:8000`. Em Linux moderno com Docker, o `host-gateway` do `docker-compose.yml` cobre esse roteamento.
