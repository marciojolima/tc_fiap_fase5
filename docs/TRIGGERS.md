# Mapa de Gatilhos do Projeto

Este documento resume os principais pontos de entrada do projeto, mostrando:

- qual evento inicia cada fluxo
- quais módulos e funções participam
- quais artefatos entram e saem
- quais fluxos são online, batch, manuais ou automáticos

## Leitura Rápida

| Gatilho | Tipo | Start | Resultado principal |
|---|---|---|---|
| Serving healthcheck | Online | `GET /health` | Retorna status do serviço |
| Serving predição | Online | `POST /predict` | Retorna probabilidade e classe |
| Métricas Prometheus | Online | `GET /metrics` | Expõe métricas do serving |
| Logging de inferência | Online passivo | `POST /predict` | Salva inferência em JSONL |
| Engenharia de features | Batch manual | `python -m src.features.feature_engineering` | Gera `interim`, `processed` e `feature_pipeline.joblib` |
| Treino | Batch manual | `python -m src.models.train` | Gera `model_current.pkl` e metadados |
| Drift monitoring | Batch manual | `python -m src.monitoring.drift` | Gera relatório e status de drift |
| Retreino por drift | Batch automático interno | Drift crítico | Gera challenger e decisão champion-challenger |
| Retreino manual | Batch manual | `python -m src.models.retraining` | Executa request de retreino |
| Export da Feature Store | Batch manual | `python -m src.feast_ops.export` | Gera parquet offline do Feast |
| Feast apply | Infra manual | `feast apply` | Registra definições da store |
| Feast materialize | Infra manual | `feast materialize-incremental` | Materializa features no Redis |
| Cenários de inferência | Batch manual | `python -m src.scenario_analysis.inference_cases` | Valida casos de negócio |
| Drift sintético | Batch manual | `python -m src.scenario_analysis.synthetic_drifts --all` | Gera lotes e artefatos de teste |
| Stack local | Infra manual | `docker compose up` | Sobe Redis, serving, MLflow, Prometheus e Grafana |

## Visão Geral

```text
Raw data
  -> feature engineering
  -> processed datasets + feature pipeline
  -> treino
  -> model_current.pkl
  -> serving /predict
  -> inference log JSONL
  -> drift monitoring
  -> retraining request
  -> challenger + promotion decision
```

## 1. Gatilhos de Serving

### 1.1 `GET /health`

**Tipo:** online  
**Start:** chamada HTTP para `/health`

**Cadeia**

`GET /health`
-> rota [`healthcheck`](../serving/routes.py)
-> retorna `{"status": "ok"}`

**Arquivos envolvidos**

- [`src/serving/app.py`](../serving/app.py)
- [`src/serving/routes.py`](../serving/routes.py)

### 1.2 `POST /predict`

**Tipo:** online  
**Start:** chamada HTTP para `/predict`

**Cadeia**

`POST /predict`
-> schema [`ChurnPredictionRequest`](../serving/schemas.py)
-> rota [`predict_churn`](../serving/routes.py)
-> função [`load_serving_config`](../serving/pipeline.py)
-> função [`prepare_inference_dataframe`](../serving/pipeline.py)
-> função [`load_feature_pipeline`](../serving/pipeline.py)
-> aplica `artifacts/models/feature_pipeline.joblib`
-> função [`predict_from_dataframe`](../serving/pipeline.py)
-> função [`load_prediction_model`](../serving/pipeline.py)
-> usa `artifacts/models/model_current.pkl`
-> aplica `predict_proba`
-> threshold definido em [`configs/training/model_current.yaml`](../../configs/training/model_current.yaml)
-> retorna [`ChurnPredictionResponse`](../serving/schemas.py)

**Entradas**

- payload HTTP validado
- [`configs/training/model_current.yaml`](../../configs/training/model_current.yaml)
- `artifacts/models/feature_pipeline.joblib`
- `artifacts/models/model_current.pkl`

**Saídas**

- resposta JSON com:
  - `churn_probability`
  - `churn_prediction`
  - `model_name`
  - `threshold`

**Observações**

- O serving não chama `src/features` nem `src/models` diretamente na requisição.
- Ele reaproveita artefatos gerados offline por esses módulos.

### 1.3 `POST /predict` -> métricas Prometheus

**Tipo:** online passivo  
**Start:** a mesma chamada do `/predict`

**Cadeia**

`POST /predict`
-> função [`start_predict_request_for_monitor`](../monitoring/metrics.py)
-> incrementa gauge de requisições em andamento
-> executa inferência
-> função [`finish_predict_request_for_monitor`](../monitoring/metrics.py)
-> atualiza:
  - contador de requisições
  - histograma de latência
  - gauge de requisições em andamento

**Consulta**

`GET /metrics`
-> função [`register_prometheus_metrics`](../monitoring/metrics.py)
-> expõe métricas para scrape

**Arquivos envolvidos**

- [`src/serving/app.py`](../serving/app.py)
- [`src/serving/routes.py`](../serving/routes.py)
- [`src/monitoring/metrics.py`](../monitoring/metrics.py)

**Observações**

- O código da API expõe as métricas.
- O armazenamento histórico não é feito pela API; ele é responsabilidade do Prometheus.

### 1.4 `POST /predict` -> logging de inferência

**Tipo:** online passivo  
**Start:** a mesma chamada do `/predict`

**Cadeia**

`POST /predict`
-> rota [`predict_churn`](../serving/routes.py)
-> função [`log_prediction_for_monitoring`](../monitoring/inference_log.py)
-> monta registro com payload + probabilidade + classe
-> append em `artifacts/monitoring/inference_logs/predictions.jsonl`

**Arquivos envolvidos**

- [`src/serving/routes.py`](../serving/routes.py)
- [`src/monitoring/inference_log.py`](../monitoring/inference_log.py)
- [`configs/monitoring_config.yaml`](../../configs/monitoring_config.yaml)

**Observações**

- Este fluxo não calcula drift.
- Ele apenas produz a base current consumida depois pelo monitoramento batch.

## 2. Gatilhos de Features

### 2.1 Engenharia de features manual

**Tipo:** batch manual  
**Start:** `python -m src.features.feature_engineering` ou `task mlfeateng`

**Cadeia**

`feature_engineering`
-> [`load_raw_data`](../common/data_loader.py)
-> valida schema raw em [`validate_raw_dataset_schema`](../features/schema_validation.py)
-> remove identificadores diretos
-> limpa duplicados e ausências
-> salva `data/interim/cleaned.parquet`
-> divide treino e teste
-> ajusta pipeline de transformação
-> transforma treino e teste
-> salva:
  - `data/processed/train.parquet`
  - `data/processed/test.parquet`
  - `data/processed/feature_columns.json`
  - `data/processed/schema_report.json`
  - `artifacts/models/feature_pipeline.joblib`

**Função principal**

- [`main`](../features/feature_engineering.py)

**Arquivos envolvidos**

- [`src/features/feature_engineering.py`](../features/feature_engineering.py)
- [`src/features/pipeline_components.py`](../features/pipeline_components.py)
- [`src/features/schema_validation.py`](../features/schema_validation.py)
- [`configs/pipeline_global_config.yaml`](../../configs/pipeline_global_config.yaml)

### 2.2 Engenharia de features via DVC

**Tipo:** batch manual orquestrado  
**Start:** `dvc repro featurize`

**Cadeia**

`dvc repro featurize`
-> stage `featurize` em [`dvc.yaml`](../../dvc.yaml)
-> executa `python -m src.features.feature_engineering`

**Observação**

- A lógica é a mesma do fluxo manual.
- O DVC apenas organiza dependências e saídas.

## 3. Gatilhos de Treino

### 3.1 Treino manual

**Tipo:** batch manual  
**Start:** `python -m src.models.train` ou `task mltrain`

**Cadeia**

`train`
-> carrega config do experimento em [`load_experiment_training_config`](../models/train.py)
-> lê:
  - `data/processed/train.parquet`
  - `data/processed/test.parquet`
-> instancia algoritmo via [`build_model`](../models/catalog.py)
-> treina modelo
-> calcula métricas
-> registra run no MLflow
-> salva:
  - `artifacts/models/model_current.pkl`
  - `artifacts/models/model_current_metadata.json`

**Função principal**

- [`run_training`](../models/train.py)

**Arquivos envolvidos**

- [`src/models/train.py`](../models/train.py)
- [`src/models/catalog.py`](../models/catalog.py)
- [`configs/training/model_current.yaml`](../../configs/training/model_current.yaml)

### 3.2 Treino via DVC

**Tipo:** batch manual orquestrado  
**Start:** `dvc repro train`

**Cadeia**

`dvc repro train`
-> stage `train` em [`dvc.yaml`](../../dvc.yaml)
-> executa `python -m src.models.train`

### 3.3 Treino múltiplo de experimentos

**Tipo:** batch manual composto  
**Start:** `task mlrunall`

**Cadeia**

`task mlrunall`
-> executa `task mlfeateng`
-> roda múltiplos `python -m src.models.train --config ...`
-> executa cenários de inferência

**Arquivo envolvido**

- [`pyproject.toml`](../../pyproject.toml)

## 4. Gatilhos de Monitoramento

### 4.1 Drift monitoring manual

**Tipo:** batch manual  
**Start:** `python -m src.monitoring.drift` ou `task mldrift`

**Cadeia**

`drift`
-> lê config em [`configs/monitoring_config.yaml`](../../configs/monitoring_config.yaml)
-> carrega dataset de referência:
  - `data/processed/train.parquet`
-> carrega dataset current:
  - `artifacts/monitoring/inference_logs/predictions.jsonl`
-> resolve matriz de features
-> usa `artifacts/models/feature_pipeline.joblib` quando necessário
-> calcula PSI por feature
-> calcula prediction drift
-> gera relatório Evidently
-> decide status:
  - `ok`
  - `warning`
  - `critical`
  - `insufficient_data`
-> salva:
  - `artifacts/monitoring/drift/drift_report.html`
  - `artifacts/monitoring/drift/drift_metrics.json`
  - `artifacts/monitoring/drift/drift_status.json`
  - `artifacts/monitoring/drift/drift_runs.jsonl`

**Função principal**

- [`run_drift_monitoring`](../monitoring/drift.py)

**Arquivos envolvidos**

- [`src/monitoring/drift.py`](../monitoring/drift.py)
- [`configs/monitoring_config.yaml`](../../configs/monitoring_config.yaml)

### 4.2 Drift demo

**Tipo:** batch manual de demonstração  
**Start:** `task mldriftdemo`

**Cadeia**

`task mldriftdemo`
-> executa `python -m src.monitoring.drift --current data/processed/test.parquet`

**Observação**

- Serve para testar o monitoramento mesmo sem usar o log real de inferências.

## 5. Gatilhos de Retreino

### 5.1 Retreino automático interno por drift crítico

**Tipo:** batch automático interno  
**Start:** execução de drift com status `critical`

**Cadeia**

`python -m src.monitoring.drift`
-> função [`run_drift_monitoring`](../monitoring/drift.py)
-> função [`maybe_trigger_retraining`](../monitoring/drift.py)
-> cria `artifacts/monitoring/retraining/retrain_request.json`
-> como `trigger_mode` atual é `auto_train_manual_promote`
-> chama [`run_retraining_request`](../models/retraining.py)
-> treina challenger
-> avalia promoção champion-challenger
-> salva:
  - `artifacts/monitoring/retraining/retrain_run.json`
  - `artifacts/monitoring/retraining/promotion_decision.json`

**Observações**

- O retreino pode ser automático.
- A promoção do challenger para substituir o champion atual continua manual.

### 5.2 Retreino manual

**Tipo:** batch manual  
**Start:** `python -m src.models.retraining` ou `task mlretrain`

**Cadeia**

`models.retraining`
-> lê `retrain_request.json`
-> cria config temporária de challenger
-> chama [`run_training`](../models/train.py)
-> salva challenger em `artifacts/models/challengers/`
-> compara champion vs challenger
-> grava decisão de promoção

**Função principal**

- [`run_retraining_request`](../models/retraining.py)

**Arquivos envolvidos**

- [`src/models/retraining.py`](../models/retraining.py)
- [`src/models/promotion.py`](../models/promotion.py)
- [`src/models/train.py`](../models/train.py)

## 6. Gatilhos da Feature Store

### 6.1 Export manual da Feature Store

**Tipo:** batch manual  
**Start:** `python -m src.feast_ops.export` ou `task feastexport`

**Cadeia**

`feast export`
-> lê dado bruto
-> reaproveita `artifacts/models/feature_pipeline.joblib`
-> transforma features
-> seleciona colunas online
-> gera:
  - `data/feature_store/customer_features.parquet`
  - `data/feature_store/export_metadata.json`

**Função principal**

- [`export_features_for_feast`](../feast_ops/export.py)

**Arquivos envolvidos**

- [`src/feast_ops/export.py`](../feast_ops/export.py)
- [`src/feast_ops/config.py`](../feast_ops/config.py)
- [`docs/FEATURE_STORE.md`](../../docs/FEATURE_STORE.md)

### 6.2 Export via DVC

**Tipo:** batch manual orquestrado  
**Start:** `dvc repro export_feature_store`

**Cadeia**

`dvc repro export_feature_store`
-> stage `export_feature_store` em [`dvc.yaml`](../../dvc.yaml)
-> executa `python -m src.feast_ops.export`

### 6.3 Feast apply

**Tipo:** infra manual  
**Start:** `task feastapply`

**Cadeia**

`feast -c feature_store apply`
-> registra definições da Feature Store

### 6.4 Feast materialize

**Tipo:** infra manual  
**Start:** `task feastmaterialize`

**Cadeia**

`feast -c feature_store materialize-incremental ...`
-> materializa features no Redis

### 6.5 Feast demo

**Tipo:** batch/manual de demonstração  
**Start:** `task feastdemo`

**Cadeia**

`python -m src.feast_ops.demo --customer-id ...`
-> consulta features online no Redis

**Observação importante**

- A API FastAPI atual ainda não consulta o Feast em produção.
- Isso está documentado em [`docs/FEATURE_STORE.md`](../../docs/FEATURE_STORE.md).

## 7. Gatilhos de Cenários e Validação

### 7.1 Cenários de inferência

**Tipo:** batch manual  
**Start:** `python -m src.scenario_analysis.inference_cases` ou `task mlscenarios`

**Cadeia**

`scenario_analysis.inference_cases`
-> lê suíte YAML de cenários
-> monta payloads
-> usa o mesmo pipeline de serving
-> produz previsões para cenários de negócio

**Arquivos envolvidos**

- [`src/scenario_analysis/inference_cases.py`](../scenario_analysis/inference_cases.py)
- [`configs/scenario_analysis/inference_cases.yaml`](../../configs/scenario_analysis/inference_cases.yaml)

### 7.2 Drift sintético

**Tipo:** batch manual  
**Start:** `python -m src.scenario_analysis.synthetic_drifts --all` ou `task mlsyntheticdrift`

**Cadeia**

`scenario_analysis.synthetic_drifts`
-> gera lotes sintéticos
-> carrega pipeline e modelo atuais
-> calcula previsões do lote
-> gera registros compatíveis com monitoramento
-> produz artefatos para validar o comportamento do drift batch

**Arquivos envolvidos**

- [`src/scenario_analysis/synthetic_drifts.py`](../scenario_analysis/synthetic_drifts.py)
- [`docs/SYNTHETIC_PREDICTIONS_GENERATOR.md`](../../docs/SYNTHETIC_PREDICTIONS_GENERATOR.md)

## 8. Gatilhos de Infraestrutura Local

### 8.1 Stack local

**Tipo:** infra manual  
**Start:** `task appstack`

**Cadeia**

`docker compose up -d redis serving mlflow prometheus grafana`
-> sobe:
  - Redis
  - serving
  - MLflow
  - Prometheus
  - Grafana

**Arquivos envolvidos**

- [`docker-compose.yml`](../../docker-compose.yml)
- [`pyproject.toml`](../../pyproject.toml)

### 8.2 MLflow local

**Tipo:** infra manual  
**Start:** `task mlflow`

**Cadeia**

`mlflow server ...`
-> sobe servidor local de tracking para treino e retreino

## 9. Mapa por Natureza do Gatilho

### Online

- `GET /health`
- `POST /predict`
- `GET /metrics`

### Batch manual

- `python -m src.features.feature_engineering`
- `python -m src.models.train`
- `python -m src.monitoring.drift`
- `python -m src.models.retraining`
- `python -m src.feast_ops.export`
- `python -m src.scenario_analysis.inference_cases`
- `python -m src.scenario_analysis.synthetic_drifts --all`

### Batch manual com DVC

- `dvc repro featurize`
- `dvc repro train`
- `dvc repro export_feature_store`

### Infra manual

- `task appstack`
- `task mlflow`
- `task feastapply`
- `task feastmaterialize`
- `task feastdemo`

### Automático interno

- drift crítico
  -> abre request de retreino
  -> executa retreino challenger
  -> gera decisão de promoção

## 10. O Que Não Existe Hoje

Atualmente o repositório **não possui**:

- Airflow
- cron formal
- scheduler interno próprio
- promoção automática do challenger para substituir `model_current.pkl`
- serving consultando Feast online em produção

## 11. Resumo Executivo

Se reduzirmos o projeto aos gatilhos centrais, o mapa fica assim:

`serving`
-> `/predict`
-> carrega config atual
-> carrega pipeline de features
-> usa `model_current.pkl`
-> retorna predição
-> atualiza métricas
-> salva inferência para monitoramento

`engenharia de features`
-> comando manual ou DVC
-> lê raw
-> sanitiza
-> gera interim
-> faz split treino/teste
-> transforma features
-> salva datasets processados e pipeline

`treino`
-> comando manual ou DVC
-> lê processed
-> treina
-> registra MLflow
-> salva modelo atual

`monitoramento`
-> comando manual
-> lê referência e current
-> calcula drift
-> salva relatório e status
-> se crítico, pode acionar retreino

`retreino`
-> automático por drift crítico ou manual por comando
-> treina challenger
-> compara com champion
-> gera decisão de promoção

`feature store`
-> export manual ou DVC
-> apply
-> materialize
-> demo online via Redis

