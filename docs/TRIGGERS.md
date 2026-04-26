# Mapa de Gatilhos do Projeto

## Índice

- [Leitura Rápida](#leitura-rápida)
- [Visão Geral](#visão-geral)
- [1. Gatilhos de Serving](#1-gatilhos-de-serving)
- [2. Gatilhos de Features](#2-gatilhos-de-features)
- [3. Gatilhos de Treino](#3-gatilhos-de-treino)
- [4. Gatilhos de Monitoramento](#4-gatilhos-de-monitoramento)
- [5. Gatilhos de Retreino](#5-gatilhos-de-retreino)
- [6. Gatilhos da Feature Store](#6-gatilhos-da-feature-store)
- [7. Gatilhos de Cenários e Validação](#7-gatilhos-de-cenários-e-validação)
- [8. Gatilhos de Infraestrutura Local](#8-gatilhos-de-infraestrutura-local)
- [9. Gatilhos de Avaliação LLM](#9-gatilhos-de-avaliação-llm)
- [10. Mapa por Natureza do Gatilho](#10-mapa-por-natureza-do-gatilho)
- [11. O Que Não Existe Hoje](#11-o-que-não-existe-hoje)
- [12. Resumo Executivo](#12-resumo-executivo)

Este documento resume os principais pontos de entrada do projeto, mostrando:

- qual evento inicia cada fluxo
- quais módulos e funções participam
- quais artefatos entram e saem
- quais fluxos são online, batch, manuais ou automáticos

## Leitura Rápida

| Gatilho | Tipo | Start | Resultado principal |
|---|---|---|---|
| Serving healthcheck | Online | `GET /health` | Retorna status do serviço |
| Serving predição online | Online | `POST /predict` | Retorna probabilidade e classe a partir do `customer_id` |
| Serving predição legada | Online | `POST /predict/raw` | Retorna probabilidade e classe a partir do payload bruto |
| Métricas Prometheus | Online | `GET /metrics` | Expõe métricas do serving |
| Logging de inferência | Online passivo | `POST /predict` | Salva inferência em JSONL |
| Engenharia de features | Batch manual | `python -m src.feature_engineering.feature_engineering` | Gera `interim`, `processed` e `feature_pipeline.joblib` |
| Treino | Batch manual | `python -m src.model_lifecycle.train` | Gera `model_current.pkl` e metadados |
| Drift monitoring | Batch manual | `python -m src.evaluation.model.drift.drift` | Gera relatório e status de drift |
| Retreino por drift | Batch automático interno | Drift crítico | Gera challenger e decisão champion-challenger |
| Retreino manual | Batch manual | `python -m src.model_lifecycle.retraining` | Executa request de retreino |
| Export da Feature Store | Batch manual | `python -m src.feast_ops.export` | Gera parquet offline do Feast |
| Feast apply | Infra manual | `feast apply` | Registra definições da store |
| Feast materialize | Infra manual | `feast materialize-incremental` | Materializa features no Redis |
| Cenários de inferência | Batch manual | `python -m src.scenario_experiments.inference_cases` | Valida casos de negócio |
| Drift sintético | Batch manual | `python -m src.scenario_experiments.synthetic_drifts --all` | Gera lotes e artefatos de teste |
| Prompt A/B | Batch manual | `task eval_ab_test_prompts` | Compara 3 variantes de prompt no golden set |
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
-> rota [`healthcheck`](../src/serving/routes.py)
-> retorna `{"status": "ok"}`

**Arquivos envolvidos**

- [`src/serving/app.py`](../src/serving/app.py)
- [`src/serving/routes.py`](../src/serving/routes.py)

### 1.2 `POST /predict`

**Tipo:** online  
**Start:** chamada HTTP para `/predict`

**Cadeia**

`POST /predict`
-> schema [`ChurnCustomerLookupRequest`](../src/serving/schemas.py)
-> rota [`predict_churn`](../src/serving/routes.py)
-> função [`load_serving_config`](../src/serving/pipeline.py)
-> função [`prepare_online_inference_payload`](../src/serving/pipeline.py)
-> função [`fetch_online_features_from_feast`](../src/serving/pipeline.py)
-> carrega o repositório Feast local
-> resolve o `FeatureService` do modelo ativo
-> consulta a online store Redis pelo `customer_id`
-> função [`predict_from_dataframe`](../src/serving/pipeline.py)
-> função [`load_prediction_model`](../src/serving/pipeline.py)
-> usa `artifacts/models/model_current.pkl`
-> aplica `predict_proba`
-> threshold definido em [`configs/model_lifecycle/model_current.yaml`](../configs/model_lifecycle/model_current.yaml)
-> retorna [`ChurnPredictionResponse`](../src/serving/schemas.py)

**Entradas**

- payload HTTP contendo `customer_id`
- [`configs/model_lifecycle/model_current.yaml`](../configs/model_lifecycle/model_current.yaml)
- [`feature_store/repo.py`](../feature_store/repo.py)
- `feature_store/data/registry.db`
- Redis com features materializadas
- `artifacts/models/model_current.pkl`

**Saídas**

- resposta JSON com:
  - `churn_probability`
  - `churn_prediction`
  - `model_name`
  - `threshold`
  - `feature_source`
  - `customer_id`

**Observações**

- O serving não recalcula features na requisição.
- O fluxo principal consulta a camada online da Feature Store antes de aplicar o modelo.

### 1.3 `POST /predict/raw`

**Tipo:** online  
**Start:** chamada HTTP para `/predict/raw`

**Cadeia**

`POST /predict/raw`
-> schema [`ChurnPredictionRequest`](../src/serving/schemas.py)
-> rota [`predict_churn_from_raw`](../src/serving/routes.py)
-> função [`load_serving_config`](../src/serving/pipeline.py)
-> função [`prepare_request_inference_payload`](../src/serving/pipeline.py)
-> função [`prepare_inference_dataframe`](../src/serving/pipeline.py)
-> função [`load_feature_pipeline`](../src/serving/pipeline.py)
-> aplica `artifacts/models/feature_pipeline.joblib`
-> função [`predict_from_dataframe`](../src/serving/pipeline.py)
-> usa `artifacts/models/model_current.pkl`
-> retorna [`ChurnPredictionResponse`](../src/serving/schemas.py)

**Observações**

- Esta rota foi mantida como fallback legado e apoio didático.
- O fluxo preferencial do projeto para inferência online é o `/predict` baseado em `customer_id`.

### 1.4 `POST /predict` -> métricas Prometheus

**Tipo:** online passivo  
**Start:** a mesma chamada do `/predict`

**Cadeia**

`POST /predict`
-> função [`start_predict_request_for_monitor`](../src/monitoring/metrics.py)
-> incrementa gauge de requisições em andamento
-> executa inferência
-> função [`finish_predict_request_for_monitor`](../src/monitoring/metrics.py)
-> atualiza:
  - contador de requisições
  - histograma de latência
  - gauge de requisições em andamento

**Consulta**

`GET /metrics`
-> função [`register_prometheus_metrics`](../src/monitoring/metrics.py)
-> expõe métricas para scrape

**Arquivos envolvidos**

- [`src/serving/app.py`](../src/serving/app.py)
- [`src/serving/routes.py`](../src/serving/routes.py)
- [`src/monitoring/metrics.py`](../src/monitoring/metrics.py)

**Observações**

- O código da API expõe as métricas.
- O armazenamento histórico não é feito pela API; ele é responsabilidade do Prometheus.

### 1.5 `POST /predict` -> logging de inferência

**Tipo:** online passivo  
**Start:** a mesma chamada do `/predict`

**Cadeia**

`POST /predict`
-> rota [`predict_churn`](../src/serving/routes.py)
-> função [`log_prediction_for_monitoring`](../src/evaluation/model/drift/prediction_logger.py)
-> monta registro com features usadas + probabilidade + classe + metadados de origem
-> append em `artifacts/logs/inference/predictions.jsonl`

**Arquivos envolvidos**

- [`src/serving/routes.py`](../src/serving/routes.py)
- [`src/evaluation/model/drift/prediction_logger.py`](../src/evaluation/model/drift/prediction_logger.py)
- [`configs/monitoring/global_monitoring.yaml`](../configs/monitoring/global_monitoring.yaml)

**Observações**

- Este fluxo não calcula drift.
- Ele apenas produz a base current consumida depois pelo monitoramento batch.
- Quando a inferência vem do Feast, o log registra `feature_source=feast_online_store`.

## 2. Gatilhos de Features

### 2.1 Engenharia de features manual

**Tipo:** batch manual  
**Start:** `python -m src.feature_engineering.feature_engineering` ou `task mlfeateng`

**Cadeia**

`feature_engineering`
-> [`load_raw_data`](../src/common/data_loader.py)
-> valida schema raw em [`validate_raw_dataset_schema`](../src/feature_engineering/schema_validation.py)
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

- [`main`](../src/feature_engineering/feature_engineering.py)

**Arquivos envolvidos**

- [`src/feature_engineering/feature_engineering.py`](../src/feature_engineering/feature_engineering.py)
- [`src/feature_engineering/pipeline_components.py`](../src/feature_engineering/pipeline_components.py)
- [`src/feature_engineering/schema_validation.py`](../src/feature_engineering/schema_validation.py)
- [`configs/pipeline_global_config.yaml`](../configs/pipeline_global_config.yaml)

### 2.2 Engenharia de features via DVC

**Tipo:** batch manual orquestrado  
**Start:** `dvc repro featurize`

**Cadeia**

`dvc repro featurize`
-> stage `featurize` em [`dvc.yaml`](../dvc.yaml)
-> executa `python -m src.feature_engineering.feature_engineering`

**Observação**

- A lógica é a mesma do fluxo manual.
- O DVC apenas organiza dependências e saídas.

## 3. Gatilhos de Treino

### 3.1 Treino manual

**Tipo:** batch manual  
**Start:** `python -m src.model_lifecycle.train` ou `task mltrain`

**Cadeia**

`train`
-> carrega config do experimento em [`load_experiment_training_config`](../src/model_lifecycle/train.py)
-> lê:
  - `data/processed/train.parquet`
  - `data/processed/test.parquet`
-> instancia algoritmo via [`build_model`](../src/model_lifecycle/catalog.py)
-> treina modelo
-> calcula métricas
-> registra run no MLflow
-> salva:
  - `artifacts/models/model_current.pkl`
  - `artifacts/models/model_current_metadata.json`

**Função principal**

- [`run_training`](../src/model_lifecycle/train.py)

**Arquivos envolvidos**

- [`src/model_lifecycle/train.py`](../src/model_lifecycle/train.py)
- [`src/model_lifecycle/catalog.py`](../src/model_lifecycle/catalog.py)
- [`configs/model_lifecycle/model_current.yaml`](../configs/model_lifecycle/model_current.yaml)

### 3.2 Treino via DVC

**Tipo:** batch manual orquestrado  
**Start:** `dvc repro train`

**Cadeia**

`dvc repro train`
-> stage `train` em [`dvc.yaml`](../dvc.yaml)
-> executa `python -m src.model_lifecycle.train`

### 3.3 Treino múltiplo de experimentos

**Tipo:** batch manual composto  
**Start:** `task mlrunall`

**Cadeia**

`task mlrunall`
-> executa `task mlfeateng`
-> roda múltiplos `python -m src.model_lifecycle.train --config ...`
-> executa cenários de inferência

**Arquivo envolvido**

- [`pyproject.toml`](../pyproject.toml)

## 4. Gatilhos de Monitoramento

### 4.1 Drift monitoring manual

**Tipo:** batch manual  
**Start:** `python -m src.evaluation.model.drift.drift` ou `task mldrift`

**Cadeia**

`drift`
-> lê config em [`configs/monitoring/global_monitoring.yaml`](../configs/monitoring/global_monitoring.yaml)
-> carrega dataset de referência:
  - `data/processed/train.parquet`
-> carrega dataset current:
  - `artifacts/logs/inference/predictions.jsonl`
-> resolve matriz de features
-> usa `artifacts/models/feature_pipeline.joblib` quando necessário
-> calcula PSI por feature
-> calcula prediction drift
-> gera relatório oficial coerente com `drift_metrics.json`
-> gera relatório auxiliar do Evidently
-> decide status:
  - `ok`
  - `warning`
  - `critical`
  - `insufficient_data`
-> salva:
  - `artifacts/evaluation/model/drift/drift_report.html`
  - `artifacts/evaluation/model/drift/drift_report_evidently.html`
  - `artifacts/evaluation/model/drift/drift_metrics.json`
  - `artifacts/evaluation/model/drift/drift_status.json`
  - `artifacts/evaluation/model/drift/drift_runs.jsonl`

**Função principal**

- [`run_drift_monitoring`](../src/evaluation/model/drift/drift.py)

**Arquivos envolvidos**

- [`src/evaluation/model/drift/drift.py`](../src/evaluation/model/drift/drift.py)
- [`configs/monitoring/global_monitoring.yaml`](../configs/monitoring/global_monitoring.yaml)

### 4.2 Drift demo

**Tipo:** batch manual de demonstração  
**Start:** `task mldriftdemo`

**Cadeia**

`task mldriftdemo`
-> executa `python -m src.evaluation.model.drift.drift --current data/processed/test.parquet`

**Observação**

- Serve para testar o monitoramento mesmo sem usar o log real de inferências.

## 5. Gatilhos de Retreino

### 5.1 Retreino automático interno por drift crítico

**Tipo:** batch automático interno  
**Start:** execução de drift com status `critical`

**Cadeia**

`python -m src.evaluation.model.drift.drift`
-> função [`run_drift_monitoring`](../src/evaluation/model/drift/drift.py)
-> função [`maybe_trigger_retraining`](../src/evaluation/model/drift/drift.py)
-> cria `artifacts/evaluation/model/retraining/retrain_request.json`
-> como `trigger_mode` atual é `auto_train_manual_promote`
-> chama [`run_retraining_request`](../src/model_lifecycle/retraining.py)
-> treina challenger
-> avalia promoção champion-challenger
-> salva:
  - `artifacts/evaluation/model/retraining/retrain_run.json`
  - `artifacts/evaluation/model/retraining/promotion_decision.json`

**Observações**

- O retreino pode ser automático.
- A promoção do challenger para substituir o champion atual continua manual.

### 5.2 Retreino manual

**Tipo:** batch manual  
**Start:** `python -m src.model_lifecycle.retraining` ou `task mlretrain`

**Cadeia**

`model_lifecycle.retraining`
-> lê `retrain_request.json`
-> cria config temporária de challenger
-> chama [`run_training`](../src/model_lifecycle/train.py)
-> salva challenger em `artifacts/models/challengers/`
-> compara champion vs challenger
-> grava decisão de promoção

**Função principal**

- [`run_retraining_request`](../src/model_lifecycle/retraining.py)

**Arquivos envolvidos**

- [`src/model_lifecycle/retraining.py`](../src/model_lifecycle/retraining.py)
- [`src/model_lifecycle/promotion.py`](../src/model_lifecycle/promotion.py)
- [`src/model_lifecycle/train.py`](../src/model_lifecycle/train.py)

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

- [`export_features_for_feast`](../src/feast_ops/export.py)

**Arquivos envolvidos**

- [`src/feast_ops/export.py`](../src/feast_ops/export.py)
- [`src/feast_ops/config.py`](../src/feast_ops/config.py)
- [`docs/FEATURE_STORE.md`](FEATURE_STORE.md)

### 6.2 Export via DVC

**Tipo:** batch manual orquestrado  
**Start:** `dvc repro export_feature_store`

**Cadeia**

`dvc repro export_feature_store`
-> stage `export_feature_store` em [`dvc.yaml`](../dvc.yaml)
-> executa `python -m src.feast_ops.export`

**Observação importante**

- Esse passo prepara a camada offline do Feast, mas nao substitui `feast apply`
  nem `feast materialize-incremental`.
- Depois do `dvc repro`, o fluxo operacional esperado continua sendo:
  `task feastapply` -> `task feastmaterialize` -> uso do serving.

### 6.3 Feast apply

**Tipo:** infra manual  
**Start:** `task feastapply`

**Cadeia**

`feast -c feature_store apply`
-> lê [`feature_store/feature_store.yaml`](../feature_store/feature_store.yaml)
-> carrega definições em [`feature_store/repo.py`](../feature_store/repo.py)
-> registra `Entity`, `FeatureView` e `FeatureServices`
-> atualiza o registry local do Feast

### 6.4 Feast materialize

**Tipo:** infra manual  
**Start:** `task feastmaterialize`

**Cadeia**

`feast -c feature_store materialize-incremental ...`
-> lê `data/feature_store/customer_features.parquet`
-> identifica a janela incremental disponível
-> publica no Redis apenas o recorte necessário para serving online

**Observações**

- Este fluxo é manual no projeto atual.
- A online store não é atualizada automaticamente a cada `dvc repro`.
- O objetivo é evitar um fluxo destrutivo de limpeza total e recarga completa.
- O serving depende de o registry ja existir e de a online store ja estar
  materializada; ele nao deve assumir a responsabilidade de bootstrapar o Feast.

### 6.5 Feast demo

**Tipo:** batch/manual de demonstração  
**Start:** `task feastdemo`

**Cadeia**

`python -m src.feast_ops.demo --customer-id ...`
-> consulta features online no Redis

**Observação importante**

- A mesma base conceitual desse fluxo já é usada pela API no endpoint `/predict`.
- Para o endpoint online funcionar corretamente, a sequência recomendada e:
  `dvc repro export_feature_store` -> `task feastapply` ->
  `task feastmaterialize` -> subir/usar o serving.

## 7. Gatilhos de Cenários e Validação

### 7.1 Cenários de inferência

**Tipo:** batch manual  
**Start:** `python -m src.scenario_experiments.inference_cases` ou `task mlscenarios`

**Cadeia**

`scenario_experiments.inference_cases`
-> lê suíte YAML de cenários
-> monta payloads
-> usa o mesmo pipeline de serving
-> produz previsões para cenários de negócio

**Arquivos envolvidos**

- [`src/scenario_experiments/inference_cases.py`](../src/scenario_experiments/inference_cases.py)
- [`configs/scenario_experiments/inference_cases.yaml`](../configs/scenario_experiments/inference_cases.yaml)

### 7.2 Drift sintético

**Tipo:** batch manual  
**Start:** `python -m src.scenario_experiments.synthetic_drifts --all` ou `task mlsyntheticdrift`

**Cadeia**

`src.evaluation.model.drift.synthetic_drifts`
-> gera lotes sintéticos
-> carrega pipeline e modelo atuais
-> calcula previsões do lote
-> gera registros compatíveis com monitoramento
-> produz artefatos para validar o comportamento do drift batch

**Arquivos envolvidos**

- [`src/evaluation/model/drift/synthetic_drifts.py`](../src/evaluation/model/drift/synthetic_drifts.py)
- [`docs/SYNTHETIC_PREDICTIONS_GENERATOR.md`](SYNTHETIC_PREDICTIONS_GENERATOR.md)

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

- [`docker-compose.yml`](../docker-compose.yml)
- [`pyproject.toml`](../pyproject.toml)

### 8.2 MLflow local

**Tipo:** infra manual  
**Start:** `task mlflow`

**Cadeia**

`mlflow server ...`
-> sobe servidor local de tracking para treino e retreino

### 8.3 Índice RAG em container

**Tipo:** batch manual  
**Start:** `task rag_index_rebuild_docker`

**Cadeia**

`docker compose --profile rag run --rm rag-index`
-> usa a imagem do serving com FastEmbed/ONNX
-> descobre `README.md`, `docs/**/*.md` e JSON relevantes
-> gera embeddings dos chunks
-> salva `artifacts/rag/cache/index.joblib`
-> reaproveita `artifacts/rag/fastembed_model_cache`

**Arquivos envolvidos**

- [`docker-compose.yml`](../docker-compose.yml)
- [`infra/dockerfiles/serving/Dockerfile`](../infra/dockerfiles/serving/Dockerfile)
- [`src/agent/rag_pipeline.py`](../src/agent/rag_pipeline.py)
- [`pyproject.toml`](../pyproject.toml)

## 9. Gatilhos de Avaliação LLM

### 9.1 Prompt A/B offline

**Tipo:** batch manual  
**Start:** `python -m src.evaluation.llm_agent.ab_test_prompts` ou `task eval_ab_test_prompts`

**Cadeia**

`src.evaluation.llm_agent.ab_test_prompts`
-> carrega [`data/golden-set.json`](../data/golden-set.json)
-> para cada pergunta:
-> usa [`retrieve_contexts`](../src/agent/rag_pipeline.py)
-> executa 3 variantes de prompt com o mesmo `llm_provider`
-> calcula `keyword_coverage` contra a resposta de referência
-> opcionalmente roda `judge_one` com `--with-judge`
-> agrega ranking das variantes
-> salva `artifacts/evaluation/llm_agent/results/prompt_ab_results.json`
-> registra histórico em `artifacts/evaluation/llm_agent/runs/prompt_ab_runs.jsonl`

**Arquivos envolvidos**

- [`evaluation/ab_test_prompts.py`](../src/evaluation/llm_agent/ab_test_prompts.py)
- [`data/golden-set.json`](../data/golden-set.json)
- [`src/agent/rag_pipeline.py`](../src/agent/rag_pipeline.py)
- [`evaluation/llm_judge.py`](../src/evaluation/llm_agent/llm_judge.py)

**Observações**

- Este fluxo não participa da resposta online do agente.
- Ele funciona como benchmark offline e gatilho de qualidade para decidir qual
  prompt vale promover.
- O modo `--with-judge` aproxima o A/B dos critérios da etapa 3, mas continua
  sendo avaliação offline.

### 9.2 RAGAS em container dedicado

**Tipo:** batch manual  
**Start:** `task eval_ragas_docker`

**Cadeia**

`docker compose --profile evaluation run --rm evaluation python -m src.evaluation.llm_agent.ragas_eval`
-> usa a imagem `tc-fiap-evaluation`
-> mantém `sentence-transformers` e `torch` fora da imagem do serving
-> carrega [`data/golden-set.json`](../data/golden-set.json)
-> recupera contextos via [`retrieve_contexts`](../src/agent/rag_pipeline.py)
-> calcula métricas RAGAS
-> salva `artifacts/evaluation/llm_agent/results/ragas_scores.json`

**Arquivos envolvidos**

- [`docker-compose.yml`](../docker-compose.yml)
- [`infra/dockerfiles/evaluation/Dockerfile`](../infra/dockerfiles/evaluation/Dockerfile)
- [`src/evaluation/llm_agent/ragas_eval.py`](../src/evaluation/llm_agent/ragas_eval.py)
- [`pyproject.toml`](../pyproject.toml)

## 10. Mapa por Natureza do Gatilho

### Online

- `GET /health`
- `POST /predict`
- `GET /metrics`

### Batch manual

- `python -m src.feature_engineering.feature_engineering`
- `python -m src.model_lifecycle.train`
- `python -m src.evaluation.model.drift.drift`
- `python -m src.model_lifecycle.retraining`
- `python -m src.feast_ops.export`
- `python -m src.scenario_experiments.inference_cases`
- `python -m src.scenario_experiments.synthetic_drifts --all`
- `python -m src.evaluation.llm_agent.ab_test_prompts`
- `task eval_ab_test_prompts`

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

## 11. O Que Não Existe Hoje

Atualmente o repositório **não possui**:

- Airflow
- cron formal
- scheduler interno próprio
- promoção automática do challenger para substituir `model_current.pkl`
- serving consultando Feast online em produção

## 12. Resumo Executivo

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
