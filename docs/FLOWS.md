# Mapa de Flows do Projeto

## Índice

- [Leitura Rápida](#leitura-rápida)
- [Visão Geral](#visão-geral)
- [Quadro Manual x Automático](#quadro-manual-x-automático)
- [Quadro de Criticidade e Ciclo Sugerido](#quadro-de-criticidade-e-ciclo-sugerido)
- [1. Flows Automáticos da API](#1-flows-automáticos-da-api)
- [2. Flows de Feature Engineering e Feature Store](#2-flows-de-feature-engineering-e-feature-store)
- [3. Flows de Models: Treino, Retreino e Promoção](#3-flows-de-models-treino-retreino-e-promoção)
- [4. Flows de Experimentos e Cenários](#4-flows-de-experimentos-e-cenários)
- [5. Flows do Agente LLM ReAct e RAG](#5-flows-do-agente-llm-react-e-rag)
- [6. Flows de Avaliação e Monitoramento](#6-flows-de-avaliação-e-monitoramento)
- [7. Flows Utilitários e Infraestrutura Local](#7-flows-utilitários-e-infraestrutura-local)
- [8. O Que Não Existe](#8-o-que-não-existe)
- [9. Resumo Executivo](#9-resumo-executivo)

Este documento mapeia os principais **flows** do projeto: pontos de entrada,
cadeias de execução, entradas, saídas e artefatos gerados. O termo "flow" é usado
de propósito porque o projeto mistura pipelines batch, rotas online, tarefas DVC,
scripts operacionais, monitoramento passivo e automações internas.

## Leitura Rápida

| Flow | Grupo | Modo | Start | Resultado principal |
|---|---|---:|---|---|
| Healthcheck da API | API | Automático online | `GET /health` | Retorna status do serviço |
| Predição online por `customer_id` | API | Automático online | `POST /predict` | Consulta Feast online e retorna classe/probabilidade |
| Predição legada por payload bruto | API | Automático online | `POST /predict/raw` | Aplica pipeline local e retorna classe/probabilidade |
| Métricas Prometheus | API/monitoramento | Automático passivo | `GET /metrics` e `/predict` | Expõe contadores, gauges e latência |
| Logging de inferência | API/monitoramento | Automático passivo | `POST /predict` | Salva `predictions.jsonl` |
| Engenharia de features | Feature engineering | Manual ou DVC | `task mlfeateng` / `dvc repro featurize` | Gera `interim`, `processed` e `feature_pipeline.joblib` |
| Export da Feature Store | Feature store | Manual ou DVC | `task feastexport` / `dvc repro export_feature_store` | Gera parquet offline do Feast |
| Feast apply | Feature store/infra | Manual | `task feastapply` | Registra definições no registry |
| Feast materialize | Feature store/infra | Manual | `task feastmaterialize` | Materializa features no Redis |
| Treino champion | Models | Manual ou DVC | `task mlflowrain` / `dvc repro train` | Gera `model_current.pkl` e metadados |
| Treino múltiplo de experimentos | Experimentos/models | Manual composto | `task mlflowrunexperiments` | Registra runs comparáveis no MLflow |
| Cenários de inferência | Experimentos | Manual | `task mlflowscenarios` | Valida casos de negócio |
| Drift sintético | Experimentos/monitoramento | Manual | `task mlflowsyntheticdrift` | Gera lotes e relatórios de drift sintético |
| Drift monitoring | Avaliação/monitoramento | Manual batch | `task mldrift` | Gera relatórios, métricas e status de drift |
| Retreino por drift crítico | Models/monitoramento | Automático interno | status `critical` no drift | Treina challenger e grava decisão |
| Retreino manual | Models | Manual | `task mlflowretrain` | Executa request de retreino existente |
| Índice RAG | Agente LLM | Manual batch | `task rag_index_rebuild_docker` | Gera cache vetorial do RAG |
| Avaliação LLM completa | Avaliação LLM | Manual batch | `task eval_all` | Gera RAGAS, LLM-as-judge e Prompt A/B |
| Stack local | Infra | Manual | `task appstack` | Sobe Redis, serving, MLflow, Prometheus e Grafana |

## Visão Geral

```text
data/raw
  -> feature engineering
  -> data/interim + data/processed + artifacts/models/feature_pipeline.joblib
  -> treino
  -> artifacts/models/model_current.pkl
  -> export Feature Store
  -> Feast registry + Redis
  -> serving /predict
  -> logs e métricas
  -> drift monitoring
  -> retreino challenger
  -> decisão champion-challenger
```

```text
docs + data/golden-set.json
  -> índice RAG
  -> agente ReAct online
  -> avaliações offline LLM
  -> resultados em artifacts/evaluation/llm_agent
```

## Quadro Manual x Automático

| Natureza | Flows | Observação operacional |
|---|---|---|
| Automático online | `/health`, `/predict`, `/predict/raw`, `/metrics`, rotas LLM | Acontecem quando a API recebe chamadas HTTP. |
| Automático passivo | métricas Prometheus e logging de inferência | São efeitos colaterais do serving; não iniciam treino nem drift sozinhos. |
| Automático interno | retreino por drift crítico | Ocorre dentro do flow batch de drift quando a configuração permite. |
| Manual batch | feature engineering, treino, export Feast, drift, avaliações LLM, cenários | Dependem de comando local, DVC, taskipy ou container dedicado. |
| Manual infra | `appstack`, `feastapply`, `feastmaterialize`, `mlflow`, RAG em container | Preparam serviços e stores usados pelos flows online/batch. |
| Manual orquestrado por DVC | `featurize`, `train`, `export_feature_store` | Controla dependências e saídas, mas não substitui cron, Airflow ou scheduler. |

## Quadro de Criticidade e Ciclo Sugerido

| Flow | Deve executar ao menos uma vez? | Ciclo sugerido para DAG/cron futuro | Por quê |
|---|---:|---|---|
| Feature engineering | Sim | Sob demanda ou a cada nova base raw | Produz datasets processados e pipeline de features. |
| Treino champion | Sim | Sob demanda, release de modelo ou nova safra relevante | Produz o modelo ativo usado pelo serving. |
| Export Feature Store | Sim, se usar `/predict` | Após feature engineering e antes da materialização | Prepara a base offline consumida pelo Feast. |
| Feast apply | Sim, se usar `/predict` | Após mudança em `feature_store/repo.py` ou bootstrap | Cria/atualiza registry do Feast. |
| Feast materialize | Sim, se usar `/predict` | Após export ou em janela incremental recorrente | Publica features online no Redis. |
| Stack local | Sim para uso local completo | Sob demanda | Sobe os serviços necessários para API e observabilidade. |
| Drift monitoring | Recomendado | Diário, semanal ou por volume mínimo de inferências | Calcula saúde do modelo e pode abrir retreino. |
| Retreino | Condicional | Quando drift for crítico ou métrica degradar | Gera challenger e decisão auditável. |
| Promoção champion | Condicional e manual | Após revisão da decisão champion-challenger | Evita troca automática de `model_current.pkl`. |
| Cenários de inferência | Recomendado | Pré-release ou após mudança de modelo/features | Valida comportamento em casos de negócio. |
| Avaliação LLM | Recomendado | Pré-release do agente ou mudança de prompt/RAG | Mede qualidade do agente contra golden set. |
| Índice RAG | Sim, se usar agente com RAG | Após mudança relevante em docs/dados indexados | Atualiza contexto recuperável pelo agente. |

## 1. Flows Automáticos da API

### 1.1 Healthcheck

**Tipo:** online automático
**Start:** chamada HTTP para `GET /health`

**Cadeia**

`GET /health`
-> rota [`healthcheck`](../src/serving/routes.py)
-> retorna `{"status": "ok"}`

**Entradas**

- processo do serving ativo

**Saídas**

- resposta JSON de saúde da API

**Arquivos envolvidos**

- [`src/serving/app.py`](../src/serving/app.py)
- [`src/serving/routes.py`](../src/serving/routes.py)

### 1.2 Predição online por `customer_id`

**Tipo:** online automático
**Start:** chamada HTTP para `POST /predict`

**Cadeia**

`POST /predict`
-> schema [`ChurnCustomerLookupRequest`](../src/serving/schemas.py)
-> rota [`predict_churn`](../src/serving/routes.py)
-> função [`load_serving_config`](../src/serving/pipeline.py)
-> função [`prepare_online_inference_payload`](../src/serving/pipeline.py)
-> função [`fetch_online_features_from_feast`](../src/serving/pipeline.py)
-> resolve o `FeatureService` do modelo ativo
-> consulta a online store Redis pelo `customer_id`
-> função [`predict_from_dataframe`](../src/serving/pipeline.py)
-> função [`load_prediction_model`](../src/serving/pipeline.py)
-> usa `artifacts/models/model_current.pkl`
-> aplica `predict_proba`
-> usa threshold em [`configs/model_lifecycle/model_current.yaml`](../configs/model_lifecycle/model_current.yaml)
-> retorna [`ChurnPredictionResponse`](../src/serving/schemas.py)

**Entradas**

- payload HTTP contendo `customer_id`
- [`configs/model_lifecycle/model_current.yaml`](../configs/model_lifecycle/model_current.yaml)
- [`feature_store/repo.py`](../feature_store/repo.py)
- `feature_store/data/registry.db`
- Redis com features materializadas
- `artifacts/models/model_current.pkl`

**Saídas**

- resposta JSON com `churn_probability`, `churn_prediction`, `model_name`,
  `threshold`, `feature_source` e `customer_id`
- métricas Prometheus atualizadas
- registro em `artifacts/logs/inference/predictions.jsonl`

**Observações**

- O serving não recalcula features na requisição.
- Este é o flow preferencial de inferência online.
- A API depende de Feast registry e Redis preparados pelos flows de Feature Store.

### 1.3 Predição legada por payload bruto

**Tipo:** online automático
**Start:** chamada HTTP para `POST /predict/raw`

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

**Entradas**

- payload bruto com atributos do cliente
- `artifacts/models/feature_pipeline.joblib`
- `artifacts/models/model_current.pkl`
- [`configs/model_lifecycle/model_current.yaml`](../configs/model_lifecycle/model_current.yaml)

**Saídas**

- resposta JSON de predição

**Observações**

- Foi mantido como fallback legado e apoio didático.
- Não é o caminho principal quando a Feature Store online está disponível.

### 1.4 Métricas Prometheus

**Tipo:** online/passivo automático
**Start:** chamadas em `/predict` e scrape em `GET /metrics`

**Cadeia**

`POST /predict`
-> função [`start_predict_request_for_monitor`](../src/monitoring/metrics.py)
-> executa inferência
-> função [`finish_predict_request_for_monitor`](../src/monitoring/metrics.py)
-> atualiza contador de requisições, histograma de latência e gauge de requisições
em andamento

`GET /metrics`
-> função [`register_prometheus_metrics`](../src/monitoring/metrics.py)
-> expõe métricas para scrape

**Entradas**

- tráfego da API
- scrape do Prometheus configurado em [`configs/monitoring/prometheus.yml`](../configs/monitoring/prometheus.yml)

**Saídas**

- endpoint `/metrics`
- séries temporais armazenadas pelo Prometheus

**Arquivos envolvidos**

- [`src/serving/app.py`](../src/serving/app.py)
- [`src/serving/routes.py`](../src/serving/routes.py)
- [`src/monitoring/metrics.py`](../src/monitoring/metrics.py)

### 1.5 Logging de inferência para drift

**Tipo:** online/passivo automático
**Start:** chamada HTTP para `POST /predict`

**Cadeia**

`POST /predict`
-> rota [`predict_churn`](../src/serving/routes.py)
-> função [`log_prediction_for_monitoring`](../src/evaluation/model/drift/prediction_logger.py)
-> monta registro com features, probabilidade, classe e metadados de origem
-> append em `artifacts/logs/inference/predictions.jsonl`

**Entradas**

- payload e features resolvidas durante a inferência
- [`configs/monitoring/global_monitoring.yaml`](../configs/monitoring/global_monitoring.yaml)

**Saídas**

- `artifacts/logs/inference/predictions.jsonl`

**Observações**

- Este flow não calcula drift.
- Ele produz a base current consumida pelo monitoramento batch.
- Quando a inferência vem do Feast, o log registra `feature_source=feast_online_store`.

### 1.6 Rotas online do Agente LLM

**Tipo:** online automático
**Start:** chamadas HTTP para rotas LLM do serving

**Cadeia**

rota LLM
-> [`src/serving/llm_routes.py`](../src/serving/llm_routes.py)
-> agente ReAct em [`src/agent/react_agent.py`](../src/agent/react_agent.py)
-> ferramentas em [`src/agent/tools.py`](../src/agent/tools.py)
-> provider LLM via [`src/agent/llm_gateway/factory.py`](../src/agent/llm_gateway/factory.py)

**Entradas**

- pergunta do usuário
- configuração global em [`configs/pipeline_global_config.yaml`](../configs/pipeline_global_config.yaml)
- índice RAG em `artifacts/rag/cache/index.joblib`, quando disponível
- artefatos consultados pelas tools, como drift, cenários e documentação

**Saídas**

- resposta do agente
- diagnósticos de status do provider, quando a rota consultada for de status

**Observações**

- O agente pode consultar ferramentas que leem artefatos do projeto.
- O índice RAG é preparado por flow manual; a rota online apenas consome o cache.

## 2. Flows de Feature Engineering e Feature Store

### 2.1 Engenharia de features

**Tipo:** batch manual ou manual orquestrado por DVC
**Start:** `python -m src.feature_engineering.feature_engineering`, `task mlfeateng`
ou `dvc repro featurize`

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
-> salva artefatos processados

**Entradas**

- `data/raw/Customer-Churn-Records.csv`
- [`configs/pipeline_global_config.yaml`](../configs/pipeline_global_config.yaml)
- [`src/feature_engineering/feature_engineering.py`](../src/feature_engineering/feature_engineering.py)
- [`src/feature_engineering/pipeline_components.py`](../src/feature_engineering/pipeline_components.py)
- [`src/feature_engineering/schema_validation.py`](../src/feature_engineering/schema_validation.py)

**Saídas**

- `data/interim/cleaned.parquet`
- `data/processed/train.parquet`
- `data/processed/test.parquet`
- `data/processed/feature_columns.json`
- `data/processed/schema_report.json`
- `artifacts/models/feature_pipeline.joblib`

**DVC**

- stage `featurize` em [`dvc.yaml`](../dvc.yaml)
- o DVC organiza dependências e saídas, mas a lógica executada é a mesma do flow manual

### 2.2 Export offline da Feature Store

**Tipo:** batch manual ou manual orquestrado por DVC
**Start:** `python -m src.feast_ops.export`, `task feastexport` ou
`dvc repro export_feature_store`

**Cadeia**

`feast export`
-> lê dado bruto
-> reaproveita `artifacts/models/feature_pipeline.joblib`
-> transforma features
-> seleciona colunas online
-> gera parquet offline e metadados

**Entradas**

- `data/raw/Customer-Churn-Records.csv`
- `artifacts/models/feature_pipeline.joblib`
- [`configs/pipeline_global_config.yaml`](../configs/pipeline_global_config.yaml)
- [`src/feast_ops/export.py`](../src/feast_ops/export.py)
- [`src/feast_ops/config.py`](../src/feast_ops/config.py)

**Saídas**

- `data/feature_store/customer_features.parquet`
- `data/feature_store/export_metadata.json`

**DVC**

- stage `export_feature_store` em [`dvc.yaml`](../dvc.yaml)
- depois do `dvc repro`, o fluxo operacional esperado e:
  `task feastapply` -> `task feastmaterialize` -> uso do serving

### 2.3 Feast apply

**Tipo:** infra manual
**Start:** `task feastapply`

**Cadeia**

`feast -c feature_store apply`
-> lê [`feature_store/feature_store.yaml`](../feature_store/feature_store.yaml)
-> carrega definições em [`feature_store/repo.py`](../feature_store/repo.py)
-> registra `Entity`, `FeatureView` e `FeatureServices`
-> atualiza o registry local do Feast

**Entradas**

- [`feature_store/feature_store.yaml`](../feature_store/feature_store.yaml)
- [`feature_store/repo.py`](../feature_store/repo.py)
- conexão Redis configurada pela variável `FEAST_REDIS_CONNECTION_STRING`

**Saídas**

- `feature_store/data/registry.db`

### 2.4 Feast materialize

**Tipo:** infra manual
**Start:** `task feastmaterialize`

**Cadeia**

`feast -c feature_store materialize-incremental ...`
-> lê `data/feature_store/customer_features.parquet`
-> identifica a janela incremental disponível
-> publica no Redis o recorte necessário para serving online

**Entradas**

- `data/feature_store/customer_features.parquet`
- `feature_store/data/registry.db`
- Redis ativo

**Saídas**

- features materializadas na online store Redis

**Observações**

- Este flow e manual.
- A online store não é atualizada automaticamente a cada `dvc repro`.
- O serving depende de o registry existir e de a online store estar materializada.

### 2.5 Feast demo

**Tipo:** manual de demonstração
**Start:** `task feastdemo`

**Cadeia**

`python -m src.feast_ops.demo --customer-id ...`
-> consulta features online no Redis

**Entradas**

- Redis com features materializadas
- Feast registry local
- [`src/feast_ops/demo.py`](../src/feast_ops/demo.py)

**Saídas**

- amostra de features online para um `customer_id`

## 3. Flows de Models: Treino, Retreino e Promoção

### 3.1 Treino champion

**Tipo:** batch manual ou manual orquestrado por DVC
**Start:** `python -m src.model_lifecycle.train`, `task mlflowrain` ou
`dvc repro train`

**Cadeia**

`train`
-> carrega config do experimento em [`load_experiment_training_config`](../src/model_lifecycle/train.py)
-> lê `data/processed/train.parquet` e `data/processed/test.parquet`
-> instancia algoritmo via [`build_model`](../src/model_lifecycle/catalog.py)
-> treina modelo
-> calcula métricas
-> registra run no MLflow
-> salva modelo e metadados atuais

**Entradas**

- `data/processed/train.parquet`
- `data/processed/test.parquet`
- [`configs/model_lifecycle/model_current.yaml`](../configs/model_lifecycle/model_current.yaml)
- [`configs/pipeline_global_config.yaml`](../configs/pipeline_global_config.yaml)
- [`src/model_lifecycle/train.py`](../src/model_lifecycle/train.py)
- [`src/model_lifecycle/catalog.py`](../src/model_lifecycle/catalog.py)

**Saídas**

- `artifacts/models/model_current.pkl`
- `artifacts/models/model_current_metadata.json`
- run no MLflow, quando o tracking server estiver configurado

**DVC**

- stage `train` em [`dvc.yaml`](../dvc.yaml)

### 3.2 Retreino automático interno por drift crítico

**Tipo:** batch automático interno
**Start:** execução de drift com status `critical`

**Cadeia**

`python -m src.evaluation.model.drift.drift`
-> função [`run_drift_monitoring`](../src/evaluation/model/drift/drift.py)
-> função [`maybe_trigger_retraining`](../src/evaluation/model/drift/drift.py)
-> cria `artifacts/evaluation/model/retraining/retrain_request.json`
-> como `trigger_mode` e `auto_train_manual_promote`
-> chama [`run_retraining_request`](../src/model_lifecycle/retraining.py)
-> treina challenger
-> avalia promoção champion-challenger
-> salva decisão

**Entradas**

- status crítico gerado pelo flow de drift
- [`configs/monitoring/global_monitoring.yaml`](../configs/monitoring/global_monitoring.yaml)
- `data/processed/train.parquet`
- `data/processed/test.parquet`
- `artifacts/models/model_current.pkl`

**Saídas**

- `artifacts/evaluation/model/retraining/retrain_request.json`
- `artifacts/evaluation/model/retraining/retrain_run.json`
- `artifacts/evaluation/model/retraining/promotion_decision.json`
- challenger em `artifacts/models/challengers/`

**Observações**

- O retreino pode ser automático dentro da execução do drift.
- A promoção do challenger para substituir o champion permanece manual.

### 3.3 Retreino manual

**Tipo:** batch manual
**Start:** `python -m src.model_lifecycle.retraining` ou `task mlflowretrain`

**Cadeia**

`model_lifecycle.retraining`
-> lê `retrain_request.json`
-> cria config temporária de challenger
-> chama [`run_training`](../src/model_lifecycle/train.py)
-> salva challenger em `artifacts/models/challengers/`
-> compara champion vs challenger
-> grava decisão de promoção

**Entradas**

- `artifacts/evaluation/model/retraining/retrain_request.json`
- [`src/model_lifecycle/retraining.py`](../src/model_lifecycle/retraining.py)
- [`src/model_lifecycle/promotion.py`](../src/model_lifecycle/promotion.py)
- [`src/model_lifecycle/train.py`](../src/model_lifecycle/train.py)

**Saídas**

- `artifacts/evaluation/model/retraining/retrain_run.json`
- `artifacts/evaluation/model/retraining/promotion_decision.json`
- challenger em `artifacts/models/challengers/`

### 3.4 Promoção champion-challenger

**Tipo:** manual/auditável
**Start:** revisão da decisão em `promotion_decision.json`

**Cadeia**

decisão de promoção
-> revisão humana
-> promoção manual do challenger, se aprovado
-> atualização do modelo ativo usado pelo serving

**Entradas**

- `artifacts/evaluation/model/retraining/promotion_decision.json`
- `artifacts/evaluation/model/retraining/retrain_run.json`
- `artifacts/models/challengers/`
- [`src/model_lifecycle/promotion.py`](../src/model_lifecycle/promotion.py)

**Saídas**

- potencial atualização de `artifacts/models/model_current.pkl`
- potencial atualização de `artifacts/models/model_current_metadata.json`

**Observações**

- O projeto não promove automaticamente o challenger.
- A separação é intencional para manter auditoria e controle operacional.

## 4. Flows de Experimentos e Cenários

### 4.1 Treino múltiplo de experimentos

**Tipo:** batch manual composto
**Start:** `task mlflowrunexperiments` ou `task mlflowrunall`

**Cadeia**

`task mlflowrunexperiments`
-> executa treino do modelo atual
-> roda configs em [`configs/model_lifecycle/experiments/`](../configs/model_lifecycle/experiments)
-> registra runs comparáveis no MLflow

`task mlflowrunall`
-> executa a sequência de experimentos
-> executa cenários de inferência

**Entradas**

- `data/processed/train.parquet`
- `data/processed/test.parquet`
- configs de experimento em [`configs/model_lifecycle/experiments/`](../configs/model_lifecycle/experiments)
- tasks definidas em [`pyproject.toml`](../pyproject.toml)

**Saídas**

- runs no MLflow
- modelos e métricas conforme cada execução
- cenários de inferência, quando usar `task mlflowrunall`

### 4.2 Cenários de inferência

**Tipo:** batch manual
**Start:** `python -m src.scenario_experiments.inference_cases` ou
`task mlflowscenarios`

**Cadeia**

`scenario_experiments.inference_cases`
-> lê suíte YAML de cenários
-> monta payloads
-> usa o mesmo pipeline de serving
-> produz previsões para cenários de negócio

**Entradas**

- [`configs/scenario_experiments/inference_cases.yaml`](../configs/scenario_experiments/inference_cases.yaml)
- [`src/scenario_experiments/inference_cases.py`](../src/scenario_experiments/inference_cases.py)
- `artifacts/models/feature_pipeline.joblib`
- `artifacts/models/model_current.pkl`

**Saídas**

- previsões e evidências dos cenários de negócio
- runs no MLflow, quando configurado

### 4.3 Drift sintético

**Tipo:** batch manual
**Start:** `python -m src.evaluation.model.drift.synthetic_drifts --all` ou
`task mlflowsyntheticdrift`

**Cadeia**

`src.evaluation.model.drift.synthetic_drifts`
-> gera lotes sintéticos
-> carrega pipeline e modelo atuais
-> calcula previsões do lote
-> gera registros compatíveis com monitoramento
-> produz artefatos para validar o comportamento do drift batch

**Entradas**

- [`src/evaluation/model/drift/synthetic_drifts.py`](../src/evaluation/model/drift/synthetic_drifts.py)
- `data/processed/test.parquet`
- `artifacts/models/feature_pipeline.joblib`
- `artifacts/models/model_current.pkl`
- [`configs/monitoring/global_monitoring.yaml`](../configs/monitoring/global_monitoring.yaml)

**Saídas**

- lotes JSONL sintéticos em `artifacts/evaluation/model/scenario_experiments/drift/`
- relatórios HTML por cenário
- histórico no MLflow, quando configurado

**Documentação relacionada**

- [`docs/SYNTHETIC_PREDICTIONS_GENERATOR.md`](SYNTHETIC_PREDICTIONS_GENERATOR.md)
- [`docs/SCENARIO_ANALYSIS.md`](SCENARIO_ANALYSIS.md)

## 5. Flows do Agente LLM ReAct e RAG

### 5.1 Índice RAG

**Tipo:** batch manual
**Start:** `python -m src.agent.rag_pipeline`, `task rag_index_rebuild` ou
`task rag_index_rebuild_docker`

**Cadeia**

`rag_pipeline`
-> descobre `README.md`, `docs/**/*.md` e JSON relevantes
-> quebra documentos em chunks
-> gera embeddings
-> salva cache vetorial reutilizável

**Entradas**

- [`README.md`](../README.md)
- [`docs/`](.)
- `data/golden-set.json`
- [`src/agent/rag_pipeline.py`](../src/agent/rag_pipeline.py)
- configs LLM/RAG em [`configs/pipeline_global_config.yaml`](../configs/pipeline_global_config.yaml)

**Saídas**

- `artifacts/rag/cache/index.joblib`
- `artifacts/rag/fastembed_model_cache`

### 5.2 Agente ReAct online

**Tipo:** online automático
**Start:** rota LLM no serving

**Cadeia**

rota LLM
-> [`src/agent/react_agent.py`](../src/agent/react_agent.py)
-> seleção de tool
-> [`src/agent/tools.py`](../src/agent/tools.py)
-> provider em [`src/agent/llm_gateway/providers/`](../src/agent/llm_gateway/providers)
-> resposta final

**Entradas**

- pergunta do usuário
- índice RAG, quando necessário para `rag_search`
- `artifacts/evaluation/model/drift/drift_status.json`, quando a tool `drift_status` for usada
- pipeline e modelo atuais, quando a tool `predict_churn` ou `scenario_prediction` for usada

**Saídas**

- resposta estruturada do agente
- chamadas internas às tools, sem alterar o modelo champion

**Observações**

- As tools do agente reutilizam componentes do projeto em vez de duplicar lógica.
- O agente online depende da preparação prévia dos artefatos usados pelas tools.

## 6. Flows de Avaliação e Monitoramento

### 6.1 Drift monitoring

**Tipo:** batch manual
**Start:** `python -m src.evaluation.model.drift.drift`, `task mldrift` ou
`task mldriftdemo`

**Cadeia**

`drift`
-> lê config em [`configs/monitoring/global_monitoring.yaml`](../configs/monitoring/global_monitoring.yaml)
-> carrega dataset de referência em `data/processed/train.parquet`
-> carrega dataset current em `artifacts/logs/inference/predictions.jsonl`
-> resolve matriz de features
-> usa `artifacts/models/feature_pipeline.joblib` quando necessário
-> calcula PSI por feature
-> calcula prediction drift
-> gera relatório oficial coerente com `drift_metrics.json`
-> gera relatório auxiliar do Evidently
-> decide status
-> pode acionar retreino automático interno se o status for crítico

**Entradas**

- `data/processed/train.parquet`
- `artifacts/logs/inference/predictions.jsonl`
- `artifacts/models/feature_pipeline.joblib`
- [`configs/monitoring/global_monitoring.yaml`](../configs/monitoring/global_monitoring.yaml)
- [`src/evaluation/model/drift/drift.py`](../src/evaluation/model/drift/drift.py)

**Saídas**

- `artifacts/evaluation/model/drift/drift_report.html`
- `artifacts/evaluation/model/drift/drift_report_evidently.html`
- `artifacts/evaluation/model/drift/drift_metrics.json`
- `artifacts/evaluation/model/drift/drift_status.json`
- `artifacts/evaluation/model/drift/drift_runs.jsonl`
- artefatos de retreino, quando o status crítico acionar o flow interno

**Status possíveis**

- `ok`
- `warning`
- `critical`
- `insufficient_data`

**Observações**

- `task mldriftdemo` executa o drift com `data/processed/test.parquet` como current
  para demonstração, mesmo sem log real de inferências.
- Este é o principal candidato a cron/DAG recorrente no projeto.

### 6.2 Avaliação LLM completa

**Tipo:** batch manual
**Start:** `task eval_all`, `task eval_all_offline`, `task eval_all_docker` ou
variantes `sample`

**Cadeia**

`task eval_all`
-> executa [`src/evaluation/llm_agent/ragas_eval.py`](../src/evaluation/llm_agent/ragas_eval.py)
-> executa [`src/evaluation/llm_agent/llm_judge.py`](../src/evaluation/llm_agent/llm_judge.py)
-> executa [`src/evaluation/llm_agent/ab_test_prompts.py`](../src/evaluation/llm_agent/ab_test_prompts.py)

**Entradas**

- [`data/golden-set.json`](../data/golden-set.json)
- [`src/agent/rag_pipeline.py`](../src/agent/rag_pipeline.py)
- provider LLM configurado em [`configs/pipeline_global_config.yaml`](../configs/pipeline_global_config.yaml)
- tasks em [`pyproject.toml`](../pyproject.toml)

**Saídas**

- `artifacts/evaluation/llm_agent/results/ragas_scores.json`
- `artifacts/evaluation/llm_agent/results/llm_judge_scores.json`
- `artifacts/evaluation/llm_agent/results/prompt_ab_results.json`
- histórico em `artifacts/evaluation/llm_agent/runs/`

### 6.3 Prompt A/B offline

**Tipo:** batch manual
**Start:** `python -m src.evaluation.llm_agent.ab_test_prompts` ou
`task eval_ab_test_prompts`

**Cadeia**

`src.evaluation.llm_agent.ab_test_prompts`
-> carrega [`data/golden-set.json`](../data/golden-set.json)
-> para cada pergunta, usa [`retrieve_contexts`](../src/agent/rag_pipeline.py)
-> executa 3 variantes de prompt com o mesmo `llm_provider`
-> calcula `keyword_coverage` contra a resposta de referência
-> opcionalmente roda `judge_one` com `--with-judge`
-> agrega ranking das variantes
-> salva resultado e histórico

**Entradas**

- [`data/golden-set.json`](../data/golden-set.json)
- [`src/evaluation/llm_agent/ab_test_prompts.py`](../src/evaluation/llm_agent/ab_test_prompts.py)
- [`src/evaluation/llm_agent/llm_judge.py`](../src/evaluation/llm_agent/llm_judge.py)
- [`src/agent/rag_pipeline.py`](../src/agent/rag_pipeline.py)

**Saídas**

- `artifacts/evaluation/llm_agent/results/prompt_ab_results.json`
- `artifacts/evaluation/llm_agent/runs/prompt_ab_runs.jsonl`

**Observações**

- Este flow não participa da resposta online do agente.
- Ele funciona como benchmark offline e controle de qualidade para decidir qual
  prompt vale promover.

### 6.4 RAGAS em container dedicado

**Tipo:** batch manual
**Start:** `task eval_ragas_docker`

**Cadeia**

`docker compose --profile evaluation run --rm -e RAGAS_SERVING_BASE_URL=http://serving:8000 evaluation python -m src.evaluation.llm_agent.ragas_eval`
-> usa a imagem `tc-fiap-evaluation`
-> usa FastEmbed para embeddings, sem `sentence-transformers` nem `torch`
-> carrega [`data/golden-set.json`](../data/golden-set.json)
-> chama `POST /llm/chat` no serving avaliado
-> extrai os contextos da trace de `rag_search`
-> calcula métricas RAGAS
-> salva resultados

**Entradas**

- [`docker-compose.yml`](../docker-compose.yml)
- [`infra/dockerfiles/evaluation/Dockerfile`](../infra/dockerfiles/evaluation/Dockerfile)
- [`src/evaluation/llm_agent/ragas_eval.py`](../src/evaluation/llm_agent/ragas_eval.py)
- [`pyproject.toml`](../pyproject.toml)

**Saídas**

- `artifacts/evaluation/llm_agent/results/ragas_scores.json`
- histórico em `artifacts/evaluation/llm_agent/runs/ragas_runs.jsonl`

### 6.5 Dashboards e observabilidade local

**Tipo:** infra/monitoramento manual
**Start:** `task appstack` e scrape do Prometheus

**Cadeia**

`task appstack`
-> sobe Prometheus e Grafana
-> Prometheus lê [`configs/monitoring/prometheus.yml`](../configs/monitoring/prometheus.yml)
-> coleta `/metrics` do serving
-> Grafana provisiona dashboards e datasource

**Entradas**

- [`configs/monitoring/prometheus.yml`](../configs/monitoring/prometheus.yml)
- [`configs/monitoring/grafana/provisioning/dashboards/dashboards.yml`](../configs/monitoring/grafana/provisioning/dashboards/dashboards.yml)
- [`configs/monitoring/grafana/provisioning/datasources/prometheus.yml`](../configs/monitoring/grafana/provisioning/datasources/prometheus.yml)
- [`src/monitoring/metrics.py`](../src/monitoring/metrics.py)

**Saídas**

- métricas Prometheus
- dashboards Grafana provisionados

## 7. Flows Utilitários e Infraestrutura Local

### 7.1 Stack local

**Tipo:** infra manual
**Start:** `task appstack`, `task appstack_dev`, `task appstack_ollama` ou variantes
`*_rebuild`

**Cadeia**

`docker compose up`
-> sobe Redis
-> sobe serving
-> sobe MLflow
-> sobe Prometheus
-> sobe Grafana
-> opcionalmente sobe Ollama, conforme override

**Entradas**

- [`docker-compose.yml`](../docker-compose.yml)
- [`docker-compose.dev.yml`](../docker-compose.dev.yml)
- [`docker-compose.ollama.yml`](../docker-compose.ollama.yml)
- [`infra/dockerfiles/serving/Dockerfile`](../infra/dockerfiles/serving/Dockerfile)
- [`infra/dockerfiles/mlflow/Dockerfile`](../infra/dockerfiles/mlflow/Dockerfile)
- tasks em [`pyproject.toml`](../pyproject.toml)

**Saídas**

- serviços locais ativos

### 7.2 MLflow local

**Tipo:** infra manual
**Start:** `task mlflow`

**Cadeia**

`mlflow server ...`
-> sobe servidor local de tracking para treino, experimentos e retreino

**Entradas**

- task definida em [`pyproject.toml`](../pyproject.toml)

**Saídas**

- servidor MLflow em `http://127.0.0.1:5000`
- diretório local `mlruns/`

### 7.3 Scripts utilitários de geração e diagnóstico

**Tipo:** manual utilitário
**Start:** scripts em [`scripts/`](../scripts)

**Flows relevantes**

| Script | Uso principal | Saída típica |
|---|---|---|
| [`scripts/generate_business_features.py`](../scripts/generate_business_features.py) | Apoio à criação de features de negócio | artefatos auxiliares de features |
| [`scripts/generate_metadatastore_features.py`](../scripts/generate_metadatastore_features.py) | Geração auxiliar para metadados/features | artefatos auxiliares |
| [`scripts/generate_synthetic_predictions.py`](../scripts/generate_synthetic_predictions.py) | Apoio à geração de predições sintéticas | arquivos compatíveis com monitoramento |
| [`scripts/list_ollama_models.py`](../scripts/list_ollama_models.py) | Diagnóstico de modelos no Ollama | lista de modelos disponíveis |

**Observações**

- São flows de apoio, não substitutos dos flows oficiais em `src/`.
- O task `ollama_list` usa `scripts/list_ollama_models.py` quando o provider ativo
  é Ollama.

## 8. O Que Não Existe

O repositório **não possui**:

- Airflow
- cron formal versionado no projeto
- scheduler interno próprio
- DAG operacional que encadeie todos os flows
- promoção automática do challenger para substituir `model_current.pkl`
- materialização automática do Feast após `dvc repro export_feature_store`
- serving responsável por bootstrapar Feast registry ou Redis

## 9. Resumo Executivo

Se reduzirmos o projeto aos flows centrais, o mapa fica assim:

`API`
-> `/predict`
-> consulta Feast online
-> usa `model_current.pkl`
-> retorna predição
-> atualiza métricas
-> salva inferência para monitoramento

`feature engineering`
-> comando manual ou DVC
-> lê raw
-> sanitiza
-> gera interim
-> faz split treino/teste
-> transforma features
-> salva datasets processados e pipeline

`feature store`
-> export manual ou DVC
-> apply
-> materialize
-> Redis passa a atender o serving online

`models`
-> treino manual ou DVC
-> registra MLflow
-> salva champion
-> retreino pode gerar challenger
-> promoção permanece manual

`experimentos`
-> treinos múltiplos
-> cenários de inferência
-> drift sintético
-> evidências para comparação e validação

`agente LLM`
-> índice RAG manual
-> ReAct online via serving
-> tools consultam artefatos do projeto
-> avaliações offline medem qualidade

`monitoramento`
-> logging online
-> drift batch
-> status e relatórios
-> se crítico, pode iniciar retreino automático interno
