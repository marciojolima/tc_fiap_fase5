# Datathon Fase 5
# Previsão de Churn Bancário com Machine Learning + Agente LLM
# FIAP Pós-Tech MLET | Grupo 30 | Maio 2026  

## Tecnologias Utilizadas

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-modeling-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-EC6B23?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-3.10.1-0194E2?style=for-the-badge&logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=for-the-badge&logo=fastapi)
![Uvicorn](https://img.shields.io/badge/Uvicorn-serving-4051B5?style=for-the-badge)
![DVC](https://img.shields.io/badge/DVC-data%20versioning-13ADC7?style=for-the-badge&logo=dvc)
![Evidently](https://img.shields.io/badge/Evidently-drift%20monitoring-6E56CF?style=for-the-badge)
![Prometheus](https://img.shields.io/badge/Prometheus-monitoring-E6522C?style=for-the-badge&logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-observability-F46800?style=for-the-badge&logo=grafana)
![Feast](https://img.shields.io/badge/Feast-feature%20store-2F855A?style=for-the-badge)
![Redis](https://img.shields.io/badge/Redis-online%20store-DC382D?style=for-the-badge&logo=redis)
![Pandera](https://img.shields.io/badge/Pandera-data%20validation-7A3E9D?style=for-the-badge)
![Poetry](https://img.shields.io/badge/Poetry-dependencies-60A5FA?style=for-the-badge&logo=poetry)
![Pytest](https://img.shields.io/badge/Pytest-tests-0A9EDC?style=for-the-badge&logo=pytest)
![Ruff](https://img.shields.io/badge/Ruff-lint%20%26%20format-D7FF64?style=for-the-badge)
![Docker Compose](https://img.shields.io/badge/Docker%20Compose-local%20orchestration-2496ED?style=for-the-badge&logo=docker)

## Instalação e Execução

### Pré-requisitos

- Python 3.13
- Poetry 2.x (obrigatório)
- NVIDIA GPU com CUDA 12.x (recomendado)
- Docker

### 1. Clone do repositório

Clone o repositório e acesse a pasta do projeto:

```bash
git clone https://github.com/marciojolima/tc_fiap_fase5.git
cd tc_fiap_fase5
```

### 2. Instalação completa do ambiente

Informe ao Poetry qual versão do Python deve ser usada no ambiente virtual:

```bash
poetry env use python3.13
```

Instale todas as dependências do projeto. O ambiente virtual é criado automaticamente quando necessário:

```bash
poetry install --all-extras
```

Se quiser ativar um shell dentro do ambiente virtual, instale o plugin `poetry-plugin-shell`:

```bash
poetry self add poetry-plugin-shell
```

Depois, abra o shell do Poetry:

```bash
poetry shell
```

Crie também o arquivo `.env` a partir do modelo de referência:

```bash
cp .env.example .env
```

O provider do modelo LLM usado pelo agente ReAct é definido em `configs/pipeline_global_config.yaml`, na chave `llm.active_provider`. As opções válidas são `ollama`, `claude` e `openai`.

Exemplo:

```yaml
llm:
  active_provider: claude
```

Se o provider ativo for externo, preencha no `.env` a chave correspondente:

```bash
OPENAI_API_KEY=<sua-chave>
ANTHROPIC_API_KEY=<sua-chave>
```

Observações importantes:

- `openai` usa `OPENAI_API_KEY`
- `claude` usa `ANTHROPIC_API_KEY`
- `ollama` não exige chave de API, mas requer uma instância do Ollama acessível pela `base_url` configurada

#### Carga inicial de dados e geração de artefatos

Faça o pull dos dados versionados no storage via DVC:

```bash
poetry run dvc pull
```

Suba a infraestrutura mínima para execução local, com Redis e MLflow:

```bash
poetry run task infra_up_only_one_time
```

Execute o pipeline principal para gerar os artefatos do projeto, incluindo engenharia de features, treinamento, indexação de embeddings, experimentos pré-configurados, análise de cenários e geração de dados sintéticos para simulação de drift:

```bash
poetry run dvc repro
```

### 3. Sincronização de dados versionados

O projeto utiliza DVC para dados e artefatos versionados. O remote padrão está
definido em `.dvc/config`, enquanto credenciais e segredos locais devem ficar em
`.dvc/config.local`.

Ao longo deste README, os exemplos usam `poetry run dvc ...`.

Na máquina local, configure o acesso ao storage remoto conforme o backend usado
no projeto. Para isso, consulte a documentação oficial do DVC sobre remotes e
storage providers, incluindo opções como Google Drive.

Depois disso, baixe os dados com:

```bash
poetry run dvc pull
```

### 4. Stack local com Docker Compose

```bash
cp .env.example .env
poetry run task appstack
```

ou

```bash
docker compose up -d
```

A stack local sobe os seguintes serviços de forma integrada:

- serving FastAPI
- Redis
- MLflow server
- Prometheus
- Grafana

Quando o `llm_provider` ativo for `ollama`, use o override [docker-compose.ollama.yml](docker-compose.ollama.yml) via `poetry run task appstack_ollama` ou `poetry run task appstack_ollama_rebuild`. Nesse modo, a stack adiciona:

- **Ollama** (volume `ollama_data` para modelos)
- job **one-shot** `ollama-pull`, que executa `ollama pull` do modelo definido em `llm.providers.ollama.model_name`

Com a stack em execução, a documentação interativa do FastAPI fica disponível no endpoint padrão de documentação do ambiente local (incluindo rotas tabulares, `/train` e `/llm/*`).

Para encerrar os serviços:

```bash
poetry run task appstack_down
```

### 5. Execução manual isolada

## Problema de Negócio

Identificar clientes com alta probabilidade de evasão (churn) para permitir ações de retenção proativas pelo banco.

## Métrica de negócio  
**≥ 70%** dos clientes que realmente evadem devem estar entre os 20% com maior risco previsto (recall@top20% ≥ 0.70).


## Estratégia de seleção de modelo  
A escolha do modelo não é fixa, sendo orientada pelo objetivo de negócio e pelas restrições operacionais.

- Em cenários onde o objetivo é maximizar a retenção e evitar perda de clientes a qualquer custo, são priorizados modelos com **maior recall**.
- Em cenários com limitação de capacidade operacional (ex: equipe de retenção reduzida), são priorizados modelos com **maior precisão (precision)**, garantindo maior eficiência nas ações.

Dessa forma, diferentes experimentos (variações de hiperparâmetros e algoritmos) podem ser promovidos a modelo em produção conforme o critério de negócio vigente, caracterizando uma abordagem orientada a valor e não apenas a métricas técnicas isoladas.

## Sumário

- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação e Execução](#instalação-e-execução)
- [Problema de Negócio](#problema-de-negócio)
- [Métrica de negócio](#métrica-de-negócio)
- [Estratégia de seleção de modelo](#estratégia-de-seleção-de-modelo)
- [Sobre o Projeto](#sobre-o-projeto)
- [O que o Projeto Entrega](#o-que-o-projeto-entrega)
- [Endpoints da API](#endpoints-da-api)
- [Arquitetura da Solução](#arquitetura-da-solução)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Feature Store](#feature-store)
- [LLM, agente ReAct e llm_provider](#llm-agente-react-e-llm_provider)
- [Monitoramento e Observabilidade](#monitoramento-e-observabilidade)
- [Stack local reproduzível](#stack-local-reproduzível)
- [Fluxo sugerido para validação local](#fluxo-sugerido-para-validação-local)
- [Artefatos Relevantes](#artefatos-relevantes)
- [Documentação](#documentação)
- [Autores](#autores)
- [Referências](#referências)
- [Licença](#licença)

## Sobre o Projeto

Este projeto foi organizado como uma plataforma de machine learning aplicada a churn bancário. A proposta é cobrir uma trilha de ponta a ponta, desde dados versionados e engenharia de features até serving, monitoramento de drift e retreino auditável.

Um ponto importante da narrativa do repositório é a transformação de um experimento centrado em notebook em uma solução mais robusta de engenharia de ML. O notebook [notebooks/churn_bancario_sem_mlops.ipynb](notebooks/churn_bancario_sem_mlops.ipynb) representa a base exploratória executada em Jupyter ou Colab. O restante do repositório organiza essa base em uma estrutura com separação de responsabilidades, versionamento de dados, treino rastreável, serving, monitoramento, governança e documentação operacional.

Em outras palavras, este repositório não busca apenas mostrar um modelo de churn funcionando, mas também evidenciar a diferença entre um experimento isolado e uma solução com preocupações reais de MLOps.

O foco principal está em demonstrar práticas de engenharia de ML esperadas no contexto do Datathon:

- pipeline de dados reproduzível
- consistência entre treino e inferência
- treinamento rastreável com MLflow
- serving desacoplado via FastAPI
- cenários de negócio versionados
- monitoramento batch de drift com artefatos auditáveis
- feature store local com Feast + Redis para materialização online incremental
- stack local reproduzível com serving, MLflow, Prometheus e Grafana

## O que o Projeto Entrega

O repositório reúne uma base funcional e demonstrável nas seguintes frentes:

### 1. Dados, features e preparação

- versionamento de dados com DVC
- separação entre camadas `raw`, `interim` e `processed`
- pipeline de engenharia de features em [src/feature_engineering/feature_engineering.py](src/feature_engineering/feature_engineering.py)
- componentes reutilizáveis em [src/feature_engineering/pipeline_components.py](src/feature_engineering/pipeline_components.py)
- validação estrutural com Pandera em [src/feature_engineering/schema_validation.py](src/feature_engineering/schema_validation.py)
- persistência de datasets preparados e artefatos auxiliares para reuso

### 2. Treinamento e gestão de modelo

- treinamento principal em [src/model_lifecycle/train.py](src/model_lifecycle/train.py)
- rastreamento de parâmetros, métricas e artefatos com MLflow
- múltiplas configurações de experimento em `configs/model_lifecycle/experiments/`
- persistência do modelo champion, challengers e metadados em `artifacts/models/`
- apoio a promoção champion-challenger em [src/model_lifecycle/promotion.py](src/model_lifecycle/promotion.py)

### 3. Serving e inferência

- API FastAPI em [src/serving/app.py](src/serving/app.py)
- endpoints de serving e treino síncrono em [src/serving/routes.py](src/serving/routes.py)
- contratos de entrada e saída em [src/serving/schemas.py](src/serving/schemas.py)
- carregamento compartilhado do pipeline de features e do modelo em [src/serving/pipeline.py](src/serving/pipeline.py)
- endpoint `POST /train` com validação Pydantic, treino síncrono e sem promoção automática do modelo ativo

### 4. Cenários de negócio e validação

- suíte de cenários versionados em [configs/scenario_experiments/inference_cases.yaml](configs/scenario_experiments/inference_cases.yaml)
- execução automatizada em [src/scenario_experiments/inference_cases.py](src/scenario_experiments/inference_cases.py)
- geração de lotes sintéticos de drift em [src/evaluation/model/drift/synthetic_drifts.py](src/evaluation/model/drift/synthetic_drifts.py)

### 5. Monitoramento e operação

- logging de inferências em [src/evaluation/model/drift/prediction_logger.py](src/evaluation/model/drift/prediction_logger.py)
- métricas operacionais expostas em [src/monitoring/metrics.py](src/monitoring/metrics.py)
- detecção batch de drift com Evidently e PSI em [src/evaluation/model/drift/drift.py](src/evaluation/model/drift/drift.py)
- relatórios HTML e arquivos JSON para auditoria em `artifacts/evaluation/model/`
- stack local reproduzível com serving, MLflow, Prometheus e Grafana
- workflow básico de CI em [.github/workflows/ci.yml](.github/workflows/ci.yml)

### 6. LLM, agente ReAct, RAG e segurança

O projeto inclui uma trilha conversacional com provider LLM configurável, agente ReAct, RAG local e guardrails, integrada à API sem alterar o contrato tabular de `/predict`.

O agente não substitui o modelo de churn. Seu papel é transformar predições, cenários e sinais operacionais em respostas mais acessíveis para análise e apoio à decisão.

- **API LLM:** `GET /llm/health`, `GET /llm/status` e `POST /llm/chat`.
- **Agente e tools:** [src/agent/react_agent.py](src/agent/react_agent.py) e [src/agent/tools.py](src/agent/tools.py), com `rag_search`, `predict_churn`, `drift_status` e `scenario_prediction`.
- **RAG e segurança:** [src/agent/rag_pipeline.py](src/agent/rag_pipeline.py), [src/security/guardrails.py](src/security/guardrails.py) e [src/security/pii_detection.py](src/security/pii_detection.py).
- **Avaliação:** golden set em [data/golden-set.json](data/golden-set.json), RAGAS, LLM-as-judge e benchmark de prompts em `src/evaluation/llm_agent/`.
- **Configuração:** [configs/pipeline_global_config.yaml](configs/pipeline_global_config.yaml) e `.env` para chaves externas.

Detalhes de arquitetura, operação e avaliação estão em [docs/AGENT_REACT.md](docs/AGENT_REACT.md), [docs/RAG_EXPLANATION.md](docs/RAG_EXPLANATION.md) e [docs/EVALUATION_RAGAS.md](docs/EVALUATION_RAGAS.md).

Essa trilha já permite demonstrar comportamento conversacional, recuperação contextual e avaliação estruturada do agente em execução real.

## Endpoints da API

| Método | Endpoint | Objetivo | Observações |
|---|---|---|---|
| `GET` | `/health` | Healthcheck simples da API tabular | Retorna `{"status":"ok"}`. |
| `POST` | `/predict` | Predição online por `customer_id` | Aceita objeto único ou array; com 1 item retorna objeto, com 2+ retorna `items` + `summary`. |
| `POST` | `/predict/raw` | Predição por payload bruto mínimo | Aceita objeto único ou array; com 1 item retorna objeto, com 2+ retorna `items` + `summary`. |
| `POST` | `/train` | Treino síncrono de um experimento individual | Valida o schema com Pydantic, recebe JSON no formato lógico do config de treino, salva challenger e retorna o tempo de treino em segundos. |
| `GET` | `/metrics` | Exposição de métricas Prometheus | Foco atual em `/predict` e `/llm/chat`. |
| `GET` | `/llm/health` | Healthcheck do router LLM | Diagnóstico rápido das rotas LLM. |
| `GET` | `/llm/status` | Status do provider LLM e do RAG | Mostra provider ativo, modelo esperado e estado do índice. |
| `POST` | `/llm/chat` | Chat com agente ReAct | Pode usar RAG e tools do domínio. |
| `GET` | `/llm/playground` | Playground HTML do chat LLM | Interface simples para simular perguntas, respostas e trace do agente. |


## Arquitetura da Solução

O projeto parte de dados versionados com DVC, aplica engenharia e validação de
features e gera bases prontas para treino e inferência. O treinamento é
rastreado no MLflow, enquanto o serving desacoplado em FastAPI expõe os
endpoints tabulares e a trilha conversacional com agente LLM.

Na operação, o sistema registra inferências, expõe métricas para observabilidade
e executa monitoramento batch de drift. Quando necessário, essa trilha pode
abrir um fluxo auditável de retreino e comparação champion-challenger.

## Estrutura do Repositório

```text
tc_fiap_fase5/
├── artifacts/              # modelos, relatórios de drift e saídas de retreino
├── configs/                # treino, cenários, monitoramento e observabilidade
├── data/                   # camadas raw, interim e processed
├── docs/                   # documentação técnica e de governança
├── feature_store/          # repositório Feast e definições da feature store
├── notebooks/              # notebooks exploratórios e de apoio
├── scripts/                # utilitários auxiliares
├── src/
│   ├── agent/              # agente, RAG e gateway de LLM
│   ├── common/             # utilidades compartilhadas
│   ├── evaluation/         # avaliação de LLM e de modelo/drift
│   ├── feature_engineering/ # engenharia e validação de features
│   ├── model_lifecycle/    # treino, promoção e retreino
│   ├── monitoring/         # métricas operacionais
│   ├── scenario_experiments/ # cenários de negócio
│   ├── security/           # guardrails e proteção básica de PII
│   └── serving/            # aplicação FastAPI e pipeline de inferência
├── tests/                  # suíte de testes automatizados
├── docker-compose.yml
├── pyproject.toml
└── README.md
```


Se você quiser subir somente um componente fora do Compose durante desenvolvimento local:

### Feature Store

A integração com Feature Store usa Feast + Redis para separar publicação offline de consulta online e manter o contrato de features consistente entre treino e serving.

O detalhamento da arquitetura, do fluxo operacional e dos comandos dessa trilha está em [docs/FEATURE_STORE.md](docs/FEATURE_STORE.md).

Serving:

```bash
poetry run uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

MLflow:

```bash
poetry run task mlflow
```

### 6. Monitoramento e demonstração de drift

Monitoramento batch:

```bash
poetry run task mldrift
```

Execução demonstrável com base de teste:

```bash
poetry run task mldriftdemo
```

Geração de cenários sintéticos:

```bash
poetry run task mlflowsyntheticdrift
```

### 7. Testes

```bash
poetry run task test
```

## LLM, agente ReAct e llm_provider

O detalhamento da trilha de LLM foi extraído para
[docs/AGENT_REACT.md](docs/AGENT_REACT.md), com a visão do agente ReAct, das
tools, do RAG, da configuração por provider e da operação local com Docker.

Para a avaliação dessa trilha em execução real, veja também
[docs/EVALUATION_RAGAS.md](docs/EVALUATION_RAGAS.md), que documenta RAGAS,
LLM-as-judge e benchmark de prompts sobre o endpoint `/llm/chat`.

## Monitoramento e Observabilidade

O projeto combina métricas operacionais, logging de inferências, monitoramento
batch de drift e trilha auditável de retreino para acompanhar a saúde do
modelo em execução.

O detalhamento dessa camada está em
[docs/MONITORING_OBSERVABILITY.md](docs/MONITORING_OBSERVABILITY.md).

## Stack local reproduzível

Quando a stack é iniciada com `poetry run task appstack`, os serviços ficam disponíveis em:

| Serviço | URL | Papel |
|---|---|---|
| FastAPI | `http://127.0.0.1:8000` | Serving da aplicação de inferência |
| Swagger UI | `http://127.0.0.1:8000/docs` | Teste interativo dos endpoints tabulares, `/train` e rotas LLM |
| MLflow UI | `http://127.0.0.1:<MLFLOW_PORT>` | Rastreamento de experimentos |
| Prometheus | `http://localhost:9090` | Coleta e exploração das métricas |
| Grafana | `http://localhost:3000` | Dashboards operacionais |

O Compose monta `configs/`, `artifacts/` e `mlruns/` com caminhos compatíveis com o código do projeto. Com isso, o serving carrega o mesmo modelo champion e o mesmo pipeline de features materializados localmente, enquanto o MLflow expõe os experimentos rastreados no SQLite local `mlruns/mlflow.db`. A porta publicada da UI vem de `MLFLOW_PORT` no `.env` e usa `5000` apenas como padrão.

## Fluxo sugerido para validação local

1. Copie `.env.example` para `.env`, se quiser customizar portas ou credenciais.
2. Suba a stack com `poetry run task appstack`.
3. Gere tráfego pelo Swagger ou por chamadas aos endpoints de serving.
4. Consulte as métricas no Prometheus.
5. Abra o Grafana para visualizar os painéis provisionados.
6. Abra o MLflow para revisar runs, parâmetros, métricas e artefatos.
7. Rode `poetry run task mldriftdemo` ou `poetry run task mldrift` para produzir uma execução de drift.
8. Abra os relatórios HTML e os arquivos JSON em `artifacts/evaluation/model/` para inspecionar as evidências geradas.

## Artefatos Relevantes

Os principais artefatos de dados, modelos, monitoramento e avaliação foram
extraídos para [docs/ARTIFACTS.md](docs/ARTIFACTS.md), que concentra a visão do
papel de cada arquivo relevante na operação do projeto.

## Documentação

- [docs/ADR/ADR-001.md](docs/ADR/ADR-001.md)
- [docs/ADR/ADR-002.md](docs/ADR/ADR-002.md)
- [docs/ADR/ADR-003.md](docs/ADR/ADR-003.md)
- [docs/ADR/ADR-004.md](docs/ADR/ADR-004.md)
- [docs/ADR/ADR-005.md](docs/ADR/ADR-005.md)
- [docs/ADR/ADR-006.md](docs/ADR/ADR-006.md)
- [docs/ADR/ADR-007.md](docs/ADR/ADR-007.md)
- [docs/ADR/ADR-008.md](docs/ADR/ADR-008.md)
- [docs/ADR/ADR-009.md](docs/ADR/ADR-009.md)
- [docs/ADR/ADR-010.md](docs/ADR/ADR-010.md)
- [docs/ADR/ADR-011.md](docs/ADR/ADR-011.md)
- [docs/AGENT_REACT.md](docs/AGENT_REACT.md)
- [docs/ARTIFACTS.md](docs/ARTIFACTS.md)
- [docs/DRIFT_MONITORING.md](docs/DRIFT_MONITORING.md)
- [docs/EVALUATION.md](docs/EVALUATION.md)
- [docs/EVALUATION_MODEL_METRICS.md](docs/EVALUATION_MODEL_METRICS.md)
- [docs/EVALUATION_RAGAS.md](docs/EVALUATION_RAGAS.md)
- [docs/FEATURE_STORE.md](docs/FEATURE_STORE.md)
- [docs/FLOWS.md](docs/FLOWS.md)
- [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md)
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- [docs/MODEL_VERSIONING.md](docs/MODEL_VERSIONING.md)
- [docs/MONITORING_OBSERVABILITY.md](docs/MONITORING_OBSERVABILITY.md)
- [docs/OPERATIONS_DASHBOARD.md](docs/OPERATIONS_DASHBOARD.md)
- [docs/OWASP_MAPPING.md](docs/OWASP_MAPPING.md)
- [docs/RAG_EXPLANATION.md](docs/RAG_EXPLANATION.md)
- [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md)
- [docs/SCENARIO_ANALYSIS.md](docs/SCENARIO_ANALYSIS.md)
- [docs/SYNTHETIC_PREDICTIONS_GENERATOR.md](docs/SYNTHETIC_PREDICTIONS_GENERATOR.md)
- [docs/SYSTEM_CARD.md](docs/SYSTEM_CARD.md)

## Autores

**Turma 6MLET - FIAP**

- Luca Poit - RM365678
- Gabriel Jordan - RM365606
- Luciana Ferreira - RM366171
- Marcio Lima - RM365919

## Referências

1. OWASP (2025). *Top 10 for Large Language Model Applications*.
2. Brasil (2018). *Lei n° 13.709/2018 (LGPD)*.
3. Kaggle. *Bank Customer Churn*. Disponível em: <https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn>.
4. Kaggle. *Bank Customer Churn - Discussion*. Disponível em: <https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/discussion?sort=undefined>.

Nota sobre a referência do Kaggle:

- a base foi utilizada principalmente para fins educacionais, experimentais e de demonstração técnica no contexto do projeto
- a página do dataset indica licença do tipo `Other (specified in description)`, então a eventual redistribuição ou uso fora do contexto acadêmico deve considerar essa ressalva e ser validada diretamente na origem
- a citação acima documenta a fonte de obtenção da base, mas não substitui a verificação de licença, procedência e condições de uso

## Licença

MIT
