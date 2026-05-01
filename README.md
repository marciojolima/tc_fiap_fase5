# Datathon Fase 5
# Previsão de Churn Bancário com Machine Learning + Agente LLM
# FIAP Pós-Tech MLET | Grupo 30 | Maio 2026  

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![MLflow](https://img.shields.io/badge/MLflow-3.10.1-0194E2?style=for-the-badge&logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=for-the-badge&logo=fastapi)
![DVC](https://img.shields.io/badge/DVC-data%20versioning-13ADC7?style=for-the-badge&logo=dvc)
![Evidently](https://img.shields.io/badge/Evidently-drift%20monitoring-6E56CF?style=for-the-badge)
![Prometheus](https://img.shields.io/badge/Prometheus-monitoring-E6522C?style=for-the-badge&logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-observability-F46800?style=for-the-badge&logo=grafana)
![Feast](https://img.shields.io/badge/Feast-feature%20store-2F855A?style=for-the-badge)
![Redis](https://img.shields.io/badge/Redis-online%20store-DC382D?style=for-the-badge&logo=redis)
![Poetry](https://img.shields.io/badge/Poetry-dependencies-60A5FA?style=for-the-badge&logo=poetry)

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

- [Sobre o Projeto](#sobre-o-projeto)
- [Instalação e Execução](#instalação-e-execução)
- [O que o Projeto Entrega](#o-que-o-projeto-entrega)
- [Endpoints da API](#endpoints-da-api)
- [Arquitetura da Solução](#arquitetura-da-solução)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [LLM, agente ReAct e llm_provider](docs/AGENT_REACT.md)
- [Feature Store](#feature-store)
- [Monitoramento e Observabilidade](#monitoramento-e-observabilidade)
- [Artefatos Relevantes](#artefatos-relevantes)
- [Documentação Complementar](#documentação-complementar)
- [Referências](#referências)
- [Autores](#autores)
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

Implementação alinhada a uma camada de provider LLM configurável, integrada à API FastAPI sem alterar o contrato do endpoint tabular `/predict`.

Nesta arquitetura, o agente ReAct não substitui o modelo tabular de churn nem
melhora diretamente a métrica principal de negócio. O ganho principal continua
vindo da qualidade da predição. O papel do agente é indireto, mas relevante:
transformar score, simulações de cenário e sinais operacionais em suporte mais
acessível para decisão, investigação e ação de retenção. Em outras palavras, o
modelo identifica risco; o agente ajuda a tornar esse risco utilizável pelas
áreas de negócio, operação e acompanhamento do ciclo de vida do modelo.

- **API (FastAPI)**  
  - `GET /llm/health` — health do router LLM.  
  - `GET /llm/status` — provider ativo (`llm.active_provider`), modelo esperado, diagnóstico específico do provider e status do RAG.  
  - `POST /llm/chat` — pergunta do usuário, resposta do agente, lista de tools usadas e trace opcional.

- **Agente ReAct** — [src/agent/react_agent.py](src/agent/react_agent.py): loop no estilo pensar → agir → observar, com limite de iterações e integração com guardrails de entrada e saída.

- **Tools (≥4)** — [src/agent/tools.py](src/agent/tools.py): `rag_search` (contexto sobre documentação e metadados do projeto), `predict_churn` (mesmo contrato do `/predict/raw`, com payload bruto), `drift_status` (artefatos de drift), `scenario_prediction` (cenários hipotéticos).

- **RAG** — [src/agent/rag_pipeline.py](src/agent/rag_pipeline.py): recuperação vetorial local com FastEmbed/ONNX e rerank lexical leve sobre arquivos versionados (por exemplo `README.md`, docs e metadados em `data/processed/` quando existirem).
  O modelo de embeddings do RAG usa cache persistente local em `artifacts/rag/fastembed_model_cache`, reduzindo downloads repetidos entre reinícios da stack.

- **Segurança** — [src/security/guardrails.py](src/security/guardrails.py) e [src/security/pii_detection.py](src/security/pii_detection.py): validação básica de input e mascaramento de PII na resposta.

- **Configuração** — blocos `llm`, `agent`, `rag` e `security` em [configs/pipeline_global_config.yaml](configs/pipeline_global_config.yaml); o `llm_provider` ativo, `model_name`, `base_url`, `max_tokens` e demais parâmetros ficam no YAML. Chaves de API de providers externos ficam no `.env` (ex.: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
  Em ambiente Docker, `LLM_BASE_URL` pode sobrescrever a `base_url` do provider ativo quando for necessário ajustar o endpoint interno do container.

- **Testes** — [tests/test_agent.py](tests/test_agent.py), [tests/test_guardrails.py](tests/test_guardrails.py), [tests/test_llm_routes.py](tests/test_llm_routes.py).

- **Utilitário** — [scripts/list_ollama_models.py](scripts/list_ollama_models.py) (task `ollama_list`): diagnóstico opcional para ambientes em que o `llm_provider` ativo é `ollama`.

- **Golden set (RAG / judge):** [data/golden-set.json](data/golden-set.json) — 24 pares `query` / `expected_answer` alinhados ao domínio (churn, MLOps, API, observabilidade, RAG/LLM). Validação mínima em [tests/test_golden_set.py](tests/test_golden_set.py).

- **RAGAS (4 métricas):** [src/evaluation/llm_agent/ragas_eval.py](src/evaluation/llm_agent/ragas_eval.py) — calcula *faithfulness*, *answer relevancy*, *context precision* e *context recall* sobre o golden set chamando o endpoint real `POST /llm/chat`; os contextos vêm da trace de `rag_search`. Embeddings multilingues via FastEmbed, sem `sentence-transformers` nem `torch`. Execução local: `poetry run task eval_ragas` (requer serving e provider LLM configurados). Saída típica: `artifacts/evaluation/llm_agent/results/ragas_scores.json`, com histórico em `artifacts/evaluation/llm_agent/runs/ragas_runs.jsonl`.

- **LLM-as-judge (3 critérios):** [src/evaluation/llm_agent/llm_judge.py](src/evaluation/llm_agent/llm_judge.py) — avalia respostas do RAG nos critérios `adequacao_negocio`, `correcao_conteudo` e `clareza_utilidade`. Execução local: `poetry run task eval_llm_judge`. Saída típica: `artifacts/evaluation/llm_agent/results/llm_judge_scores.json`, com histórico em `artifacts/evaluation/llm_agent/runs/llm_judge_runs.jsonl`.

- **Prompt A/B (3 variantes):** [src/evaluation/llm_agent/ab_test_prompts.py](src/evaluation/llm_agent/ab_test_prompts.py) — benchmark offline com três variantes de prompt sobre o golden set, comparando cobertura lexical mínima da resposta e, opcionalmente, notas do `llm_judge`. Execução local: `poetry run task eval_ab_test_prompts` ou `poetry run python -m src.evaluation.llm_agent.ab_test_prompts --with-judge`. Saída típica: `artifacts/evaluation/llm_agent/results/prompt_ab_results.json`, com histórico em `artifacts/evaluation/llm_agent/runs/prompt_ab_runs.jsonl`.

- **Execução completa:** `poetry run task eval_all` executa RAGAS, LLM-as-judge e Prompt A/B em sequência. O RAGAS reutiliza o cache local do FastEmbed configurado para o RAG e, por padrão, chama `http://127.0.0.1:8000/llm/chat` (`RAGAS_SERVING_BASE_URL` permite sobrescrever).

  As tasks de avaliação usam o provider configurado em `configs/pipeline_global_config.yaml`. Para providers externos, a chave pode estar exportada no shell ou preenchida no `.env` local (`ANTHROPIC_API_KEY` para Claude, `OPENAI_API_KEY` para OpenAI).

**Extensões previstas para essa trilha:** ampliação do CI/CD e documentação agregada de resultados de avaliação.

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

Notas de serving:

- `POST /predict` e `POST /predict/raw` mantêm compatibilidade com o contrato antigo para payload unitário.
- Se a requisição tiver apenas 1 item, envie um objeto JSON e a API responderá com um objeto JSON de predição.
- Se a requisição tiver 2 ou mais itens, envie um array JSON e a API responderá com um objeto de lote contendo `items` e `summary`.
- Em lote, cada item é processado de forma isolada, permitindo sucesso parcial e erro parcial no mesmo request.
- `POST /predict/raw` continua recebendo apenas o payload mínimo de inferência, não o registro bruto completo com as 18 colunas do CSV original.
- `POST /train` não sobrescreve `artifacts/models/current.pkl` e retorna erro se o payload tentar apontar para o modelo champion ativo.
- `POST /train` reutiliza o módulo de treino existente, registra métricas/artefatos no MLflow e retorna `training_time_seconds`, mas não limpa cache nem altera o modelo carregado pelo serving em memória.
- Em Docker, o serviço `serving` também precisa do volume `./mlruns:/app/mlruns`, porque o endpoint registra runs diretamente no backend SQLite do MLflow.
- O fluxo recomendado para servir um novo modelo continua sendo treino de challenger, avaliação e promoção explícita.

## Arquitetura da Solução

O fluxo principal do projeto pode ser resumido em seis etapas:

1. **Versionamento e ingestão dos dados**  
   Os dados brutos são mantidos em `data/raw/`, com apoio do DVC para rastreabilidade e sincronização entre ambientes.

2. **Engenharia e validação de features**  
   A pipeline gera artefatos intermediários e finais em `data/interim/` e `data/processed/`, além de evidências de validação estrutural.

3. **Treinamento rastreável**  
   O treinamento registra parâmetros, métricas, tags e artefatos no MLflow, mantendo trilha de execução reproduzível.

4. **Serving desacoplado**  
   A API FastAPI carrega o pipeline persistido de features e o modelo atual para servir inferências com o mesmo contrato do treino.

5. **Observabilidade e logging de inferências**  
   As predições podem ser registradas para posterior monitoramento, enquanto métricas operacionais ficam disponíveis para Prometheus.

6. **Drift e retreino auditável**  
   O monitoramento compara dados de referência e correntes, gera relatórios de drift e pode acionar o fluxo de retreino com decisão champion-challenger.

## Tecnologias Utilizadas

- **Linguagem:** Python 3.13
- **Gerenciamento de dependências:** Poetry
- **Treinamento e modelagem:** Scikit-learn, XGBoost
- **Experiment tracking:** MLflow
- **Versionamento de dados:** DVC
- **Feature Store:** Feast
- **Online store:** Redis
- **Validação de dados:** Pandera
- **Serving:** FastAPI, Uvicorn
- **Monitoramento de drift:** Evidently
- **Observabilidade:** Prometheus, Grafana
- **Qualidade de código:** Pytest, Ruff
- **Orquestração local:** Docker Compose

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
│   ├── security/           # guardrails e PII em evolução
│   └── serving/            # aplicação FastAPI e pipeline de inferência
├── tests/                  # suíte de testes automatizados
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── STATUS_ATUAL_PROJETO.md
```

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

Faça o pull dos dados versionados no storage via DVC, incluindo arquivos como `data/raw/Customer-Churn-Records.csv`:

Observação sobre a base de churn bancário:

- a base utilizada no projeto foi obtida via Kaggle, a partir do dataset `Bank Customer Churn`
- o uso neste repositório é voltado principalmente a fins educacionais, experimentais e de demonstração técnica
- a referência de obtenção e a licença/procedência da base devem ser verificadas diretamente na página do dataset antes de qualquer redistribuição ou uso fora de contexto acadêmico
- por esse motivo, este projeto não deve apresentar essa base como dado operacional real de clientes

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

O projeto utiliza DVC para dados e artefatos versionados. O remote padrão está definido em `.dvc/config` com o nome `datathon_remote` e apontando para um storage no Google Drive.

Ao longo deste README, os exemplos usam `poetry run dvc ...`.

#### Como a configuração está organizada

- `.dvc/config`: arquivo versionado no Git com a configuração compartilhada do remote, como nome e URL
- `.dvc/config.local`: arquivo local, não versionado, usado para credenciais e segredos de cada máquina

#### 1. Configure o remote

A configuração compartilhada aponta para o remote `datathon_remote`. Para recriá-lo manualmente em outra máquina:

```bash
poetry run dvc remote add -d datathon_remote gdrive://<REMOTE_ID>
```

#### 2. Configure as credenciais locais do Google Drive

As credenciais OAuth não devem ir para o Git. Por isso, elas devem ser gravadas localmente com `--local`, o que escreve em `.dvc/config.local`:

```bash
poetry run dvc remote modify --local datathon_remote gdrive_client_id <ID>
poetry run dvc remote modify --local datathon_remote gdrive_client_secret <SECRET>
```

#### 3. Configure o OAuth no Google Cloud Console

Para que o DVC possa autenticar no Google Drive via OAuth, é necessário existir um cliente OAuth configurado no Google Cloud Console. Em linhas gerais:

1. crie ou selecione um projeto no Google Cloud Console
2. habilite a Google Drive API para esse projeto
3. configure a tela de consentimento OAuth
4. crie credenciais do tipo OAuth Client ID
5. use o `client_id` e o `client_secret` gerados nos comandos `dvc remote modify --local ...`

#### 4. Baixe os dados

Depois do remote e das credenciais estarem corretos, baixe os dados com:

```bash
poetry run dvc pull
```

#### Observações importantes

- não versione `.dvc/config.local`
- não publique `client_id` e `client_secret` em README, issue, commit ou pull request
- `.dvc/config` define a configuração compartilhada do remote; `.dvc/config.local` guarda segredos e ajustes locais da máquina
- ** para maiores detalhes consulte o arquivo: [`dvc.yaml`](dvc.yaml)**

### 4. Stack local com Docker Compose

```bash
cp .env.example .env
poetry run task appstack
```

Se preferir executar a stack sem as tasks do projeto, o modo normal também pode ser iniciado diretamente com:

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

**Modo desenvolvimento:** use `poetry run task appstack_dev` para subir a stack com o override [docker-compose.dev.yml](docker-compose.dev.yml). Nesse modo, o `serving` monta `./src` em `/app/src` e roda o Uvicorn com `--reload`, então alterações em arquivos Python dentro de `src/` não exigem rebuild da imagem.

**Quando usar rebuild:** reconstrua a stack apenas quando mudar Dockerfile, `pyproject.toml`, `poetry.lock`, dependências ou alguma estrutura relevante de build. Para a stack base, use `poetry run task appstack_rebuild`; para desenvolvimento, use `poetry run task appstack_dev_rebuild`; para o cenário com Ollama local, use `poetry run task appstack_ollama_rebuild`.

**Quando não precisa rebuild:** no modo desenvolvimento, mudanças em `src/` são recarregadas pelo Uvicorn. Configurações, dados e artefatos ficam disponíveis por volumes do Compose principal, incluindo `configs/`, `data/processed/`, `data/feature_store/`, `artifacts/` e `feature_store/`.

Para encerrar os serviços:

```bash
poetry run task appstack_down
```

### 5. Execução manual isolada

Se você quiser subir somente um componente fora do Compose durante desenvolvimento local:

### Feature Store

Uma Feature Store local baseada em Feast, com Redis como online store, separa a camada offline, usada para preparo e materialização, da camada online, usada para consulta de baixa latência.

O consumo é governado por `FeatureServices` por versão de modelo. Isso deixa explícito qual contrato de features cada modelo usa no treino e no serving, mesmo quando diferentes versões compartilham a mesma `FeatureView` base.

Fluxo recomendado:

```bash
poetry run dvc repro featurize
poetry run dvc repro train
poetry run dvc repro create_fs_offline
docker compose up -d redis
poetry run task feastapply
poetry run task feastmaterialize
poetry run task feastdemo
```

Arquivos principais dessa integração:

- `feature_store/feature_store.yaml`
- `feature_store/repo.py`
- `src/feast_ops/export.py`
- `src/feast_ops/demo.py`
- `docs/FEATURE_STORE.md`

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

Monitoramento técnico para a solução tabular de churn, combinando métricas operacionais, logging de inferências, detecção de drift e fluxo de retreino auditável.

### Escopo de monitoramento

#### Métricas operacionais do serving

As métricas expostas pela aplicação permitem acompanhar o comportamento da API em execução, com foco inicial em:

- volume de requisições
- latência
- taxa de erro
- requisições em andamento

Essas métricas são consumidas pela stack local configurada em `configs/monitoring/` e orquestrada pelo Docker Compose junto com o serving e o MLflow.

#### Logging de inferências

As inferências podem ser registradas em `artifacts/logs/inference/predictions.jsonl`, criando uma trilha de execução útil para:

- auditoria das features efetivamente servidas ao modelo
- composição do dataset corrente de monitoramento
- análise posterior de drift
- apoio a ciclos de retreino

O contrato desse arquivo prioriza as features transformadas e monitoráveis
consumidas pelo modelo em produção, com metadados mínimos de predição e origem.

#### Monitoramento batch de drift

O fluxo em [src/evaluation/model/drift/drift.py](src/evaluation/model/drift/drift.py) compara uma base de referência com dados correntes e produz evidências operacionais em:

- `artifacts/evaluation/model/drift/drift_report.html`
- `artifacts/evaluation/model/drift/drift_metrics.json`
- `artifacts/evaluation/model/drift/drift_status.json`
- `artifacts/evaluation/model/drift/drift_runs.jsonl`

Na prática, isso permite:

- inspecionar visualmente o comportamento das distribuições
- calcular PSI por feature
- consolidar um status geral de drift
- manter histórico das execuções de monitoramento

O relatório HTML destaca no topo o resumo operacional do projeto,
incluindo thresholds de `warning` e `critical` definidos no YAML e o status
final calculado pelo pipeline batch. Esse arquivo representa a visão
oficial do projeto para drift, baseada no PSI persistido em
`drift_metrics.json`, enquanto o Evidently fica disponível em um relatório
auxiliar separado para diagnóstico complementar.

#### Gatilho auditável de retreino

Quando o monitoramento identifica condição crítica, o fluxo abre uma trilha auditável de retreino, com artefatos como:

- `artifacts/evaluation/model/retraining/retrain_request.json`
- `artifacts/evaluation/model/retraining/retrain_run.json`
- `artifacts/evaluation/model/retraining/promotion_decision.json`
- `artifacts/evaluation/model/retraining/generated_configs/`

Essa trilha documenta:

- motivo do disparo
- configuração usada no retreino
- resultado consolidado da execução
- decisão final de promoção ou manutenção do champion

### Stack local reproduzível

Quando a stack é iniciada com `poetry run task appstack`, os serviços ficam disponíveis em:

| Serviço | URL | Papel |
|---|---|---|
| FastAPI | `http://127.0.0.1:8000` | Serving da aplicação de inferência |
| Swagger UI | `http://127.0.0.1:8000/docs` | Teste interativo dos endpoints tabulares, `/train` e rotas LLM |
| MLflow UI | `http://127.0.0.1:<MLFLOW_PORT>` | Rastreamento de experimentos |
| Prometheus | `http://localhost:9090` | Coleta e exploração das métricas |
| Grafana | `http://localhost:3000` | Dashboards operacionais |

O Compose monta `configs/`, `artifacts/` e `mlruns/` com caminhos compatíveis com o código do projeto. Com isso, o serving carrega o mesmo modelo champion e o mesmo pipeline de features materializados localmente, enquanto o MLflow expõe os experimentos rastreados no SQLite local `mlruns/mlflow.db`. A porta publicada da UI vem de `MLFLOW_PORT` no `.env` e usa `5000` apenas como padrão.

### Fluxo sugerido para validação local

1. Copie `.env.example` para `.env`, se quiser customizar portas ou credenciais.
2. Suba a stack com `poetry run task appstack`.
3. Gere tráfego pelo Swagger ou por chamadas aos endpoints de serving.
4. Consulte as métricas no Prometheus.
5. Abra o Grafana para visualizar os painéis provisionados.
6. Abra o MLflow para revisar runs, parâmetros, métricas e artefatos.
7. Rode `poetry run task mldriftdemo` ou `poetry run task mldrift` para produzir uma execução de drift.
8. Abra os relatórios HTML e os arquivos JSON em `artifacts/evaluation/model/` para inspecionar as evidências geradas.

## Artefatos Relevantes

Os arquivos abaixo ajudam a demonstrar reprodutibilidade, rastreabilidade e operação do projeto. Eles também servem como evidência objetiva da estrutura e dos fluxos documentados no repositório.

| Artefato | Papel no projeto |
|---|---|
| `data/interim/cleaned.parquet` | Base saneada da camada `interim`, com remoção de identificadores diretos, deduplicação, tratamento de nulos e validação de schema, antes da conversão para o formato final de modelagem. |
| `data/processed/train.parquet` | Base final de treino da camada `processed`, com split, criação de features derivadas, remoção de leakage, encoding e scaling, pronta para consumo pelos algoritmos. |
| `data/processed/test.parquet` | Base final de teste da camada `processed`, gerada com o mesmo pipeline do treino e mantida separada para validação sem vazamento. |
| `data/processed/feature_columns.json` | Registra a ordem e os nomes finais das features, ajudando a manter consistência entre treino e inferência. |
| `data/processed/schema_report.json` | Evidência da validação estrutural dos dados processados, reforçando a etapa de qualidade de dados. |
| `artifacts/models/feature_pipeline.joblib` | Pipeline de transformação persistido para reutilização no serving, evitando divergência entre treino e produção. |
| `artifacts/models/current.pkl` | Modelo champion mantido como versão principal para inferência. |
| `artifacts/models/current_metadata.json` | Metadados do champion, incluindo informações de versão, configuração e métricas relevantes. |
| `artifacts/models/challengers/` | Diretório reservado para challengers gerados em ciclos de retreino e comparados antes de eventual promoção. |
| `artifacts/logs/inference/predictions.jsonl` | Log de inferências usado como base para monitoramento posterior. O contrato registra principalmente as features transformadas efetivamente servidas ao modelo, com metadados mínimos de predição e origem. |
| `artifacts/evaluation/model/drift/drift_report.html` | Relatório HTML oficial do projeto para drift, coerente com `drift_metrics.json` e com a decisão operacional baseada em PSI. |
| `artifacts/evaluation/model/drift/drift_report_evidently.html` | Relatório auxiliar do Evidently, mantido para diagnóstico visual complementar das distribuições e widgets estatísticos. |
| `artifacts/evaluation/model/drift/drift_metrics.json` | Consolidação das métricas de drift, incluindo PSI por feature e resumo para automação de decisão. |
| `artifacts/evaluation/model/drift/drift_status.json` | Estado mais recente do monitoramento de drift, com classificação para apoio ao gatilho de retreino. |
| `artifacts/evaluation/model/drift/drift_runs.jsonl` | Histórico de execuções do monitoramento, útil para trilha de auditoria e acompanhamento temporal. |
| `artifacts/evaluation/model/retraining/retrain_request.json` | Registro do pedido de retreino, com motivação e contexto do disparo do processo. |
| `artifacts/evaluation/model/retraining/retrain_run.json` | Resultado consolidado da execução do retreino, incluindo status, motivo, métricas e decisão final. |
| `artifacts/evaluation/model/retraining/promotion_decision.json` | Decisão champion-challenger com regra de promoção explícita e deltas de métricas entre os modelos comparados. |
| `artifacts/evaluation/model/retraining/generated_configs/` | Configurações geradas automaticamente para retreinos auditáveis e reproduzíveis. |
| `configs/scenario_experiments/inference_cases.yaml` | Suíte versionada de cenários de inferência usada para validar comportamento do modelo em casos de negócio. |
| `artifacts/evaluation/model/scenario_experiments/drift/*.jsonl` | Lotes sintéticos construídos para simular diferentes perfis de drift e testar o fluxo de monitoramento. |
| `artifacts/evaluation/model/scenario_experiments/drift/*_report.html` | Relatórios HTML dos cenários sintéticos, usados para demonstração e validação do processo de drift. |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Índice principal das avaliações do projeto: modelo tabular, cenários, drift, retreino e RAG/LLM. |

## Documentação Complementar

- [docs/EVALUATION.md](docs/EVALUATION.md)
- [docs/DRIFT_MONITORING.md](docs/DRIFT_MONITORING.md)
- [docs/OPERATIONS_DASHBOARD.md](docs/OPERATIONS_DASHBOARD.md)
- [docs/MODEL_VERSIONING.md](docs/MODEL_VERSIONING.md)
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- [docs/SCENARIO_ANALYSIS.md](docs/SCENARIO_ANALYSIS.md)
- [docs/EVALUATION_MODEL_METRICS.md](docs/EVALUATION_MODEL_METRICS.md)
- [docs/AGENT_REACT.md](docs/AGENT_REACT.md)
- [docs/EVALUATION_RAGAS.md](docs/EVALUATION_RAGAS.md)
- [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md)
- [docs/SYNTHETIC_PREDICTIONS_GENERATOR.md](docs/SYNTHETIC_PREDICTIONS_GENERATOR.md)

## Referências

1. OWASP (2025). *Top 10 for Large Language Model Applications*.
2. Brasil (2018). *Lei n° 13.709/2018 (LGPD)*.
3. Kaggle. *Bank Customer Churn*. Disponível em: <https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn>.
4. Kaggle. *Bank Customer Churn - Discussion*. Disponível em: <https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/discussion?sort=undefined>.

Nota sobre a referência do Kaggle:

- a base foi utilizada principalmente para fins educacionais, experimentais e de demonstração técnica no contexto do projeto
- a página do dataset indica licença do tipo `Other (specified in description)`, então a eventual redistribuição ou uso fora do contexto acadêmico deve considerar essa ressalva e ser validada diretamente na origem
- a citação acima documenta a fonte de obtenção da base, mas não substitui a verificação de licença, procedência e condições de uso

## Autores

**Turma 6MLET - FIAP**

- Luca Poit - RM365678
- Gabriel Jordan - RM365606
- Luciana Ferreira - RM366171
- Marcio Lima - RM365919

## Licença

Este projeto adota a licença MIT para o código-fonte do repositório.
