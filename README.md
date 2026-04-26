# TC FIAP Fase 5

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

Projeto integrador da Fase 05 do curso MLET da FIAP, desenvolvido no formato de Datathon. O repositório implementa uma solução de predição de churn bancário com foco em MLOps, rastreabilidade, observabilidade e evolução arquitetural para componentes com LLMs e agentes.

O `README` apresenta o projeto, a arquitetura e a forma de execução. O acompanhamento de aderência aos requisitos, entregas concluídas e pendências fica centralizado em [STATUS_ATUAL_PROJETO.md](STATUS_ATUAL_PROJETO.md).

## Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [O que o Projeto Entrega](#o-que-o-projeto-entrega)
- [Arquitetura da Solução](#arquitetura-da-solução)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Instalação e Execução](#instalação-e-execução)
- [LLM, agente ReAct e llm_provider](#llm-agente-react-e-llm_provider)
- [Feature Store](#feature-store)
- [Monitoramento e Observabilidade](#monitoramento-e-observabilidade)
- [Artefatos Relevantes](#artefatos-relevantes)
- [Documentação Complementar](#documentação-complementar)
- [Autores](#autores)

## Sobre o Projeto

Este projeto foi organizado como uma plataforma de machine learning aplicada a churn bancário. A proposta é cobrir uma trilha de ponta a ponta, desde dados versionados e engenharia de features até serving, monitoramento de drift e retreino auditável.

Um ponto importante da narrativa do repositório é a transformação de um experimento centrado em notebook para uma solução mais robusta de engenharia de ML. O notebook [notebooks/churn_bancario_sem_mlops.ipynb](notebooks/churn_bancario_sem_mlops.ipynb) representa essa base inicial, mais próxima de um fluxo exploratório executado em Jupyter ou Colab. A partir dele, o projeto evolui para uma estrutura com separação de responsabilidades, versionamento de dados, treino rastreável, serving, monitoramento, governança e documentação operacional.

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

Além da trilha tabular principal, o repositório inclui uma trilha **LLM** (agente ReAct, RAG, guardrails e integração com `llm_provider`) já utilizável via API; avaliação formal (RAGAS, benchmark com várias configs) e CI/CD específicos do agente são os próximos passos planejados. O andamento frente aos requisitos do Datathon continua detalhado em [STATUS_ATUAL_PROJETO.md](STATUS_ATUAL_PROJETO.md).

## O que o Projeto Entrega

Hoje o repositório já possui uma base funcional e demonstrável nas seguintes frentes:

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
- persistência do modelo atual, challengers e metadados em `artifacts/models/`
- apoio a promoção champion-challenger em [src/model_lifecycle/promotion.py](src/model_lifecycle/promotion.py)

### 3. Serving e inferência

- API FastAPI em [src/serving/app.py](src/serving/app.py)
- endpoint de predição em [src/serving/routes.py](src/serving/routes.py)
- contratos de entrada e saída em [src/serving/schemas.py](src/serving/schemas.py)
- carregamento compartilhado do pipeline de features e do modelo em [src/serving/pipeline.py](src/serving/pipeline.py)

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

  Se `artifacts/rag/` tiver sido criado por um container antigo com outro usuário, corrija a posse uma vez antes da execução local:

  ```bash
  sudo chown -R "$(id -u):$(id -g)" artifacts/rag
  ```

**Próximos passos planejados (ainda não concluídos no repositório):** extensão do CI/CD para essa trilha e documentação agregada de resultados de avaliação.

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
- Poetry 2.x
- Docker e Docker Compose, para Redis, serving, MLflow, Prometheus, Grafana e, opcionalmente, Ollama
- acesso ao remote DVC no Google Drive, caso vá baixar os dados versionados em vez de reconstruir a partir de arquivos locais

### 1. Clone do repositório

Comece clonando o projeto e entrando na raiz do repositório:

```bash
git clone https://github.com/marciojolima/tc_fiap_fase5.git
cd tc_fiap_fase5
```

Se estiver validando uma branch específica, troque para ela antes da instalação:

```bash
git checkout <nome-da-branch>
```

### 2. Instalação completa do ambiente

Entre na raiz do repositório e instale as dependências do projeto. Para gerar todos os artefatos documentados neste README, use os extras opcionais de treino, serving, monitoramento, avaliação e operações:

```bash
poetry install --all-extras
```

Se a intenção for apenas trabalhar no núcleo Python sem DVC, Feast, MLflow, Evidently ou avaliação LLM, a instalação mínima também funciona:

```bash
poetry install
```

Se o Poetry informar que o `pyproject.toml` mudou significativamente desde a última geração do `poetry.lock`, a branch clonada está com o lock file desatualizado. Nesse caso, regenere o lock e repita a instalação:

```bash
poetry lock
poetry install --all-extras
```

Depois da instalação, valide se as tasks do projeto estão disponíveis:

```bash
poetry run task --list
```

Os comandos deste README usam `poetry run`, então não é obrigatório ativar o ambiente virtual manualmente. Se preferir trabalhar com o ambiente ativado no shell atual, use:

```bash
eval "$(poetry env activate)"
```

Em ambientes com o plugin `poetry-plugin-shell` instalado, a alternativa equivalente é:

```bash
poetry shell
```

Crie também o arquivo `.env` local usado pelo Docker Compose e pelos providers externos de LLM:

```bash
cp .env.example .env
```

Quando usar OpenAI ou Claude como provider ativo, preencha no `.env` as variáveis correspondentes:

```bash
OPENAI_API_KEY=<sua-chave>
ANTHROPIC_API_KEY=<sua-chave>
```

#### Poetry ainda é necessário se eu usar Docker?

Para subir apenas a stack já construída com `docker compose`, o Docker executa a API e os serviços de apoio em containers. Mesmo assim, o fluxo completo do projeto ainda usa Poetry no host para os comandos batch e de produção de artefatos, como:

- `poetry run dvc pull`
- `poetry run dvc repro featurize`
- `poetry run dvc repro train`
- `poetry run dvc repro export_feature_store`
- `poetry run task feastapply`
- `poetry run task feastmaterialize`
- `poetry run task mldrift`
- `poetry run task eval_all`

Na prática: Docker cobre serving, Redis, MLflow, Prometheus, Grafana e Ollama. Poetry cobre a orquestração local dos pipelines, DVC, geração de datasets, treino, avaliação e manutenção dos artefatos que a stack monta por volume.

### 3. Sincronização de dados versionados

O projeto utiliza DVC para dados e artefatos versionados. No repositório atual, o remote padrão já está definido em `.dvc/config` com o nome `datathon_remote` e apontando para um storage no Google Drive.

Se o DVC já estiver instalado no ambiente, você pode usar `dvc ...` diretamente. Se preferir usar as dependências gerenciadas pelo projeto, utilize `poetry run dvc ...`.

#### Como a configuração está organizada

- `.dvc/config`: arquivo versionado no Git com a configuração compartilhada do remote, como nome e URL
- `.dvc/config.local`: arquivo local, não versionado, usado para credenciais e segredos de cada máquina

Em outras palavras:

- o time pode versionar em `.dvc/config` que o remote se chama `datathon_remote`
- cada pessoa configura em `.dvc/config.local` suas próprias credenciais de acesso

#### 1. Verifique ou configure o remote

No projeto atual, a configuração compartilhada já aponta para o remote `datathon_remote`. Se você precisar recriá-lo manualmente em outra máquina, o fluxo é:

```bash
dvc remote add -d datathon_remote gdrive://<REMOTE_ID>
```

Se estiver usando o ambiente do projeto via Poetry:

```bash
poetry run dvc remote add -d datathon_remote gdrive://<REMOTE_ID>
```

#### 2. Configure as credenciais locais do Google Drive

As credenciais OAuth não devem ir para o Git. Por isso, elas devem ser gravadas localmente com `--local`, o que escreve em `.dvc/config.local`:

```bash
dvc remote modify --local datathon_remote gdrive_client_id <ID>
dvc remote modify --local datathon_remote gdrive_client_secret <SECRET>
```

Ou, usando o ambiente do projeto:

```bash
poetry run dvc remote modify --local datathon_remote gdrive_client_id <ID>
poetry run dvc remote modify --local datathon_remote gdrive_client_secret <SECRET>
```

#### 3. Garanta a permissão no Google Drive

Não basta conhecer o `client_id` e o `client_secret`. A conta Google usada na autenticação também precisa ter permissão para acessar o storage apontado pelo remote.

Na prática, isso significa que:

- a pasta ou recurso do Google Drive referenciado pelo `gdrive://...` precisa estar compartilhado com a conta que fará o `dvc pull`
- se a permissão não existir, a autenticação pode até funcionar, mas o download dos dados falhará por falta de acesso ao conteúdo

Se o time estiver centralizando os dados em uma pasta compartilhada, confirme antes que sua conta foi adicionada com acesso apropriado.

#### 4. Configure o OAuth no Google Cloud Console

Para que o DVC possa autenticar no Google Drive via OAuth, é necessário existir um cliente OAuth configurado no Google Cloud Console. Em linhas gerais:

1. crie ou selecione um projeto no Google Cloud Console
2. habilite a Google Drive API para esse projeto
3. configure a tela de consentimento OAuth
4. crie credenciais do tipo OAuth Client ID
5. use o `client_id` e o `client_secret` gerados nos comandos `dvc remote modify --local ...`

Na primeira autenticação, o DVC pode abrir um fluxo de autorização OAuth no navegador. Essa etapa vincula a conta Google local ao client OAuth configurado e concede acesso ao storage do Drive.

#### 5. Baixe os dados

Depois do remote e das credenciais estarem corretos, baixe os dados com:

```bash
dvc pull
```

Ou:

```bash
poetry run dvc pull
```

#### Resumo prático

```bash
dvc remote modify --local datathon_remote gdrive_client_id <ID>
dvc remote modify --local datathon_remote gdrive_client_secret <SECRET>
dvc pull
```

#### Observações importantes

- não versione `.dvc/config.local`
- não publique `client_id` e `client_secret` em README, issue, commit ou pull request
- se a autenticação OAuth estiver correta, mas o Drive não estiver compartilhado com sua conta, o `pull` ainda assim pode falhar
- `.dvc/config` define a configuração compartilhada do remote; `.dvc/config.local` guarda segredos e ajustes locais da máquina

### 4. Produção dos artefatos principais

Esta é a sequência recomendada para sair de um clone novo do repositório e produzir os artefatos necessários para treino, serving, Feature Store e monitoramento.

#### 4.1 Preparar dados, features e modelo champion

Execute os stages do DVC na ordem de dependência:

```bash
poetry run dvc repro featurize
poetry run dvc repro train
poetry run dvc repro export_feature_store
```

Responsabilidade de cada gatilho:

- `dvc repro featurize`: gera `data/interim/cleaned.parquet`, `data/processed/train.parquet`, `data/processed/test.parquet`, `data/processed/feature_columns.json`, `data/processed/schema_report.json` e `artifacts/models/feature_pipeline.joblib`
- `dvc repro train`: treina o modelo champion e gera `artifacts/models/model_current.pkl` e `artifacts/models/model_current_metadata.json`
- `dvc repro export_feature_store`: usa `artifacts/models/feature_pipeline.joblib` para gerar `data/feature_store/customer_features.parquet` e `data/feature_store/export_metadata.json`

Observações importantes:

- `dvc repro export_feature_store` depende do artefato `artifacts/models/feature_pipeline.joblib`, gerado no stage `featurize`
- a API de predição completa também depende de `artifacts/models/model_current.pkl`, gerado no stage `train`

#### 4.2 Registrar e materializar a Feature Store

A materialização online depende do Redis. Para subir apenas o Redis:

```bash
docker compose up -d redis
```

Depois registre as definições do Feast e materialize os dados para a online store:

```bash
poetry run task feastapply
poetry run task feastmaterialize
```

Responsabilidade de cada comando:

- `task feastapply`: registra `Entity`, `FeatureView` e `FeatureServices` no registry local do Feast, gerando `feature_store/data/registry.db`
- `task feastmaterialize`: lê `data/feature_store/customer_features.parquet` e materializa incrementalmente as features na online store Redis
- `task feastdemo`: valida uma leitura online de exemplo para o cliente `15634602`

Validação opcional:

```bash
poetry run task feastdemo
```

#### 4.3 Produzir experimentos, cenários e trilha MLflow

Para executar o champion, os challengers configurados e a suíte de cenários de negócio em uma única sequência:

```bash
poetry run task mlrunall
```

Esse comando registra runs no MLflow local configurado em `file:./mlruns` quando nenhum `MLFLOW_TRACKING_URI` externo é informado. Ele também gera modelos experimentais em `artifacts/models/` conforme os caminhos declarados em `configs/model_lifecycle/experiments/`.

Também é possível executar partes isoladas:

```bash
poetry run task mltrain
poetry run task mlrunexperiments
poetry run task mlscenarios
```

#### 4.4 Produzir artefatos de monitoramento e drift

Com o modelo e os dados processados disponíveis, gere uma execução demonstrável de drift usando a base de teste como base corrente:

```bash
poetry run task mldriftdemo
```

Para executar o monitoramento sobre o log real de inferências da API, primeiro suba a stack, gere predições no endpoint `/predict` e depois rode:

```bash
poetry run task mldrift
```

Os principais artefatos gerados ficam em:

- `artifacts/logs/inference/predictions.jsonl`
- `artifacts/evaluation/model/drift/drift_report.html`
- `artifacts/evaluation/model/drift/drift_report_evidently.html`
- `artifacts/evaluation/model/drift/drift_metrics.json`
- `artifacts/evaluation/model/drift/drift_status.json`
- `artifacts/evaluation/model/drift/drift_runs.jsonl`
- `artifacts/evaluation/model/retraining/`, quando o gatilho de retreino é acionado

#### 4.5 Produzir artefatos de avaliação LLM

As avaliações LLM usam o provider configurado em `configs/pipeline_global_config.yaml`.
Para RAGAS, suba o serving antes, porque a avaliação chama `POST /llm/chat`.
Com o provider pronto, rode:

```bash
poetry run task eval_ragas
poetry run task eval_llm_judge
poetry run task eval_ab_test_prompts
```

Para executar a trilha de avaliação em container dedicado, use a imagem de
avaliação:

```bash
poetry run task eval_ragas_docker
poetry run task eval_all_sample_docker
```

Ou execute tudo em sequência:

```bash
poetry run task eval_all
```

Saídas esperadas:

- `artifacts/evaluation/llm_agent/results/ragas_scores.json`
- `artifacts/evaluation/llm_agent/results/llm_judge_scores.json`
- `artifacts/evaluation/llm_agent/results/prompt_ab_results.json`
- `artifacts/evaluation/llm_agent/runs/*.jsonl`

#### 4.6 Sequência curta para reproduzir os artefatos essenciais

Para uma execução local completa e objetiva:

```bash
poetry install --all-extras
cp .env.example .env
poetry run dvc pull
poetry run dvc repro featurize
poetry run dvc repro train
poetry run dvc repro export_feature_store
docker compose up -d redis
poetry run task feastapply
poetry run task feastmaterialize
poetry run task mldriftdemo
```

Se quiser validar também serving, dashboards e MLflow:

```bash
poetry run task appstack
```

### 5. Stack local com Docker Compose

O arquivo `.env.example` é apenas um modelo versionado com valores de referência.
O arquivo efetivamente lido pelo `docker compose` é o `.env`, que você cria a partir dele.

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

Com a stack em execução, a documentação interativa do FastAPI fica disponível no endpoint padrão de documentação do ambiente local (incluindo rotas `/llm/*`).

**Modo desenvolvimento:** use `poetry run task appstack_dev` para subir a stack com o override [docker-compose.dev.yml](docker-compose.dev.yml). Nesse modo, o `serving` monta `./src` em `/app/src` e roda o Uvicorn com `--reload`, então alterações em arquivos Python dentro de `src/` não exigem rebuild da imagem.

**Quando usar rebuild:** reconstrua a stack apenas quando mudar Dockerfile, `pyproject.toml`, `poetry.lock`, dependências ou alguma estrutura relevante de build. Para a stack base, use `poetry run task appstack_rebuild`; para desenvolvimento, use `poetry run task appstack_dev_rebuild`; para o cenário com Ollama local, use `poetry run task appstack_ollama_rebuild`.

**Quando não precisa rebuild:** no modo desenvolvimento, mudanças em `src/` são recarregadas pelo Uvicorn. Configurações, dados e artefatos também já ficam disponíveis por volumes do Compose principal, incluindo `configs/`, `data/processed/`, `data/feature_store/`, `artifacts/` e `feature_store/`.

**Diagnóstico LLM:** com a stack no ar, abra `http://127.0.0.1:8000/llm/status` para ver o `llm_provider` ativo, o modelo esperado e o diagnóstico específico do provider. Se o provider for `ollama`, `poetry run task ollama_list` ajuda a confirmar os modelos instalados nessa instância.

Para encerrar os serviços:

```bash
poetry run task appstack_down
```

### 6. Execução manual isolada

Se você quiser subir somente um componente fora do Compose durante desenvolvimento local:

### Feature Store

O projeto agora possui uma Feature Store local baseada em Feast, com Redis como online store. O objetivo é separar claramente a camada offline, usada para preparo e materialização, da camada online, usada para consulta de baixa latência.

Além disso, a governança de consumo foi refinada com `FeatureServices` por versão de modelo. Isso deixa explícito qual contrato de features cada modelo usa no treino e no serving, mesmo quando diferentes versões ainda compartilham a mesma `FeatureView` base.

Fluxo recomendado:

```bash
poetry run dvc repro featurize
poetry run dvc repro train
poetry run dvc repro export_feature_store
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

Detalhamento completo, decisões arquiteturais, limitações e próximos passos estão em [docs/FEATURE_STORE.md](docs/FEATURE_STORE.md).

Serving:

```bash
poetry run task serving
```

MLflow:

```bash
poetry run task mlflow
```

### 7. Monitoramento e demonstração de drift

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
poetry run task mlsyntheticdrift
```

### 8. Testes

```bash
poetry run task test
```

## LLM, agente ReAct e llm_provider

Este tópico resume o que foi implementado na trilha LLM e como operar em conjunto com o Docker Compose. O detalhamento por arquivo e endpoint está na subseção **6. LLM, agente ReAct, RAG e segurança**, em [O que o Projeto Entrega](#o-que-o-projeto-entrega).

- **Integração:** a API usa o `llm_provider` ativo definido em `configs/pipeline_global_config.yaml`, com implementação para `ollama`, `openai` e `claude`.
- **Tamanho da resposta:** `POST /llm/chat` aceita `answer_style` com `short`, `medium` (default) ou `long` para controlar a extensão da resposta final do agente.
- **Segredos:** providers externos leem `OPENAI_API_KEY` ou `ANTHROPIC_API_KEY` do `.env`.
- **Compose base:** `poetry run task appstack` sobe apenas a stack comum, sem carregar container local de modelo.
- **Índice RAG:** `poetry run task rag_index_rebuild_docker` gera o cache vetorial em `artifacts/rag/cache/index.joblib` usando a imagem leve do serving antes de subir a API.
- **Avaliação LLM isolada:** `poetry run task eval_ragas_docker` e `poetry run task eval_all_docker` usam a imagem `tc-fiap-evaluation`, mantendo a execução de avaliação separada do serving.
- **Compose com Ollama:** `poetry run task appstack_ollama` adiciona `ollama` e `ollama-pull`. Esse é o modo indicado quando `llm.active_provider=ollama`.
- **Base URL no Docker:** no cenário com Ollama local, o override do Compose injeta `LLM_BASE_URL=http://ollama:11434` no `serving`, porque `127.0.0.1` dentro do container apontaria para o próprio container da API.
- **Modelo Ollama:** use uma **tag válida** na biblioteca Ollama (por exemplo `gemma3:270m`). Nomes estilo arquivo GGUF não são tags do `ollama pull`.
- **Container `ollama-pull`:** ao subir a stack com override Ollama, ele termina com estado **Exited** após o pull — comportamento esperado para um job único. Em caso de dúvida, use `docker logs tc-fiap-ollama-pull`.
- **Rebuild da imagem da app:** no modo desenvolvimento, mudanças em `src/` são recarregadas pelo `serving` com Uvicorn `--reload`. Rebuild fica reservado para mudanças em Dockerfile, `pyproject.toml`, `poetry.lock`, dependências ou estrutura relevante de build.

## Monitoramento e Observabilidade

O projeto já implementa uma trilha concreta de monitoramento técnico para a solução tabular de churn, combinando métricas operacionais, logging de inferências, detecção de drift e fluxo de retreino auditável.

### O que já é monitorado

#### Métricas operacionais do serving

As métricas expostas pela aplicação permitem acompanhar o comportamento da API em execução, com foco inicial em:

- volume de requisições
- latência
- taxa de erro
- requisições em andamento

Essas métricas são consumidas pela stack local configurada em `configs/monitoring/`, agora orquestrada pelo Docker Compose junto com o serving e o MLflow.

#### Logging de inferências

As inferências podem ser registradas em `artifacts/logs/inference/predictions.jsonl`, criando uma trilha de execução útil para:

- auditoria das features efetivamente servidas ao modelo
- composição do dataset corrente de monitoramento
- análise posterior de drift
- apoio a ciclos de retreino

O contrato atual desse arquivo prioriza as features transformadas e monitoráveis
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

O relatório HTML agora também destaca no topo o resumo operacional do projeto,
incluindo thresholds de `warning` e `critical` definidos no YAML e o status
final calculado pelo pipeline batch. Esse arquivo passou a representar a visão
oficial do projeto para drift, baseada no PSI persistido em
`drift_metrics.json`, enquanto o Evidently fica disponível em um relatório
auxiliar separado para diagnóstico complementar.

#### Gatilho auditável de retreino

Quando o monitoramento identifica condição crítica, o projeto já suporta uma trilha auditável de retreino, com artefatos como:

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
| Swagger UI | `http://127.0.0.1:8000/docs` | Teste interativo do endpoint |
| MLflow UI | `http://127.0.0.1:5000` | Rastreamento de experimentos |
| Prometheus | `http://localhost:9090` | Coleta e exploração das métricas |
| Grafana | `http://localhost:3000` | Dashboards operacionais |

O Compose monta `configs/`, `artifacts/` e `mlruns/` com caminhos compatíveis com o código do projeto. Com isso, o serving carrega o mesmo modelo champion e o mesmo pipeline de features já materializados localmente, enquanto o MLflow expõe os experimentos rastreados em `mlruns/`.

### Fluxo sugerido para validação local

1. Copie `.env.example` para `.env`, se quiser customizar portas ou credenciais.
2. Suba a stack com `poetry run task appstack`.
3. Gere tráfego pelo Swagger ou por chamadas ao endpoint de predição.
4. Consulte as métricas no Prometheus.
5. Abra o Grafana para visualizar os painéis provisionados.
6. Abra o MLflow para revisar runs, parâmetros, métricas e artefatos.
7. Rode `poetry run task mldriftdemo` ou `poetry run task mldrift` para produzir uma execução de drift.
8. Abra os relatórios HTML e os arquivos JSON em `artifacts/evaluation/model/` para inspecionar as evidências geradas.

Resumo rápido:

- `.env.example`: template versionado, usado como referência para o time
- `.env`: arquivo local efetivamente lido pelo `docker compose`

## Artefatos Relevantes

Os arquivos abaixo ajudam a demonstrar reprodutibilidade, rastreabilidade e operação do projeto. Eles também servem como evidência objetiva do que já foi implementado.

| Artefato | Papel no projeto |
|---|---|
| `data/interim/cleaned.parquet` | Base saneada da camada `interim`: já teve identificadores diretos removidos, passou por deduplicação, remoção de nulos e validação de schema, mas ainda não foi convertida para o formato final de modelagem. |
| `data/processed/train.parquet` | Base final de treino da camada `processed`: já passou por split, criação de features derivadas, remoção de leakage, encoding e scaling, ficando pronta para consumo pelos algoritmos. |
| `data/processed/test.parquet` | Base final de teste da camada `processed`, gerada com o mesmo pipeline do treino e mantida separada para validação sem vazamento. |
| `data/processed/feature_columns.json` | Registra a ordem e os nomes finais das features, ajudando a manter consistência entre treino e inferência. |
| `data/processed/schema_report.json` | Evidência da validação estrutural dos dados processados, reforçando a etapa de qualidade de dados. |
| `artifacts/models/feature_pipeline.joblib` | Pipeline de transformação persistido para reutilização no serving, evitando divergência entre treino e produção. |
| `artifacts/models/model_current.pkl` | Modelo champion atualmente mantido como versão principal para inferência. |
| `artifacts/models/model_current_metadata.json` | Metadados do champion atual, incluindo informações de versão, configuração e métricas relevantes. |
| `artifacts/models/challengers/` | Diretório reservado para challengers gerados em ciclos de retreino e comparados antes de eventual promoção. |
| `artifacts/logs/inference/predictions.jsonl` | Log de inferências usado como base para monitoramento posterior. No contrato atual, ele registra principalmente as features transformadas efetivamente servidas ao modelo, com metadados mínimos de predição e origem. |
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
| [docs/EVALUATION.md](docs/EVALUATION.md) | Visão consolidada das avaliações do projeto: modelo tabular, cenários, drift, retreino e trilha futura de LLM. |

## Documentação Complementar

- [docs/EVALUATION.md](docs/EVALUATION.md)
- [docs/DRIFT_MONITORING.md](docs/DRIFT_MONITORING.md)
- [docs/OPERATIONS_DASHBOARD.md](docs/OPERATIONS_DASHBOARD.md)
- [docs/MODEL_VERSIONING.md](docs/MODEL_VERSIONING.md)
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- [docs/SCENARIO_ANALYSIS.md](docs/SCENARIO_ANALYSIS.md)
- [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md)
- [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md)
- [docs/SYNTHETIC_PREDICTIONS_GENERATOR.md](docs/SYNTHETIC_PREDICTIONS_GENERATOR.md)

## Autores

**Turma 6MLET - FIAP**

- Luca Poiti - RM365678
- Gabriel Jordan - RM365606
- Luciana Ferreira - RM366171
- Marcio Lima - RM365919
