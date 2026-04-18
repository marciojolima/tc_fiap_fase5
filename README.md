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
- [Como Executar](#como-executar)
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

Além da trilha tabular principal, o repositório também mantém módulos em evolução para agente, RAG, avaliação de LLM e segurança aplicada. Esses componentes fazem parte da direção do projeto, mas seu andamento detalhado está no documento de status.

## O que o Projeto Entrega

Hoje o repositório já possui uma base funcional e demonstrável nas seguintes frentes:

### 1. Dados, features e preparação

- versionamento de dados com DVC
- separação entre camadas `raw`, `interim` e `processed`
- pipeline de engenharia de features em [src/features/feature_engineering.py](src/features/feature_engineering.py)
- componentes reutilizáveis em [src/features/pipeline_components.py](src/features/pipeline_components.py)
- validação estrutural com Pandera em [src/features/schema_validation.py](src/features/schema_validation.py)
- persistência de datasets preparados e artefatos auxiliares para reuso

### 2. Treinamento e gestão de modelo

- treinamento principal em [src/models/train.py](src/models/train.py)
- rastreamento de parâmetros, métricas e artefatos com MLflow
- múltiplas configurações de experimento em `configs/training/experiments/`
- persistência do modelo atual, challengers e metadados em `artifacts/models/`
- apoio a promoção champion-challenger em [src/models/promotion.py](src/models/promotion.py)

### 3. Serving e inferência

- API FastAPI em [src/serving/app.py](src/serving/app.py)
- endpoint de predição em [src/serving/routes.py](src/serving/routes.py)
- contratos de entrada e saída em [src/serving/schemas.py](src/serving/schemas.py)
- carregamento compartilhado do pipeline de features e do modelo em [src/serving/pipeline.py](src/serving/pipeline.py)

### 4. Cenários de negócio e validação

- suíte de cenários versionados em [configs/scenario_analysis/inference_cases.yaml](configs/scenario_analysis/inference_cases.yaml)
- execução automatizada em [src/scenario_analysis/inference_cases.py](src/scenario_analysis/inference_cases.py)
- geração de lotes sintéticos de drift em [src/scenario_analysis/synthetic_drifts.py](src/scenario_analysis/synthetic_drifts.py)

### 5. Monitoramento e operação

- logging de inferências em [src/monitoring/inference_log.py](src/monitoring/inference_log.py)
- métricas operacionais expostas em [src/monitoring/metrics.py](src/monitoring/metrics.py)
- detecção batch de drift com Evidently e PSI em [src/monitoring/drift.py](src/monitoring/drift.py)
- relatórios HTML e arquivos JSON para auditoria em `artifacts/monitoring/`
- stack local reproduzível com serving, MLflow, Prometheus e Grafana
- workflow básico de CI em [.github/workflows/ci.yml](.github/workflows/ci.yml)

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
├── evaluation/             # scripts de avaliação para trilhas com LLM
├── notebooks/              # notebooks exploratórios e de apoio
├── scripts/                # utilitários auxiliares
├── src/
│   ├── agent/              # componentes em evolução para agente e RAG
│   ├── common/             # utilidades compartilhadas
│   ├── features/           # engenharia e validação de features
│   ├── models/             # treino, promoção e retreino
│   ├── monitoring/         # drift, métricas e logging de inferências
│   ├── scenario_analysis/  # cenários de negócio e geração de batches sintéticos
│   ├── security/           # guardrails e PII em evolução
│   └── serving/            # aplicação FastAPI e pipeline de inferência
├── tests/                  # suíte de testes automatizados
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── STATUS_ATUAL_PROJETO.md
```

## Como Executar

### Pré-requisitos

- Python 3.13
- Poetry
- Docker e Docker Compose, caso queira subir a stack de observabilidade

### 1. Instalação

```bash
poetry install
```

### 2. Sincronização de dados versionados

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

### 3. Pipeline principal

Engenharia de features:

```bash
poetry run task mlfeateng
```

Treinamento:

```bash
poetry run task mltrain
```

Execução ampliada com múltiplos experimentos e cenários:

```bash
poetry run task mlrunall
```

### 4. Stack local com Docker Compose

O arquivo `.env.example` é apenas um modelo versionado com valores de referência.
O arquivo efetivamente lido pelo `docker compose` é o `.env`, que você cria a partir dele.

```bash
cp .env.example .env
poetry run task appstack
```

A stack local sobe os seguintes serviços de forma integrada:

- Redis
- serving FastAPI
- MLflow server
- Prometheus
- Grafana

Com a stack em execução, a documentação interativa do FastAPI fica disponível no endpoint padrão de documentação do ambiente local.

Para encerrar os serviços:

```bash
poetry run task appstack_down
```

### 5. Execução manual isolada

Se você quiser subir somente um componente fora do Compose durante desenvolvimento local:

### Feature Store

O projeto agora possui uma Feature Store local baseada em Feast, com Redis como online store. O objetivo é separar claramente a camada offline, usada para preparo e materialização, da camada online, usada para consulta de baixa latência.

Fluxo recomendado:

```bash
poetry run task mlfeateng
poetry run task feastexport
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
poetry run task mlsyntheticdrift
```

### 7. Testes

```bash
poetry run task test
```

## Monitoramento e Observabilidade

O projeto já implementa uma trilha concreta de monitoramento técnico para a solução tabular de churn, combinando métricas operacionais, logging de inferências, detecção de drift e fluxo de retreino auditável.

### O que já é monitorado

#### Métricas operacionais do serving

As métricas expostas pela aplicação permitem acompanhar o comportamento da API em execução, com foco inicial em:

- volume de requisições
- latência
- taxa de erro
- requisições em andamento

Essas métricas são consumidas pela stack local configurada em `configs/observability/`, agora orquestrada pelo Docker Compose junto com o serving e o MLflow.

#### Logging de inferências

As inferências podem ser registradas em `artifacts/monitoring/inference_logs/predictions.jsonl`, criando uma trilha de execução útil para:

- auditoria de predições
- composição do dataset corrente de monitoramento
- análise posterior de drift
- apoio a ciclos de retreino

#### Monitoramento batch de drift

O fluxo em [src/monitoring/drift.py](src/monitoring/drift.py) compara uma base de referência com dados correntes e produz evidências operacionais em:

- `artifacts/monitoring/drift/drift_report.html`
- `artifacts/monitoring/drift/drift_metrics.json`
- `artifacts/monitoring/drift/drift_status.json`
- `artifacts/monitoring/drift/drift_runs.jsonl`

Na prática, isso permite:

- inspecionar visualmente o comportamento das distribuições
- calcular PSI por feature
- consolidar um status geral de drift
- manter histórico das execuções de monitoramento

#### Gatilho auditável de retreino

Quando o monitoramento identifica condição crítica, o projeto já suporta uma trilha auditável de retreino, com artefatos como:

- `artifacts/monitoring/retraining/retrain_request.json`
- `artifacts/monitoring/retraining/retrain_run.json`
- `artifacts/monitoring/retraining/promotion_decision.json`
- `artifacts/monitoring/retraining/generated_configs/`

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
8. Abra os relatórios HTML e os arquivos JSON em `artifacts/monitoring/` para inspecionar as evidências geradas.

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
| `artifacts/monitoring/inference_logs/predictions.jsonl` | Log de inferências usado como base para monitoramento posterior, principalmente nos fluxos de drift. |
| `artifacts/monitoring/drift/drift_report.html` | Relatório HTML do Evidently para inspeção visual do comportamento das features e das distribuições monitoradas. |
| `artifacts/monitoring/drift/drift_metrics.json` | Consolidação das métricas de drift, incluindo PSI por feature e resumo para automação de decisão. |
| `artifacts/monitoring/drift/drift_status.json` | Estado mais recente do monitoramento de drift, com classificação para apoio ao gatilho de retreino. |
| `artifacts/monitoring/drift/drift_runs.jsonl` | Histórico de execuções do monitoramento, útil para trilha de auditoria e acompanhamento temporal. |
| `artifacts/monitoring/retraining/retrain_request.json` | Registro do pedido de retreino, com motivação e contexto do disparo do processo. |
| `artifacts/monitoring/retraining/retrain_run.json` | Resultado consolidado da execução do retreino, incluindo status, motivo, métricas e decisão final. |
| `artifacts/monitoring/retraining/promotion_decision.json` | Decisão champion-challenger com regra de promoção explícita e deltas de métricas entre os modelos comparados. |
| `artifacts/monitoring/retraining/generated_configs/` | Configurações geradas automaticamente para retreinos auditáveis e reproduzíveis. |
| `configs/scenario_analysis/inference_cases.yaml` | Suíte versionada de cenários de inferência usada para validar comportamento do modelo em casos de negócio. |
| `artifacts/scenario_analysis/drift/*.jsonl` | Lotes sintéticos construídos para simular diferentes perfis de drift e testar o fluxo de monitoramento. |
| `artifacts/scenario_analysis/drift/*_report.html` | Relatórios HTML dos cenários sintéticos, usados para demonstração e validação do processo de drift. |
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
