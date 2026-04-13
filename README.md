# TC FIAP Fase 5

Projeto integrador da Fase 05 do curso MLET da FIAP, desenvolvido no formato de Datathon. O repositório reúne uma solução de predição de churn bancário com foco em engenharia de machine learning, rastreabilidade, observabilidade e evolução arquitetural para componentes de LLMOps e agentes.

O objetivo deste `README` é apresentar o projeto, sua estrutura e a forma de execução. O acompanhamento de progresso, aderência aos requisitos e pendências fica concentrado em [STATUS_ATUAL_PROJETO.md](STATUS_ATUAL_PROJETO.md).

## Grupo

Turma 6MLET - FIAP

- Luca Poiti - RM365678
- Gabriel Jordan - RM365606
- Luciana Ferreira - RM366171
- Marcio Lima - RM365919

## Visão Geral

O projeto foi organizado como uma plataforma de ML aplicada a churn bancário. A trilha principal hoje cobre:

- versionamento de dados com DVC
- engenharia de features com validação de schema
- treinamento rastreável com MLflow
- serving de inferência com FastAPI
- análise de cenários de negócio
- monitoramento batch de drift com Evidently e PSI
- fluxo auditável de retreino com avaliação champion-challenger
- observabilidade operacional com Prometheus e Grafana

Além da trilha principal tabular, o repositório também possui módulos em evolução para agente, RAG, avaliação de LLM e segurança aplicada. Esses componentes fazem parte da direção arquitetural do projeto, mas o andamento detalhado deles está documentado no arquivo de status.

## Objetivo do Projeto

Construir uma solução reproduzível de churn bancário que demonstre práticas esperadas de MLOps no contexto do Datathon:

- pipeline de dados versionado e reexecutável
- preparação de dados e features com consistência entre treino e inferência
- treinamento com rastreabilidade de parâmetros, métricas e artefatos
- API para consumo do modelo
- monitoramento de comportamento do modelo após inferência
- documentação técnica e de governança para apoio à apresentação

## Arquitetura Resumida

O fluxo principal do projeto pode ser resumido assim:

1. dados brutos são versionados em `data/raw/` com apoio do DVC
2. a engenharia de features gera artefatos em `data/interim/` e `data/processed/`
3. o treinamento registra execuções no MLflow e publica artefatos em `artifacts/models/`
4. a API FastAPI carrega o pipeline de features e o modelo atual para servir predições
5. as inferências alimentam arquivos de monitoramento em `data/monitoring/`
6. o monitoramento calcula drift, gera relatórios e pode acionar retreino auditável

## Estrutura do Repositório

```text
tc_fiap_fase5/
├── artifacts/          # modelos, relatórios de drift e saídas de retreino
├── configs/            # configurações de treino, monitoramento e observabilidade
├── data/               # camadas raw, interim, processed e logs de monitoramento
├── docs/               # documentação técnica e de governança
├── evaluation/         # scripts de avaliação para trilhas com LLM
├── notebooks/          # notebooks exploratórios e de apoio
├── scripts/            # utilitários auxiliares
├── src/
│   ├── agent/          # componentes em evolução para agente e RAG
│   ├── common/         # utilidades compartilhadas
│   ├── features/       # engenharia e validação de features
│   ├── models/         # treino, promoção e retreino
│   ├── monitoring/     # drift, métricas e logging de inferências
│   ├── scenario_analysis/
│   ├── security/       # guardrails e PII em evolução
│   └── serving/        # aplicação FastAPI e pipeline de inferência
├── tests/              # suíte de testes automatizados
├── README.md
└── STATUS_ATUAL_PROJETO.md
```

## Componentes Principais

### Dados e features

- [src/features/feature_engineering.py](src/features/feature_engineering.py): pipeline principal de preparação dos dados
- [src/features/schema_validation.py](src/features/schema_validation.py): validação estrutural com Pandera
- [src/features/pipeline_components.py](src/features/pipeline_components.py): componentes reutilizáveis do pipeline

### Treinamento e gestão do modelo

- [src/models/train.py](src/models/train.py): treinamento principal com MLflow
- [src/models/promotion.py](src/models/promotion.py): decisão de promoção champion-challenger
- [src/models/retraining.py](src/models/retraining.py): fluxo de retreino auditável

### Serving

- [src/serving/app.py](src/serving/app.py): aplicação FastAPI
- [src/serving/routes.py](src/serving/routes.py): rotas HTTP
- [src/serving/schemas.py](src/serving/schemas.py): contratos de entrada e saída
- [src/serving/pipeline.py](src/serving/pipeline.py): carregamento do pipeline e do modelo

### Monitoramento e observabilidade

- [src/monitoring/drift.py](src/monitoring/drift.py): monitoramento batch de drift
- [src/monitoring/inference_log.py](src/monitoring/inference_log.py): trilha de inferências
- [src/monitoring/metrics.py](src/monitoring/metrics.py): métricas expostas para observabilidade
- [configs/observability/prometheus.yml](configs/observability/prometheus.yml): configuração do Prometheus

### Cenários e avaliação

- [src/scenario_analysis/inference_cases.py](src/scenario_analysis/inference_cases.py): execução de cenários de inferência
- [configs/scenario_analysis/inference_cases.yaml](configs/scenario_analysis/inference_cases.yaml): suíte versionada de casos
- `evaluation/`: scripts de avaliação para frentes com LLM e prompts

## Como Executar

### 1. Instalar dependências

```bash
poetry install
```

### 2. Sincronizar dados versionados

O projeto usa DVC para versionamento de dados e artefatos. Após instalar as dependências, sincronize os arquivos versionados:

```bash
poetry run dvc pull
```

Se o remote não estiver configurado no ambiente local, configure-o antes de executar o `pull`.

### 3. Rodar a pipeline principal

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

### 4. Subir a API

```bash
poetry run task serving
```

Com a aplicação em execução, a documentação interativa do FastAPI fica disponível na porta padrão do ambiente local.

### 5. Rodar monitoramento e observabilidade

Drift batch:

```bash
poetry run task mldrift
```

Execução demonstrável com base de teste:

```bash
poetry run task mldriftdemo
```

Observabilidade local:

```bash
poetry run task observability
```

Serviços locais:

- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- MLflow UI: `http://127.0.0.1:5000`

Para abrir o MLflow:

```bash
poetry run task mlflow
```

### 6. Executar testes

```bash
poetry run task test
```

## Artefatos Relevantes

- `data/processed/train.parquet` e `data/processed/test.parquet`: bases finais para treino e validação
- `data/processed/feature_columns.json`: ordem final das features
- `data/processed/schema_report.json`: evidência de validação estrutural
- `artifacts/models/model_current.pkl`: modelo atual usado no serving
- `artifacts/models/feature_pipeline.joblib`: pipeline de transformação compartilhado entre treino e inferência
- `data/monitoring/current/predictions.jsonl`: trilha de inferências para monitoramento
- `artifacts/monitoring/drift/`: relatórios e métricas de drift
- `artifacts/monitoring/retraining/`: pedidos de retreino, decisão de promoção e saídas auditáveis

## Documentação Complementar

- [docs/DRIFT_MONITORING.md](docs/DRIFT_MONITORING.md)
- [docs/OPERATIONS_DASHBOARD.md](docs/OPERATIONS_DASHBOARD.md)
- [docs/MODEL_VERSIONING.md](docs/MODEL_VERSIONING.md)
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- [docs/SCENARIO_ANALYSIS.md](docs/SCENARIO_ANALYSIS.md)
- [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md)
- [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md)
- [docs/SYNTHETIC_PREDICTIONS_GENERATOR.md](docs/SYNTHETIC_PREDICTIONS_GENERATOR.md)

## Status do Projeto

Como o projeto ainda está em andamento, o acompanhamento de entregas concluídas, lacunas frente ao Datathon e pendências atuais fica em [STATUS_ATUAL_PROJETO.md](STATUS_ATUAL_PROJETO.md).
