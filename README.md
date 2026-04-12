# TC FIAP Fase 5

Projeto integrador da Fase 05 do curso MLET da FIAP, estruturado como uma entrega de Datathon com foco em churn bancário, MLOps, observabilidade, governança e preparação para evolução com LLMs e agentes.

Este README foi revisado com base no estado real do repositório e tomando `REQUISITOS_DATATHON.md` como referência principal de aceite.

## Grupo

Turma 6MLET - FIAP

- Luca Poiti - RM365678
- Gabriel Jordan - RM365606
- Luciana Ferreira - RM366171
- Marcio Lima - RM365919

## Objetivo

Construir uma solução de predição de churn com:

- pipeline de dados reproduzível e versionado
- treino rastreável com MLflow
- API de inferência com FastAPI
- análise de cenários para validação de negócio
- detecção de drift com gatilho auditável de retreino
- documentação de governança e segurança em evolução

## Status Executivo

O projeto já entrega de forma consistente a base de dados, features, treino, serving e monitoramento batch de drift. Em termos da régua do Datathon, a entrega está mais madura em:

- Etapa 1: dados, baseline e MLOps inicial
- Etapa 3: observabilidade de drift para modelo tabular
- fundamentos de governança e rastreabilidade

Os principais itens ainda não concluídos para aderência mais forte ao guia são:

- baseline adicional em PyTorch
- notebook de EDA e golden set formal
- agente ReAct funcional com tools reais
- pipeline RAG e integração com LLM de serving
- RAGAS, LLM-as-judge e telemetria LLM operacionais
- guardrails e PII com implementação efetiva
- fairness audit automatizada e explicabilidade formal
- pipeline de deploy/staging e coverage gate explícito

## O Que Está Implementado

### Dados, features e treino

- versionamento de dados com DVC
- pipeline de engenharia de features em [src/features/feature_engineering.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/features/feature_engineering.py)
- validação de schema com Pandera em [src/features/schema_validation.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/features/schema_validation.py)
- split treino/teste antes do fit do pipeline de transformação
- persistência de `train.parquet`, `test.parquet`, `feature_columns.json` e `feature_pipeline.joblib`
- treino com MLflow, tags de governança e versionamento de dados processados em [src/models/train.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/models/train.py)
- múltiplos YAMLs de experimento em [configs/training/experiments/random_forest_v1.yaml](/home/marcio/dev/projects/python/tc_fiap_fase5/configs/training/experiments/random_forest_v1.yaml) e [configs/training/experiments/random_forest_v2.yaml](/home/marcio/dev/projects/python/tc_fiap_fase5/configs/training/experiments/random_forest_v2.yaml)

### Serving e análise

- API FastAPI em [src/serving/app.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/serving/app.py)
- rota `/predict` com schemas Pydantic em [src/serving/routes.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/serving/routes.py) e [src/serving/schemas.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/serving/schemas.py)
- reutilização do mesmo pipeline de features no treino e na inferência em [src/serving/pipeline.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/serving/pipeline.py)
- análise de cenários com logging dedicado no MLflow em [src/scenario_analysis/inference_cases.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/scenario_analysis/inference_cases.py)

### Monitoramento e governança técnica

- registro de inferências para monitoramento em [src/monitoring/inference_log.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/inference_log.py)
- detecção batch de drift com Evidently e PSI em [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py)
- geração de drifts sintéticos para validar o fluxo experimental em [src/scenario_analysis/synthetic_drifts.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/scenario_analysis/synthetic_drifts.py)
- gatilho auditável de retreino via `artifacts/monitoring/retraining/retrain_request.json`
- estratégia e fluxo documentados em [docs/DRIFT_MONITORING.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/DRIFT_MONITORING.md)
- versionamento de metadados de modelo em [docs/MODEL_VERSIONING.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/MODEL_VERSIONING.md)

## O Que Está Parcial ou Placeholder

Os arquivos abaixo existem, mas ainda não caracterizam entrega completa segundo `REQUISITOS_DATATHON.md`:

- agente e tools: [src/agent/react_agent.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/agent/react_agent.py), [src/agent/tools.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/agent/tools.py)
- RAG: [src/agent/rag_pipeline.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/agent/rag_pipeline.py)
- guardrails e PII: [src/security/guardrails.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/security/guardrails.py), [src/security/pii_detection.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/security/pii_detection.py)
- dashboard operacional Prometheus/Grafana e métricas do serving em [src/monitoring/metrics.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/metrics.py)
- avaliação RAG/LLM: [evaluation/ragas_eval.py](/home/marcio/dev/projects/python/tc_fiap_fase5/evaluation/ragas_eval.py), [evaluation/llm_judge.py](/home/marcio/dev/projects/python/tc_fiap_fase5/evaluation/llm_judge.py), [evaluation/ab_test_prompts.py](/home/marcio/dev/projects/python/tc_fiap_fase5/evaluation/ab_test_prompts.py)
- documentação ainda muito inicial: [docs/SYSTEM_CARD.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/SYSTEM_CARD.md), [docs/OWASP_MAPPING.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/OWASP_MAPPING.md), [docs/RED_TEAM_REPORT.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/RED_TEAM_REPORT.md)

## Estrutura do Repositório

```text
tc_fiap_fase5/
├── .github/
├── artifacts/
├── configs/
├── data/
├── docs/
├── evaluation/
├── src/
│   ├── agent/
│   ├── common/
│   ├── features/
│   ├── inference/
│   ├── models/
│   ├── monitoring/
│   ├── scenario_analysis/
│   ├── security/
│   └── serving/
├── tests/
├── docker-compose.yml
├── dvc.lock
├── dvc.yaml
├── Makefile
├── pyproject.toml
├── README.md
└── REQUISITOS_DATATHON.md
```

## Como Executar

### Instalação

```bash
poetry install
```

### Engenharia de features

```bash
poetry run task mlfeateng
```

### Treino do modelo atual

```bash
poetry run task mltrain
```

### Rodar experimentos e cenários

```bash
poetry run task mlrunall
```

### API de serving

```bash
poetry run task serving
```

### Dashboard operacional

```bash
poetry run task observability
```

Depois acesse:

- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

O dashboard provisionado acompanha inicialmente a latência, o volume, a taxa de erro e as requisições em andamento do endpoint `/predict`.

### Drift batch

```bash
poetry run task mldrift
```

Para gerar uma execução demonstrável sem tráfego real:

```bash
poetry run task mldriftdemo
```

### Lotes sintéticos de drift

```bash
poetry run task mlsyntheticdrift
```

### MLflow UI

```bash
poetry run task mlflow
```

### Testes

```bash
poetry run task test
```

## Artefatos Relevantes

- `data/interim/cleaned.parquet`: camada intermediária limpa para auditoria e reprodutibilidade
- `data/processed/train.parquet`: base final de treino
- `data/processed/test.parquet`: base final de teste
- `data/processed/feature_columns.json`: ordem e nomes finais das features
- `data/processed/schema_report.json`: evidência de validação do pipeline
- `artifacts/models/feature_pipeline.joblib`: pipeline de transformação reutilizado em produção
- `data/monitoring/current/predictions.jsonl`: log das inferências para monitoramento
- `artifacts/monitoring/drift/drift_report.html`: relatório do Evidently
- `artifacts/monitoring/drift/drift_metrics.json`: PSI por feature e consolidação
- `artifacts/monitoring/drift/drift_status.json`: status do monitoramento
- `artifacts/monitoring/retraining/retrain_request.json`: gatilho auditável de retreino
- `artifacts/monitoring/retraining/retrain_run.json`: resultado auditável da execução do retreino
- `configs/scenario_analysis/inference_cases.yaml`: suíte versionada de cenários de inferência
- `artifacts/scenario_analysis/drift/*.jsonl`: cenários sintéticos para simulação de drift
- `artifacts/scenario_analysis/drift/*_report.html`: relatórios HTML do Evidently para drifts sintéticos
- `docs/DRIFT_MONITORING.md`: estratégia atual de drift, PSI e fluxo de retreino
- `docs/OPERATIONS_DASHBOARD.md`: instruções do dashboard operacional Prometheus/Grafana

## Documentação Disponível

### Mais consistentes hoje

- [docs/MODEL_CARD.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/MODEL_CARD.md)
- [docs/LGPD_PLAN.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/LGPD_PLAN.md)
- [docs/MODEL_VERSIONING.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/MODEL_VERSIONING.md)
- [docs/SCENARIO_ANALYSIS.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/SCENARIO_ANALYSIS.md)
- [docs/EVALUATION_METRICS.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/EVALUATION_METRICS.md)
- [docs/DRIFT_MONITORING.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/DRIFT_MONITORING.md)
- [docs/OPERATIONS_DASHBOARD.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/OPERATIONS_DASHBOARD.md)

### Ainda precisam evoluir para o padrão esperado da banca

- [docs/SYSTEM_CARD.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/SYSTEM_CARD.md)
- [docs/OWASP_MAPPING.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/OWASP_MAPPING.md)
- [docs/RED_TEAM_REPORT.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/RED_TEAM_REPORT.md)

## Checklist Revisado da Datathon

### Etapa 1: Dados, baseline e MLOps inicial

- [x] Estrutura de projeto Python com `pyproject.toml`
- [x] Dados versionados com DVC no repositório
- [x] Pipeline de engenharia de features com separação entre raw, interim e processed
- [x] Validação de schema com Pandera
- [x] Split treino/teste antes do fit do pipeline de transformação
- [x] Treino com MLflow e metadados padronizados
- [x] Persistência de datasets processados e pipeline de features
- [x] Execução de múltiplos experimentos de treino por configuração
- [ ] Baseline adicional em PyTorch
- [ ] Notebook de EDA incluído no repositório
- [ ] Golden set formal em `data/golden_set/`
- [ ] Feature store compartilhada ou materialização incremental entre modelos

### Etapa 2: API, LLM e agente

- [x] API FastAPI para serving
- [x] Schemas de entrada e saída para inferência
- [x] Análise de cenários com payloads versionados
- [ ] Agente ReAct funcional com pelo menos 3 tools
- [ ] Tools de negócio implementadas
- [ ] Pipeline RAG operacional
- [ ] Integração com LLM de serving
- [ ] Endpoints adicionais para agente ou RAG

### Etapa 3: Avaliação e observabilidade

- [x] Estrutura de avaliação criada em `evaluation/`
- [x] Configuração de monitoramento dedicada em `configs/monitoring_config.yaml`
- [x] Detecção de drift implementada com Evidently, PSI e relatório HTML
- [x] Gatilho auditável de retreino quando drift fica crítico
- [x] Lotes sintéticos para validar o fluxo de drift
- [ ] Dashboard operacional Prometheus/Grafana
- [ ] RAGAS com 4 métricas efetivamente executadas
- [ ] LLM-as-judge com pelo menos 3 critérios efetivamente executados
- [ ] Alertas automáticos integrados a canal externo
- [ ] Observabilidade LLM com Langfuse ou TruLens

### Etapa 4: Segurança e governança

- [x] Plano inicial de LGPD documentado
- [x] Minimização de identificadores diretos no pipeline de dados
- [x] Governança explícita para `Geography`
- [x] Registro de metadados de versão, `risk_level`, `fairness_checked` e `git_sha` no treino
- [ ] System Card completo no padrão esperado da banca
- [ ] OWASP mapping com cenários detalhados
- [ ] Red team report com cenários adversariais detalhados
- [ ] Guardrails de input/output implementados de forma efetiva
- [ ] Detecção e sanitização de PII implementadas
- [ ] Fairness audit automatizada e anexada ao ciclo de treino
- [ ] Explicabilidade formal da predição

### Engenharia de software e qualidade

- [x] Type hints nas partes principais do projeto
- [x] Logging estruturado nas etapas centrais
- [x] Testes unitários para features, treino, serving pipeline, monitoramento e cenários
- [x] Configuração de lint com Ruff
- [x] Organização por módulos de domínio
- [x] Workflow básico de CI em [/.github/workflows/ci.yml](/home/marcio/dev/projects/python/tc_fiap_fase5/.github/workflows/ci.yml)
- [ ] Hooks automatizados de pre-commit efetivamente configurados
- [ ] Teste de integração HTTP real para a API FastAPI
- [ ] Coverage gate formal, por exemplo `--cov-fail-under`
- [ ] Pipeline de deploy/staging

## Leitura Honesta do Projeto Frente ao Guia

Comparando com `REQUISITOS_DATATHON.md`, o repositório hoje demonstra melhor aderência aos requisitos de maturidade MLOps do que aos requisitos de LLMOps/agentes. Isso não é um problema em si, mas precisa ficar transparente para a apresentação:

- a trilha de modelo tabular para churn está funcional e defendível
- a trilha de observabilidade de drift também está demonstrável
- a trilha de agente, RAG e segurança aplicada ainda está em preparação

Se a intenção for maximizar aderência à banca, as próximas entregas de maior impacto são:

1. implementar agente ReAct com 3 tools reais e um endpoint dedicado
2. fechar guardrails e sanitização de PII com testes não-placeholder
3. transformar `SYSTEM_CARD`, `OWASP_MAPPING` e `RED_TEAM_REPORT` em documentos completos
4. adicionar golden set e avaliação automatizada
5. evoluir o CI atual para incluir coverage gate e deploy de staging

## Observações Importantes

- O `Makefile` ainda está apenas como placeholder e não representa os comandos reais do projeto.
- O workflow de CI existente executa lint, checagem sintática e testes, mas não faz deploy nem possui gate de cobertura.
- O arquivo [/.pre-commit-config.yaml](/home/marcio/dev/projects/python/tc_fiap_fase5/.pre-commit-config.yaml) existe, porém ainda está vazio na prática e não configura hooks úteis.
- Os testes de agente e guardrails ainda são placeholders e não devem ser apresentados como evidência de funcionalidade.
- A documentação de governança existe, porém parte dela ainda está em nível inicial e precisa ser aprofundada antes da entrega final.

## Fluxo Recomendado para Demonstração

1. `poetry install`
2. `poetry run task mlfeateng`
3. `poetry run task mltrain`
4. `poetry run task mlscenario`
5. `poetry run task serving`
6. `poetry run task mldriftdemo`
7. `poetry run task mlflow`

## Licença e Uso

Licença ainda não definida no repositório.
