# TC FIAP Fase 5

Projeto integrador da Fase 05 do curso MLET da FIAP, estruturado como uma entrega de Datathon com foco em churn bancário, MLOps, LLMOps, governança, segurança e rastreabilidade.

Este README foi organizado para funcionar ao mesmo tempo como:
- visão executiva do projeto
- guia rápido de execução local
- checklist de entrega da Datathon
- mapa de pendências e próximos passos

## Grupo

### 👥 Autores
Turma 6MLET - FIAP

- Luca Poiti - RM365678
- Gabriel Jordan - RM365606
- Luciana Ferreira - RM366171
- Marcio Lima - RM365919

## Objetivo

Construir uma solução de predição de churn com base em dados bancários, com pipeline de dados reproduzível, treino rastreável em MLflow, API de inferência, análise de cenários, documentação de governança e base preparada para evolução de agente, RAG, observabilidade e guardrails de produção.

## Estado Atual do Projeto

Hoje o repositório já entrega:
- pipeline de engenharia de features com validação por estágio, minimização LGPD, split treino/teste e persistência de artefatos
- pipeline reutilizável de transformação de features compartilhado entre treino e inferência
- treino de modelos com MLflow, metadados padronizados e versionamento de dados processados
- API FastAPI para serving de churn
- análise de cenários com rastreabilidade em MLflow
- documentação inicial de Model Card, System Card, LGPD, OWASP, Red Team e versionamento
- suíte de testes unitários para partes centrais do fluxo

Ainda existem componentes em estado parcial ou placeholder:
- agente ReAct
- tools do agente
- RAG pipeline
- guardrails de input/output
- detecção de PII
- monitoramento operacional e de drift em produção
- CI/CD de staging

## Arquitetura Resumida

Fluxo principal atualmente implementado:

1. Leitura do dataset bruto
2. Validação de schema bruto
3. Minimização LGPD com remoção de identificadores diretos
4. Limpeza da camada interim
5. Validação de schema interim
6. Engenharia de features com pipeline declarativo reutilizável
7. Persistência de `train.parquet`, `test.parquet` e `feature_pipeline.joblib`
8. Treino de modelos com tracking em MLflow
9. Inferência via FastAPI usando o mesmo pipeline de features do treino
10. Análise de cenários com logging dedicado em MLflow

## Estrutura do Repositório

```text
tc_fiap_fase5/
├── configs/
├── data/
├── docs/
├── evaluation/
├── examples/
├── src/
│   ├── agent/
│   ├── common/
│   ├── features/
│   ├── inference/
│   ├── models/
│   ├── monitoring/
│   ├── security/
│   └── serving/
├── tests/
├── docker-compose.yml
├── dvc.lock
├── dvc.yaml
├── Makefile
├── pyproject.toml
└── REQUISITOS_DATATHON.md
```

## Principais Módulos

- [src/features/feature_engineering.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/features/feature_engineering.py): orquestra a engenharia de features, validação, split e persistência
- [src/features/pipeline_components.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/features/pipeline_components.py): transformers customizados e pipeline de transformação reutilizável
- [src/features/schema_validation.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/features/schema_validation.py): schemas raw e interim com `pandera.DataFrameModel`
- [src/models/train.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/models/train.py): treino, métricas e logging em MLflow
- [src/serving/app.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/serving/app.py): aplicação FastAPI
- [src/serving/pipeline.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/serving/pipeline.py): preparação de inferência e aplicação do pipeline de features
- [src/inference/scenario_analysis.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/inference/scenario_analysis.py): execução de cenários hipotéticos
- [src/security/guardrails.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/security/guardrails.py): ponto de evolução para guardrails
- [src/security/pii_detection.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/security/pii_detection.py): ponto de evolução para detecção de PII

## Stack Atual

- Python 3.13
- Pandas
- Scikit-learn
- Pandera
- MLflow
- FastAPI
- DVC
- Pytest
- Ruff
- Taskipy

## Como Executar

### Instalação

Exemplo com Poetry:

```bash
poetry install
```

### Engenharia de Features

```bash
poetry run task mlfeateng
```

### Treino do Modelo Atual

```bash
poetry run task mltrain
```

### Pipeline Completo de Treino e Cenários

```bash
poetry run task mlrunall
```

### API de Serving

```bash
poetry run task serving
```

### MLflow UI

```bash
poetry run task mlflow
```

### Testes

```bash
poetry run task test
```

## Artefatos Gerados

Atualmente o projeto persiste os seguintes artefatos por motivos de MLOps e governança:

- `data/interim/cleaned.parquet`
  Mantém uma versão intermediária já limpa, sem duplicidades, sem nulos e após minimização de identificadores diretos. Isso ajuda na rastreabilidade da preparação dos dados, facilita auditoria do pipeline e permite inspecionar a camada anterior à modelagem sem depender de reprocessar tudo.

- `data/processed/train.parquet`
  Registra o conjunto final de treino já transformado e pronto para modelagem. Esse artefato garante reprodutibilidade experimental, desacopla a etapa de engenharia de features da etapa de treino e permite comparar modelos diferentes sobre exatamente a mesma base processada.

- `data/processed/test.parquet`
  Registra o conjunto final de teste usado na avaliação. Isso é importante para consistência de benchmark, reprodutibilidade de métricas e auditoria posterior sobre quais dados sustentaram a validação do modelo.

- `data/processed/feature_columns.json`
  Documenta a ordem e o nome final das features geradas pelo pipeline. Esse controle evita ambiguidades entre treino e inferência, ajuda debugging e dá transparência sobre o espaço final de entrada do modelo.

- `data/processed/schema_report.json`
  Guarda evidência de que o dado passou pelas validações esperadas no pipeline. Mesmo sendo simples hoje, esse artefato funciona como ponto inicial de compliance operacional e pode evoluir para relatórios mais detalhados de qualidade de dados.

- `artifacts/feature_pipeline.joblib`
  Persiste o pipeline completo de transformação de features para reutilização em produção. Isso reduz `training-serving skew`, garante que treino e inferência usem exatamente a mesma lógica de transformação e melhora a sustentabilidade da arquitetura.

- artefatos de modelo definidos nos YAMLs de treino
  Permitem versionar e promover modelos treinados por experimento. Isso é essencial para governança, comparação entre versões, rollback controlado e rastreabilidade do que foi efetivamente treinado.

- runs e métricas no MLflow
  Registram parâmetros, métricas, tags e contexto de cada experimento. Isso sustenta boas práticas de experiment tracking, lineage, auditoria técnica e documentação da evolução do projeto ao longo do tempo.

## Documentação Disponível

- [docs/MODEL_CARD.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/MODEL_CARD.md)
- [docs/SYSTEM_CARD.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/SYSTEM_CARD.md)
- [docs/LGPD_PLAN.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/LGPD_PLAN.md)
- [docs/OWASP_MAPPING.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/OWASP_MAPPING.md)
- [docs/RED_TEAM_REPORT.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/RED_TEAM_REPORT.md)
- [docs/MODEL_VERSIONING.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/MODEL_VERSIONING.md)
- [docs/SCENARIO_ANALYSIS.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/SCENARIO_ANALYSIS.md)
- [docs/EVALUATION_METRICS.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/EVALUATION_METRICS.md)

## Checklist da Datathon

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
- [x] Módulos base de drift e métricas presentes em `src/monitoring/`
- [ ] Dashboard operacional Prometheus/Grafana
- [ ] Drift detection operacional e automatizado
- [ ] RAGAS com 4 métricas efetivamente executadas
- [ ] LLM-as-judge com pelo menos 3 critérios efetivamente executados
- [ ] Alertas automáticos
- [ ] Observabilidade LLM com Langfuse ou TruLens

### Etapa 4: Segurança e governança

- [x] Plano inicial de LGPD documentado
- [x] Minimização de identificadores diretos no pipeline de dados
- [x] Governança explícita para `Geography`
- [x] Model Card e System Card versionados
- [x] Mapeamento OWASP documentado
- [x] Red Team Report documentado
- [ ] Guardrails de input/output implementados de forma efetiva
- [ ] Detecção e sanitização de PII implementadas
- [ ] Fairness audit automatizada e anexada ao ciclo de treino
- [ ] Explicabilidade formal da predição

### Engenharia de software e qualidade

- [x] Type hints nas partes principais do projeto
- [x] Logging estruturado nas etapas centrais
- [x] Testes unitários de features, serving, modelos e cenários
- [x] Configuração de lint com Ruff
- [x] Organização por módulos de domínio
- [ ] Pipeline CI/CD com gates de lint, test e deploy
- [ ] Coverage gate formal
- [ ] Hooks automatizados de pre-commit

## O Que Já Foi Além do Mínimo

Além do mínimo da Datathon, este repositório já incorporou decisões arquiteturais importantes:
- pipeline de features reutilizado em treino e inferência, reduzindo `training-serving skew`
- schemas distintos por estágio de dado
- centralização de categorias de negócio em configuração
- governança inicial de LGPD incorporada no próprio fluxo de dados
- análise de cenários com experimento dedicado em MLflow
- documentação de governança distribuída em arquivos específicos, não concentrada em um único relatório

## Pendências e Placeholders

Esta seção deve continuar sendo usada como checklist vivo do grupo.

### Placeholders técnicos atuais

- `src/agent/react_agent.py`
- `src/agent/tools.py`
- `src/agent/rag_pipeline.py`
- `src/security/guardrails.py`
- `src/security/pii_detection.py`
- `src/monitoring/drift.py`
- `src/monitoring/metrics.py`
- `evaluation/ragas_eval.py`
- `evaluation/llm_judge.py`
- `evaluation/ab_test_prompts.py`

### Pendências recomendadas de curto prazo

- [ ] implementar guardrails reais de entrada e saída
- [ ] implementar detecção de PII com sanitização de logs
- [ ] completar o fluxo de monitoramento e drift
- [ ] registrar evidência prática de fairness
- [ ] fechar teste de rota FastAPI isolado no ambiente local
- [ ] revisar `Makefile` para refletir comandos reais do projeto
- [ ] adicionar CI/CD
- [ ] adicionar instruções de setup com `.env`, Docker e dados de exemplo

### Pendências recomendadas de apresentação

- [ ] inserir link do vídeo ou demo final
- [ ] inserir arquitetura final em diagrama consolidado
- [ ] inserir evidências visuais de MLflow
- [ ] inserir evidências visuais da API
- [ ] inserir evidências visuais dos relatórios de segurança e governança

## Observações Importantes

- Este README foi escrito para acompanhar o estado real do repositório, então o checklist deve ser atualizado sempre que uma etapa evoluir.
- Componentes documentados como placeholder não devem ser apresentados como concluídos.

## Referência de Execução Recomendada

Fluxo sugerido para demonstração local:

1. `poetry install`
2. `poetry run task mlfeateng`
3. `poetry run task mltrain`
4. `poetry run task mlscenario`
5. `poetry run task serving`
6. `poetry run task mlflow`

## Licença e Uso

`[PLACEHOLDER]` Definir licença do projeto antes da entrega final.
