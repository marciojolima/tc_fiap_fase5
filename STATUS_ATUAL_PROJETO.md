# Status Atual do Projeto

Última revisão: 2026-04-19

O objetivo aqui é ser honesto sobre o que já está de pé, o que está parcial e
o que ainda falta para a banca.

## Leitura Executiva

O projeto está claramente mais maduro na trilha de MLOps para churn tabular do
que na trilha de LLMOps, agente e segurança aplicada. Hoje já existe uma base
defensável de:

- DVC para dados
- pipeline de features com validação
- treino rastreável com MLflow
- serving com FastAPI
- cenários de inferência
- drift com Evidently e PSI
- retreino auditável com champion-challenger
- feature store com Feast + Redis e materialização incremental
- stack local com Prometheus e Grafana
- suíte relevante de testes automatizados

Os maiores gaps frente ao que a live enfatizou continuam em:

- baseline adicional em PyTorch
- golden set formal
- agente ReAct com tools reais
- pipeline RAG operacional
- guardrails e PII efetivos
- fairness audit e explicabilidade formal
- CI/CD com deploy e gate formal de cobertura

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
- [x] Notebook de EDA incluído no repositório
- [ ] Golden set formal em `data/golden_set/`
- [x] Feature store com Feast introduzida no projeto
- [x] Redis configurado como online store local via Docker Compose
- [x] Camada offline da feature store derivada do pipeline atual, sem duplicar regras de features
- [x] Materialização incremental implementada no Feast
- [x] Demo de leitura online por `customer_id`
- [x] Serving integrado ao Feast para inferência online por `customer_id`
- [x] Contrato de features versionado por modelo com `FeatureService`

Observações:

- O notebook existe em `notebooks/churn_bancario_sem_mlops.ipynb`, mas ainda não
  é o elemento central da entrega.
- O professor deixou claro na live que o modelo não é o foco principal, mas a
  plataforma precisa estar bem estruturada.
- A trilha de feature store está funcional e bem aderente ao gap de
  `Feature Management` enfatizado na live: há separação offline/online, Redis
  em container, materialização incremental e serving consultando a online store.
- O projeto já superou o estágio de "feature store single-model" porque agora
  existe governança de contrato com `FeatureServices` por versão de modelo,
  ainda que todos reaproveitem a mesma `FeatureView` base nesta etapa.

### Etapa 2: API, LLM e agente

- [x] API FastAPI para serving
- [x] Schemas de entrada e saída para inferência
- [x] Análise de cenários com payloads versionados
- [ ] Agente ReAct funcional com pelo menos 3 tools
- [ ] Tools de negócio implementadas
- [ ] Pipeline RAG operacional
- [ ] Integração com LLM de serving
- [ ] Endpoints adicionais para agente ou RAG

Observações:

- A parte tabular de inferência está implementada e testada.
- `src/agent/` existe, mas ainda é placeholder e não deve ser apresentada como
  entrega funcional completa.

### Etapa 3: Avaliação e observabilidade

- [x] Estrutura de avaliação criada em `evaluation/`
- [x] Configuração de monitoramento dedicada em `configs/monitoring_config.yaml`
- [x] Módulos base de drift e métricas presentes em `src/monitoring/`
- [x] Dashboard operacional Prometheus/Grafana
- [ ] Drift detection operacional e automatizado
- [ ] RAGAS com 4 métricas efetivamente executadas
- [ ] LLM-as-judge com pelo menos 3 critérios efetivamente executados
- [ ] Alertas automáticos
- [ ] Observabilidade LLM com Langfuse ou TruLens

Observações:

- O drift está implementado e operacional em modo batch, com Evidently, PSI,
  `drift_report.html`, `drift_metrics.json`, `drift_status.json` e
  `drift_runs.jsonl`.
- O retreino já é disparado pelo fluxo de drift e gera `retrain_request.json`,
  `retrain_run.json` e `promotion_decision.json`.
- Ainda não existe agendamento/cron formal nem canal de alerta externo, então a
  automação operacional ainda não está completa no sentido mais forte da live.

### Etapa 4: Segurança e governança

- [x] Plano inicial de LGPD documentado
- [x] Minimização de identificadores diretos no pipeline de dados
- [x] Governança explícita para `Geography`
- [x] Model Card versionado
- [ ] System Card efetivamente preenchido
- [ ] Mapeamento OWASP documentado de forma substantiva
- [ ] Red Team Report documentado de forma substantiva
- [ ] Guardrails de input/output implementados de forma efetiva
- [ ] Detecção e sanitização de PII implementadas
- [ ] Fairness audit automatizada e anexada ao ciclo de treino
- [ ] Explicabilidade formal da predição

Observações:

- `docs/SYSTEM_CARD.md`, `docs/OWASP_MAPPING.md` e
  `docs/RED_TEAM_REPORT.md` existem, mas hoje estão essencialmente vazios e não
  sustentam banca como entrega concluída.
- Os módulos `src/security/guardrails.py` e `src/security/pii_detection.py`
  ainda não configuram segurança aplicada de ponta a ponta.

### Engenharia de software e qualidade

- [x] Type hints nas partes principais do projeto
- [x] Logging estruturado nas etapas centrais
- [x] Testes unitários de features, serving, modelos e cenários
- [x] Configuração de lint com Ruff
- [x] Organização por módulos de domínio
- [ ] Pipeline CI/CD com gates de lint, test e deploy
- [ ] Coverage gate formal
- [ ] Hooks automatizados de pre-commit

Observações:

- O workflow atual em `.github/workflows/ci.yml` já roda checkout, install,
  lint, compile, test e `pip check`.
- Ainda não há deploy/staging nem `--cov-fail-under`.
- `.pre-commit-config.yaml` existe, mas está praticamente vazia.

## O Que Já Está Forte para a Banca

Hoje a narrativa mais forte do projeto é:

- solução tabular de churn com pipeline de dados reproduzível
- treino rastreável com MLflow e metadados
- serving local consistente
- feature store local com Feast + Redis integrada ao serving
- contrato de features versionado por modelo
- cenários de inferência
- observabilidade operacional básica
- detecção de drift com software funcionando
- retreino auditável com comparação champion-challenger

Isso conversa muito bem com a fala do professor de que a avaliação está mais
interessada em engenharia de machine learning do que em “ter o melhor modelo”.

## O Que Ainda É Risco na Apresentação

Os pontos abaixo não devem ser “vendidos como prontos” sem ressalva:

- agente ReAct
- tools de negócio
- RAG
- LLM-as-judge
- RAGAS
- guardrails efetivos
- PII sanitization
- fairness automatizada
- System Card / OWASP / Red Team como governança madura
- feature divergence real entre versões de `FeatureService`
- ingestion jobs orquestrados por scheduler externo

## Avaliação da Feature Store

Considerando `REQUISITOS_DATATHON.md` e `REQUISITOS_DATATHON_LIVE_EXPLANATION.md`,
a parte de feature store pode ser considerada **cumprida de forma funcional e
defensável para a banca**.

O que já foi entregue:

- repositório Feast mínimo dentro do projeto
- offline source em parquet derivado do pipeline atual
- Redis como online store local via container
- materialização incremental sem padrão destrutivo
- leitura online por `customer_id`
- integração real do serving com Feast
- documentação técnica e operacional da solução
- versionamento do contrato de consumo por modelo com `FeatureService`

Lacunas remanescentes, mas não bloqueantes para considerar a etapa atendida:

- os `FeatureServices` já estão versionados por modelo, mas ainda reutilizam a
  mesma `FeatureView`, sem conjuntos de features realmente divergentes
- a ingestão continua em jobs locais/batch (`feature_engineering` ->
  `export_feature_store` -> `materialize`), sem orquestrador dedicado
- o timestamp usado para materialização é sintético, por limitação do dataset
  acadêmico
- não há ainda um fluxo de batch scoring/offline retrieval usando Feast como
  interface de consumo para treino; o treino segue lendo `data/processed/`

## Prioridades Recomendadas

Se a intenção for maximizar aderência aos requisitos com menor risco, a ordem
mais segura hoje parece ser:

1. consolidar a narrativa da trilha tabular já funcional
2. completar documentação de governança crítica
3. fortalecer segurança aplicada com evidência concreta
4. decidir se vale fechar um agente mínimo funcional ou tirar essa promessa da narrativa
5. amadurecer CI/CD e gates de qualidade

## Evidências-Chave do Repositório

- `artifacts/monitoring/drift/drift_report.html`
- `artifacts/monitoring/drift/drift_metrics.json`
- `artifacts/monitoring/drift/drift_status.json`
- `artifacts/monitoring/drift/drift_runs.jsonl`
- `artifacts/monitoring/retraining/retrain_request.json`
- `artifacts/monitoring/retraining/retrain_run.json`
- `artifacts/monitoring/retraining/promotion_decision.json`
- `artifacts/models/model_current.pkl`
- `artifacts/models/model_current_metadata.json`
- `artifacts/models/challengers/`

## Conclusão

O projeto já demonstra um ciclo relevante de engenharia de machine learning para
modelo tabular, com drift, retreino, feature store e governança operacional
básica. A parte mais frágil frente aos requisitos continua sendo a trilha de IA
generativa, segurança aplicada e governança documental profunda.
