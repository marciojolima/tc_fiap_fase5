# Status Atual do Projeto

Última revisão: 2026-04-24

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
- retreino auditável com gatilho automático local e champion-challenger
- feature store com Feast + Redis e materialização incremental
- stack local com Prometheus e Grafana
- suíte relevante de testes automatizados

Os maiores gaps frente ao que a live enfatizou continuam em:

- baseline adicional em PyTorch
- guardrails, PII e red team efetivos
- fairness audit e explicabilidade formal
- CI/CD com deploy e gate formal de cobertura
- execução e reporte formal de RAGAS, LLM-as-judge e benchmark RAG/LLM

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
- [x] Golden set formal em `data/golden-set.json`
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
- [x] Agente ReAct funcional com pelo menos 3 tools
- [x] Tools de negócio implementadas
- [x] Pipeline RAG operacional
- [x] Integração com LLM de serving
- [x] Endpoints adicionais para agente ou RAG
- [x] LLM servido via API com provider gerenciado Claude
- [ ] Quantização aplicada em provider local opcional

Observações:

- A parte tabular de inferência está implementada e testada.
- A trilha `src/agent/` agora ja nao e mais placeholder: o agente usa ReAct com
  quatro tools de dominio e um RAG operacional com embeddings em memoria.
- O corpus e descoberto automaticamente a partir de `README.md`,
  `docs/**/*.md` e JSON hardcoded relevantes; novos `.md` entram no indice no
  proximo startup da stack.
- O RAG possui cache persistido em `artifacts/rag/cache/` com manifesto de
  fontes e historico em `artifacts/rag/index_build_history.jsonl`.
- A decisão arquitetural atual é consumir Claude como serviço gerenciado. Nesse
  caminho, o projeto integra o LLM via API, mas não controla nem comprova
  quantização interna do modelo. O caminho local com Ollama existe como opção,
  e é nele que uma evidência de quantização poderia ser documentada, se a equipe
  decidir demonstrar esse requisito também.

### Etapa 3: Avaliação e observabilidade

- [x] Estrutura de avaliação criada em `evaluation/`
- [x] Configuração de monitoramento dedicada em `configs/monitoring/global_monitoring.yaml`
- [x] Módulos base de drift e métricas presentes em `src/monitoring/`
- [x] Dashboard operacional Prometheus/Grafana
- [x] Drift detection operacional e automatizado em fluxo local batch
- [x] Gatilho auditável de retreino para drift crítico
- [x] Comparação champion-challenger com decisão persistida de promoção
- [ ] RAGAS com 4 métricas efetivamente executadas
- [ ] LLM-as-judge com pelo menos 3 critérios efetivamente executados
- [ ] Benchmark RAG/LLM com 3 configurações consolidado em artefato
- [ ] Alertas automáticos
- [ ] Observabilidade LLM com Langfuse ou TruLens

Observações:

- O drift está implementado e operacional em modo batch, com Evidently, PSI,
  `drift_report.html`, `drift_metrics.json`, `drift_status.json` e
  `drift_runs.jsonl`.
- O retreino já é disparado pelo fluxo de drift no modo
  `auto_train_manual_promote` e gera `retrain_request.json`,
  `retrain_run.json` e `promotion_decision.json`.
- A decisão de promoção já compara champion e challenger por regra explícita de
  métrica primária e melhoria mínima, mas a promoção final continua manual.
- Ainda não existe agendamento/cron formal nem canal de alerta externo, então a
  automação operacional ainda não está completa no sentido mais forte da live.
- A observabilidade da trilha de LLM foi reforcada com metricas Prometheus e um
  dashboard dedicado ao RAG, cobrindo corpus, chunks, bytes, memoria estimada,
  delta de RSS, tempo por etapa de startup, cache hit e latencia da busca.
- `evaluation/ragas_eval.py`, `evaluation/llm_judge.py` e
  `evaluation/ab_test_prompts.py` existem e têm testes de suporte. As saídas
  foram padronizadas para `artifacts/evaluation/llm_agent/results/`, com histórico em
  `artifacts/evaluation/llm_agent/runs/`, mas ainda precisam ser executadas e reportadas
  formalmente.

### Etapa 4: Segurança e governança

- [x] Plano inicial de LGPD documentado
- [x] Minimização de identificadores diretos no pipeline de dados
- [x] Governança explícita para `Geography`
- [x] Model Card versionado
- [ ] System Card efetivamente preenchido
- [ ] Mapeamento OWASP documentado de forma substantiva
- [ ] Red Team Report documentado de forma substantiva
- [ ] Guardrails de input/output robustos e evidenciados por cenários adversariais
- [ ] Detecção e sanitização de PII aplicadas de ponta a ponta
- [ ] Fairness audit automatizada e anexada ao ciclo de treino
- [ ] Explicabilidade formal da predição

Observações:

- `docs/SYSTEM_CARD.md`, `docs/OWASP_MAPPING.md` e
  `docs/RED_TEAM_REPORT.md` existem, mas hoje estão essencialmente vazios e não
  sustentam banca como entrega concluída.
- Os módulos `src/security/guardrails.py` e `src/security/pii_detection.py`
  já implementam uma camada básica: bloqueio de alguns padrões de prompt
  injection, limite de tamanho de input e mascaramento simples de e-mail,
  telefone e CPF. Isso é útil como base, mas ainda não configura segurança
  aplicada de ponta a ponta nem substitui OWASP mapping, red team e relatório de
  mitigação.

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
- A verificação local mais recente executou `poetry run ruff check` com sucesso e
  `poetry run pytest -q` com 114 testes aprovados.
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
- gatilho local de retreino para drift crítico
- retreino auditável com comparação champion-challenger e decisão persistida

Isso conversa muito bem com a fala do professor de que a avaliação está mais
interessada em engenharia de machine learning do que em “ter o melhor modelo”.

## O Que Ainda É Risco na Apresentação

Os pontos abaixo não devem ser “vendidos como prontos” sem ressalva:

- LLM-as-judge
- RAGAS
- benchmark RAG/LLM com 3 configurações
- quantização aplicada em provider local opcional
- guardrails efetivos e red team
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
2. executar e reportar RAGAS, LLM-as-judge e benchmark de prompts/RAG
3. completar documentação de governança crítica
4. fortalecer segurança aplicada com evidência concreta
5. amadurecer CI/CD e gates de qualidade

## Evidências-Chave do Repositório

- `artifacts/evaluation/model/drift/drift_report.html`
- `artifacts/evaluation/model/drift/drift_metrics.json`
- `artifacts/evaluation/model/drift/drift_status.json`
- `artifacts/evaluation/model/drift/drift_runs.jsonl`
- `artifacts/evaluation/model/retraining/retrain_request.json`
- `artifacts/evaluation/model/retraining/retrain_run.json`
- `artifacts/evaluation/model/retraining/promotion_decision.json`
- `artifacts/models/current.pkl`
- `artifacts/models/current_metadata.json`
- `artifacts/models/challengers/`
- `artifacts/rag/cache/manifest.json`
- `artifacts/rag/index_build_history.jsonl`
- `artifacts/evaluation/llm_agent/results/`
- `artifacts/evaluation/llm_agent/runs/`
- `data/golden-set.json`
- `evaluation/ragas_eval.py`
- `evaluation/llm_judge.py`
- `evaluation/ab_test_prompts.py`

## Conclusão

O projeto já demonstra um ciclo relevante de engenharia de machine learning para
modelo tabular, com drift, gatilho de retreino, challenger, feature store e
governança operacional básica. A trilha de IA generativa agora possui agente,
RAG, rotas e scripts de avaliação, mas ainda precisa transformar esses scripts
em resultados reportados. A parte mais frágil frente aos requisitos continua
sendo segurança aplicada, red team, governança documental profunda, fairness e
explicabilidade.
