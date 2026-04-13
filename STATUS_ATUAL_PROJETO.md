# Status Atual do Projeto

Última revisão: 2026-04-13

Este documento concentra o andamento do projeto frente aos arquivos [REQUISITOS_DATATHON.md](REQUISITOS_DATATHON.md) e [REQUISITOS_DATATHON_LIVE_EXPLANATION.md](REQUISITOS_DATATHON_LIVE_EXPLANATION.md). O `README.md` foi reposicionado como documentação tradicional do repositório; aqui ficam somente status, aderência, lacunas e prioridades.

## Leitura Executiva

O projeto está mais maduro na trilha de MLOps para modelo tabular de churn do que na trilha de LLMOps, agentes e segurança aplicada. Hoje o repositório já demonstra, de forma consistente:

- dados versionados com DVC
- pipeline de features com validação
- treinamento rastreável com MLflow
- serving com FastAPI
- análise de cenários de inferência
- monitoramento batch de drift com Evidently e PSI
- fluxo auditável de retreino com decisão champion-challenger
- base inicial de observabilidade com Prometheus e Grafana
- suíte relevante de testes automatizados

Os maiores gaps frente ao que a banca enfatizou na live continuam concentrados em:

- baseline adicional em PyTorch
- golden set formal
- agente ReAct funcional com tools reais
- pipeline RAG operacional
- avaliação RAGAS e LLM-as-judge executada de ponta a ponta
- guardrails e tratamento de PII efetivamente integrados
- fairness audit e explicabilidade formal
- esteira de deploy mais madura, com staging e gates mais fortes

## Status por Eixo de Requisito

### 1. Dados, baseline e pipeline de ML

Atendido ou bem encaminhado:

- estrutura Python organizada com `pyproject.toml`
- dados versionados com DVC
- camadas `raw`, `interim` e `processed`
- pipeline de engenharia de features implementado
- validação de schema com Pandera
- separação treino/teste antes do ajuste do pipeline
- artefatos processados persistidos para reuso
- experimentos configuráveis por YAML
- treinamento com MLflow e metadados de governança

Parcial:

- notebook exploratório existe em `notebooks/`, mas ainda não representa claramente a EDA no formato mais forte para apresentação

Pendente:

- baseline adicional em PyTorch
- golden set formal em estrutura dedicada
- feature store compartilhada ou materialização incremental entre múltiplos modelos

### 2. API, serving e cenários de negócio

Atendido ou bem encaminhado:

- API FastAPI implementada
- endpoint `/predict`
- schemas Pydantic de entrada e saída
- pipeline de features reutilizado entre treino e inferência
- análise de cenários com suíte versionada

Parcial:

- a arquitetura já suporta expansão, mas ainda não há endpoints específicos para agente ou RAG

Pendente:

- trilha de serving para LLM/agente
- integrações funcionais com tools de negócio

### 3. Monitoramento, retreino e observabilidade

Atendido ou bem encaminhado:

- logging de inferências para monitoramento
- drift batch com Evidently e PSI
- relatórios HTML e métricas em JSON
- lotes sintéticos para demonstração de drift
- gatilho auditável de retreino
- comparação champion-challenger
- dashboard e stack local de observabilidade configurados

Parcial:

- a base de observabilidade existe, mas ainda não representa uma operação completa com alertas externos e maior profundidade de acompanhamento
- o workflow de CI atual cobre lint, checagem sintática e testes, mas ainda não fecha uma esteira completa de staging/deploy

Pendente:

- alertas automáticos integrados a canal externo
- observabilidade específica para componentes com LLM
- pipeline de deploy com ambientes mais explícitos
- coverage gate formal no CI

### 4. LLMOps, agentes e avaliação de IA generativa

Existe estrutura inicial no repositório, mas ainda não pode ser tratada como entrega concluída frente aos requisitos da live.

Hoje há base de código em:

- `src/agent/react_agent.py`
- `src/agent/tools.py`
- `src/agent/rag_pipeline.py`
- `evaluation/ragas_eval.py`
- `evaluation/llm_judge.py`
- `evaluation/ab_test_prompts.py`

Status atual:

- estrutura criada
- testes e módulos iniciais presentes
- funcionalidade fim a fim ainda não consolidada como evidência de entrega

Pendente:

- agente ReAct funcional com pelo menos 3 tools reais
- pipeline RAG operacional
- integração com LLM de serving
- RAGAS com execução real
- LLM-as-judge com critérios operacionais
- telemetria de LLM

### 5. Segurança, governança e documentação

Atendido ou bem encaminhado:

- plano inicial de LGPD documentado
- model card e documentação de versionamento existentes
- registro de metadados relevantes no treinamento
- documentação de drift, dashboard e métricas disponível

Parcial:

- `docs/SYSTEM_CARD.md`, `docs/OWASP_MAPPING.md` e `docs/RED_TEAM_REPORT.md` existem, mas ainda precisam aprofundamento para a régua da banca
- módulos de `guardrails` e `pii_detection` existem, mas ainda não devem ser apresentados como implementação completa em produção

Pendente:

- system card completo
- OWASP mapping detalhado
- red team report com cenários adversariais
- sanitização e detecção de PII efetivas
- guardrails de entrada e saída realmente integrados
- fairness audit automatizada
- explicabilidade formal da predição

### 6. Engenharia de software e qualidade

Atendido ou bem encaminhado:

- organização modular do código
- type hints nas partes principais
- logging estruturado em fluxos centrais
- testes automatizados para features, modelos, serving, monitoramento e cenários
- lint com Ruff
- CI básico em `.github/workflows/ci.yml`

Parcial:

- `.pre-commit-config.yaml` existe, mas está praticamente vazio

Pendente:

- hooks de pre-commit úteis
- gate explícito de cobertura, por exemplo `--cov-fail-under`
- pipeline de deploy/staging

## Evidências Importantes do Estado Atual

- o fluxo de drift já gera `artifacts/monitoring/drift/drift_status.json`, `drift_metrics.json` e `drift_report.html`
- o fluxo de retreino já gera `retrain_request.json`, `retrain_run.json` e `promotion_decision.json`
- existe challenger registrado em `artifacts/models/challengers/`
- o modelo atual e seus metadados estão materializados em `artifacts/models/model_current.pkl` e `model_current_metadata.json`
- o `Makefile` ainda é placeholder e não representa a forma principal de operação do projeto

## Prioridades Recomendadas

Se a intenção for maximizar aderência aos requisitos com o menor risco, as próximas prioridades mais valiosas são:

1. consolidar a narrativa final da trilha tabular já funcional para demonstração
2. fechar um agente ReAct mínimo, com tools reais e evidência executável
3. transformar segurança aplicada em entrega concreta, com guardrails e PII testáveis
4. completar os documentos de governança mais cobrados pela banca
5. fortalecer o CI com gate de cobertura e caminho claro para staging

## Observação Final

A leitura mais honesta do projeto hoje é: a entrega principal de churn tabular com MLOps está defendível e bem mais madura; a trilha de LLMOps e agentes já tem direção arquitetural no repositório, mas ainda precisa de fechamento para ser apresentada como requisito plenamente cumprido.
