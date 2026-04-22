# Status Check List

Última revisão: 2026-04-19

Este arquivo funciona como uma checagem rápida de entrega. A regra aqui é
simples:

- `[x]` somente para o que já está efetivamente cumprido no repositório
- `[ ]` para o que ainda está pendente ou apenas parcial

O [STATUS_ATUAL_PROJETO.md](STATUS_ATUAL_PROJETO.md) continua sendo o documento
mais analítico. Este checklist existe como uma visão executiva e objetiva.

## Checklist de Entrega Final

Use este checklist como guia antes do Demo Day.

### Etapa 1 — Dados + Baseline

- [x] EDA documentada com insights relevantes para o problema da empresa
- [x] Baseline treinado e métricas reportadas no MLflow
- [x] Pipeline versionado (DVC + Docker) e reprodutível
- [x] Métricas de negócio mapeadas para métricas técnicas
- [x] `pyproject.toml` com todas as dependências
- [x] Feature Store com Feast integrada ao projeto
- [x] Redis como online store local
- [x] Materialização incremental de features
- [x] Serving consultando a online store por `customer_id`
- [x] `FeatureService` versionado por modelo

Observações:

- A EDA existe no notebook do projeto, mas ainda não é o eixo central da entrega.
- O pipeline está reproduzível localmente com DVC, Docker Compose e tasks do projeto.
- A etapa de feature store já está atendida de forma funcional para a banca.
- A principal ressalva é que os `FeatureServices` ainda compartilham a mesma
  `FeatureView`, o que limita a divergência real entre versões nesta fase.

### Etapa 2 — LLM + Agente

- [ ] LLM servido via API com quantização aplicada
- [ ] Agente ReAct funcional com ≥ 3 tools relevantes ao domínio
- [ ] RAG retornando contexto relevante dos dados fornecidos
- [x] CI/CD pipeline funcional (GitHub Actions)
- [ ] Benchmark documentado com ≥ 3 configurações

Observações:

- O CI atual já roda install, lint, compile, test e `pip check`, mas ainda não
  tem deploy nem gates mais maduros.
- A trilha de LLM, agente e RAG ainda está em placeholder e não deve ser
  contada como entregue.

### Etapa 3 — Avaliação + Observabilidade

- [ ] Golden set com ≥ 20 pares relevantes ao domínio
- [ ] RAGAS: 4 métricas calculadas e reportadas
- [ ] LLM-as-judge com ≥ 3 critérios (incluindo critério de negócio)
- [x] Telemetria e dashboard funcionando end-to-end
- [x] Detecção de drift implementada e documentada
- [x] Gatilho auditável de retreino implementado
- [x] Comparação champion vs challenger com decisão persistida

Observações:

- A stack local com FastAPI, Prometheus, Grafana e MLflow já funciona de ponta a ponta.
- O drift está implementado com Evidently, PSI, histórico de execuções, bloqueio
  por amostra insuficiente, retreino auditável e comparação champion-challenger.
- O modo configurado hoje é `auto_train_manual_promote`: drift crítico elegível
  abre solicitação, executa retreino local e persiste a decisão de promoção.
- A parte de avaliação para LLM ainda não está operacional.

### Etapa 4 — Segurança + Governança

- [ ] OWASP mapping com ≥ 5 ameaças e mitigações
- [ ] Guardrails de input e output funcionais
- [ ] ≥ 5 cenários adversariais testados e documentados
- [ ] Plano LGPD aplicado ao caso real
- [ ] Explicabilidade e fairness documentados
- [ ] System Card completo

Observações:

- Existe documentação inicial de LGPD, Model Card e governança, mas a trilha de
  segurança aplicada ainda não está suficientemente madura para marcar como concluída.
- `fairness_checked` continua `false` no metadata atual do champion.
- `System Card`, `OWASP Mapping` e `Red Team Report` ainda precisam ser
  aprofundados para sustentar banca como entrega efetiva.

### Demo Day

- [ ] Pitch ≤ 10 min: Problema → Abordagem → Demo → Resultados → Impacto
- [ ] Ensaio prévio com timer
- [ ] Backup: slides offline caso a demo falhe
- [ ] Preparação para Q&A técnico e de negócio

Observações:

- Estes itens dependem mais da preparação final da apresentação do que do código do repositório.

## Resumo Rápido

Hoje o projeto está mais forte em:

- pipeline tabular de churn
- MLflow e rastreabilidade
- serving com FastAPI
- feature store com Feast + Redis
- contrato de features versionado por modelo
- observabilidade local
- drift + retreino auditável
- decisão de promoção champion-challenger

Os maiores gaps ainda estão em:

- trilha de LLM/agente/RAG
- golden set e avaliação LLM
- segurança aplicada
- governança documental mais profunda
- orquestração formal dos ingestion jobs
- preparação final de Demo Day
