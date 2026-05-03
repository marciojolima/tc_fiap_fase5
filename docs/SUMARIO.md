# Sumário da Documentação

Este sumário centraliza a navegação dos documentos do projeto por tema, com
uma descrição curta do papel de cada arquivo.

## ADRs

- Contexto de domínio e problema base: [ADRs/ADR-001.md](ADRs/ADR-001.md)
- LLM via API e requisito de quantização: [ADRs/ADR-002.md](ADRs/ADR-002.md)
- Agente ReAct com RAG e tools de domínio: [ADRs/ADR-003.md](ADRs/ADR-003.md)
- Estratégia de avaliação em camadas: [ADRs/ADR-004.md](ADRs/ADR-004.md)
- RAGAS com quatro métricas calculadas e reportadas: [ADRs/ADR-005.md](ADRs/ADR-005.md)
- Model registry simplificado com MLflow e sidecars: [ADRs/ADR-006.md](ADRs/ADR-006.md)
- Telemetria LLM centrada em Prometheus e Grafana: [ADRs/ADR-007.md](ADRs/ADR-007.md)
- Métricas de negócio registradas explicitamente no treino: [ADRs/ADR-008.md](ADRs/ADR-008.md)
- CI com gates de qualidade, sem deploy para staging: [ADRs/ADR-009.md](ADRs/ADR-009.md)
- Fairness e explicabilidade mantidas em nível documental: [ADRs/ADR-010.md](ADRs/ADR-010.md)
- RAG com índice vetorial em memória e cache local: [ADRs/ADR-011.md](ADRs/ADR-011.md)
- Lacunas residuais de maturidade em capacidades já entregues: [ADRs/ADR-012.md](ADRs/ADR-012.md)

## Arquitetura e Operação

- Visão do agente conversacional e provider LLM: [AGENT_REACT.md](AGENT_REACT.md)
- Fluxos do projeto ponta a ponta: [FLOWS.md](FLOWS.md)
- Artefatos relevantes e onde encontrá-los: [ARTIFACTS.md](ARTIFACTS.md)
- Dashboard operacional e leitura das métricas: [OPERATIONS_DASHBOARD.md](OPERATIONS_DASHBOARD.md)
- Observabilidade e monitoramento da plataforma: [MONITORING_OBSERVABILITY.md](MONITORING_OBSERVABILITY.md)
- Monitoramento de drift e gatilhos de retreino: [DRIFT_MONITORING.md](DRIFT_MONITORING.md)
- Feature Store com Feast e Redis: [FEATURE_STORE.md](FEATURE_STORE.md)

## Modelagem e Avaliação

- Estratégia geral de avaliação: [EVALUATION.md](EVALUATION.md)
- Métricas técnicas do modelo tabular de churn: [EVALUATION_MODEL_METRICS.md](EVALUATION_MODEL_METRICS.md)
- Avaliação RAGAS e trilha LLM: [EVALUATION_RAGAS.md](EVALUATION_RAGAS.md)
- Análises de cenários do modelo: [SCENARIO_ANALYSIS.md](SCENARIO_ANALYSIS.md)
- Model card do champion atual: [MODEL_CARD.md](MODEL_CARD.md)
- Versionamento e governança do modelo: [MODEL_VERSIONING.md](MODEL_VERSIONING.md)

## Segurança, Governança e Apoio

- System card e visão sistêmica: [SYSTEM_CARD.md](SYSTEM_CARD.md)
- Plano LGPD e tratamento de dados: [LGPD_PLAN.md](LGPD_PLAN.md)
- Mapeamento OWASP para a solução: [OWASP_MAPPING.md](OWASP_MAPPING.md)
- Relatório de red teaming: [RED_TEAM_REPORT.md](RED_TEAM_REPORT.md)
- Explicação da camada RAG: [RAG_EXPLANATION.md](RAG_EXPLANATION.md)
- Gerador de predições sintéticas para simulações: [SYNTHETIC_PREDICTIONS_GENERATOR.md](SYNTHETIC_PREDICTIONS_GENERATOR.md)

[Voltar ao README](../README.md)
