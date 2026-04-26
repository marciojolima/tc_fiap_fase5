# System Card — Datathon Fase 05

## Visão Geral do Sistema

| Campo | Valor |
|---|---|
| **Nome** | Datathon Churn Analysis Platform |
| **Versão** | 0.1.0 |
| **Finalidade** | Predição de churn + análise assistida por agente LLM |
| **Autores** | Grupo FIAP Pós-Tech MLET |

## Componentes

### 1. Pipeline de Dados (Etapa 1)
- **Fonte**: Bank Customer Churn dataset (CSV local versionado)
- **Versionamento**: DVC
- **Features**: variáveis financeiras, cadastrais e derivadas para churn
- **Validação**: Pandera schema contracts

### 2. Modelo Tabular (Etapa 1)
- **Arquitetura**: modelos sklearn (champion/challenger)
- **Tracking**: MLflow com metadados padronizados
- **Métricas**: AUC, F1, Precision, Recall, Accuracy
- Ver: [MODEL_CARD.md](MODEL_CARD.md)

### 3. LLM (Etapa 2)
- **Modelo**: configurável por provider (`ollama`, `openai`, `claude`)
- **Provider local padrão na stack**: Ollama
- **Serving**: FastAPI endpoint `/llm/chat`
- **Configuração**: `configs/pipeline_global_config.yaml` + `.env`

### 4. Agente ReAct (Etapa 2)
- **Framework**: implementação própria em `src/agent/react_agent.py`
- **Tools de domínio**: `rag_search`, `predict_churn`, `drift_status`, `scenario_prediction`
- **RAG**: índice vetorial local com FastEmbed + cache em `artifacts/rag/`

### 5. Avaliação (Etapa 3)
- **RAGAS**: 4 métricas (faithfulness, answer_relevancy, context_precision, context_recall)
- **LLM-as-judge**: 3 critérios (incluindo adequação ao negócio)
- **Golden Set**: pares em `configs/evaluation/golden_set.yaml`

### 6. Observabilidade (Etapa 3)
- **Métricas**: Prometheus (latência, throughput, erros, métricas do RAG)
- **Dashboard**: Grafana
- **Drift**: Evidently + PSI em trilha tabular
- **Telemetria LLM/RAG**: status e métricas expostas pela API

### 7. Segurança (Etapa 4)
- **Guardrails**: Input (prompt injection/tamanho) + Output (PII)
- **PII**: regex masking para CPF, telefone BR e email
- **OWASP**: 5+ ameaças mapeadas — ver [OWASP_MAPPING.md](OWASP_MAPPING.md)
- **Red Team**: 5+ cenários — ver [RED_TEAM_REPORT.md](RED_TEAM_REPORT.md)

## Fluxo de Dados

```text
Dataset CSV -> DVC -> Feature Engineering -> Treino Tabular (MLflow)
                                              |
User Query -> InputGuardrail -> Agente (ReAct + Provider LLM)
                                  |                    |
                                  |               Tools de domínio
                                  |
                               RAG local
                                  |
                         OutputGuardrail -> Response
                                  |
                    Prometheus/Grafana + Drift (Evidently)
```

## Requisitos Não-Funcionais

| Requisito | Alvo |
|---|---|
| Latência de inferência tabular | baixa latência operacional |
| Latência de geração LLM | adequada para suporte interativo |
| Disponibilidade | ambiente local com stack docker |
| Qualidade de código | lint + testes automatizados |
| Segurança mínima LLM | guardrails + OWASP mapping + red team básico |

## Conformidade

- **LGPD**: plano em [LGPD_PLAN.md](LGPD_PLAN.md)
- **OWASP LLM**: mapeamento em [OWASP_MAPPING.md](OWASP_MAPPING.md)
- **Fairness**: documentação no [MODEL_CARD.md](MODEL_CARD.md)
- **Explicabilidade**: interpretação global por importância de features; explicabilidade local permanece evolução futura
