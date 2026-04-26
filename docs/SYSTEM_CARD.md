# System Card

## 1. Objetivo do Sistema

O sistema apoia duas trilhas do projeto de churn bancário:

- predição tabular de churn (`/predict` e `/predict/raw`);
- assistente LLM com agente ReAct e RAG (`/llm/chat`).

O foco é suporte à decisão e operação técnica, não automação de decisão
irreversível sem supervisão humana.

## 2. Escopo Funcional

### Componentes principais

- API FastAPI de serving (`src/serving/`).
- Pipeline tabular de inferência com modelo champion.
- Agente ReAct com tools de domínio.
- RAG com indexação de documentação e artefatos do projeto.
- Observabilidade com métricas Prometheus e dashboards.

### Principais endpoints

- `GET /health`
- `POST /predict`
- `POST /predict/raw`
- `GET /llm/health`
- `GET /llm/status`
- `POST /llm/chat`

## 3. Entradas e Saídas

### Entradas

- Payload de cliente para inferência de churn.
- Prompt em linguagem natural para o endpoint LLM.

### Saídas

- Probabilidade e classe de churn.
- Resposta textual do agente LLM, com sanitização de PII no output.
- Metadados operacionais e traço opcional do agente (`include_trace`).

## 4. Dependências Externas

- Ollama/OpenAI/Claude como providers LLM (configuração via
  `configs/pipeline_global_config.yaml` e `.env`).
- Redis para online store da feature store.
- MLflow para rastreabilidade de treino/experimentação.
- Prometheus e Grafana para observabilidade.

## 5. Segurança, Riscos e Mitigações

### Controles implementados

- Guardrail de input com padrões de prompt injection e limite de tamanho.
- Guardrail de output com redaction de PII (email, telefone, CPF).
- Mapeamento OWASP e cenários adversariais documentados.

### Documentos relacionados

- `docs/OWASP_MAPPING.md`
- `docs/RED_TEAM_REPORT.md`
- `docs/LGPD_PLAN.md`

## 6. Limitações Conhecidas

- Detecção de prompt injection baseada em regex (heurística simples).
- Sem classificador semântico avançado de jailbreak/toxicidade.
- Fairness documentado, mas sem gate automatizado obrigatório no pipeline.
- Explicabilidade local por predição ainda não implementada.

## 7. Uso Responsável

- O score de churn deve ser usado como apoio à priorização de retenção.
- O resultado do agente LLM deve ser tratado como assistivo, com validação
  humana quando envolver decisões críticas.

## 8. Operação Mínima (Checklist)

- API online: `/health` e `/llm/health` respondendo `200`.
- Provider LLM configurado corretamente em `.env` e YAML.
- Drift e métricas operacionais disponíveis nos dashboards.
- Testes de guardrail passando:
  `tests/test_guardrails.py` e `tests/test_guardrails_adversarial.py`.
