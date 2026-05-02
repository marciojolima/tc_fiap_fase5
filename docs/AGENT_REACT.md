# LLM, agente ReAct e llm_provider

## Visão geral

Esta trilha concentra a experiência conversacional do projeto. A API expõe um
fluxo de perguntas e respostas com agente ReAct, RAG local, tools de negócio e
guardrails, sem alterar o contrato do endpoint tabular `/predict`.

As decisões arquiteturais dessa trilha estão consolidadas em
[ADRs/ADR-002.md](ADRs/ADR-002.md) e [ADRs/ADR-003.md](ADRs/ADR-003.md). Este
documento fica concentrado no funcionamento operacional da camada
conversacional.

## Componentes principais

- **API LLM:** [src/serving/llm_routes.py](../src/serving/llm_routes.py) expõe `GET /llm/health`, `GET /llm/status` e `POST /llm/chat`.
- **Agente ReAct:** [src/agent/react_agent.py](../src/agent/react_agent.py) executa o ciclo pensar, agir e observar com limite de iterações.
- **Tools:** [src/agent/tools.py](../src/agent/tools.py) inclui `rag_search`, `predict_churn`, `drift_status` e `scenario_prediction`.
- **RAG:** [src/agent/rag_pipeline.py](../src/agent/rag_pipeline.py) combina busca vetorial local e reranqueamento lexical leve sobre documentação e artefatos do projeto.
- **Segurança:** [src/security/guardrails.py](../src/security/guardrails.py) e [src/security/pii_detection.py](../src/security/pii_detection.py) validam entrada e reduzem exposição de PII.

## Configuração

As definições ficam em [configs/pipeline_global_config.yaml](../configs/pipeline_global_config.yaml), nos blocos `llm`, `agent`, `rag` e `security`.

- `llm.active_provider` aceita `ollama`, `openai` ou `claude`
- chaves externas ficam no `.env`, com `OPENAI_API_KEY` e `ANTHROPIC_API_KEY`
- em Docker, `LLM_BASE_URL` pode sobrescrever a `base_url` do provider ativo

## Operação local

- `poetry run task appstack` sobe a stack base
- `poetry run task appstack_ollama` adiciona `ollama` e `ollama-pull` quando `llm.active_provider=ollama`
- `poetry run task rag_index_rebuild_docker` recria o índice vetorial antes de subir a API, quando necessário
- `http://127.0.0.1:8000/llm/status` ajuda a validar provider, modelo esperado e status do RAG

No modo de desenvolvimento, alterações em `src/` são recarregadas pelo serving.
Rebuild costuma ser necessário apenas após mudanças em Dockerfile,
`pyproject.toml`, `poetry.lock` ou dependências.

## Relação com a avaliação

O documento [EVALUATION_RAGAS.md](EVALUATION_RAGAS.md) cobre a validação dessa
trilha em execução real. As rotinas de RAGAS, LLM-as-judge e benchmark de
prompts chamam o endpoint `POST /llm/chat`, passando pelo mesmo agente, pelas
mesmas tools e pelo mesmo provider configurado para a aplicação.
