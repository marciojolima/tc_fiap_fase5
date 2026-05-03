# Avaliação

## Indice

- [Objetivo](#objetivo)
- [Mapa de documentos](#mapa-de-documentos)
- [Taxonomia de avaliacao](#taxonomia-de-avaliacao)
- [Artefatos principais](#artefatos-principais)
- [Como reproduzir](#como-reproduzir)
- [Resumo executivo](#resumo-executivo)

## Objetivo

Este documento e o ponto de entrada da avaliacao do projeto. Ele organiza as
avaliacoes tabulares, operacionais e generativas sem misturar conceitos:

- modelo supervisionado de churn
- cenarios de negocio
- drift, retreino e champion-challenger
- RAG, RAGAS, LLM-as-judge e benchmark de prompts

## Mapa de documentos

| Tema | Documento |
| ---- | --------- |
| Metricas do modelo de churn | [EVALUATION_MODEL_METRICS.md](EVALUATION_MODEL_METRICS.md) |
| RAGAS, golden set e LLM-as-judge | [EVALUATION_RAGAS.md](EVALUATION_RAGAS.md) |
| Monitoramento de drift | [DRIFT_MONITORING.md](DRIFT_MONITORING.md) |
| Fluxos operacionais | [FLOWS.md](FLOWS.md) |
| RAG e agente LLM | [RAG_EXPLANATION.md](RAG_EXPLANATION.md) |

## Taxonomia de avaliacao

### 1. Modelo tabular

A avaliacao supervisionada acontece durante o treino em:

- [src/model_lifecycle/train.py](../src/model_lifecycle/train.py)

Metricas calculadas:

- `auc`
- `f1`
- `precision`
- `recall`
- `accuracy`
- `churn_recall_top`
- `churn_precision_top`

Essas metricas comparam execucoes, sustentam o champion e entram no fluxo de
retreino. A prioridade de negocio para churn esta detalhada em
[EVALUATION_MODEL_METRICS.md](EVALUATION_MODEL_METRICS.md).

### 2. Cenarios de negocio

A avaliacao por cenarios verifica se a inferencia se comporta de forma coerente
em casos controlados:

- [src/scenario_experiments/inference_cases.py](../src/scenario_experiments/inference_cases.py)
- [configs/scenario_experiments/inference_cases.yaml](../configs/scenario_experiments/inference_cases.yaml)

Ela ajuda a responder se perfis de alto risco, thresholds e simulacoes de
mudanca fazem sentido para o dominio bancario.

### 3. Drift e operacao

A avaliacao operacional verifica se os dados correntes continuam parecidos com o
dominio usado no treino:

- [src/evaluation/model/drift/drift.py](../src/evaluation/model/drift/drift.py)
- [src/evaluation/model/drift/prediction_logger.py](../src/evaluation/model/drift/prediction_logger.py)

Sinais principais:

- `data drift`
- `prediction drift`
- PSI por feature
- PSI das probabilidades previstas
- tamanho minimo de amostra para decisao operacional

### 4. Retreino e champion-challenger

Quando o drift justifica acao, o projeto avalia se um challenger deve substituir
o champion:

- [src/model_lifecycle/retraining.py](../src/model_lifecycle/retraining.py)
- [src/model_lifecycle/promotion.py](../src/model_lifecycle/promotion.py)

A decisao de promocao considera a metrica principal configurada e um ganho minimo
explicito, mantendo a promocao final auditavel.

### 5. Avaliacao RAG/LLM

A avaliacao generativa fica concentrada em:

- [src/evaluation/llm_agent/ragas_eval.py](../src/evaluation/llm_agent/ragas_eval.py)
- [src/evaluation/llm_agent/llm_judge.py](../src/evaluation/llm_agent/llm_judge.py)
- [src/evaluation/llm_agent/ab_test_prompts.py](../src/evaluation/llm_agent/ab_test_prompts.py)

Ela cobre as frentes principais da trilha generativa:

- golden set com perguntas relevantes ao dominio
- RAGAS com 4 metricas calculadas e reportadas
- LLM-as-judge com 3 criterios, incluindo criterio de negocio
- benchmark de prompts com 3 configuracoes

A explicacao completa esta em [EVALUATION_RAGAS.md](EVALUATION_RAGAS.md).

## Artefatos principais

Modelo e drift:

- `artifacts/evaluation/model/drift/drift_report.html`
- `artifacts/evaluation/model/drift/drift_metrics.json`
- `artifacts/evaluation/model/drift/drift_status.json`
- `artifacts/evaluation/model/drift/drift_runs.jsonl`
- `artifacts/evaluation/model/retraining/retrain_request.json`
- `artifacts/evaluation/model/retraining/retrain_run.json`
- `artifacts/evaluation/model/retraining/promotion_decision.json`

LLM/RAG:

- `data/golden-set.json`
- `artifacts/evaluation/llm_agent/results/ragas_scores.json`
- `artifacts/evaluation/llm_agent/results/llm_judge_scores.json`
- `artifacts/evaluation/llm_agent/results/prompt_ab_results.json`
- `artifacts/evaluation/llm_agent/runs/*.jsonl`

## Como reproduzir

Modelo e operacao:

```bash
poetry run task mlflowrunexperiments
poetry run task mldriftdemo
```

RAGAS precisa do serving ativo, porque chama o endpoint real `POST /llm/chat`:

```bash
poetry run task appstack
poetry run task eval_ragas
poetry run task eval_llm_judge
poetry run task eval_ab_test_prompts
```

Em Docker, o RAGAS usa `RAGAS_SERVING_BASE_URL=http://serving:8000` nas tasks
`eval_ragas_docker`, `eval_ragas_sample_docker`, `eval_all_docker` e
`eval_all_sample_docker`.

## Resumo executivo

O projeto possui avaliacao em camadas: qualidade supervisionada do modelo,
sanidade por cenarios, saude operacional por drift, decisao de retreino e
avaliacao RAG/LLM. A leitura recomendada para banca e:

- [EVALUATION_MODEL_METRICS.md](EVALUATION_MODEL_METRICS.md) para defender o modelo
- [EVALUATION_RAGAS.md](EVALUATION_RAGAS.md) para defender golden set, RAGAS e judge
- [DRIFT_MONITORING.md](DRIFT_MONITORING.md) para defender observabilidade e retreino
