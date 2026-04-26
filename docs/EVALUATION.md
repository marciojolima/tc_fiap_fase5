# Evaluation

## Índice

- [Objetivo](#objetivo)
- [Taxonomia de Avaliação do Projeto](#taxonomia-de-avaliação-do-projeto)
- [Resumo Executivo](#resumo-executivo)

## Objetivo

Este documento organiza, de forma taxativa, os diferentes tipos de avaliação
que já acontecem no projeto. A ideia é evitar ambiguidade entre:

- avaliação de modelo tabular
- avaliação por cenários de negócio
- avaliação operacional via drift e monitoramento
- avaliação de retreino e promoção
- avaliação futura para trilhas com LLM

Hoje o projeto já possui avaliação real em várias camadas, mesmo que a pasta
`src/evaluation/llm_agent/` ainda esteja reservada principalmente para a trilha futura de IA
generativa.

## Taxonomia de Avaliação do Projeto

### 1. Avaliação de desempenho do modelo tabular

Este é o eixo mais tradicional de avaliação supervisionada do projeto.

Ele acontece durante o treino em:

- [src/model_lifecycle/train.py](../src/model_lifecycle/train.py)

As métricas calculadas hoje incluem:

- `auc`
- `f1`
- `precision`
- `recall`
- `accuracy`

Essas métricas são usadas para:

- comparar execuções de treino
- registrar qualidade do champion
- sustentar o sidecar de metadata do modelo
- alimentar comparação com challengers em ciclos de retreino

Para a leitura detalhada da importância e prioridade dessas métricas no contexto
de churn, consulte:

- [EVALUATION_METRICS.md](EVALUATION_METRICS.md)

### 2. Avaliação por cenários de negócio

O projeto também avalia o comportamento do modelo em cenários hipotéticos
controlados, que funcionam como testes de sanidade orientados a negócio.

Esse fluxo acontece em:

- [src/scenario_experiments/inference_cases.py](../src/scenario_experiments/inference_cases.py)
- [configs/scenario_experiments/inference_cases.yaml](../configs/scenario_experiments/inference_cases.yaml)

Essa avaliação responde perguntas como:

- o modelo reage de forma coerente em perfis de alto risco?
- o threshold está gerando a classificação esperada?
- o comportamento observado ainda faz sentido para os casos simulados?

Os resultados são registrados no MLflow e também ajudam na narrativa da banca,
porque mostram avaliação além das métricas agregadas de treino.

### 3. Avaliação operacional por monitoramento de drift

Outra camada de avaliação no projeto é a avaliação operacional do modelo em
produção simulada ou local.

Esse fluxo acontece em:

- [src/evaluation/model/drift/drift.py](../src/evaluation/model/drift/drift.py)
- [src/evaluation/model/drift/prediction_logger.py](../src/evaluation/model/drift/prediction_logger.py)

Os principais sinais avaliados hoje são:

- `data drift` por feature
- `prediction drift`
- PSI por feature
- PSI das probabilidades previstas
- elegibilidade operacional da decisão com base no tamanho da amostra

Os artefatos gerados por esse processo incluem:

- `artifacts/evaluation/model/drift/drift_report.html`
- `artifacts/evaluation/model/drift/drift_metrics.json`
- `artifacts/evaluation/model/drift/drift_status.json`
- `artifacts/evaluation/model/drift/drift_runs.jsonl`

Essa camada não mede “qualidade supervisionada” no sentido clássico. Ela mede:

- se os dados correntes continuam compatíveis com o domínio treinado
- se há mudança suficiente para justificar atenção ou retreino

### 4. Avaliação de retreino e comparação champion-challenger

Quando o drift justifica retreino, o projeto entra em uma avaliação de
substituição de modelo.

Esse fluxo acontece em:

- [src/model_lifecycle/retraining.py](../src/model_lifecycle/retraining.py)
- [src/model_lifecycle/promotion.py](../src/model_lifecycle/promotion.py)

Os artefatos principais são:

- `artifacts/evaluation/model/retraining/retrain_request.json`
- `artifacts/evaluation/model/retraining/retrain_run.json`
- `artifacts/evaluation/model/retraining/promotion_decision.json`

Aqui a avaliação não é mais apenas “o modelo treinou”. Ela passa a responder:

- o challenger foi gerado com sucesso?
- ele manteve ou melhorou a métrica principal?
- ele ficou elegível para promoção?
- o champion deve ser mantido?

Hoje, a regra inicial de promoção usa:

- `auc` como métrica principal
- `minimum_improvement` explícito na configuração

Isso transforma a avaliação em uma parte ativa do fluxo de governança, e não
apenas em documentação.

### 5. Avaliação sintética de robustez sob drift

O projeto também possui uma trilha de avaliação sintética, útil para demonstração
e teste do comportamento do monitoramento.

Esse fluxo acontece em:

- [src/evaluation/model/drift/synthetic_drifts.py](../src/evaluation/model/drift/synthetic_drifts.py)

Ele permite:

- gerar lotes artificiais com diferentes perfis de drift
- produzir relatórios HTML específicos por cenário
- verificar como o pipeline reage a mudanças controladas

Essa camada é importante porque ajuda a validar o software de monitoramento sem
depender apenas de tráfego real.

### 6. Avaliação para trilhas com LLM

A pasta `src/evaluation/llm_agent/` agora concentra três trilhas complementares de avaliação
para IA generativa:

- [src/evaluation/llm_agent/ragas_eval.py](../src/evaluation/llm_agent/ragas_eval.py)
- [src/evaluation/llm_agent/llm_judge.py](../src/evaluation/llm_agent/llm_judge.py)
- [src/evaluation/llm_agent/ab_test_prompts.py](../src/evaluation/llm_agent/ab_test_prompts.py)

Esses módulos não fazem parte do fluxo online do agente. Eles funcionam como
gatilhos offline de benchmark e controle de qualidade:

- RAGAS
- LLM-as-judge
- A/B test de prompts

Hoje a leitura correta do projeto é:

- a avaliação tabular já existe e está viva
- a avaliação operacional de drift já existe e está viva
- a avaliação champion-challenger já existe e está viva
- a avaliação de LLM já existe como benchmark offline sobre o golden set

#### 6.1 Prompt A/B

O benchmark A/B de prompts acontece em:

- [src/evaluation/llm_agent/ab_test_prompts.py](../src/evaluation/llm_agent/ab_test_prompts.py)

Ele compara tres variantes de prompt sobre o mesmo golden set, sempre com o
mesmo `retrieve_contexts()` e o mesmo `llm_provider` configurado, para responder:

- qual prompt cobre melhor os termos esperados da referência?
- qual variante se sai melhor quando enriquecida com `LLM-as-judge`?
- qual prompt vale promover como candidato principal para a trilha RAG/LLM?

O fluxo gera:

- `artifacts/evaluation/llm_agent/results/prompt_ab_results.json`
- `artifacts/evaluation/llm_agent/runs/prompt_ab_runs.jsonl`

Métricas usadas hoje:

- `keyword_coverage` determinística contra a referência
- `mean_judge_score` opcional, quando o benchmark é executado com `--with-judge`

Isso permite transformar “troca de prompt” em benchmark reproduzível, e não em
ajuste subjetivo manual.

## Resumo Executivo

Hoje o projeto já possui avaliação real em pelo menos quatro frentes:

1. desempenho supervisionado do classificador
2. cenários de negócio para inferência
3. avaliação operacional via drift
4. avaliação de promoção via champion-challenger

A pasta `src/evaluation/llm_agent/` ainda não é o centro de toda avaliação do projeto.
Ela está mais associada à trilha futura de LLM evaluation. Por isso, a taxonomia
mais correta neste momento é entender “evaluation” no projeto como um conceito
distribuído por vários módulos, e não como uma única pasta funcional.
