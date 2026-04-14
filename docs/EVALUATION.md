# Evaluation

## Objetivo

Este documento organiza, de forma taxativa, os diferentes tipos de avaliação
que já acontecem no projeto. A ideia é evitar ambiguidade entre:

- avaliação de modelo tabular
- avaliação por cenários de negócio
- avaliação operacional via drift e monitoramento
- avaliação de retreino e promoção
- avaliação futura para trilhas com LLM

Hoje o projeto já possui avaliação real em várias camadas, mesmo que a pasta
`evaluation/` ainda esteja reservada principalmente para a trilha futura de IA
generativa.

## Taxonomia de Avaliação do Projeto

### 1. Avaliação de desempenho do modelo tabular

Este é o eixo mais tradicional de avaliação supervisionada do projeto.

Ele acontece durante o treino em:

- [src/models/train.py](../src/models/train.py)

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

- [src/scenario_analysis/inference_cases.py](../src/scenario_analysis/inference_cases.py)
- [configs/scenario_analysis/inference_cases.yaml](../configs/scenario_analysis/inference_cases.yaml)

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

- [src/monitoring/drift.py](../src/monitoring/drift.py)
- [src/monitoring/inference_log.py](../src/monitoring/inference_log.py)

Os principais sinais avaliados hoje são:

- `data drift` por feature
- `prediction drift`
- PSI por feature
- PSI das probabilidades previstas
- elegibilidade operacional da decisão com base no tamanho da amostra

Os artefatos gerados por esse processo incluem:

- `artifacts/monitoring/drift/drift_report.html`
- `artifacts/monitoring/drift/drift_metrics.json`
- `artifacts/monitoring/drift/drift_status.json`
- `artifacts/monitoring/drift/drift_runs.jsonl`

Essa camada não mede “qualidade supervisionada” no sentido clássico. Ela mede:

- se os dados correntes continuam compatíveis com o domínio treinado
- se há mudança suficiente para justificar atenção ou retreino

### 4. Avaliação de retreino e comparação champion-challenger

Quando o drift justifica retreino, o projeto entra em uma avaliação de
substituição de modelo.

Esse fluxo acontece em:

- [src/models/retraining.py](../src/models/retraining.py)
- [src/models/promotion.py](../src/models/promotion.py)

Os artefatos principais são:

- `artifacts/monitoring/retraining/retrain_request.json`
- `artifacts/monitoring/retraining/retrain_run.json`
- `artifacts/monitoring/retraining/promotion_decision.json`

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

- [src/scenario_analysis/synthetic_drifts.py](../src/scenario_analysis/synthetic_drifts.py)

Ele permite:

- gerar lotes artificiais com diferentes perfis de drift
- produzir relatórios HTML específicos por cenário
- verificar como o pipeline reage a mudanças controladas

Essa camada é importante porque ajuda a validar o software de monitoramento sem
depender apenas de tráfego real.

### 6. Avaliação futura para trilhas com LLM

A pasta `evaluation/` existe hoje como espaço reservado para avaliação de IA
generativa, mas seus módulos ainda estão em placeholder:

- [evaluation/ragas_eval.py](../evaluation/ragas_eval.py)
- [evaluation/llm_judge.py](../evaluation/llm_judge.py)
- [evaluation/ab_test_prompts.py](../evaluation/ab_test_prompts.py)

No estado atual, eles não representam avaliação operacional do sistema. Servem
mais como taxonomia reservada para:

- RAGAS
- LLM-as-judge
- A/B test de prompts

Por isso, a leitura correta do projeto hoje é:

- a avaliação tabular já existe e está viva
- a avaliação operacional de drift já existe e está viva
- a avaliação champion-challenger já existe e está viva
- a avaliação de LLM ainda não está implementada de ponta a ponta

## Resumo Executivo

Hoje o projeto já possui avaliação real em pelo menos quatro frentes:

1. desempenho supervisionado do classificador
2. cenários de negócio para inferência
3. avaliação operacional via drift
4. avaliação de promoção via champion-challenger

A pasta `evaluation/` ainda não é o centro de toda avaliação do projeto.
Ela está mais associada à trilha futura de LLM evaluation. Por isso, a taxonomia
mais correta neste momento é entender “evaluation” no projeto como um conceito
distribuído por vários módulos, e não como uma única pasta funcional.
