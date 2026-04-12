# Monitoramento de Drift e Gatilho de Retreino

## Objetivo

Este documento descreve a estratégia atual de monitoramento de drift do projeto,
quais tipos de drift estão sendo observados no momento e como o gatilho de
retreino funciona na implementação atual.

O foco desta etapa é responder a uma pergunta central de operação:

- os dados e o comportamento preditivo do modelo continuam compatíveis com o
  padrão usado no treinamento?

Se a resposta começar a ser "não", o sistema precisa registrar isso com
rastreabilidade e iniciar o fluxo de retreino.

## Estratégia Atual

O projeto utiliza uma estratégia de monitoramento batch baseada na comparação
entre:

- uma base de referência
- uma base corrente

Hoje, a base de referência é o conjunto processado de treino e a base corrente
é formada a partir do log de inferências da API.

Arquivos principais do fluxo:

- [configs/monitoring_config.yaml](/home/marcio/dev/projects/python/tc_fiap_fase5/configs/monitoring_config.yaml:1)
- [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:1)
- [src/monitoring/inference_log.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/inference_log.py:1)
- [src/models/retraining.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/models/retraining.py:1)
- [src/models/train.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/models/train.py:1)

O fluxo atual é:

1. a API registra as inferências em `data/monitoring/current/predictions.jsonl`
2. a rotina de drift carrega base de referência e base corrente
3. o sistema compara distribuições de features e, quando habilitado, das
   probabilidades previstas
4. o sistema valida se a base corrente já atingiu o tamanho mínimo para decisão
5. o sistema consolida um status final: `ok`, `warning`, `critical` ou
   `insufficient_data`
6. em caso crítico e com amostra elegível, é gerada uma solicitação auditável
   de retreino
6. dependendo do modo configurado, o retreino pode ser executado automaticamente

## Como Pensar no Passo 1

Antes mesmo de executar a rotina de drift, o primeiro passo para entender o
raciocínio do projeto é subir o serving.

Comando:

```bash
poetry run task serving
```

Isso é importante porque a base corrente do monitoramento não nasce sozinha.
Ela é construída a partir das predições recebidas pela API.

No desenho atual do projeto, o drift compara:

- a base de referência: `data/processed/train.parquet`
- a base corrente: `data/monitoring/current/predictions.jsonl`

Ou seja: sem serving e sem predição, não existe base corrente para comparar.

Mapeamento no código:

- rota de predição: [src/serving/routes.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/serving/routes.py:51)
- preparação da configuração de serving: [src/serving/pipeline.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/serving/pipeline.py:31)
- log de inferência que alimenta a base corrente: [src/monitoring/inference_log.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/inference_log.py:43)

Esse log é justamente o insumo que depois será lido pela rotina de drift.

## Passo a Passo Operacional

O fluxo principal de análise de drift no projeto hoje pode ser resumido assim:

1. subir o serving
2. gerar predições reais pela API
3. acumular a base corrente em `predictions.jsonl`
4. executar a rotina batch de drift
5. ler os artefatos gerados
6. verificar se houve abertura e execução de retreino

Comandos típicos:

```bash
poetry run task serving
poetry run task mldrift
```

## O Que Acontece em Cada Passo

### 1. Serving

O serving responde à rota `/predict` e registra uma linha por inferência.

Arquivos envolvidos:

- [src/serving/routes.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/serving/routes.py:51)
- [src/serving/pipeline.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/serving/pipeline.py:31)
- [src/monitoring/inference_log.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/inference_log.py:44)

O arquivo gerado nessa fase é:

- `data/monitoring/current/predictions.jsonl`

Esse arquivo contém:

- metadados de inferência como `timestamp`, `model_name`, `model_version`
- `churn_probability` e `churn_prediction`
- payload de entrada usado na predição

### 2. Execução do Drift

A rotina batch é executada por:

```bash
poetry run task mldrift
```

Ponto de entrada:

- [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:429)

Essa rotina:

- carrega a base de referência
- carrega a base corrente
- prepara as features no mesmo padrão do treino
- gera relatório HTML com Evidently
- calcula PSI por feature
- calcula `prediction_psi`, quando habilitado
- aplica a política de amostra mínima para decisão
- consolida o status em `ok`, `warning`, `critical` ou `insufficient_data`

Mapeamento principal:

- carga de datasets: [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:58)
- preparação dos dados de monitoramento: [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:127)
- cálculo de PSI numérico: [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:156)
- cálculo de PSI categórico: [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:193)
- PSI por feature: [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:223)
- consolidação do status: [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:246)
- bloqueio operacional por amostra pequena: [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:284)

### 3. Artefatos Gerados

Após rodar o drift, os arquivos principais são:

- `artifacts/monitoring/drift/drift_report.html`
- `artifacts/monitoring/drift/drift_metrics.json`
- `artifacts/monitoring/drift/drift_status.json`
- `artifacts/monitoring/drift/drift_runs.jsonl`

Onde isso acontece:

- gravação dos JSONs principais: [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:478)
- histórico de execuções: [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:517)

Leitura rápida:

- `drift_report.html`: evidência visual detalhada
- `drift_metrics.json`: métricas detalhadas da última execução
- `drift_status.json`: resumo operacional da última execução
- `drift_runs.jsonl`: histórico das execuções de monitoramento

### 4. Gatilho de Retreino

Se o status final for `critical`, o sistema abre uma solicitação auditável de
retreino e, no modo atual, também executa o retreino.

Antes disso, existe uma guarda operacional importante: se a base corrente tiver
menos linhas do que o mínimo configurado em
`minimum_current_sample_size_for_decision`, o PSI ainda é calculado e salvo
para observabilidade, mas o status vira `insufficient_data` e nenhum retreino é
aberto.

Configuração:

- [configs/monitoring_config.yaml](/home/marcio/dev/projects/python/tc_fiap_fase5/configs/monitoring_config.yaml:1)

Implementação:

- abertura da solicitação: [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:367)
- política de disparo: [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:401)
- executor dedicado: [src/models/retraining.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/models/retraining.py:130)

Artefatos dessa fase:

- `artifacts/monitoring/retraining/retrain_request.json`
- `artifacts/monitoring/retraining/retrain_run.json`

### 5. Interpretação Correta do Resultado

Ao testar com poucas predições, o mais importante é validar o fluxo
operacional, não tirar conclusão estatística forte sobre o negócio.

Em amostras pequenas:

- o PSI pode ficar artificialmente alto
- o projeto continua registrando métricas e histórico
- o status operacional passa a ser `insufficient_data`
- o retreino fica bloqueado até haver volume mínimo

Hoje o mínimo operacional está configurado em:

```yaml
minimum_current_sample_size_for_decision: 30
```

Isso evita que uma amostra muito pequena dispare retreino automático com base
em um PSI estatisticamente frágil.

Na prática, o pipeline já demonstra:

- captura online de inferências
- monitoramento batch separado do serving
- cálculo de drift com PSI
- histórico de execuções
- gatilho auditável de retreino
- execução automática do retreino no modo atual

## PSI em Termos Intuitivos

O PSI, no contexto deste projeto, responde a uma pergunta simples:

- a distribuição atual está parecida com a distribuição que o modelo viu no
  treinamento?

Se a resposta for "sim", o PSI tende a ficar baixo.
Se a resposta for "não", o PSI sobe.

Leitura prática adotada hoje:

- `PSI < 0.10`: sem sinal forte de drift
- `0.10 <= PSI < 0.20`: alerta
- `PSI >= 0.20`: drift crítico

Mas essa régua só vale para decisão operacional quando a base corrente já
atingiu o tamanho mínimo configurado. Antes disso, o PSI é tratado como sinal
exploratório e não como justificativa para retreino.

Esses thresholds são os mesmos usados pela rotina de decisão operacional e
estão definidos em [configs/monitoring_config.yaml](/home/marcio/dev/projects/python/tc_fiap_fase5/configs/monitoring_config.yaml:12).

## Tipos de Drift Monitorados Hoje

Atualmente o projeto monitora dois tipos principais:

### 1. Data Drift

Este é o eixo principal do monitoramento atual.

O sistema calcula PSI por feature para verificar mudança de distribuição entre
referência e produção. A decisão consolidada usa os thresholds configurados no
arquivo de monitoramento.

Trecho atual do YAML:

```yaml
data_drift:
  enabled: true
  warning_threshold: 0.10
  critical_threshold: 0.20
```

Na prática:

- `PSI < 0.10`: situação normal
- `0.10 <= PSI < 0.20`: alerta
- `PSI >= 0.20`: situação crítica

### 2. Prediction Drift

O projeto também monitora mudança de distribuição das probabilidades previstas
(`churn_probability`), usando PSI.

Trecho atual do YAML:

```yaml
prediction_drift:
  enabled: true
  warning_threshold: 0.10
  critical_threshold: 0.20
```

Esse monitoramento ajuda a responder se o comportamento do modelo em produção
está se afastando do comportamento esperado, mesmo antes de termos um ciclo
completo de ground truth em produção.

## O Que Ainda Nao Esta Sendo Tratado

Neste momento, o projeto ainda nao implementa concept drift de forma completa.

Ou seja, ainda nao estamos comparando:

- previsao feita em `T-1`
- verdade observada em `T`
- degradacao real de performance ao longo do tempo

Esse tipo de monitoramento depende de verdade rotulada posterior e deve entrar
nas etapas seguintes da evolucao.

## Base de Referencia e Base Corrente

Trecho central da configuracao atual:

```yaml
drift:
  enabled: true
  reference_data_path: data/processed/train.parquet
  current_data_path: data/monitoring/current/predictions.jsonl
  feature_columns_path: data/processed/feature_columns.json
  feature_pipeline_path: artifacts/models/feature_pipeline.joblib
  model_path: artifacts/models/model_current.pkl
```

Interpretacao:

- `reference_data_path`: base considerada padrao historico do modelo
- `current_data_path`: dados mais recentes para comparacao
- `feature_columns_path`: define a ordem das features esperadas
- `feature_pipeline_path`: garante que a transformacao usada no drift seja
  compatível com a usada no treino e no serving
- `model_path`: usado para recuperar probabilidades de referencia, quando
  necessario

## Metricas e Ferramentas

O monitoramento atual usa:

- `Evidently` para gerar o relatorio HTML de drift
- `PSI` como metrica principal de estabilidade de distribuicao

Os artefatos gerados hoje incluem:

- `artifacts/monitoring/drift/drift_report.html`
- `artifacts/monitoring/drift/drift_metrics.json`
- `artifacts/monitoring/drift/drift_status.json`

## Consolidacao do Status

Ao final da rotina, o sistema consolida um status unico:

- `ok`
- `warning`
- `critical`

A consolidacao considera:

- maior PSI entre as features
- participacao de features afetadas
- PSI das probabilidades previstas, quando habilitado

Esse status e a base do gatilho de retreino.

## Gatilho de Retreino

Quando o status consolidado e `critical`, o projeto considera que ha evidencias
suficientes para abrir um fluxo de retreino.

Trecho atual do YAML:

```yaml
retraining:
  enabled: true
  trigger_mode: auto_train_manual_promote
  training_config_path: configs/training/model_current.yaml
  request_path: artifacts/monitoring/retraining/retrain_request.json
  run_path: artifacts/monitoring/retraining/retrain_run.json
```

Interpretacao dos campos:

- `enabled`: liga ou desliga a abertura do fluxo de retreino
- `trigger_mode`: define o nivel de automacao
- `training_config_path`: qual experimento de treino sera executado
- `request_path`: onde fica a solicitacao auditavel
- `run_path`: onde fica o resultado da execucao do retreino

## Estrategia Atual de Automacao

No estado atual, o modo configurado e:

```yaml
trigger_mode: auto_train_manual_promote
```

Isso significa:

- o drift critico abre a solicitacao de retreino
- o retreino pode ser executado automaticamente
- a promocao do novo modelo ainda nao e automatica

Essa decisao e proposital. Ela reduz risco e combina melhor com o momento atual
do projeto, em que o foco ainda e consolidar a base de monitoramento e
rastreabilidade antes de automatizar a promocao do challenger.

## Estrutura da Solicitacao de Retreino

Quando o drift fica critico, o sistema gera um `retrain_request.json`.

Exemplo conceitual:

```json
{
  "request_id": "uuid",
  "status": "requested",
  "reason": "critical_data_or_prediction_drift",
  "model_path": "artifacts/models/model_current.pkl",
  "training_config_path": "configs/training/model_current.yaml",
  "created_at": "2026-04-12T00:00:00+00:00",
  "trigger_mode": "auto_train_manual_promote",
  "promotion_policy": "manual_approval_required",
  "drift_status": "critical",
  "max_feature_psi": 0.25,
  "prediction_psi": 0.14,
  "drifted_features": ["Age", "Balance"]
}
```

Esse arquivo funciona como contrato entre:

- monitoramento
- executor de retreino
- futuras etapas de promocao

## Estrutura do Resultado do Retreino

O executor de retreino gera um `retrain_run.json` com o resumo da execucao.

Exemplo conceitual:

```json
{
  "request_id": "uuid",
  "status": "completed",
  "started_at": "...",
  "completed_at": "...",
  "reason": "critical_data_or_prediction_drift",
  "trigger_mode": "auto_train_manual_promote",
  "promotion_policy": "manual_approval_required",
  "drift_status": "critical",
  "training_config_path": "configs/training/model_current.yaml",
  "experiment_name": "random_forest_current",
  "model_output_path": "artifacts/models/model_current.pkl",
  "model_version": "0.2.0",
  "metrics": {
    "auc": 0.91,
    "f1": 0.80
  }
}
```

## Leitura Atual do Core

Mesmo antes das proximas etapas, o core do monitoramento de drift do projeto
hoje pode ser resumido assim:

- monitoramos `data drift` e `prediction drift`
- usamos `PSI` como metrica principal de decisao
- classificamos o estado em `ok`, `warning` e `critical`
- `critical` abre solicitacao de retreino
- no modo atual, o retreino pode ser disparado automaticamente
- a promocao ainda permanece manual e auditavel

Essa base deve continuar a mesma nas proximas etapas. O que deve evoluir depois
e principalmente o "depois do treino": comparacao champion vs challenger e
promocao controlada.
