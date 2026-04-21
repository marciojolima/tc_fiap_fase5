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

- [configs/monitoring_config.yaml](../configs/monitoring_config.yaml)
- [src/monitoring/drift.py](../src/monitoring/drift.py)
- [src/monitoring/inference_log.py](../src/monitoring/inference_log.py)
- [src/models/retraining.py](../src/models/retraining.py)
- [src/models/train.py](../src/models/train.py)

O fluxo atual é:

1. a API registra as inferências em `artifacts/monitoring/inference_logs/predictions.jsonl`
2. a rotina de drift carrega base de referência e base corrente
3. o sistema compara distribuições de features e, quando habilitado, das
   probabilidades previstas
4. o sistema valida se a base corrente já atingiu o tamanho mínimo para decisão
5. o sistema consolida um status final: `ok`, `warning`, `critical` ou
   `insufficient_data`
6. em caso crítico e com amostra elegível, é gerada uma solicitação auditável
   de retreino
7. dependendo do modo configurado, o retreino pode ser executado automaticamente

## Como Pensar no Passo 1

Antes mesmo de executar a rotina de drift, o primeiro passo para entender o
raciocínio do projeto é subir o serving.

Comando:

```bash
poetry run task appstack
```

Isso é importante porque a base corrente do monitoramento não nasce sozinha.
Ela é construída a partir das predições recebidas pela API.

No desenho atual do projeto, o drift compara:

- a base de referência: `data/processed/train.parquet`
- a base corrente: `artifacts/monitoring/inference_logs/predictions.jsonl`

Ou seja: sem serving e sem predição, não existe base corrente para comparar.

Mapeamento no código:

- rota de predição: [src/serving/routes.py](../src/serving/routes.py)
- preparação da configuração de serving: [src/serving/pipeline.py](../src/serving/pipeline.py)
- log de inferência que alimenta a base corrente: [src/monitoring/inference_log.py](../src/monitoring/inference_log.py)

Esse log é justamente o insumo que depois será lido pela rotina de drift.
No contrato atual, ele representa principalmente as features transformadas
efetivamente servidas ao modelo, e não mais apenas o payload bruto recebido pela
API.

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
poetry run task appstack
poetry run task mldrift
```

## O Que Acontece em Cada Passo

### 1. Serving

O serving responde à rota `/predict` e registra uma linha por inferência.

No fluxo recomendado do projeto, ele sobe junto com MLflow, Prometheus e Grafana via Docker Compose.

Arquivos envolvidos:

- [src/serving/routes.py](../src/serving/routes.py)
- [src/serving/pipeline.py](../src/serving/pipeline.py)
- [src/monitoring/inference_log.py](../src/monitoring/inference_log.py)

O arquivo gerado nessa fase é:

- `artifacts/monitoring/inference_logs/predictions.jsonl`

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

- [src/monitoring/drift.py](../src/monitoring/drift.py)

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

- carga de datasets: [src/monitoring/drift.py](../src/monitoring/drift.py)
- preparação dos dados de monitoramento: [src/monitoring/drift.py](../src/monitoring/drift.py)
- cálculo de PSI numérico: [src/monitoring/drift.py](../src/monitoring/drift.py)
- cálculo de PSI categórico: [src/monitoring/drift.py](../src/monitoring/drift.py)
- PSI por feature: [src/monitoring/drift.py](../src/monitoring/drift.py)
- consolidação do status: [src/monitoring/drift.py](../src/monitoring/drift.py)
- bloqueio operacional por amostra pequena: [src/monitoring/drift.py](../src/monitoring/drift.py)

### 3. Artefatos Gerados

Após rodar o drift, os arquivos principais são:

- `artifacts/monitoring/drift/drift_report.html`
- `artifacts/monitoring/drift/drift_metrics.json`
- `artifacts/monitoring/drift/drift_status.json`
- `artifacts/monitoring/drift/drift_runs.jsonl`

Onde isso acontece:

- gravação dos JSONs principais: [src/monitoring/drift.py](../src/monitoring/drift.py)
- histórico de execuções: [src/monitoring/drift.py](../src/monitoring/drift.py)

Leitura rápida:

- `drift_report.html`: evidência visual detalhada, agora com resumo operacional
  do projeto no topo exibindo thresholds de `warning` e `critical`, além do
  Evidently configurado com `PSI` para reduzir divergência com o monitoramento batch
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

- [configs/monitoring_config.yaml](../configs/monitoring_config.yaml)

Implementação:

- abertura da solicitação: [src/monitoring/drift.py](../src/monitoring/drift.py)
- política de disparo: [src/monitoring/drift.py](../src/monitoring/drift.py)
- executor dedicado: [src/models/retraining.py](../src/models/retraining.py)

Artefatos dessa fase:

- `artifacts/monitoring/retraining/retrain_request.json`
- `artifacts/monitoring/retraining/retrain_run.json`

Formas atuais de acionamento:

- fluxo batch automático, quando o drift crítico atende a política configurada
- chamada explícita do executor

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

PSI significa **Population Stability Index**, ou **Índice de Estabilidade
Populacional**.

No contexto deste projeto, ele serve para medir a estabilidade do modelo ao
longo do tempo, comparando a distribuição das variáveis de entrada e, quando
habilitado, das probabilidades previstas entre:

- a base de referência, associada ao treino
- a base corrente, associada ao comportamento mais recente em produção

De forma intuitiva, o PSI responde a uma pergunta simples:

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
estão definidos em [configs/monitoring_config.yaml](../configs/monitoring_config.yaml).

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
  current_data_path: artifacts/monitoring/inference_logs/predictions.jsonl
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
  promotion_decision_path: artifacts/monitoring/retraining/promotion_decision.json
  promotion_rules:
    primary_metric: auc
    minimum_improvement: 0.005
```

Interpretacao dos campos:

- `enabled`: liga ou desliga a abertura do fluxo de retreino
- `trigger_mode`: define o nivel de automacao
- `training_config_path`: qual experimento de treino sera executado
- `request_path`: onde fica a solicitacao auditavel
- `run_path`: onde fica o resultado da execucao do retreino
- `promotion_decision_path`: onde fica a decisao auditavel champion vs challenger
- `promotion_rules`: regra minima para considerar o challenger elegivel

## Estrategia Atual de Automacao

No estado atual, o modo configurado e:

```yaml
trigger_mode: auto_train_manual_promote
```

Isso significa:

- o drift critico abre a solicitacao de retreino
- o retreino pode ser executado automaticamente
- o retreino gera um challenger separado
- o challenger e comparado com o champion atual
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
  "promotion_decision_path": "artifacts/monitoring/retraining/promotion_decision.json",
  "promotion_rules": {
    "primary_metric": "auc",
    "minimum_improvement": 0.005
  },
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
  "challenger_training_config_path": "artifacts/monitoring/retraining/generated_configs/retrain_<request_id>.yaml",
  "experiment_name": "random_forest_current",
  "model_output_path": "artifacts/models/challengers/model_current_<request_id>.pkl",
  "model_version": "0.2.0-challenger-<request_id>",
  "metrics": {
    "auc": 0.91,
    "f1": 0.80
  },
  "promotion_decision": {
    "status": "eligible",
    "eligible_for_promotion": true,
    "recommended_action": "manual_review_for_promotion"
  }
}
```

## Comparacao Champion vs Challenger

Depois que o retreino termina, o projeto agora executa uma comparacao auditavel
entre:

- o champion atual, representado pelo metadata sidecar de `model_current.pkl`
- o challenger recem-treinado, salvo em `artifacts/models/challengers/`

Arquivos envolvidos:

- [src/models/retraining.py](../src/models/retraining.py)
- [src/models/promotion.py](../src/models/promotion.py)
- [artifacts/models/model_current_metadata.json](../artifacts/models/model_current_metadata.json)

O resultado dessa etapa vai para:

- `artifacts/monitoring/retraining/promotion_decision.json`

Essa decisao ainda nao promove nada sozinha. Ela apenas responde:

- o challenger ficou elegivel para promocao?
- qual foi a regra usada?
- qual o delta entre champion e challenger?

No estado atual, a regra minima padrao e:

- `primary_metric = auc`
- `minimum_improvement = 0.005`

## Leitura Atual do Core

Mesmo antes das proximas etapas, o core do monitoramento de drift do projeto
hoje pode ser resumido assim:

- monitoramos `data drift` e `prediction drift`
- usamos `PSI` como metrica principal de decisao
- classificamos o estado em `ok`, `warning`, `critical` ou `insufficient_data`
- `critical` abre solicitacao de retreino
- no modo atual, o retreino pode ser disparado automaticamente
- o retreino agora gera um challenger separado
- o challenger e comparado com o champion antes de qualquer promocao
- a promocao ainda permanece manual e auditavel

Essa base deve continuar a mesma nas proximas etapas. O que deve evoluir depois
e principalmente a promocao controlada do challenger aprovado.
