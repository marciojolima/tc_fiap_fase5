# Monitoramento e Observabilidade

Este documento detalha a trilha de monitoramento e observabilidade do projeto,
com foco em métricas operacionais, logging de inferências, detecção de drift e
retreino auditável.

## Escopo de monitoramento

### Métricas operacionais do serving

As métricas expostas pela aplicação permitem acompanhar o comportamento da API
em execução, com foco inicial em:

- volume de requisições
- latência
- taxa de erro
- requisições em andamento

Essas métricas são consumidas pela stack local configurada em
`configs/monitoring/` e orquestrada pelo Docker Compose junto com o serving e o
MLflow.

### Logging de inferências

As inferências podem ser registradas em
`artifacts/logs/inference/predictions.jsonl`, criando uma trilha de execução
útil para:

- auditoria das features efetivamente servidas ao modelo
- composição do dataset corrente de monitoramento
- análise posterior de drift
- apoio a ciclos de retreino

O contrato desse arquivo prioriza as features transformadas e monitoráveis
consumidas pelo modelo em produção, com metadados mínimos de predição e origem.

### Monitoramento batch de drift

O fluxo em [src/evaluation/model/drift/drift.py](../src/evaluation/model/drift/drift.py)
compara uma base de referência com dados correntes e produz evidências
operacionais em:

- `artifacts/evaluation/model/drift/drift_report.html`
- `artifacts/evaluation/model/drift/drift_metrics.json`
- `artifacts/evaluation/model/drift/drift_status.json`
- `artifacts/evaluation/model/drift/drift_runs.jsonl`

Na prática, isso permite:

- inspecionar visualmente o comportamento das distribuições
- calcular PSI por feature
- consolidar um status geral de drift
- manter histórico das execuções de monitoramento

O relatório HTML destaca no topo o resumo operacional do projeto, incluindo
thresholds de `warning` e `critical` definidos no YAML e o status final
calculado pelo pipeline batch. Esse arquivo representa a visão oficial do
projeto para drift, baseada no PSI persistido em `drift_metrics.json`,
enquanto o Evidently fica disponível em um relatório auxiliar separado para
diagnóstico complementar.

### Gatilho auditável de retreino

Quando o monitoramento identifica condição crítica, o fluxo abre uma trilha
auditável de retreino, com artefatos como:

- `artifacts/evaluation/model/retraining/retrain_request.json`
- `artifacts/evaluation/model/retraining/retrain_run.json`
- `artifacts/evaluation/model/retraining/promotion_decision.json`
- `artifacts/evaluation/model/retraining/generated_configs/`

Essa trilha documenta:

- motivo do disparo
- configuração usada no retreino
- resultado consolidado da execução
- decisão final de promoção ou manutenção do champion
