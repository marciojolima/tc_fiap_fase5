# Gerador de Predições Sintéticas

## Objetivo

Este módulo gera arquivos JSONL compatíveis com o contrato usado pelo
monitoramento de drift atual.

Ele é útil para:

- testar o pipeline batch de drift sem depender da API
- produzir lotes com distribuição parecida com a base original
- produzir lotes com drift intencional para validar alertas e gatilhos

O formato gerado segue o mesmo padrão do log de inferência usado em produção,
incluindo:

- `timestamp`
- `model_name`
- `model_version`
- `threshold`
- `churn_probability`
- `churn_prediction`
- features de entrada aceitas pelo serving

## Módulo

Arquivo:

- [scripts/generate_synthetic_predictions.py](/home/marcio/dev/projects/python/tc_fiap_fase5/scripts/generate_synthetic_predictions.py)

Saída padrão:

- `artifacts/monitoring/drift/experiments/predictions/synthetic_predictions_v1.jsonl`

## Como executar

Via módulo Python:

```bash
poetry run python -m scripts.generate_synthetic_predictions \
  --num-predictions 50 \
  --drift no_drift
```

Gerando um lote com drift intencional:

```bash
poetry run python -m scripts.generate_synthetic_predictions \
  --num-predictions 80 \
  --drift with_drift
```

Gerando em um caminho específico e salvando metadados:

```bash
poetry run python -m scripts.generate_synthetic_predictions \
  --num-predictions 120 \
  --drift with_drift \
  --output artifacts/monitoring/drift/experiments/predictions/synthetic_predictions_v1_with_drift_120.jsonl \
  --metadata-output artifacts/monitoring/drift/experiments/predictions/synthetic_predictions_v1_with_drift_120.metadata.json
```

## Parâmetros

- `--num-predictions`: quantidade de registros sintéticos a gerar. Obrigatório.
- `--drift`: define o tipo de lote. Valores aceitos: `no_drift` e `with_drift`. Obrigatório.
- `--input-csv`: CSV base usado para preservar o domínio das features. Padrão: `data/raw/Customer-Churn-Records.csv`.
- `--output`: arquivo JSONL de saída. Padrão: `artifacts/monitoring/drift/experiments/predictions/synthetic_predictions_v1.jsonl`.
- `--metadata-output`: caminho opcional para salvar um resumo JSON da geração.
- `--experiment-config`: YAML do modelo atual usado para calcular `churn_probability` e `churn_prediction`. Padrão: `configs/training/model_current.yaml`.
- `--seed`: seed para reprodutibilidade. Padrão: `42`.

## Modos de geração

### `no_drift`

Gera registros por amostragem com reposição a partir da base original, tentando
preservar o comportamento esperado da distribuição de treino.

### `with_drift`

Gera registros com alteração forte de distribuição em variáveis como score de
crédito, idade, saldo, salário estimado, geografia e perfil de cartão. Esse
modo reaproveita a mesma lógica de drift sintético usada nos cenários de
validação do projeto.

## Exemplo de integração com o monitoramento

Depois de gerar um arquivo sintético, você pode usá-lo como base corrente para
uma execução local do monitoramento batch, substituindo temporariamente o
`current_data_path` ou copiando o arquivo gerado para o caminho esperado pelo
monitoramento.

O artefato resultante pode ser inspecionado junto com:

- [docs/DRIFT_MONITORING.md](/home/marcio/dev/projects/python/tc_fiap_fase5/docs/DRIFT_MONITORING.md)
- [src/monitoring/inference_log.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/inference_log.py:1)
- [src/monitoring/drift.py](/home/marcio/dev/projects/python/tc_fiap_fase5/src/monitoring/drift.py:1)
