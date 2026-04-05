# Model Versioning e Governança

Este projeto registra metadados minimos de governanca no fluxo de treino
para atender ao requisito de rastreabilidade esperado no Datathon.

## Onde isso acontece

O enriquecimento de metadados acontece em
`src/models/train.py`, durante o logging da run no MLflow.

Os valores sao definidos a partir de duas fontes:
- configuracao declarativa do experimento em `configs/training/*.yaml`
- metadados tecnicos calculados automaticamente no momento do treino

## Metadados registrados no MLflow

Durante o treino, o projeto registra pelo menos:

- `model_name`
- `model_version`
- `model_type`
- `training_data_version`
- `owner`
- `phase`
- `risk_level`
- `fairness_checked`
- `git_sha`
- `feature_set`
- `threshold`

Esses campos sao registrados como `params` e/ou `tags`, para facilitar:
- lineage do modelo
- reproducao de experimentos
- montagem de Model Card / System Card
- auditoria de qual codigo e quais dados geraram um artefato

## Origem de cada campo

### Declarados no YAML do experimento

Exemplo em `configs/training/model_current.yaml`:

```yaml
experiment:
  name: random_forest_current
  run_name: random_forest_current
  version: 0.2.0

governance:
  risk_level: high
  fairness_checked: false
```

Campos vindos do contrato:
- `model_name` <- `experiment.name`
- `model_version` <- `experiment.version`
- `risk_level` <- `governance.risk_level`
- `fairness_checked` <- `governance.fairness_checked`

### Calculados automaticamente

- `training_data_version`
  - hash MD5 dos arquivos `data/processed/train.parquet` e `data/processed/test.parquet`
- `git_sha`
  - commit atual obtido via `git rev-parse HEAD`

## Como interpretar

### `model_version`

Identifica a versao funcional do experimento/modelo.

Exemplo:
- `0.1.0` para primeira versao estavel
- `0.2.0` para melhoria de arquitetura ou qualidade

### `training_data_version`

Representa a assinatura dos dados processados usados no treino.
Se os dados mudarem, esse hash muda, mesmo que o algoritmo continue igual.

### `git_sha`

Permite rastrear exatamente qual revisao do codigo gerou a run e o artefato.

### `risk_level`

Classificacao declarativa do risco do modelo para fins de governanca.
Valores sugeridos:
- `low`
- `medium`
- `high`
- `critical`

### `fairness_checked`

Indica se houve verificacao explicita de fairness/bias para a versao treinada.
Hoje o campo existe para governanca e preparo de evolucao futura.

## Objetivo desta abordagem

Esta implementacao nao substitui um Model Registry completo, mas cria a base
necessaria para:
- versionamento consistente
- promotion workflow futuro
- champion/challenger com metadata confiavel
- compliance com os requisitos de governanca do Datathon
