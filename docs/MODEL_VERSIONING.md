# Model Versioning e Governança

## Índice

- [Onde isso acontece](#onde-isso-acontece)
- [Metadados registrados no MLflow](#metadados-registrados-no-mlflow)
- [Origem de cada campo](#origem-de-cada-campo)
- [Como interpretar](#como-interpretar)
- [Objetivo desta abordagem](#objetivo-desta-abordagem)
- [Conexao com o fluxo do projeto](#conexao-com-o-fluxo-do-projeto)
- [Limites da implementacao](#limites-da-implementacao)

Este projeto registra metadados minimos de governanca no fluxo de treino para
sustentar rastreabilidade, comparacao entre versoes e auditoria do ciclo de
vida do modelo.

As decisoes arquiteturais sobre esse recorte de registry e governanca ficam em
[ADRs/ADR-006.md](ADRs/ADR-006.md). Este documento descreve como o mecanismo foi
materializado no repositorio.

## Onde isso acontece

O enriquecimento de metadados acontece em
`src/model_lifecycle/train.py`, durante o logging da run no MLflow.

Os valores sao definidos a partir de duas fontes:
- configuracao declarativa do experimento em `configs/model_lifecycle/*.json`
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

### Declarados no contrato do experimento

Exemplo em `configs/model_lifecycle/current.json`:

```json
{
  "experiment": {
    "name": "current",
    "run_name": "current",
    "version": "0.2.0"
  },
  "governance": {
    "risk_level": "high",
    "fairness_checked": false
  }
}
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
  - commit obtido via `git rev-parse HEAD`

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
O campo reforca a trilha de governanca documentada para cada versao treinada.

## Objetivo desta abordagem

Esta implementacao nao substitui um Model Registry completo, mas cria a base
necessaria para:
- versionamento consistente
- promotion workflow auditavel
- champion/challenger com metadata confiavel
- compliance com os requisitos de governanca do Datathon

## Conexao com o fluxo do projeto

Esse versionamento sustenta partes do fluxo operacional:

- o champion possui sidecar de metadados em
  `artifacts/models/current_metadata.json`
- o retreino gera challengers em `artifacts/models/challengers/`
- o monitoramento de drift pode abrir um retreino auditavel
- a comparacao champion-challenger produz `promotion_decision.json`

Ou seja, a metadata nao e apenas decorativa. Ela participa da trilha de:

- reproducao
- auditoria
- comparacao entre versoes
- decisao de manutencao do champion

## Limites da implementacao

Nao existe um Model Registry completo com:

- approval workflow formal
- aliases de producao gerenciados por registry
- promocao automatica entre ambientes
- rollback operacional baseado em registry externo

Por isso, a implementacao deve ser lida como:

- boa base de governanca e rastreabilidade
- promotion workflow inicial e auditavel
- maturidade abaixo de um registry operacional completo
