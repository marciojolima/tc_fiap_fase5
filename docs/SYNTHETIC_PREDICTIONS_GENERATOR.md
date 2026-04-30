# Gerador de Predições Sintéticas

## Índice

- [Objetivo](#objetivo)
- [Módulo](#módulo)
- [Como executar](#como-executar)
- [Parâmetros](#parâmetros)
- [Modos de geração](#modos-de-geração)
- [Preciso analisar a camada raw?](#preciso-analisar-a-camada-raw)
- [Exemplo de integração com o monitoramento](#exemplo-de-integração-com-o-monitoramento)
- [Relação com o serving via Feast](#relação-com-o-serving-via-feast)

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
- `monitoring_contract`
- `model_name`
- `model_version`
- `threshold`
- `churn_probability`
- `churn_prediction`
- `feature_source`
- features transformadas monitoráveis compatíveis com o drift batch

O gerador foi alinhado ao contrato de monitoramento
`transformed_features_v1`. Isso o torna compatível com o pipeline batch de
drift, mas não idêntico ao fluxo operacional do endpoint `/predict` baseado em
Feast.

## Módulo

Arquivo:

- [scripts/generate_synthetic_predictions.py](../scripts/generate_synthetic_predictions.py)

Saída padrão:

- `artifacts/evaluation/model/drift/experiments/predictions/synthetic_predictions_v1.jsonl`

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
  --output artifacts/evaluation/model/drift/experiments/predictions/synthetic_predictions_v1_with_drift_120.jsonl \
  --metadata-output artifacts/evaluation/model/drift/experiments/predictions/synthetic_predictions_v1_with_drift_120.metadata.json
```

## Parâmetros

- `--num-predictions`: quantidade de registros sintéticos a gerar. Obrigatório.
- `--drift`: define o tipo de lote. Valores aceitos: `no_drift` e `with_drift`. Obrigatório.
- `--input-csv`: CSV base usado para preservar o domínio das features. Padrão: `data/raw/Customer-Churn-Records.csv`.
- `--output`: arquivo JSONL de saída. Padrão: `artifacts/evaluation/model/drift/experiments/predictions/synthetic_predictions_v1.jsonl`.
- `--metadata-output`: caminho opcional para salvar um resumo JSON da geração.
- `--experiment-config`: config estruturado do modelo usado para calcular `churn_probability` e `churn_prediction`. Padrão: `configs/model_lifecycle/current.json`.
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

Mais especificamente, o `with_drift` não cria linhas totalmente artificiais do
zero. Ele segue esta estratégia:

1. carrega a base bruta de referência em `data/raw/Customer-Churn-Records.csv`
2. mantém apenas as colunas aceitas pelo pipeline de geração e monitoramento
3. faz uma amostragem inicial com reposição para preservar o domínio real das
   features
4. aplica deslocamentos fortes e intencionais em colunas que influenciam a
   distribuição observada pelo monitoramento
5. aplica o mesmo `feature_pipeline` do projeto para gerar a matriz
   transformada monitorável
6. usa o modelo atual para calcular `churn_probability` e `churn_prediction`

Isso é importante porque o objetivo do modo `with_drift` não é apenas gerar
inputs válidos, mas gerar uma base corrente cuja distribuição fique
propositalmente diferente da base de referência usada no treino.

Na implementação atual, o gerador reaproveita o cenário
`build_mixed_extreme_drift_batch` de
[src/evaluation/model/drift/synthetic_drifts.py](../src/evaluation/model/drift/synthetic_drifts.py).
Esse cenário foi pensado para deslocar a distribuição de forma suficientemente
forte para que o monitoramento batch tenha alta chance de detectar drift.

As principais mudanças aplicadas hoje são:

- `CreditScore`: deslocado para uma distribuição com valores mais baixos e mais
  dispersos
- `Age`: deslocada para uma população significativamente mais velha
- `Balance`: concentrado em extremos, combinando valores próximos de zero e
  valores muito altos
- `NumOfProducts`: redistribuído para privilegiar combinações menos comuns
- `IsActiveMember`: maior concentração de clientes inativos
- `EstimatedSalary`: mistura de faixas muito baixas e muito altas
- `Geography`: concentração maior em `Germany` e `Spain`
- `Card Type`: concentração em perfis mais específicos
- `Point Earned`: deslocamento para extremos baixos e altos

Esse deslocamento tende a impactar:

- o PSI das features monitoradas
- a distribuição das probabilidades previstas
- a taxa de predições positivas no lote corrente

## Preciso analisar a camada raw?

Para usar o módulo, não.

O script já usa a camada raw como base de domínio para:

- preservar categorias válidas como `Geography`, `Gender` e `Card Type`
- manter o formato esperado pelo contrato atual de monitoramento
- evitar geração de combinações absurdas ou fora do contrato

Mas para avaliar a qualidade do cenário `with_drift`, vale a pena analisar a
camada raw e comparar:

- distribuição original das features na base bruta
- distribuição do lote gerado com `no_drift`
- distribuição do lote gerado com `with_drift`

Essa análise ajuda a responder se o drift sintético está:

- forte o suficiente para ser detectado
- realista para o domínio do problema
- coerente com o comportamento que você quer demonstrar no monitoramento

Em resumo:

- a raw não precisa ser reestudada para executar o gerador
- a raw é usada como base de domínio pelo próprio script
- a análise comparativa da raw é recomendada quando você quiser calibrar melhor
  a intensidade e o realismo do `with_drift`

## Exemplo de integração com o monitoramento

Depois de gerar um arquivo sintético, você pode usá-lo como base corrente para
uma execução local do monitoramento batch, substituindo temporariamente o
`current_data_path` ou copiando o arquivo gerado para o caminho esperado pelo
monitoramento.

O artefato resultante pode ser inspecionado junto com:

- [DRIFT_MONITORING.md](DRIFT_MONITORING.md)
- [src/evaluation/model/drift/prediction_logger.py](../src/evaluation/model/drift/prediction_logger.py)
- [src/evaluation/model/drift/drift.py](../src/evaluation/model/drift/drift.py)

## Relação com o serving via Feast

O lote sintético é compatível com o monitoramento batch, mas não replica todo o
fluxo operacional do `/predict`.

Diferenças importantes:

- o serving online principal consulta a Feature Store por `customer_id`
- o log real do `/predict` tende a usar `feature_source=feast_online_store`
- o lote sintético normalmente usa outra proveniência e pode não carregar
  `customer_id`

Isso não impede o uso do gerador para testar o `mldrift`. A diferença é mais
de rastreabilidade operacional do que de compatibilidade com o contrato de
drift atual.
