# Scenario Analysis

O modulo `src.inference.scenario_analysis` executa cenarios hipoteticos
de churn e registra cada execucao em um experimento separado no MLflow.

## Executar um cenario unico

```bash
python -m src.inference.scenario_analysis \
  --config configs/training/model_current.yaml \
  --scenario-name high_churn_manual \
  --payload-file examples/scenario_analysis/high_churn_customer.json
```

## Executar uma suite de cenarios

```bash
python -m src.inference.scenario_analysis \
  --config configs/training/model_current.yaml \
  --suite-file examples/scenario_analysis/churn_validation_suite.json
```

## Formato da suite

```json
{
  "scenarios": [
    {
      "name": "high_risk_profile",
      "payload": {
        "Age": 92,
        "Balance": 0,
        "Card Type": "SILVER",
        "CreditScore": 350,
        "EstimatedSalary": 11.58,
        "Gender": "Female",
        "Geography": "Germany",
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "NumOfProducts": 1,
        "Point Earned": 119,
        "Tenure": 0
      }
    }
  ]
}
```

Cada cenario registra no MLflow:
- payload de entrada como params
- threshold utilizado
- probabilidade prevista
- predicao final
- payload e resultado como artifacts JSON

## Nota importante sobre intuicao de negocio

Os cenarios desta suite foram calibrados com base na distribuicao real do
dataset de treino, e nao apenas em intuicao de negocio.

Em particular, este dataset apresenta um comportamento contraintuitivo em
`NumOfProducts`:
- `1` produto: churn medio aproximado de `27.7%`
- `2` produtos: churn medio aproximado de `7.6%`
- `3` produtos: churn medio aproximado de `82.7%`
- `4` produtos: churn medio aproximado de `100%`

Isso significa que, neste projeto, aumentar o numero de produtos nao
necessariamente reduz churn. Por isso:
- o cenario `high_risk_profile` usa `NumOfProducts = 4`
- o cenario `low_risk_profile` usa `NumOfProducts = 2`

Se novos modelos ou novos datasets forem introduzidos, a suite deve ser
reavaliada antes de ser usada como teste de sanidade.
