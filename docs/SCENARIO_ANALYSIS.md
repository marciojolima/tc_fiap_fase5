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
        "Age": 60,
        "Balance": 0,
        "Card Type": "SILVER",
        "CreditScore": 400,
        "EstimatedSalary": 30000,
        "Gender": "Female",
        "Geography": "Germany",
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "NumOfProducts": 1,
        "Point Earned": 100,
        "Tenure": 1
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
