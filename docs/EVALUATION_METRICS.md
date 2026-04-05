# Importância das Métricas de ML — Churn Bancário

Ordenadas da mais importante para a menos importante no contexto de churn bancário,
onde **perder um cliente sem perceber** custa mais do que acionar um cliente fiel por engano.

| # | Métrica | Importância no churn | O que mede | Por que essa posição |
|---|---------|---------------------|------------|----------------------|
| 1 | **Recall** | ★★★★★ Crítica | Dos clientes que *realmente* vão embora, quantos o modelo detectou | Cliente churnou sem ser detectado = receita perdida para sempre. Cada falso negativo é um cliente que saiu sem que o banco pudesse agir. |
| 2 | **AUC** | ★★★★☆ Alta | Qualidade do ranqueamento de risco, independente do threshold | Permite comparar modelos e ajustar o ponto de corte depois. Um AUC alto significa que o modelo *sabe quem é mais arriscado*, mesmo que você mude o threshold conforme o budget de retenção. |
| 3 | **F1** | ★★★★☆ Alta | Equilíbrio entre precision e recall | Métrica de síntese: penaliza modelos que sacrificam um lado. Indispensável quando as classes são desbalanceadas (muito mais clientes fiéis do que churners), como é típico em churn bancário. |
| 4 | **Precision** | ★★★☆☆ Moderada | Dos que o modelo acusou de churnar, quantos realmente foram | Controla o desperdício de orçamento de retenção. Importante, mas secundária: é melhor gastar uma oferta a mais do que perder um cliente de vez. |
| 5 | **Accuracy** | ★☆☆☆☆ Baixa | Percentual de acertos totais (churners + fiéis) | Métrica enganosa em churn: se 80% dos clientes ficam, um modelo burro que diz "ninguém sai" já chega a 80% de accuracy — sem detectar nenhum churner. Útil só como referência geral. |

---

## Regra de ouro

No churn bancário, a ordem de prioridade real é:

```
recall → AUC → F1 → precision → accuracy
```

**Exceção:** se o banco tiver budget de retenção muito limitado (poucos gestores para ligar,
custo alto por ação), a precision sobe de importância e chega perto do F1 — porque acionar
o cliente errado tem custo real.

> No projeto com `class_weight=balanced`, o modelo já compensa o desequilíbrio das classes
> e dá mais peso ao recall — que é a escolha certa para churn bancário.