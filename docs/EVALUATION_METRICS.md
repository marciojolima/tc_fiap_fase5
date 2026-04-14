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

# 📊 Análise de Modelos de Classificação — Churn Bancário

## 🎯 Objetivo

Avaliar diferentes modelos de classificação para previsão de churn, considerando métricas como **Accuracy, Precision, Recall, F1-score e AUC**, além do impacto do **threshold de decisão**.

Mais do que escolher “o maior número”, o objetivo destas métricas no projeto é
apoiar decisões operacionais:

- qual modelo deve seguir como champion
- qual threshold é mais coerente com a estratégia de retenção
- quando um challenger realmente supera o champion

Essa leitura conversa diretamente com o fluxo implementado de drift, retreino e
comparação champion-challenger.

---

## 🧠 Modelos Avaliados

* Random Forest (v1, v2, v3)
* Gradient Boosting (v1)
* XGBoost (v1)

Todos utilizando o mesmo conjunto de dados (`processed_v1`) e divisão de treino/teste (80/20).

No estado atual do projeto, o champion operacional está alinhado ao
`random_forest_current`, cuja leitura prática é próxima da família de resultados
mais equilibrados mostrada nesta tabela.

---

## 🏆 Resultados Consolidados

| Modelo               | Accuracy | AUC   | F1        | Precision | Recall    |
| -------------------- | -------- | ----- | --------- | --------- | --------- |
| random_forest_v2     | 0.852    | 0.872 | **0.642** | 0.634     | 0.650     |
| random_forest_v3     | 0.740    | 0.873 | 0.567     | 0.429     | **0.833** |
| gradient_boosting_v1 | 0.867    | 0.879 | 0.600     | **0.772** | 0.490     |
| xgboost_v1           | 0.837    | 0.857 | 0.610     | 0.596     | 0.625     |
| random_forest_v1     | 0.843    | 0.871 | 0.631     | 0.606     | 0.657     |

---

## 🏅 Melhor Modelo Geral

### ✅ Random Forest v2

* Melhor **F1-score (0.642)**
* Equilíbrio entre **precision** e **recall**
* Boa capacidade de separação (**AUC 0.872**)

📌 **Conclusão:**

> Modelo mais adequado para uso geral, equilibrando corretamente falsos positivos e falsos negativos.

Essa ideia de equilíbrio é importante porque, no projeto, a promoção de um novo
challenger não depende apenas de “treinar de novo”. Ela depende de o modelo
novo realmente manter ou melhorar o comportamento esperado frente ao champion.

---

## 🔥 Modelo Orientado a Recall

### ⚠️ Random Forest v3 (threshold = 0.4)

* Recall: **0.833** (muito alto)
* Precision: **0.429** (baixo)

📌 **Interpretação:**

> Modelo mais agressivo, priorizando identificar clientes com risco de churn, ao custo de maior número de falsos positivos.

📌 **Uso recomendado:**

* Estratégias de retenção agressiva
* Quando o custo de perder cliente é alto

---

## 🎯 Modelo Orientado a Precision

### ⚠️ Gradient Boosting v1

* Precision: **0.772**
* Recall: **0.490**

📌 **Interpretação:**

> Modelo conservador, evitando falsos positivos, porém deixando escapar clientes com risco de churn.

📌 **Uso recomendado:**

* Quando ações sobre clientes têm custo elevado
* Evitar abordagens desnecessárias

---

## ⚖️ Impacto do Threshold

Foi observado que a alteração do threshold (limiar de decisão) impacta significativamente o comportamento do modelo:

| Threshold | Comportamento                     |
| --------- | --------------------------------- |
| 0.3       | Alto Recall (mais sensível)       |
| 0.5       | Equilíbrio                        |
| 0.7       | Alta Precision (mais conservador) |

📌 **Insight chave:**

> O threshold influencia diretamente o trade-off entre precision e recall, podendo adaptar o mesmo modelo para diferentes estratégias de negócio.

Esse ponto é central para a banca: muitas vezes dois modelos com AUC parecida
podem produzir comportamentos operacionais muito diferentes por causa do
threshold escolhido.

---

## 📈 Observação sobre AUC

* Os modelos apresentaram AUC semelhantes (~0.87)
* Isso indica que possuem **capacidade similar de separação entre classes**

📌 **Importante:**

> AUC mede a qualidade do ranking, não a decisão final.

---

## ⚠️ Limitações da Accuracy

Apesar de alguns modelos apresentarem alta accuracy:

* Gradient Boosting: 0.867
* Random Forest v2: 0.852

📌 **Observação:**

> Accuracy pode ser enganosa em problemas desbalanceados como churn, sendo necessário avaliar também Precision, Recall e F1.

---

## 🧠 Conclusões Finais

* **Random Forest v2** é o melhor modelo geral
* **Threshold é um fator crítico de decisão**
* O mesmo modelo pode ser adaptado para diferentes estratégias:

  * **Recall alto → retenção agressiva**
  * **Precision alta → abordagem conservadora**
* **AUC semelhante entre modelos indica que o ganho está mais na decisão (threshold) do que no modelo em si**

## Conexão com o Fluxo Atual de Retreino

No fluxo implementado hoje, essas métricas não ficam apenas em documentação:

- elas ajudam a registrar o champion atual
- entram na análise do challenger gerado em retreino
- sustentam a decisão auditável em `promotion_decision.json`

No estágio atual, a regra de promoção usa `auc` como métrica principal com
delta mínimo explícito. Isso não significa que AUC seja a única métrica
relevante para negócio, mas ela funciona bem como critério padronizado de
comparação inicial entre versões do modelo.
