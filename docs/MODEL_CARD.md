# Model Card

## 📋 Dicionário de Dados — Bank Customer Churn

| Coluna | Tradução (PT-BR) | Tipo | Restrições | Explicação |
|---|---|---|---|---|
| CreditScore | Pontuação de Crédito | int | 300–850 | Confiabilidade financeira do cliente |
| Geography | País | str | France, Germany, Spain | País de residência |
| Gender | Gênero | str | Male, Female | Gênero do cliente |
| Age | Idade | int | 18–100 | Idade em anos |
| Tenure | Tempo de Casa | int | 0–10 | Anos como cliente do banco |
| Balance | Saldo | float | ≥ 0 | Valor disponível em conta |
| NumOfProducts | Nº de Produtos | int | 1–4 | Serviços bancários ativos |
| HasCrCard | Possui Cartão | int | 0 ou 1 | 1 = Sim |
| IsActiveMember | Membro Ativo | int | 0 ou 1 | 1 = Movimenta a conta |
| EstimatedSalary | Salário Estimado | float | > 0 | Rendimento anual estimado |
| **Exited** | **Churn (Target)** | int | 0 ou 1 | **1 = Saiu, 0 = Ficou** |
| Complain | Reclamação | int | 0 ou 1 | 1 = Registrou reclamação |
| Satisfaction Score | Score de Satisfação | int | 1–5 | Nota dada pelo cliente |
| Card Type | Tipo de Cartão | str | DIAMOND, GOLD, SILVER, PLATINUM | Categoria do cartão |
| Point Earned | Pontos de Fidelidade | int | ≥ 0 | Pontos acumulados |

### 🔍 Interpretação das Features Mais Importantes (XGBoost - Gain - exemplo)

O gráfico de **Feature Importance** (usando o critério "Gain") revela quais variáveis o modelo considerou mais relevantes para prever o churn dos clientes. Quanto maior o valor de Gain, maior a contribuição da feature nas decisões do modelo.

#### Top 5 Features Mais Importantes

| Posição | Feature              | Importância (%) | Interpretação de Negócio |
|---------|----------------------|------------------|---------------------------|
| 1º     | **NumOfProducts**    | 26.59%          | **Mais importante do modelo**. Clientes com poucos produtos bancários (ex: apenas conta) têm risco muito maior de churn. Estratégia de cross-selling pode ser muito eficaz. |
| 2º     | **IsActiveMember**   | 15.34%          | Clientes inativos saem muito mais do banco. Manter o engajamento (uso de app, transações, etc.) é uma das chaves para reduzir o churn. |
| 3º     | **Age**              | 10.31%          | A idade tem influência significativa. Geralmente clientes muito jovens ou mais velhos apresentam maior probabilidade de churn. |
| 4º     | **Geo_Germany**      | 7.60%           | Clientes da Alemanha possuem maior risco de churn em relação à França (categoria referência). Pode refletir diferenças de mercado ou concorrência no país. |
| 5º     | **Balance**          | 4.81%           | O saldo em conta influencia a decisão de permanência. Clientes com saldos baixos tendem a ser mais propensos a sair. |