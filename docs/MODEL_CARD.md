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