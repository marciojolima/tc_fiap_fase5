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

#### Features por Importância
| Posição | Feature               | Impacto no Churn | Interpretação de Negócio |
|--------|----------------------|------------------|---------------------------|
| 1º     | NumOfProducts        | 🔥 Muito alto     | Clientes com poucos produtos têm maior probabilidade de churn. Cross-sell é essencial para retenção. |
| 2º     | IsActiveMember       | 🔥 Muito alto     | Clientes inativos apresentam alto risco de churn. Engajamento é chave. |
| 3º     | Age                  | ⚠️ Alto           | Idades extremas (jovens ou mais velhos) tendem a churn maior. |
| 4º     | Geography (Germany)  | ⚠️ Alto           | Clientes da Alemanha apresentam maior churn comparado a outros países. |
| 5º     | Balance              | ⚠️ Alto           | Baixo saldo indica menor vínculo com o banco → maior churn. |
| 6º     | Complain             | 🔥 Muito alto     | Clientes que reclamaram têm altíssimo risco de churn. |
| 7º     | Satisfaction Score   | 🔥 Muito alto     | Baixa satisfação está diretamente ligada ao churn. |
| 8º     | Tenure               | ⚠️ Médio          | Clientes novos têm maior risco de sair por falta de vínculo. |
| 9º     | CreditScore          | ⚠️ Médio          | Score baixo pode indicar instabilidade financeira → maior churn. |
| 10º    | Card Type            | ⚠️ Médio          | Clientes com cartões premium tendem a churn menor. |
| 11º    | Point Earned         | ⚠️ Médio          | Mais pontos acumulados indicam maior engajamento e menor churn. |
| 12º    | EstimatedSalary      | ⚠️ Baixo          | Salário influencia pouco isoladamente no churn. |
| 13º    | HasCrCard            | ⚠️ Baixo          | Ter cartão ajuda na retenção, mas com impacto limitado. |
| 14º    | Gender              | ⚠️ Muito baixo     | Baixo poder preditivo isolado; pode refletir padrões, mas com pouco impacto direto. |