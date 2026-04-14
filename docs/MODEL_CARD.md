# Model Card

## Visão Geral

Este documento descreve o modelo tabular atualmente mantido como champion no
projeto de churn bancário. O objetivo não é só listar métricas e features, mas
explicar como essas informações devem ser lidas no contexto de negócio.

No problema de churn, a pergunta central é:

- quais clientes apresentam maior risco de encerrar relacionamento com o banco?

Esse tipo de modelo é útil para apoiar:

- campanhas de retenção
- priorização de atendimento
- leitura de perfis com maior propensão de saída

Ao mesmo tempo, ele não substitui análise humana. O modelo organiza sinais de
risco a partir dos dados históricos, mas pode reagir mal a padrões novos, raros
ou fora do domínio observado no treinamento.

## Modelo Atual

As informações abaixo refletem o modelo champion hoje materializado em
`artifacts/models/model_current.pkl` e no sidecar
`artifacts/models/model_current_metadata.json`.

| Campo | Valor atual |
|---|---|
| `experiment_name` | `random_forest_current` |
| `algorithm` | `random_forest` |
| `flavor` | `sklearn` |
| `model_version` | `0.2.0` |
| `feature_set` | `processed_v1` |
| `threshold` | `0.5` |
| `risk_level` | `high` |
| `fairness_checked` | `false` |

### Métricas atuais do champion

| Métrica | Valor |
|---|---|
| AUC | `0.8720` |
| F1 | `0.6416` |
| Precision | `0.6340` |
| Recall | `0.6495` |
| Accuracy | `0.8520` |

Essas métricas mostram um modelo equilibrado para a trilha tabular do projeto,
com boa separação de risco e comportamento razoável entre precision e recall.
Como já documentado em `docs/EVALUATION_METRICS.md`, a leitura de churn deve
dar mais peso a recall, AUC e F1 do que à accuracy isolada.

## Contexto do Domínio

No conjunto de churn bancário, a variável alvo é `Exited`, onde:

- `1` significa que o cliente saiu
- `0` significa que o cliente permaneceu

O domínio do problema mistura sinais de:

- relacionamento com o banco
- engajamento operacional
- perfil financeiro
- características cadastrais

Isso significa que o modelo não está tentando “adivinhar intenção” de forma
abstrata. Ele aprende padrões históricos associados à saída de clientes com
base em variáveis observáveis no momento do treinamento.

Em termos de negócio, a leitura das features deve sempre considerar que churn é
um fenômeno multifatorial. Um cliente pode sair por preço, experiência,
concorrência, mudança de país, insatisfação ou baixa utilização dos produtos.
As variáveis do dataset são aproximações estruturadas desse contexto.

## 📋 Dicionário de Dados — Bank Customer Churn

Antes de olhar importâncias e interpretações, vale registrar o papel de cada
campo no domínio. A tabela abaixo ajuda a conectar o nome técnico da feature ao
significado de negócio que ela representa no problema de churn.

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

## Domínio das Features

Para facilitar a interpretação, as features podem ser agrupadas por papel no
negócio:

- Perfil financeiro: `CreditScore`, `Balance`, `EstimatedSalary`
- Relacionamento com o banco: `Tenure`, `NumOfProducts`, `HasCrCard`,
  `Card Type`, `Point Earned`
- Engajamento: `IsActiveMember`
- Perfil cadastral: `Geography`, `Gender`, `Age`
- Sinais de atrito ou experiência: `Complain`, `Satisfaction Score`

Essa organização é importante porque o modelo não lê apenas variáveis isoladas.
Ele aprende combinações. Por exemplo:

- poucos produtos + baixa atividade pode sugerir vínculo fraco
- reclamação + baixa satisfação pode reforçar risco de saída
- saldo baixo isoladamente pode não significar muito, mas junto com baixa
  atividade e pouco tempo de casa pode indicar maior fragilidade

## 🔍 Interpretação das Features Mais Importantes

A tabela abaixo é uma leitura de negócio das features mais relevantes em um
cenário típico de churn. Ela ajuda a explicar por que certas variáveis tendem a
ganhar destaque em modelos tabulares desse domínio.

Importante: a ordem de importância não deve ser lida como causalidade. Ela
resume o quanto uma feature ajudou o modelo a separar classes no histórico
observado. Isso não significa, por si só, que a feature “causa churn”.

### Features por importância

| Posição | Feature | Impacto no churn | Interpretação de negócio |
|---|---|---|---|
| 1º | NumOfProducts | Muito alto | Clientes com poucos produtos tendem a ter vínculo mais fraco. Estratégias de cross-sell podem aumentar retenção. |
| 2º | IsActiveMember | Muito alto | Clientes inativos costumam apresentar maior risco. Engajamento operacional é um sinal importante de permanência. |
| 3º | Age | Alto | Faixas etárias diferentes podem reagir de forma distinta a preço, experiência digital e relacionamento com o banco. |
| 4º | Geography (Germany) | Alto | Diferenças geográficas podem refletir comportamento de mercado, oferta, concorrência ou composição da base. |
| 5º | Balance | Alto | Baixo saldo pode indicar menor vínculo financeiro com a instituição. |
| 6º | Complain | Muito alto | Reclamações tendem a sinalizar atrito no relacionamento. |
| 7º | Satisfaction Score | Muito alto | Baixa satisfação está alinhada ao aumento de risco de saída. |
| 8º | Tenure | Médio | Clientes mais novos podem ter menos vínculo e menor barreira para saída. |
| 9º | CreditScore | Médio | Pode refletir estabilidade financeira e perfil de relacionamento. |
| 10º | Card Type | Médio | Produtos premium podem estar associados a maior retenção, mas isso depende do recorte da base. |
| 11º | Point Earned | Médio | Acúmulo de pontos pode indicar uso recorrente e maior engajamento. |
| 12º | EstimatedSalary | Baixo | Sozinho, costuma ter poder explicativo mais limitado. |
| 13º | HasCrCard | Baixo | Ajuda no vínculo, mas nem sempre muda fortemente a predição isoladamente. |
| 14º | Gender | Muito baixo | Baixo poder preditivo isolado; pode refletir padrões da base, mas não deve ser lido como explicação causal. |

## Como Ler a Importância de Features com Cuidado

A importância de features é útil para explicar o comportamento geral do modelo,
mas não deve ser confundida com regra estável de negócio. Há pelo menos quatro
cuidados importantes:

1. A importância mede utilidade estatística no treino, não causalidade.
2. O modelo pode reagir mal a combinações raras que não apareceram com volume
   suficiente no histórico.
3. A interpretação de negócio pode deixar de valer quando o dado observado sai
   do domínio treinado.
4. Mudanças estruturais em produção podem quebrar a leitura intuitiva das
   importâncias sem mudar o ranking histórico das features.

Exemplos práticos:

- um cliente com `NumOfProducts` muito alto ou em faixa pouco observada no
  treino pode gerar uma resposta que não reflete a interpretação clássica de
  “mais produtos = mais vínculo”
- um novo país surgindo em produção, ou uma categoria fora das esperadas em
  `Geography`, pode confundir a generalização do modelo
- mudanças de portfólio, produto ou comportamento da base podem tornar a
  interpretação histórica menos confiável

Por isso, este projeto combina:

- interpretação de features
- monitoramento de drift
- retreino auditável
- comparação champion-challenger

Essa combinação é mais segura do que tratar importância de feature como verdade
imutável do negócio.

## Limitações Conhecidas

- O modelo atual é tabular e supervisionado; ele não captura causalidade real.
- O conjunto de treino tem domínio conhecido e restrito; novas categorias ou
  padrões podem reduzir confiabilidade.
- `Geography` é mantida sob governança explícita e exige leitura cuidadosa.
- `fairness_checked` ainda está como `false`, então não há auditoria formal de
  viés concluída no ciclo atual.
- A explicabilidade atual é interpretativa e documental; ainda não há camada
  formal de explicabilidade local por predição.

## Uso Recomendado

Este modelo deve ser usado como suporte à decisão para:

- priorização de clientes para retenção
- leitura comparativa de risco
- análise operacional do comportamento da carteira

Ele não deve ser usado como mecanismo único e automático para decisões
irreversíveis. O uso mais maduro continua sendo:

- combinar score do modelo com contexto de negócio
- monitorar drift e degradação
- reavaliar o champion quando houver mudança relevante nos dados
