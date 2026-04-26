# Model Card

## Índice

- [Visão Geral](#visão-geral)
- [Modelo Atual](#modelo-atual)
- [Contexto do Domínio](#contexto-do-domínio)
- [Dicionário de Dados — Bank Customer Churn](#dicionário-de-dados--bank-customer-churn)
- [Domínio das Features](#domínio-das-features)
- [Contagem e Evolução das Colunas](#contagem-e-evolução-das-colunas)
- [Interpretação das Features Mais Importantes](#interpretação-das-features-mais-importantes)
- [Como Ler a Importância de Features com Cuidado](#como-ler-a-importância-de-features-com-cuidado)
- [Limitações Conhecidas](#limitações-conhecidas)
- [Uso Recomendado](#uso-recomendado)

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

Essa tabela resume o estado operacional do champion a partir dos metadados
persistidos ao final do treino e do rastreamento associado no MLflow. Em outras
palavras, ela não é uma tabela “manual”: ela representa o contrato mínimo de
identificação do modelo atualmente usado no serving e no monitoramento.

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

Leitura dos campos:

- `experiment_name`: nome lógico do experimento/modelo dentro do projeto.
- `algorithm`: família do algoritmo treinado, útil para leitura técnica e comparação.
- `flavor`: stack de serialização e serving do modelo, aqui `sklearn`.
- `model_version`: versão funcional do champion atual.
- `feature_set`: conjunto de features esperado pelo treino e pela inferência.
- `threshold`: ponto de corte usado para converter probabilidade em classe.
- `risk_level`: classificação documental do risco do caso de uso.
- `fairness_checked`: indica se já houve auditoria formal de fairness anexada ao ciclo atual.

No contexto deste projeto, **fairness** é a análise de possíveis diferenças
indevidas de comportamento do modelo entre grupos. Em churn, isso pode incluir,
por exemplo, verificar se o modelo apresenta desempenho ou propensão de score
muito diferentes entre recortes de `Gender` ou `Geography` sem justificativa
adequada do domínio. O valor `false` aqui significa que essa auditoria ainda
não foi concluída de forma automatizada e versionada no pipeline atual.

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
Como ja documentado em [EVALUATION_MODEL_METRICS.md](EVALUATION_MODEL_METRICS.md), a
leitura de churn deve dar mais peso a recall, AUC e F1 do que à accuracy
isolada.

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

## Dicionário de Dados — Bank Customer Churn

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

Além das colunas originais, o pipeline também cria features derivadas para
capturar relações mais úteis do que os valores absolutos isolados:

| Feature derivada | Fórmula resumida | Papel de negócio |
|---|---|---|
| `BalancePerProduct` | `Balance / NumOfProducts` | Ajuda a distinguir clientes com mesmo saldo total, mas com distribuição diferente de relacionamento entre produtos. |
| `PointsPerSalary` | `Point Earned / EstimatedSalary` | Aproxima a intensidade de engajamento relativo, evitando ler pontos acumulados sem contexto econômico. |

Essa organização é importante porque o modelo não lê apenas variáveis isoladas.
Ele aprende combinações. Por exemplo:

- poucos produtos + baixa atividade pode sugerir vínculo fraco
- reclamação + baixa satisfação pode reforçar risco de saída
- saldo baixo isoladamente pode não significar muito, mas junto com baixa
  atividade e pouco tempo de casa pode indicar maior fragilidade

## Contagem e Evolução das Colunas

O dataset bruto validado na entrada possui `18` colunas. Esse total inclui
identificadores diretos, variáveis candidatas a feature, sinais com risco de
vazamento e a própria variável alvo.

Antes do treino, algumas colunas deixam de compor o espaço de atributos por
motivos de governança, LGPD e consistência estatística:

- `RowNumber`, `CustomerId` e `Surname` são removidas por minimização de dados e
  por não representarem sinal de negócio útil para generalização.
- `Exited` sai do conjunto de entrada porque é a variável alvo, usada apenas
  como rótulo supervisionado.
- `Complain` e `Satisfaction Score` são excluídas do conjunto de treino por
  risco de leakage, já que podem refletir eventos muito próximos ou posteriores
  à decisão de churn.

Ao mesmo tempo, o pipeline acrescenta `2` features derivadas:

- `BalancePerProduct`
- `PointsPerSalary`

Também há uma transformação estrutural importante: `Geography` deixa de existir
como uma única coluna textual e passa a ser representada por `2` colunas
binárias, `Geo_Germany` e `Geo_Spain`, com `France` funcionando como categoria
de referência no one-hot encoding com `drop_first=True`.

Com isso, a contagem final fica:

- `18` colunas no dado bruto
- `-3` colunas de identificação direta
- `-1` coluna alvo removida de `X`
- `-2` colunas excluídas por leakage
- `+2` features derivadas novas
- `Geography`: sai `1` coluna e entram `2` colunas codificadas

O resultado final é um conjunto com `15` features usadas no treino:
`Card Type`, `Gender`, `Geo_Germany`, `Geo_Spain`, `CreditScore`, `Age`,
`Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`,
`EstimatedSalary`, `Point Earned`, `BalancePerProduct` e `PointsPerSalary`.

## Interpretação das Features Mais Importantes

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
- uma feature derivada como `BalancePerProduct` pode passar a carregar um
  significado diferente se o portfólio de produtos mudar ou se aparecerem
  combinações pouco vistas no treino
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
