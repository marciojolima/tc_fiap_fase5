# Feature Store com Feast e Redis

## Objetivo no contexto do projeto

Esta evolução adiciona uma Feature Store ao projeto de churn bancário sem reescrever o pipeline já existente. A ideia é aproximar a arquitetura de um cenário produtivo, mantendo a execução local simples, didática e defensável em banca.

No desenho atual:

- o pipeline principal continua gerando os datasets de treino em `data/processed/`
- uma camada-ponte exporta as features já prontas para `data/feature_store/customer_features.parquet`
- o Feast usa esse parquet como offline source
- o Redis, em container, funciona como online store para serving de baixa latência

## Por que esta abordagem foi escolhida

O projeto já possui um pipeline de feature engineering centralizado e persistido em `artifacts/models/feature_pipeline.joblib`. Em vez de duplicar regras em um segundo fluxo, a integração com o Feast reaproveita esse pipeline para produzir um dataset offline compatível com feature store.

Isso atende a dois objetivos importantes do Datathon:

- evita duplicidade de lógica de transformação
- mantém coerência entre treino e serving

## Offline store x Online store

### Offline store

No contexto deste projeto, a camada offline da Feature Store é o parquet gerado em `data/feature_store/customer_features.parquet`.

Esse dataset contém:

- `customer_id` como chave da entidade
- `event_timestamp` e `created_timestamp`
- as features já transformadas e alinhadas com o modelo atual

O Feast usa essa fonte para registrar as definições e materializar dados para a camada online.

### Online store

A camada online é o Redis local, executado via `docker compose`.

O Redis armazena apenas o estado mais recente das features materializadas. Isso é suficiente para a demo de serving online e reforça a narrativa de produção: treino e histórico ficam na camada offline; leitura de baixa latência fica na camada online.

## Features expostas na Feature Store

Foi decidido publicar na Feature Store as features já usadas pelo modelo atual:

- `CreditScore`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Point Earned`
- `BalancePerProduct`
- `PointsPerSalary`
- `Geo_Germany`
- `Geo_Spain`
- `Gender`
- `Card Type`

Justificativa:

- são exatamente as features consumidas pelo modelo hoje
- já passaram pelo pipeline oficial de transformação
- evitam expor na online store colunas que não entram na inferência, como target, leakage e identificadores diretos

Observação importante:

- `Gender` e `Card Type` ficam armazenadas em formato numérico porque o pipeline atual aplica `OrdinalEncoder`
- `Geo_Germany` e `Geo_Spain` já representam a versão one-hot de `Geography`

Ou seja, a Feature Store publica um conjunto de atributos pronto para inferência, e não a cópia literal das colunas brutas.

## Materialização incremental

O fluxo foi preparado para usar a materialização incremental nativa do Feast. Isso evita o anti-padrão de limpar toda a store online e recarregar tudo do zero.

No dataset acadêmico de churn não existe um timestamp operacional real. Por isso, a camada-ponte cria um `event_timestamp` determinístico e estável, apenas para permitir:

- uso correto do Feast
- materialização incremental
- demonstração arquitetural coerente em ambiente local

Essa decisão é uma adaptação didática, explicitamente registrada, e não pretende simular um CDC real de produção.

## Fluxo local recomendado

### 1. Instalar dependências

```bash
poetry install
```

### 2. Subir o Redis

```bash
docker compose up -d redis
```

### 3. Gerar features do pipeline principal

```bash
poetry run python -m src.features.feature_engineering
```

### 4. Exportar a camada offline do Feast

```bash
poetry run python -m src.feast_ops.export
```

Se preferir reaproveitar o DVC:

```bash
poetry run dvc repro export_feature_store
```

### 5. Aplicar as definições do Feast

```bash
poetry run feast -c feature_store apply
```

### 6. Materializar para o Redis

```bash
poetry run feast -c feature_store materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

### 7. Ler features online por `customer_id`

```bash
poetry run python -m src.feast_ops.demo --customer-id 15634602
```

## Integração com a narrativa MLOps

Esta evolução se conecta ao restante da plataforma desta forma:

- `DVC`: rastreia o artefato offline exportado da feature store como parte do pipeline local
- `MLflow`: continua sendo o tracking de experimentos e lineage de treino; a feature store complementa o serving
- `Feature engineering`: segue centralizado no pipeline já existente, sem reimplementação paralela
- `Serving`: passa a ter uma rota clara para futura leitura online de features antes da predição
- `Docker Compose`: ganha um Redis local simples, suficiente para demonstração

## Limitações assumidas

- o dataset de churn é estático, então o `event_timestamp` é sintético
- o fluxo atual demonstra batch-to-online materialization, não streaming
- não há autenticação nem TLS no Redis local, por escolha deliberada de simplicidade
- a API FastAPI atual ainda não consulta o Feast em produção; a demo foi entregue por script utilitário para não acoplar mudanças maiores agora

## Próximos passos naturais

- integrar a leitura online do Feast à camada de serving
- separar feature services por versão de modelo
- substituir timestamp sintético por data operacional real, caso o dataset evolua
- adicionar testes de integração específicos para `apply`, materialização e leitura online
