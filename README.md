# TC FIAP Fase 5

Projeto integrador da Fase 05 do curso MLET da FIAP, estruturado como uma entrega de Datathon com foco em churn bancario, MLOps, observabilidade, governanca e evolucao gradual para componentes com LLMs e agentes.

## Grupo

Turma 6MLET - FIAP

- Luca Poiti - RM365678
- Gabriel Jordan - RM365606
- Luciana Ferreira - RM366171
- Marcio Lima - RM365919

## Objetivo do Projeto

Construir uma solucao de predicao de churn com base em praticas de engenharia de machine learning esperadas no Datathon:

- pipeline de dados reproduzivel e versionado
- engenharia de features com validacao
- treino rastreavel com MLflow
- API de inferencia com FastAPI
- cenarios de negocio para validacao do comportamento do modelo
- monitoramento de drift com Evidently e PSI
- gatilho auditavel de retreino
- documentacao tecnica e de governanca em evolucao

## Visao Geral do Estado Atual

Este README reflete a documentacao e o desenho implementado do projeto como um
todo. Em particular, o fluxo de retreino ja existe e funciona de forma batch e
auditavel, mesmo sem um endpoint HTTP dedicado de `retrain` no serving.

Hoje o projeto entrega com boa consistencia:

- pipeline de dados, features e treino
- experimento rastreado com MLflow
- serving via FastAPI
- testes automatizados para partes importantes do fluxo
- monitoramento batch de drift
- fluxo auditavel de retreino com decisao de promocao champion-challenger
- observabilidade inicial com Prometheus e Grafana

Ao mesmo tempo, ainda faltam varias entregas para aderencia mais forte ao guia do Datathon e ao que foi enfatizado na live, especialmente nas frentes de LLMOps, agentes, guardrails efetivos, fairness e maturidade de deploy.

## Funcionalidades Implementadas

### 1. Dados, features e treino

- versionamento de dados com DVC
- organizacao das camadas `raw`, `interim` e `processed`
- pipeline de engenharia de features em `src/features/feature_engineering.py`
- componentes de pipeline em `src/features/pipeline_components.py`
- validacao de schema com Pandera em `src/features/schema_validation.py`
- separacao treino/teste antes do ajuste do pipeline de transformacao
- persistencia de datasets processados e pipeline de features
- multiplos experimentos configurados em YAML
- treino e rastreabilidade com MLflow em `src/models/train.py`
- artefatos de modelo e metadados versionados em `artifacts/models/`

### 2. Serving e inferencia

- API FastAPI em `src/serving/app.py`
- endpoint `/predict`
- schemas de entrada e saida com Pydantic
- reutilizacao do mesmo pipeline de features entre treino e inferencia
- testes de API e serving em `tests/test_api.py` e `tests/test_serving.py`

### 3. Analise de cenarios

- suite versionada de cenarios em `configs/scenario_analysis/inference_cases.yaml`
- execucao de cenarios em `src/scenario_analysis/inference_cases.py`
- logging de execucoes no MLflow
- documentacao de apoio em `docs/SCENARIO_ANALYSIS.md`

### 4. Monitoramento, drift e retreino

- registro de inferencias para monitoramento em `src/monitoring/inference_log.py`
- metricas operacionais para serving em `src/monitoring/metrics.py`
- deteccao batch de drift com Evidently e PSI em `src/monitoring/drift.py`
- geracao de lotes sinteticos para simulacao de drift em `src/scenario_analysis/synthetic_drifts.py`
- relatorios HTML e arquivos JSON de drift em `artifacts/monitoring/drift/`
- fluxo de retreino batch e auditavel em `src/models/retraining.py`
- decisao de promocao champion-challenger em `src/models/promotion.py`
- artefatos auditaveis em `artifacts/monitoring/retraining/`

Observacao importante:

- o retreino ja acontece no projeto por meio do fluxo de monitoramento e do executor dedicado de retreino
- hoje isso nao depende de um endpoint HTTP especifico de `retrain` no serving
- o serving segue focado em inferencia; o retreino atual e disparado pelo fluxo batch de drift

### 5. Observabilidade e operacao

- configuracao de Prometheus em `configs/observability/prometheus.yml`
- provisionamento de Grafana em `configs/observability/grafana/`
- subida local do stack de observabilidade com `docker compose`
- workflow de CI em `.github/workflows/ci.yml`
- tarefas de automacao com Taskipy em `pyproject.toml`

### 6. Documentacao e governanca

- documentacao de drift em `docs/DRIFT_MONITORING.md`
- documentacao de versionamento de modelos em `docs/MODEL_VERSIONING.md`
- model card em `docs/MODEL_CARD.md`
- plano LGPD em `docs/LGPD_PLAN.md`
- metricas de avaliacao em `docs/EVALUATION_METRICS.md`
- dashboard operacional em `docs/OPERATIONS_DASHBOARD.md`

## Aderencia Atual aos Requisitos

Com base no cruzamento entre `REQUISITOS_DATATHON.md`, `REQUISITOS_DATATHON_LIVE_EXPLANATION.md` e o codigo do repositorio, a situacao atual pode ser resumida assim:

### Entregue ou bem encaminhado

- estrutura de projeto Python organizada
- uso de DVC para dados
- MLflow para experiment tracking
- FastAPI para serving
- testes automatizados com `pytest`
- monitoramento de drift implementado
- evidencias auditaveis de retreino e decisao de promocao
- observabilidade inicial com Prometheus e Grafana
- documentacao tecnica relevante para operacao e governanca

### Parcial

- CI/CD: existe workflow, mas ainda nao representa uma esteira completa com staging e deploy
- model management: ha versionamento e metadados, mas sem um registry completo com approval workflow formal
- observabilidade: a base existe, mas ainda nao cobre toda a maturidade sugerida na live
- agente, RAG e seguranca: ha estrutura de codigo e testes iniciais, mas nao caracterizam entrega completa

### Ainda nao concluido

- baseline adicional em PyTorch
- notebook de EDA no padrao esperado pela banca
- golden set formal
- agente ReAct funcional com tools de uso real
- pipeline RAG operacional
- avaliacao RAGAS e LLM-as-judge em fluxo real
- telemetria LLM operacional
- guardrails e deteccao de PII integrados ao fluxo de producao
- fairness automatizada
- deploy mais completo com ambientes e gates mais fortes

## Situacao Mais Recente do Fluxo de Monitoramento

Os artefatos atuais mostram que:

- houve deteccao de drift critico
- um retreino foi executado
- um challenger foi gerado
- a promocao foi rejeitada
- o champion atual foi mantido

Isso reforca que o projeto ja possui um ciclo auditavel de monitoramento e retreino, mesmo que ainda faltem evolucoes importantes para a maturidade final esperada.

## Como Executar

### Instalacao

Fluxo recomendado para preparar o ambiente local desde o clone do repositorio:

```bash
git clone https://github.com/marciojolima/tc_fiap_fase5.git
cd tc_fiap_fase5
poetry install
```

Depois da instalacao das dependencias Python, e importante sincronizar os dados versionados com o storage remoto configurado no DVC. Sem isso, parte importante do pipeline pode nao ter os arquivos necessarios para engenharia de features, treino, validacao e demonstracoes de monitoramento.

### Configuracao do data storage com DVC

O projeto usa DVC para versionar dados e apontar para um remote externo. No estado atual do repositorio, o remote padrao esta configurado como `datathon_remote` e utiliza Google Drive.

Exemplo de configuracao presente no projeto:

```ini
[core]
    remote = datathon_remote
['remote "datathon_remote"']
    url = gdrive://1tfIgv-9mikvC8EZVLUneDVQhQes5_Q7w
```

Em um ambiente novo, o fluxo esperado costuma ser:

```bash
poetry run dvc pull
```

Esse comando baixa do storage remoto os dados e artefatos versionados pelo DVC para o workspace local.

Se for necessario configurar ou reconfigurar o remote manualmente, um exemplo com Google Drive seria:

```bash
poetry run dvc remote add -d datathon_remote gdrive://1tfIgv-9mikvC8EZVLUneDVQhQes5_Q7w
poetry run dvc pull
```

Dependendo da maquina e da conta utilizada, o DVC pode solicitar autenticacao no Google Drive na primeira execucao. Essa etapa e importante porque, sem acesso ao storage, o repositorio pode ter a estrutura correta mas nao os dados reais necessarios para reproduzir o projeto.

### Sequencia minima recomendada

Depois de instalar dependencias e sincronizar os dados, a sequencia mais segura para explorar o projeto localmente e:

```bash
poetry run task mlfeateng
poetry run task mltrain
poetry run task serving
```

### Engenharia de features

```bash
poetry run task mlfeateng
```

Esse comando executa `python -m src.features.feature_engineering` e materializa a etapa de preparacao dos dados. Na pratica, ele:

- le a base de entrada versionada
- aplica limpeza e transformacoes
- valida schema e consistencia
- gera datasets intermediarios e processados usados nas etapas seguintes

E um bom primeiro passo quando queremos reconstruir os artefatos de dados a partir da materia-prima do projeto.

### Treino

```bash
poetry run task mltrain
```

Esse comando executa `python -m src.models.train` e roda o treino principal com a configuracao padrao do projeto. Ele costuma:

- carregar os dados processados
- aplicar o pipeline de features persistente ou configurado para treino
- treinar o modelo principal
- registrar metricas, parametros e artefatos no MLflow
- atualizar artefatos de modelo e metadados para uso posterior

### Rodar experimentos e cenarios

```bash
poetry run task mlrunall
```

Essa task e mais abrangente. Ela executa uma sequencia de treinos com configuracoes diferentes e, ao final, roda a analise de cenarios versionada. E util para demonstracao mais completa do projeto, comparacao entre experimentos e geracao de evidencias no MLflow.

### API de serving

```bash
poetry run task serving
```

Essa task executa `fastapi dev src/serving/app.py` e sobe a aplicacao de inferencia localmente. Ela deve ser usada quando queremos:

- testar o endpoint `/predict`
- validar o contrato de entrada e saida da API
- verificar se o modelo atual e o pipeline de features estao carregando corretamente
- integrar o serving com metricas e monitoramento local

Quando o servidor estiver no ar, a aplicacao FastAPI normalmente ficara acessivel na porta padrao do ambiente de desenvolvimento e podera ser inspecionada pela documentacao interativa do proprio FastAPI.

### Monitoramento de drift

```bash
poetry run task mldrift
```

Essa task executa `python -m src.monitoring.drift` e roda o fluxo batch de monitoramento de drift. Ela compara dados de referencia com dados correntes, calcula indicadores como PSI e gera os artefatos de monitoramento em `artifacts/monitoring/drift/`.

Para uma execucao demonstravel usando a base de teste atual:

```bash
poetry run task mldriftdemo
```

Essa variante e util quando queremos demonstrar o processo mesmo sem depender de trafego real acumulado. Ela usa a base de teste como insumo para uma execucao controlada do monitoramento.

### Geracao de drifts sinteticos

```bash
poetry run task mlsyntheticdrift
```

Esse comando cria lotes sinteticos com diferentes perfis de drift para validar o comportamento do monitoramento e produzir evidencias de teste e demonstracao.

### MLflow UI

```bash
poetry run task mlflow
```

Essa task sobe a interface local do MLflow em `http://127.0.0.1:5000`. Ela e importante para acompanhar:

- experimentos de treino
- metricas comparativas
- parametros utilizados
- artefatos salvos em cada execucao

Em geral, vale abrir o MLflow enquanto rodam treinos, cenarios ou retreinos, porque isso facilita a leitura da trilha de execucao do projeto.

### Observabilidade

```bash
poetry run task observability
```

Essa task executa `docker compose up -d prometheus grafana` e sobe os servicos auxiliares de observabilidade local. Ela nao sobe a API de serving sozinha; o ideal e usar essa task junto com `poetry run task serving` para observar a aplicacao em execucao.

Servicos locais:

- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

Na pratica:

- Prometheus faz o scraping das metricas expostas pela aplicacao
- Grafana permite visualizar dashboards para latencia, volume de requisicoes, erros e outros sinais operacionais

Para desligar esses servicos auxiliares:

```bash
poetry run task observability_down
```

### Testes

```bash
poetry run task test
```

Essa task executa a suite de testes com `pytest`, incluindo cobertura sobre o codigo em `src`. E uma etapa importante antes de abrir PR, consolidar uma entrega ou validar se mudancas locais quebraram algum fluxo essencial.

## Artefatos Relevantes

Os arquivos abaixo ajudam a demonstrar reprodutibilidade, rastreabilidade e operacao do projeto. Eles tambem servem como evidencia objetiva do que ja foi implementado.

| Artefato | Papel no projeto |
|---|---|
| `data/interim/cleaned.parquet` | Camada intermediaria ja tratada, usada como ponto de auditoria entre ingestao e preparacao final dos dados. |
| `data/processed/train.parquet` | Base final de treino gerada pelo pipeline de features e usada nos experimentos do modelo. |
| `data/processed/test.parquet` | Base final de teste usada para validacao e comparacao de desempenho. |
| `data/processed/feature_columns.json` | Registra a ordem e os nomes finais das features, ajudando a manter consistencia entre treino e inferencia. |
| `data/processed/schema_report.json` | Evidencia da validacao estrutural dos dados processados, reforcando a etapa de qualidade de dados. |
| `artifacts/models/feature_pipeline.joblib` | Pipeline de transformacao persistido para reutilizacao no serving, evitando divergencia entre treino e producao. |
| `artifacts/models/model_current.pkl` | Modelo champion atualmente mantido como versao principal para inferencia. |
| `artifacts/models/model_current_metadata.json` | Metadados do champion atual, incluindo informacoes de versao, configuracao e metricas relevantes. |
| `artifacts/models/challengers/` | Diretorio reservado para challengers gerados em ciclos de retreino e comparados antes de eventual promocao. |
| `data/monitoring/current/predictions.jsonl` | Log de inferencias usado como base para monitoramento posterior, principalmente nos fluxos de drift. |
| `artifacts/monitoring/drift/drift_report.html` | Relatorio HTML do Evidently para inspecao visual do comportamento das features e das distribuicoes monitoradas. |
| `artifacts/monitoring/drift/drift_metrics.json` | Consolidacao das metricas de drift, incluindo PSI por feature e resumo para automacao de decisao. |
| `artifacts/monitoring/drift/drift_status.json` | Estado mais recente do monitoramento de drift, com classificacao para apoio ao gatilho de retreino. |
| `artifacts/monitoring/drift/drift_runs.jsonl` | Historico de execucoes do monitoramento, util para trilha de auditoria e acompanhamento temporal. |
| `artifacts/monitoring/retraining/retrain_request.json` | Registro do pedido de retreino, com motivacao e contexto do disparo do processo. |
| `artifacts/monitoring/retraining/retrain_run.json` | Resultado consolidado da execucao do retreino, incluindo status, motivo, metricas e decisao final. |
| `artifacts/monitoring/retraining/promotion_decision.json` | Decisao champion-challenger com regra de promocao explicita e deltas de metricas entre os modelos comparados. |
| `artifacts/monitoring/retraining/generated_configs/` | Configuracoes geradas automaticamente para retreinos auditaveis e reproduziveis. |
| `configs/scenario_analysis/inference_cases.yaml` | Suite versionada de cenarios de inferencia usada para validar comportamento do modelo em casos de negocio. |
| `artifacts/scenario_analysis/drift/*.jsonl` | Lotes sinteticos construidos para simular diferentes perfis de drift e testar o fluxo de monitoramento. |
| `artifacts/scenario_analysis/drift/*_report.html` | Relatorios HTML dos cenarios sinteticos, usados para demonstracao e validacao do processo de drift. |

## Estrutura Resumida

```text
tc_fiap_fase5/
├── artifacts/
├── configs/
├── data/
├── docs/
├── evaluation/
├── scripts/
├── src/
│   ├── agent/
│   ├── common/
│   ├── features/
│   ├── inference/
│   ├── models/
│   ├── monitoring/
│   ├── scenario_analysis/
│   ├── security/
│   └── serving/
├── tests/
├── README.md
├── STATUS_ATUAL_PROJETO.md
└── REQUISITOS_DATATHON.md
```

## Documentos de Apoio

- [DRIFT_MONITORING.md](docs/DRIFT_MONITORING.md): estrategia e fluxo de monitoramento
- [OPERATIONS_DASHBOARD.md](docs/OPERATIONS_DASHBOARD.md): dashboard operacional
- [MODEL_VERSIONING.md](docs/MODEL_VERSIONING.md): versionamento e metadados de modelos
- [MODEL_CARD.md](docs/MODEL_CARD.md): resumo do modelo, contexto e limitacoes
- [SCENARIO_ANALYSIS.md](docs/SCENARIO_ANALYSIS.md): descricao da analise de cenarios e dos casos de inferencia
- [EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md): metricas usadas para avaliacao dos modelos
- [LGPD_PLAN.md](docs/LGPD_PLAN.md): direcionamento inicial para aspectos de privacidade e LGPD

## Observacao Importante

Este repositorio nao pretende afirmar que todos os requisitos do Datathon ja foram atendidos. O foco deste `README.md` e registrar com transparencia o que foi implementado ate agora e deixar claro que ainda existem frentes relevantes em aberto.
