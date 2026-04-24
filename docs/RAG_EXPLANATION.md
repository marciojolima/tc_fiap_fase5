# RAG Explanation

## Indice

- [Objetivo](#objetivo)
- [Fontes de Contexto](#fontes-de-contexto)
- [Pipeline de Inicializacao](#pipeline-de-inicializacao)
- [Embeddings e Busca Vetorial](#embeddings-e-busca-vetorial)
- [Cache Persistido](#cache-persistido)
- [Uso de Memoria](#uso-de-memoria)
- [Dashboard](#dashboard)
- [Endpoint `/llm/chat`](#endpoint-llmchat)
- [Perguntas Recomendadas para Smoke Test](#perguntas-recomendadas-para-smoke-test)
- [Exemplos por Tool](#exemplos-por-tool)
- [Resumo de Respostas Esperadas](#resumo-de-respostas-esperadas)
- [Dicas de Operacao](#dicas-de-operacao)

## Objetivo

Este documento descreve como a trilha de RAG do projeto funciona hoje, como o
indice e carregado, como o cache acelera reinicios e como interpretar o uso do
endpoint `POST /llm/chat`.

O desenho foi atualizado para ficar aderente ao que a Datathon cobra nas etapas
de:

- agente com tools de negocio
- RAG retornando contexto relevante do repositorio
- observabilidade da trilha de LLM

As referencias principais de requisito continuam sendo:

- [README.md](../README.md)
- [STATUS_ATUAL_PROJETO.md](../STATUS_ATUAL_PROJETO.md)
- [STATUS_CHECK_LIST.md](../STATUS_CHECK_LIST.md)

## Fontes de Contexto

O corpus do RAG e montado automaticamente no startup do serving com estas
fontes:

- [README.md](../README.md)
- todos os arquivos [`docs/**/*.md`](./)
- um conjunto pequeno de JSON relevantes definidos em codigo

Os JSON continuam hardcoded por design para manter governanca e evitar que
artefatos temporarios, dashboards e saidas repetitivas do MLflow entrem no
indice por engano.

Hoje os JSON mais importantes indexados sao:

- `data/processed/feature_columns.json`
- `data/processed/schema_report.json`
- `data/feature_store/export_metadata.json`
- `artifacts/monitoring/drift/drift_status.json`
- `artifacts/monitoring/drift/drift_metrics.json`
- `artifacts/monitoring/retraining/retrain_request.json`
- `artifacts/monitoring/retraining/retrain_run.json`
- `artifacts/monitoring/retraining/promotion_decision.json`

Arquivos vazios sao ignorados automaticamente no processo de descoberta.

Arquivos de implementacao relacionados:

- [src/agent/rag_pipeline.py](../src/agent/rag_pipeline.py)
- [src/agent/tools.py](../src/agent/tools.py)
- [src/agent/react_agent.py](../src/agent/react_agent.py)
- [src/serving/app.py](../src/serving/app.py)
- [src/serving/llm_routes.py](../src/serving/llm_routes.py)

## Pipeline de Inicializacao

Quando a API FastAPI sobe, o `lifespan` do serving inicializa o RAG antes do
servico ficar pronto para consumo.

O pipeline executa estas etapas:

1. descoberta das fontes
2. leitura e normalizacao de texto
3. chunking com metadados de origem
4. carga do modelo de embeddings
5. uso de cache local quando o manifesto de fontes nao mudou
6. se necessario, geracao de embeddings para todos os chunks
7. montagem do indice vetorial em memoria
8. publicacao de metricas Prometheus e escrita do historico em disco

Os chunks sao mantidos com:

- caminho da fonte
- tipo da fonte
- `chunk_id`
- texto do chunk
- tamanho em caracteres

## Embeddings e Busca Vetorial

O runtime do RAG usa embeddings locais com `sentence-transformers`. O modelo
padrao atual e configurado em
[configs/pipeline_global_config.yaml](../configs/pipeline_global_config.yaml).

O fluxo de consulta faz:

1. embedding da pergunta do usuario
2. busca por similaridade vetorial no indice em memoria
3. rerank lexical leve nos melhores candidatos
4. retorno do `top_k` com fonte e identificacao do chunk

Isso significa que a busca ja nao depende mais apenas de overlap lexical bruto.

## Cache Persistido

O RAG usa cache em disco para evitar reembeddar o corpus inteiro sempre que a
stack sobe.

Arquivos gerados:

- `artifacts/rag/cache/manifest.json`
- `artifacts/rag/cache/index.joblib`
- `artifacts/rag/index_build_history.jsonl`

### Como o cache funciona

No startup, o projeto monta um manifesto das fontes com:

- caminho relativo
- tamanho em bytes
- `mtime_ns`
- hash SHA-256

Se esse manifesto for igual ao ultimo manifesto salvo em cache, o indice vetorial
e carregado do `joblib`.

Se qualquer arquivo mudar, se um novo `.md` for adicionado em `docs/`, ou se um
JSON hardcoded for alterado, o indice e reconstruido automaticamente no proximo
startup.

Isso atende ao objetivo de manutencao simples: basta adicionar o arquivo ao
repositorio e reiniciar a stack para o corpus ser atualizado.

## Uso de Memoria

O RAG usa memoria principalmente em tres componentes:

- modelo de embeddings carregado no processo do serving
- textos dos chunks em memoria
- matriz de embeddings normalizada usada na busca vetorial

Nao ha banco vetorial separado neste desenho atual. O indice fica no proprio
processo Python do serving.

As metricas expostas incluem:

- quantidade de arquivos do corpus
- quantidade de chunks
- bytes das fontes lidas
- bytes da matriz de embeddings
- estimativa de memoria do indice
- delta de RSS do processo durante a inicializacao
- tempo total e tempo por etapa da inicializacao
- indicacao se a ultima inicializacao usou cache
- latencia da busca vetorial

## Dashboard

Foi adicionado um dashboard dedicado no Grafana:

- `RAG Operational Dashboard`

Arquivo relacionado:

- [configs/observability/grafana/dashboards/rag_operational_dashboard.json](../configs/observability/grafana/dashboards/rag_operational_dashboard.json)

Ele mostra:

- ultima inicializacao com cache ou sem cache
- tempo total de inicializacao
- tempo por etapa
- arquivos e chunks indexados
- bytes do corpus e do indice
- delta de RSS do processo
- latencia P95 da busca vetorial

Para comparar primeira e segunda inicializacao:

- na primeira execucao, espere `cache_hit = 0`
- na segunda execucao, se nenhuma fonte mudou, espere `cache_hit = 1`

O historico detalhado dessas inicializacoes tambem fica salvo em:

- `artifacts/rag/index_build_history.jsonl`

## Endpoint `/llm/chat`

Payload minimo:

```json
{
  "message": "Quais rotas HTTP o projeto expoe especificamente para o assistente LLM e diagnostico do provider LLM?",
  "include_trace": true
}
```

O que esperar na resposta:

- `answer`: resposta final do agente
- `used_tools`: lista de tools usadas
- `trace`: trilha ReAct quando `include_trace=true`

Rotas e schemas relacionados:

- [src/serving/llm_routes.py](../src/serving/llm_routes.py)
- [src/serving/schemas.py](../src/serving/schemas.py)

## Perguntas Recomendadas para Smoke Test

1. Pergunta:
```text
Quais rotas HTTP o projeto expoe especificamente para o assistente LLM e diagnostico do provider LLM?
```
   Resposta esperada:
```text
mencao a /llm/health, /llm/status e /llm/chat
```

2. Pergunta:
```text
Cite pelo menos tres tools do agente ReAct deste projeto.
```
   Resposta esperada:
```text
rag_search, predict_churn, drift_status, scenario_prediction
```

3. Pergunta:
```text
Em linhas gerais, como o RAG do projeto obtem contexto para uma pergunta?
```
   Resposta esperada:
```text
descoberta sobre README.md, docs/**/*.md, JSON relevantes, embeddings, busca vetorial em memoria e retorno com fonte
```

4. Pergunta:
```text
O que o monitoramento de drift busca identificar neste repositorio?
```
   Resposta esperada:
```text
mencao a drift, PSI, artefatos de monitoramento e apoio a retreino
```

## Exemplos por Tool

Os exemplos abaixo ajudam a explorar o agente manualmente no Swagger ou em
chamadas diretas para `POST /llm/chat`.

### Tool `rag_search`

Use estas perguntas quando quiser validar se o agente esta encontrando contexto
documental no repositorio e respondendo com base nas fontes indexadas.

1. Pergunta:
```text
Quais rotas HTTP o projeto expoe especificamente para o assistente LLM e diagnostico do provider LLM?
```
   Resposta esperada:
```text
o agente deve mencionar /llm/health, /llm/status e /llm/chat
```

2. Pergunta:
```text
Como o RAG do projeto obtem contexto para responder perguntas sobre o repositorio?
```
   Resposta esperada:
```text
o agente deve mencionar README.md, docs/**/*.md, JSON relevantes, embeddings, busca vetorial em memoria e retorno com fonte
```

3. Pergunta:
```text
O que o monitoramento de drift busca identificar neste repositorio?
```
   Resposta esperada:
```text
o agente deve mencionar data drift, prediction drift, PSI, artefatos de monitoramento e apoio a retreino
```

4. Pergunta:
```text
Quais artefatos de retreino e promocao entram no corpus do RAG?
```
   Resposta esperada:
```text
o agente deve citar arquivos como retrain_request.json, retrain_run.json e promotion_decision.json
```

### Tool `predict_churn`

Use estas perguntas para testar se o agente consegue acionar a predicao de
churn com base em dados de cliente.

1. Pergunta:
```text
Considere um cliente de 42 anos, da Alemanha, com saldo alto, dois produtos, inativo e com 650 de credit score. Qual seria a previsao de churn?
```
   Resposta esperada:
```text
o agente deve acionar predict_churn e responder com probabilidade de churn e classe prevista, deixando claro que a resposta depende do payload enviado
```

2. Pergunta:
```text
Com base neste perfil de cliente bancario, o modelo indicaria maior risco de evasao?
```
   Resposta esperada:
```text
o agente deve usar predict_churn e devolver uma conclusao objetiva sobre risco baixo ou alto, acompanhada da probabilidade prevista
```

### Tool `scenario_prediction`

Use estas perguntas para testar simulacoes e comparar cenarios de negocio.

1. Pergunta:
```text
Se o mesmo cliente passar a ser ativo e aumentar o numero de produtos, como muda a previsao de churn?
```
   Resposta esperada:
```text
o agente deve usar scenario_prediction para comparar cenarios e explicar se o risco cai ou sobe entre o estado atual e o estado ajustado
```

2. Pergunta:
```text
Simule dois cenarios para este cliente: um com saldo maior e inatividade, outro com mais produtos e atividade. Qual deles parece melhor para retencao?
```
   Resposta esperada:
```text
o agente deve comparar os cenarios, mostrar as probabilidades previstas e concluir qual combinacao parece mais favoravel para reduzir churn
```

### Tool `drift_status`

Use estas perguntas para explorar a saude operacional do modelo em producao.

1. Pergunta:
```text
Como esta a saude atual do modelo em relacao a drift?
```
   Resposta esperada:
```text
o agente deve usar drift_status e resumir o estado atual com mencao a status, PSI, possiveis alertas e impacto operacional
```

2. Pergunta:
```text
O monitoramento sugere necessidade de retreino neste momento?
```
   Resposta esperada:
```text
o agente deve usar drift_status e responder se ha sinal de retreino, mencionando o status atual e o racional associado ao drift observado
```

## Resumo de Respostas Esperadas

Se o agente estiver funcionando como esperado, o comportamento geral deve ser:

- perguntas documentais acionam `rag_search` e trazem respostas ancoradas em
  arquivos do repositorio
- perguntas de predicao acionam `predict_churn` e retornam probabilidade e
  classe prevista
- perguntas de simulacao acionam `scenario_prediction` e comparam cenarios
- perguntas de saude operacional acionam `drift_status` e resumem drift e
  possivel necessidade de retreino

## Dicas de Operacao

- Se o startup ficar mais lento na primeira execucao, isso e esperado por causa
  do carregamento do modelo de embeddings e da geracao inicial do indice.
- Se a segunda execucao nao usar cache, consulte `manifest.json` e o historico
  em `index_build_history.jsonl` para verificar se alguma fonte mudou.
- Se a resposta do agente ficar generica, use `include_trace=true` para confirmar
  se `rag_search` foi realmente acionada.
- O endpoint `GET /llm/status` agora tambem ajuda no diagnostico do estado do
  RAG, alem do estado do `llm_provider` ativo.
