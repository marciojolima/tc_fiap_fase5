# RAG Explanation

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

- `REQUISITOS_DATATHON.md`
- `REQUISITOS_DATATHON_LIVE_EXPLANATION.md`

## Fontes de Contexto

O corpus do RAG e montado automaticamente no startup do serving com estas
fontes:

- `README.md`
- todos os arquivos `docs/**/*.md`
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
padrao atual e configurado em `configs/pipeline_global_config.yaml`.

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
  "message": "Quais rotas HTTP o projeto expoe especificamente para o assistente LLM e diagnostico do Ollama?",
  "include_trace": true
}
```

### O que esperar na resposta

- `answer`: resposta final do agente
- `used_tools`: lista de tools usadas
- `trace`: trilha ReAct quando `include_trace=true`

### Perguntas recomendadas para smoke test

1. `Quais rotas HTTP o projeto expoe especificamente para o assistente LLM e diagnostico do Ollama?`
   Resposta esperada:
   mencao a `/llm/health`, `/llm/status` e `/llm/chat`

2. `Cite pelo menos tres tools do agente ReAct deste projeto.`
   Resposta esperada:
   `rag_search`, `predict_churn`, `drift_status`, `scenario_prediction`

3. `Em linhas gerais, como o RAG do projeto obtem contexto para uma pergunta?`
   Resposta esperada:
   descoberta sobre `README.md`, `docs/**/*.md`, JSON relevantes, embeddings,
   busca vetorial em memoria e retorno com fonte

4. `O que o monitoramento de drift busca identificar neste repositorio?`
   Resposta esperada:
   mencao a drift, PSI, artefatos de monitoramento e apoio a retreino

## Dicas de Operacao

- Se o startup ficar mais lento na primeira execucao, isso e esperado por causa
  do carregamento do modelo de embeddings e da geracao inicial do indice.
- Se a segunda execucao nao usar cache, consulte `manifest.json` e o historico
  em `index_build_history.jsonl` para verificar se alguma fonte mudou.
- Se a resposta do agente ficar generica, use `include_trace=true` para confirmar
  se `rag_search` foi realmente acionada.
- O endpoint `GET /llm/status` agora tambem ajuda no diagnostico do estado do
  RAG, alem do estado do Ollama.
