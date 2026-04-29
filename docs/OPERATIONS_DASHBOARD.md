# Dashboard Operacional

## Índice

- [Como interpretar o dashboard](#como-interpretar-o-dashboard)
- [Como validar rapidamente sem depender do Grafana](#como-validar-rapidamente-sem-depender-do-grafana)
- [Como subir localmente](#como-subir-localmente)
- [Acessos](#acessos)
- [Observação](#observação)

O projeto expõe métricas Prometheus no serving por meio do endpoint `GET /metrics`.

As métricas operacionais iniciais são:

- `churn_serving_predict_latency_seconds`: latência do endpoint `POST /predict`
- `churn_serving_predict_requests_total`: total de requisições por método e status HTTP
- `churn_serving_predict_requests_in_progress`: requisições de predição em andamento
- `churn_serving_llm_chat_latency_seconds`: latência do endpoint `POST /llm/chat`
- `churn_serving_llm_chat_provider_latency_seconds`: latencia da chamada ao `llm_provider` dentro do fluxo `/llm/chat` (com label `provider`)
- `churn_serving_llm_chat_requests_total`: total de requisições por método e status HTTP do chat
- `churn_serving_llm_chat_requests_in_progress`: requisições de chat em andamento

## Como interpretar o dashboard

O painel operacional foi pensado para responder perguntas simples e importantes
sobre o serving:

- o endpoint responde rapido?
- o volume de chamadas esta aumentando ou caindo?
- a API esta falhando?
- ha acúmulo de requisicoes em processamento?

### Latencia P95 /predict

O painel de latencia mostra o percentil 95 das requisicoes do endpoint
`POST /predict`.

Em termos praticos, `P95` significa:

- 95% das requisicoes terminaram em um tempo igual ou menor que esse valor
- 5% das requisicoes mais lentas ficaram acima dele

Isso e mais util do que olhar so para a media, porque a media pode esconder
algumas chamadas lentas. Em producao, a experiencia do usuario costuma ser mais
afetada por caudas de latencia do que pelo valor medio.

Exemplo:

- `P95 = 97.5 ms` significa que 95% das chamadas terminaram em ate
  aproximadamente `0.0975` segundo

Para este projeto local, esse valor e bom. Ele indica que o endpoint de
predicao esta respondendo em menos de `100 ms` na maior parte das chamadas.

Mesmo assim, "bom" e "ruim" dependem do contexto:

- em uma demo local ou API simples, algo abaixo de `100 ms` tende a ser muito bom
- entre `100 ms` e `500 ms` ainda pode ser aceitavel para varios cenarios
- acima de `1 s` com frequencia ja merece investigacao

Em um ambiente real, a avaliacao deve considerar:

- volume simultaneo de chamadas
- tamanho e complexidade do modelo
- tempo de carregamento de artefatos
- infraestrutura disponivel
- SLA esperado pelo negocio

### Volume de Requisicoes /predict

Esse painel mostra a taxa de chamadas recebidas pelo endpoint `POST /predict`.
No dashboard atual, ela aparece em requisicoes por minuto.

Ela ajuda a responder:

- a API esta recebendo trafego?
- o volume esta estavel?
- houve pico ou queda abrupta?

Em uma demo local, e normal esse numero oscilar bastante, porque as chamadas sao
manuais e pouco frequentes.

### Taxa de Erro 5xx /predict

Esse painel mostra a proporcao de respostas HTTP da familia `5xx` no endpoint
`POST /predict`.

O que significa `5xx`:

- `500`: erro interno no servidor
- `502`: erro de gateway
- `503`: servico indisponivel
- `504`: timeout no gateway

Em resumo, `5xx` representa falhas do lado do servidor, e nao erro de uso do
cliente.

Interpretacao pratica:

- `0` e o valor esperado em situacao saudavel
- qualquer valor acima de `0` indica que parte das requisicoes falhou
- se esse numero cresce, a API pode estar com excecoes, indisponibilidade ou
  dependencia quebrada

No dashboard, antes do ajuste mais recente, era possivel aparecer `No data`
quando nao havia nenhum erro `5xx`. Agora o comportamento esperado para cenario
saudavel e exibir `0`.

### Requisicoes em Andamento

Essa metrica mostra quantas chamadas ao endpoint `POST /predict` estao em
processamento exatamente no momento em que o Prometheus faz o scrape.

Por isso, em demos locais, o valor costuma ficar em `0` quase o tempo todo:

- a requisicao entra
- a predicao termina rapido
- quando o Prometheus consulta `/metrics`, ela ja acabou

Quando essa metrica pode ser maior que zero:

- quando ha varias chamadas simultaneas
- quando o modelo demora mais para responder
- quando existe fila interna ou gargalo de CPU/memoria
- quando alguma dependencia externa deixa a resposta mais lenta

Ver essa metrica em `1`, `2` ou mais nao e automaticamente um problema. O sinal
de alerta aparece quando:

- ela cresce com frequencia
- permanece alta por muito tempo
- isso acontece junto com piora de latencia ou aumento de erros

## Como validar rapidamente sem depender do Grafana

O endpoint `GET /metrics` do serving e a fonte primaria dessas metricas. Se
houver duvida sobre o dashboard, vale consultar diretamente:

```bash
curl http://127.0.0.1:8000/metrics
```

Depois de algumas chamadas para `POST /predict`, e esperado encontrar linhas
como:

```text
churn_serving_predict_requests_total{method="POST",status_code="200"} 8.0
churn_serving_predict_latency_seconds_count 8.0
churn_serving_predict_requests_in_progress 0.0
```

Se essas linhas aparecem, significa que:

- o serving esta registrando as metricas corretamente
- o Prometheus tem algo valido para coletar
- qualquer problema restante tende a estar na camada de scrape, datasource ou
  visualizacao do Grafana

## Como subir localmente

1. Suba a stack local:

```bash
cp .env.example .env
poetry run task appstack
```

2. Gere tráfego no endpoint de predição:

Use o Swagger em `http://127.0.0.1:8000/docs` ou faça chamadas para `POST /predict`.

## Acessos

- Serving: `http://127.0.0.1:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- MLflow: `http://127.0.0.1:<MLFLOW_PORT>` (padrão `5000`)
- Usuário padrão Grafana: `admin`
- Senha padrão Grafana: `admin`

Os dashboards `Serving Operacional - Churn API` e `LLM Chat Operacional - Churn API` são provisionados automaticamente.

## Observação

O Prometheus foi configurado para fazer scrape do serviço `serving` diretamente pela rede interna do Docker Compose, usando o alvo `serving:8000`.
