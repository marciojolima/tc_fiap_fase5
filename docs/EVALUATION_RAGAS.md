# Avaliacao RAGAS e LLM-as-judge

## Indice

- [Objetivo](#objetivo)
- [Requisitos atendidos](#requisitos-atendidos)
- [Golden set](#golden-set)
- [Como o RAGAS avalia o serving](#como-o-ragas-avalia-o-serving)
- [Metricas RAGAS](#metricas-ragas)
- [LLM-as-judge](#llm-as-judge)
- [Benchmark de prompts](#benchmark-de-prompts)
- [Artefatos reportados](#artefatos-reportados)
- [Como executar](#como-executar)
- [Cuidados de interpretacao](#cuidados-de-interpretacao)

## Objetivo

Este documento explica a avaliacao da trilha RAG/LLM do projeto. Ele detalha como
o golden set, o RAGAS, o LLM-as-judge e o benchmark de prompts se conectam aos
requisitos da entrega.

A visao consolidada da avaliacao do projeto fica em [EVALUATION.md](EVALUATION.md).
Para a arquitetura e a operacao da trilha conversacional, veja tambem
[AGENT_REACT.md](AGENT_REACT.md).

## Escopo implementado

| Frente | Evidencia no projeto |
| --------- | -------------------- |
| Golden set do dominio | [data/golden-set.json](../data/golden-set.json) possui 24 itens alinhados a churn bancario, MLOps, API, observabilidade e RAG/LLM. |
| RAGAS com 4 metricas | [src/evaluation/llm_agent/ragas_eval.py](../src/evaluation/llm_agent/ragas_eval.py) calcula `faithfulness`, `answer_relevancy`, `context_precision` e `context_recall`; resultado em `artifacts/evaluation/llm_agent/results/ragas_scores.json`. |
| LLM-as-judge com criterio de negocio | [src/evaluation/llm_agent/llm_judge.py](../src/evaluation/llm_agent/llm_judge.py) usa `adequacao_negocio`, `correcao_conteudo` e `clareza_utilidade`. |

## Golden set

O golden set e o conjunto de perguntas e respostas de referencia usado para medir
a qualidade do RAG/LLM.

Arquivo:

- [data/golden-set.json](../data/golden-set.json)

Contrato principal de cada item:

- `question`: pergunta usada na avaliacao
- `expected_answer`: resposta de referencia
- `contexts`: contextos curados esperados
- `expected_tools`: tools esperadas quando aplicavel
- `metadata`: categoria, dificuldade e tipo

No estado atual, o arquivo possui 24 pares relevantes ao dominio, cobrindo
perguntas sobre churn, API, observabilidade, agente e RAG.

Validador:

- [tests/test_golden_set.py](../tests/test_golden_set.py)

## Como o RAGAS avalia o serving

O RAGAS nao avalia uma resposta isolada escrita manualmente. Ele monta um dataset
a partir do golden set e das respostas geradas pelo serving real, passando pelo
mesmo fluxo documentado em [AGENT_REACT.md](AGENT_REACT.md): provider ativo,
agente ReAct, tools, guardrails e RAG.

Fluxo atual:

1. [src/evaluation/llm_agent/ragas_eval.py](../src/evaluation/llm_agent/ragas_eval.py) carrega [data/golden-set.json](../data/golden-set.json).
2. Para cada pergunta, chama `POST /llm/chat` no serving.
3. O endpoint executa o agente ReAct, as guardrails, a selecao de tool e o RAG.
4. Quando `rag_search` e usado, a trace inclui `retrieved_contexts`.
5. O avaliador monta as colunas exigidas pelo RAGAS:
   - `user_input`
   - `retrieved_contexts`
   - `reference_contexts`
   - `response`
   - `reference`
6. O RAGAS calcula as quatro metricas e persiste o relatorio.

Endpoint avaliado:

- [src/serving/llm_routes.py](../src/serving/llm_routes.py)

Tool que fornece os contextos:

- [src/agent/tools.py](../src/agent/tools.py)

Busca vetorial usada pela tool:

- [src/agent/rag_pipeline.py](../src/agent/rag_pipeline.py)

Essa decisao deixa a avaliacao mais fiel ao comportamento observado em producao
local, porque passa pelo contrato HTTP real do serving em vez de chamar apenas
funcoes internas.

## Metricas RAGAS

As quatro metricas calculadas sao:

| Metrica | O que mede | Como interpretar |
| ------- | ---------- | ---------------- |
| `faithfulness` | Se a resposta esta sustentada pelos contextos recuperados. | Baixo valor indica resposta possivelmente inventada ou pouco ancorada no RAG. |
| `answer_relevancy` | Se a resposta responde semanticamente a pergunta. | Baixo valor indica resposta tangencial, incompleta ou desalinhada. |
| `context_precision` | Se os contextos recuperados sao relevantes e aparecem bem ranqueados. | Baixo valor indica ruido no retrieval. |
| `context_recall` | Se os contextos recuperados cobrem a referencia esperada. | Baixo valor indica que o RAG deixou de recuperar evidencia importante. |

No projeto, os embeddings usados pelo RAGAS sao FastEmbed, alinhados ao runtime do
RAG. Isso evita `sentence-transformers` e `torch` na avaliacao.

Modelo de embedding configurado:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Aqui esse nome identifica o modelo de embedding; nao significa dependencia Python
`sentence-transformers`.

## LLM-as-judge

O LLM-as-judge complementa o RAGAS. Enquanto o RAGAS mede dimensoes tecnicas do
RAG, o judge avalia a utilidade da resposta para o projeto.

Arquivo:

- [src/evaluation/llm_agent/llm_judge.py](../src/evaluation/llm_agent/llm_judge.py)

Criterios:

- `adequacao_negocio`: alinhamento ao dominio de churn bancario, MLOps, dados,
  serving e observabilidade
- `correcao_conteudo`: consistencia com a referencia e plausibilidade factual
- `clareza_utilidade`: clareza, objetividade e utilidade para time tecnico e
  negocio

O criterio de negocio e `adequacao_negocio`, atendendo ao requisito de incluir um
criterio explicitamente ligado ao dominio.

Saida:

- `artifacts/evaluation/llm_agent/results/llm_judge_scores.json`
- `artifacts/evaluation/llm_agent/runs/llm_judge_runs.jsonl`

## Benchmark de prompts

O benchmark de prompts compara tres configuracoes de prompt sobre o mesmo golden
set.

Arquivo:

- [src/evaluation/llm_agent/ab_test_prompts.py](../src/evaluation/llm_agent/ab_test_prompts.py)

Variantes atuais:

- `baseline`
- `grounded_strict`
- `documental_explicit`

Metricas:

- `keyword_coverage`
- `mean_judge_score`, quando executado com `--with-judge`

Saida:

- `artifacts/evaluation/llm_agent/results/prompt_ab_results.json`
- `artifacts/evaluation/llm_agent/runs/prompt_ab_runs.jsonl`

## Artefatos reportados

RAGAS:

- `artifacts/evaluation/llm_agent/results/ragas_scores.json`
- `artifacts/evaluation/llm_agent/runs/ragas_runs.jsonl`

LLM-as-judge:

- `artifacts/evaluation/llm_agent/results/llm_judge_scores.json`
- `artifacts/evaluation/llm_agent/runs/llm_judge_runs.jsonl`

Prompt A/B:

- `artifacts/evaluation/llm_agent/results/prompt_ab_results.json`
- `artifacts/evaluation/llm_agent/runs/prompt_ab_runs.jsonl`

## Como executar

Local:

```bash
poetry run task appstack
poetry run task eval_ragas
poetry run task eval_llm_judge
poetry run task eval_ab_test_prompts
```

Tudo em sequencia:

```bash
poetry run task eval_all
```

RAGAS em Docker:

```bash
poetry run task eval_ragas_docker
```

O RAGAS usa `RAGAS_SERVING_BASE_URL`. O default local e
`http://127.0.0.1:8000`; nas tasks Docker o valor e `http://serving:8000`.

## Cuidados de interpretacao

- RAGAS depende da qualidade do LLM usado como avaliador e da estabilidade do
  endpoint `/llm/chat`.
- Como a avaliacao chama o serving real, mudancas em guardrails, prompt ReAct,
  tools, RAG ou provider LLM podem alterar os resultados.
- `context_precision` e `context_recall` avaliam retrieval; resultados baixos
  apontam mais para corpus, chunking, embedding ou ranking do que para o LLM.
- `faithfulness` avalia ancoragem; uma resposta correta por conhecimento geral,
  mas sem suporte nos contextos recuperados, pode receber nota baixa.
- `answer_relevancy` usa embeddings; por isso pode variar se o modelo de
  embedding mudar.
