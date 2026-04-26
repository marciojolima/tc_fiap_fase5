"""Avaliacao do pipeline RAG com RAGAS - 4 metricas obrigatorias.

Referência: Es et al. (2024) — RAGAS: Automated Evaluation of Retrieval
            Augmented Generation. https://arxiv.org/abs/2309.15217

Requer um provider LLM ativo (para metricas e geracao de respostas) e,
para embeddings usados pelo RAGAS, o pacote sentence-transformers
(modelo multilingual configuravel abaixo).

RAGAS 0.4 — métricas ``collections.*`` (Faithfulness instanciada etc.) herdam de
``SimpleBaseMetric``, mas ``evaluate()`` só aceita instâncias de
``ragas.metrics.base.Metric``. Por isso usamos os singletons legados
(``faithfulness``, ``answer_relevancy``, …), que são ``Metric`` e funcionam com
``InstructorLLM``. O LLM para metricas usa o SDK adequado ao provider ativo.

Embeddings: métricas legadas exigem interface LangChain; usamos
``LangchainEmbeddingsWrapper(HuggingFaceEmbeddings)``. Variáveis opcionais:
``RAGAS_LLM_MAX_TOKENS`` (default 4096, JSON estruturado),
``RAGAS_ANSWER_RELEVANCY_STRICTNESS`` (default 1 para modelos pequenos).
``RAGAS_FAITHFULNESS_NLI_BATCH_SIZE`` (default 2): NLI em lotes para JSON estável em
modelos 3B.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from langchain_community.embeddings import (
    HuggingFaceEmbeddings as LangChainHuggingFaceEmbeddings,
)
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import InstructorBaseRagasLLM, llm_factory
from ragas.metrics._answer_relevance import answer_relevancy  # noqa: PLC2701
from ragas.metrics._context_precision import context_precision  # noqa: PLC2701
from ragas.metrics._context_recall import context_recall  # noqa: PLC2701
from ragas.metrics._faithfulness import (  # noqa: PLC2701
    Faithfulness,
    NLIStatementInput,
    NLIStatementOutput,
)
from ragas.run_config import RunConfig

from agent.llm_gateway.factory import build_llm_client

# Imports do projeto (pacote instalado com poetry install -e .)
from agent.rag_pipeline import retrieve_contexts
from common.config_loader import (
    load_global_config,
    resolve_llm_api_key,
    resolve_llm_base_url,
    resolve_llm_max_tokens,
    resolve_llm_model_name,
    resolve_llm_provider,
)
from src.evaluation.llm_agent.artifacts import (
    RESULTS_DIR,
    RUNS_DIR,
    append_jsonl,
    build_run_metadata,
    persist_result_with_history,
    relative_path,
    write_json,
)

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN = "data/golden-set.json"
DEFAULT_OUT = str(RESULTS_DIR / "ragas_scores.json")
DEFAULT_HISTORY_OUT = str(RUNS_DIR / "ragas_runs.jsonl")
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Ollama local (CPU) e RAGAS (várias chamadas por métrica) costumam estourar 120–180s.
DEFAULT_TIMEOUT_SEC = 300


def _timeout_seconds(override: int | None = None) -> int:
    if override is not None:
        return max(30, override)
    raw = os.environ.get("RAGAS_TIMEOUT_SEC", "").strip()
    if raw.isdigit():
        return max(30, int(raw))
    return DEFAULT_TIMEOUT_SEC


def _instructor_llm_from_provider(
    provider: str,
    model: str,
    timeout_sec: int,
) -> InstructorBaseRagasLLM:
    """Constroi o LLM do RAGAS a partir do provider configurado."""

    max_tokens = resolve_llm_max_tokens(provider) or int(
        os.environ.get("RAGAS_LLM_MAX_TOKENS", "4096")
    )

    if provider == "ollama":
        base = (resolve_llm_base_url(provider) or "http://127.0.0.1:11434").rstrip("/")
        client = OpenAI(
            base_url=f"{base}/v1",
            api_key="ollama",
            timeout=float(timeout_sec),
        )
        return llm_factory(
            model,
            provider="openai",
            client=client,
            temperature=0,
            max_tokens=max_tokens,
        )

    if provider == "openai":
        client_kwargs: dict[str, object] = {
            "api_key": resolve_llm_api_key(provider),
            "timeout": float(timeout_sec),
        }
        base_url = resolve_llm_base_url(provider)
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        return llm_factory(
            model,
            provider="openai",
            client=client,
            temperature=0,
            max_tokens=max_tokens,
        )

    if provider == "claude":
        try:
            anthropic_module = import_module("anthropic")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Dependencia 'anthropic' nao instalada para usar ragas_eval com Claude."
            ) from exc
        anthropic_client_cls = getattr(anthropic_module, "Anthropic")
        client_kwargs: dict[str, object] = {"api_key": resolve_llm_api_key(provider)}
        base_url = resolve_llm_base_url(provider)
        if base_url:
            client_kwargs["base_url"] = base_url
        client = anthropic_client_cls(**client_kwargs)
        llm = llm_factory(
            model,
            provider="anthropic",
            client=client,
            temperature=0,
            max_tokens=max_tokens,
        )
        # RAGAS 0.4.3 cria InstructorModelArgs com temperature e top_p por
        # padrao. Anthropic rejeita chamadas que enviam ambos.
        if hasattr(llm, "model_args") and isinstance(llm.model_args, dict):
            llm.model_args.pop("top_p", None)
        return llm

    raise ValueError(f"Provider '{provider}' nao suportado pelo ragas_eval.")


def _legacy_compatible_embeddings(model_name: str) -> LangchainEmbeddingsWrapper:
    """Métricas legadas (ex.: answer_relevancy) usam embed_query/embed_documents
    (LangChain).

    ``ragas.embeddings.HuggingFaceEmbeddings`` é outra API (embed_text) e quebra no
    evaluate().
    """

    rag_cfg = load_global_config().get("rag", {})
    embedding_cache_path = Path(
        str(
            rag_cfg.get(
                "ragas_embedding_cache_dir",
                "artifacts/rag/ragas_embedding_model_cache",
            )
        )
    )
    embedding_cache_dir = str(embedding_cache_path)
    local_files_only = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    embedder = LangChainHuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=embedding_cache_dir,
        model_kwargs={"local_files_only": local_files_only},
    )
    return LangchainEmbeddingsWrapper(embedder)


@dataclass
class BatchedNLIFaithfulness(Faithfulness):
    """Faithfulness com NLI em sub-lotes: modelos pequenos falham o schema com muitas
    statements."""

    nli_batch_size: int = 2

    async def _create_verdicts(
        self,
        row: dict[str, Any],
        statements: list[str],
        callbacks: Any,
    ) -> NLIStatementOutput:
        assert self.llm is not None

        contexts_str = "\n".join(row["retrieved_contexts"])
        merged: list[Any] = []
        bs = max(1, self.nli_batch_size)
        for i in range(0, len(statements), bs):
            chunk = statements[i : i + bs]
            verdicts = await self.nli_statements_prompt.generate(
                data=NLIStatementInput(context=contexts_str, statements=chunk),
                llm=self.llm,
                callbacks=callbacks,
            )
            merged.extend(verdicts.statements)
        return NLIStatementOutput(statements=merged)


def _normalize_golden_item(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    category = item.get("category") or metadata.get("category")
    return {
        **item,
        "query": str(item.get("query") or item.get("question") or "").strip(),
        "expected_answer": str(item.get("expected_answer") or "").strip(),
        "category": category,
        "contexts": [str(context).strip() for context in item.get("contexts", [])],
        "expected_tools": list(item.get("expected_tools", [])),
        "metadata": metadata,
    }


def load_golden_items(path: str | Path) -> list[dict[str, Any]]:
    """Carrega itens do golden set JSON e normaliza o contrato de avaliação."""

    full = Path(path)
    with open(full, encoding="utf-8") as f:
        data = json.load(f)
    items = [_normalize_golden_item(item) for item in data.get("items", [])]
    if not items:
        raise ValueError(f"Nenhum item em {path}")
    missing_query = [item.get("id", "<sem-id>") for item in items if not item["query"]]
    if missing_query:
        raise ValueError(f"Itens sem pergunta no golden set: {missing_query}")
    return items


def generate_rag_answer(
    question: str,
    contexts: list[str],
) -> str:
    """Gera resposta condicionada aos contextos recuperados (avaliação RAG)."""

    system = (
        "Você é um assistente técnico do projeto de churn/MLOps. "
        "Responda em português, de forma objetiva, usando preferencialmente "
        "apenas as informações dos contextos fornecidos."
    )
    empty_ctx = "(nenhum contexto recuperado)"
    ctx_block = "\n\n---\n\n".join(contexts) if contexts else empty_ctx
    user = f"Contextos:\n{ctx_block}\n\nPergunta: {question}"
    client = build_llm_client()
    return client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )


def build_dataset(  # noqa: PLR0913
    items: list[dict[str, Any]],
    *,
    top_k: int,
    max_rows: int | None,
) -> Dataset:
    """Monta o Dataset HuggingFace com colunas exigidas pelo RAGAS."""

    rows: list[dict[str, Any]] = []
    limit = max_rows if max_rows is not None else len(items)
    for item in items[:limit]:
        q = str(item["query"]).strip()
        reference = str(item["expected_answer"]).strip()
        retrieved = retrieve_contexts(q, top_k=top_k)
        response = generate_rag_answer(q, retrieved)
        curated_contexts = item.get("contexts", [])
        rows.append(
            {
                "user_input": q,
                "retrieved_contexts": retrieved,
                "reference_contexts": curated_contexts,
                "response": response,
                "reference": reference,
                "golden_item_id": item.get("id", ""),
                "golden_category": item.get("category"),
            }
        )
    return Dataset.from_list(rows)


def run_ragas_evaluation(  # noqa: PLR0913, PLR0914
    golden_path: str | Path = DEFAULT_GOLDEN,
    *,
    top_k: int = 3,
    max_rows: int | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    timeout_sec: int | None = None,
) -> dict[str, float]:
    """Executa RAGAS com faithfulness, answer_relevancy, context_precision,
    context_recall."""

    t = _timeout_seconds(timeout_sec)
    provider = resolve_llm_provider()
    model = resolve_llm_model_name(provider)

    items = load_golden_items(golden_path)
    dataset = build_dataset(
        items,
        top_k=top_k,
        max_rows=max_rows,
    )

    llm = _instructor_llm_from_provider(provider, model, t)
    embeddings = _legacy_compatible_embeddings(embed_model)

    nli_bs = max(1, int(os.environ.get("RAGAS_FAITHFULNESS_NLI_BATCH_SIZE", "2")))
    # Faithfulness: instância dedicada
    # (NLI em lotes — evita JSON inválido em modelos pequenos).
    metrics = [
        BatchedNLIFaithfulness(nli_batch_size=nli_bs),
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    # Modelos pequenos às vezes devolvem 1 geração em vez de strictness=3.
    ar_strict = os.environ.get("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "1")
    answer_relevancy.strictness = int(ar_strict)
    run_config = RunConfig(timeout=float(t), max_retries=3)

    logger.info(
        "RAGAS: %d linhas | provider=%s | model=%s | embed=%s | timeout=%ss | "
        "NLI batch=%d",
        len(dataset),
        provider,
        model,
        embed_model,
        t,
        nli_bs,
    )

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    _log_faithfulness_diagnostics(df)
    wanted = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    scores: dict[str, float] = {}
    for name in wanted:
        if name in df.columns:
            scores[name] = float(df[name].mean(skipna=True))
    if not scores:
        for col in df.columns:
            series = df[col]
            if hasattr(series, "dtype") and series.dtype.kind in "fc":
                scores[str(col)] = float(series.mean(skipna=True))

    return scores


def _log_faithfulness_diagnostics(df: pd.DataFrame) -> None:
    """Explica faithfulness=0 sem exceção: média 0 = NLI marcou verdict 0 em todas as
    afirmações."""

    if "faithfulness" not in df.columns:
        cols = list(df.columns)
        logger.warning("Sem coluna 'faithfulness' no DataFrame. Colunas: %s", cols)
        return
    s = df["faithfulness"]
    n_nan = int(s.isna().sum())
    n_ok = int(s.notna().sum())
    logger.info(
        "faithfulness (por linha): min=%.6f max=%.6f | válidos=%d nan=%d | valores=%s",
        float(s.min(skipna=True)) if n_ok else float("nan"),
        float(s.max(skipna=True)) if n_ok else float("nan"),
        n_ok,
        n_nan,
        [float(x) if pd.notna(x) else None for x in s.tolist()],
    )
    if n_ok and float(s.max(skipna=True)) <= 0.0:
        logger.info(
            "faithfulness média 0 sem erro: o NLI considerou 0%% das afirmações da "
            "resposta como sustentadas pelo contexto (verdict só 0). Comum em modelos "
            "pequenos/conservadores; teste outro "
            "llm.providers.<provider>.model_name ou reveja se as "
            "respostas do RAG citam bem o contexto."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Avaliação RAGAS (4 métricas) sobre o golden set.",
    )
    parser.add_argument(
        "--golden",
        default=DEFAULT_GOLDEN,
        help="Caminho para golden-set.json",
    )
    parser.add_argument("--top-k", type=int, default=3, dest="top_k")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limita linhas para teste rápido (default: todas)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUT,
        help="JSON de saída com médias das métricas",
    )
    parser.add_argument(
        "--history-output",
        default=DEFAULT_HISTORY_OUT,
        help="JSONL append-only com resumo das execuções",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("RAGAS_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        help="Modelo sentence-transformers para embeddings do RAGAS",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="SEC",
        help=(
            "Timeout em segundos para geracao do dataset, cliente do provider e "
            f"RunConfig (default: env RAGAS_TIMEOUT_SEC ou {DEFAULT_TIMEOUT_SEC})"
        ),
    )
    args = parser.parse_args()

    try:
        scores = run_ragas_evaluation(
            golden_path=args.golden,
            top_k=args.top_k,
            max_rows=args.max_rows,
            embed_model=args.embed_model,
            timeout_sec=args.timeout,
        )
    except Exception:
        logger.exception("Falha na avaliação RAGAS")
        failure_metadata = build_run_metadata()
        write_json(
            args.output,
            {
                "schema": "ragas_scores_v1",
                **failure_metadata,
                "status": "failed",
                "golden": relative_path(args.golden),
                "top_k": args.top_k,
                "embed_model": args.embed_model,
                "timeout_sec": _timeout_seconds(args.timeout),
            },
        )
        append_jsonl(
            args.history_output,
            {
                "schema": "ragas_run_history_v1",
                **failure_metadata,
                "type": "ragas",
                "status": "failed",
                "output_path": relative_path(args.output),
            },
        )
        raise SystemExit(1) from None

    metadata = build_run_metadata()
    item_count = args.max_rows or len(load_golden_items(args.golden))
    payload = {
        "schema": "ragas_scores_v1",
        **metadata,
        "status": "completed",
        "metrics_mean": scores,
        "golden": relative_path(args.golden),
        "top_k": args.top_k,
        "n_items": item_count,
        "embed_model": args.embed_model,
        "timeout_sec": _timeout_seconds(args.timeout),
        "llm_provider": resolve_llm_provider(),
        "llm_model": resolve_llm_model_name(resolve_llm_provider()),
    }
    history_payload = {
        "schema": "ragas_run_history_v1",
        **metadata,
        "type": "ragas",
        "status": "completed",
        "output_path": relative_path(args.output),
        "golden": relative_path(args.golden),
        "top_k": args.top_k,
        "n_items": item_count,
        **scores,
    }
    persist_result_with_history(
        output_path=args.output,
        history_path=args.history_output,
        result_payload=payload,
        history_payload=history_payload,
    )
    logger.info("Métricas: %s", scores)
    logger.info("Salvo em %s", Path(args.output).resolve())


if __name__ == "__main__":
    main()
