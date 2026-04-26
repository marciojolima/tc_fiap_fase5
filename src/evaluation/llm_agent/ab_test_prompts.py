"""Benchmark A/B de prompts sobre o golden set do projeto.

Objetivo:
- comparar variantes de prompt de forma offline
- medir cobertura lexical mínima contra a referência
- opcionalmente enriquecer a comparação com LLM-as-judge

Este módulo não faz parte do fluxo online do agente; ele é um gatilho de
qualidade/benchmark para decidir qual prompt vale a pena promover.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.llm_gateway.factory import build_llm_client
from agent.rag_pipeline import retrieve_contexts
from src.evaluation.llm_agent.artifacts import (
    RESULTS_DIR,
    RUNS_DIR,
    append_jsonl,
    build_run_metadata,
    relative_path,
    write_json,
)
from src.evaluation.llm_agent.artifacts import (
    persist_result_with_history as persist_result,
)
from src.evaluation.llm_agent.llm_judge import judge_one, provider_chat
from src.evaluation.llm_agent.ragas_eval import load_golden_items

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN = "data/golden-set.json"
DEFAULT_OUT = str(RESULTS_DIR / "prompt_ab_results.json")
DEFAULT_HISTORY_OUT = str(RUNS_DIR / "prompt_ab_runs.jsonl")
DEFAULT_TIMEOUT_SEC = 300
DEFAULT_TOP_K = 3
MIN_TERM_LENGTH = 2
TOKEN_RE = re.compile(r"[a-z0-9_./*-]+", re.IGNORECASE)
STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "que",
    "se",
    "um",
    "uma",
}


@dataclass(frozen=True)
class PromptVariant:
    name: str
    system_prompt: str
    description: str


PROMPT_VARIANTS = (
    PromptVariant(
        name="baseline",
        description="Prompt simples orientado a contexto recuperado.",
        system_prompt=(
            "Você é um assistente técnico do projeto de churn/MLOps. "
            "Responda em português, de forma objetiva, usando preferencialmente "
            "apenas as informações dos contextos fornecidos."
        ),
    ),
    PromptVariant(
        name="grounded_strict",
        description="Prompt mais rígido, com ênfase em grounding e não invenção.",
        system_prompt=(
            "Você é um assistente técnico do projeto de churn/MLOps. "
            "Responda em português do Brasil, de forma objetiva. "
            "Use apenas os fatos presentes nos contextos fornecidos. "
            "Se a resposta não estiver suportada pelo contexto, diga isso "
            "explicitamente e não invente detalhes."
        ),
    ),
    PromptVariant(
        name="documental_explicit",
        description=(
            "Prompt que reforça rotas, tools, arquivos e artefatos do repositório."
        ),
        system_prompt=(
            "Você é um assistente técnico especializado no repositório do projeto "
            "de churn/MLOps. Para perguntas documentais, priorize mencionar "
            "explicitamente rotas HTTP, tools, arquivos, configs e artefatos quando "
            "eles aparecerem nos contextos. Responda em português do Brasil, sem "
            "inventar resultados nem omitir nomes concretos importantes."
        ),
    ),
)


def _timeout_seconds(override: int | None = None) -> int:
    if override is not None:
        return max(30, override)
    raw = os.environ.get("RAGAS_TIMEOUT_SEC", "").strip()
    if raw.isdigit():
        return max(30, int(raw))
    return DEFAULT_TIMEOUT_SEC


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    return " ".join(lowered.split())


def extract_reference_terms(reference: str) -> list[str]:
    """Extrai termos-chave mínimos da referência para scoring lexical simples."""

    normalized = _normalize_text(reference)
    seen: set[str] = set()
    terms: list[str] = []
    for token in TOKEN_RE.findall(normalized):
        cleaned_token = token.strip(".,:;!?")
        if cleaned_token in STOPWORDS:
            continue
        if len(cleaned_token) <= MIN_TERM_LENGTH and not cleaned_token.startswith("/"):
            continue
        if cleaned_token not in seen:
            seen.add(cleaned_token)
            terms.append(cleaned_token)
    return terms


def compute_keyword_coverage(answer: str, reference: str) -> dict[str, Any]:
    """Métrica determinística de cobertura lexical da resposta."""

    required_terms = extract_reference_terms(reference)
    normalized_answer = _normalize_text(answer)
    hits = [term for term in required_terms if term in normalized_answer]
    coverage = round(len(hits) / len(required_terms), 4) if required_terms else 0.0
    return {
        "required_terms": required_terms,
        "matched_terms": hits,
        "missing_terms": [term for term in required_terms if term not in hits],
        "keyword_coverage": coverage,
    }


def generate_prompt_answer(
    question: str,
    contexts: list[str],
    variant: PromptVariant,
) -> str:
    empty_ctx = "(nenhum contexto recuperado)"
    ctx_block = "\n\n---\n\n".join(contexts) if contexts else empty_ctx
    user = f"Contextos:\n{ctx_block}\n\nPergunta: {question}"
    _ = question
    return provider_chat(variant.system_prompt, user)


def run_prompt_ab_test(  # noqa: PLR0914
    golden_path: str | Path = DEFAULT_GOLDEN,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_rows: int | None = None,
    timeout_sec: int | None = None,
    include_judge: bool = False,
) -> dict[str, Any]:
    """Executa benchmark A/B com 3 variantes de prompt sobre o golden set."""

    t = _timeout_seconds(timeout_sec)
    metadata = build_llm_client().metadata()

    items = load_golden_items(golden_path)
    limit = max_rows if max_rows is not None else len(items)
    rows: list[dict[str, Any]] = []
    per_variant_coverages = {variant.name: [] for variant in PROMPT_VARIANTS}
    per_variant_judge_means = {variant.name: [] for variant in PROMPT_VARIANTS}

    for item in items[:limit]:
        question = str(item["query"]).strip()
        reference = str(item["expected_answer"]).strip()
        contexts = retrieve_contexts(question, top_k=top_k)
        variant_rows: list[dict[str, Any]] = []

        for variant in PROMPT_VARIANTS:
            answer = generate_prompt_answer(
                question,
                contexts,
                variant,
            )
            lexical = compute_keyword_coverage(answer, reference)
            per_variant_coverages[variant.name].append(lexical["keyword_coverage"])

            judge_scores: dict[str, Any] | None = None
            if include_judge:
                judge_scores = judge_one(
                    query=question,
                    reference=reference,
                    candidate=answer,
                    timeout_sec=t,
                )
                judge_mean = sum(
                    judge_scores[key]
                    for key in (
                        "adequacao_negocio",
                        "correcao_conteudo",
                        "clareza_utilidade",
                    )
                ) / 3
                per_variant_judge_means[variant.name].append(round(judge_mean, 4))

            variant_rows.append(
                {
                    "variant": variant.name,
                    "description": variant.description,
                    "answer": answer,
                    "lexical_metrics": lexical,
                    "judge_scores": judge_scores,
                }
            )

        rows.append(
            {
                "id": str(item.get("id", "")),
                "category": item.get("category"),
                "metadata": item.get("metadata", {}),
                "expected_tools": item.get("expected_tools", []),
                "curated_context_count": len(item.get("contexts", [])),
                "query": question,
                "reference": reference,
                "context_count": len(contexts),
                "variants": variant_rows,
            }
        )
        logger.info("Prompt A/B item processado: %s", item.get("id", "sem-id"))

    aggregate_variants: list[dict[str, Any]] = []
    for variant in PROMPT_VARIANTS:
        coverages = per_variant_coverages[variant.name]
        judge_means = per_variant_judge_means[variant.name]
        aggregate_variants.append(
            {
                "variant": variant.name,
                "description": variant.description,
                "mean_keyword_coverage": (
                    round(sum(coverages) / len(coverages), 4) if coverages else 0.0
                ),
                "mean_judge_score": (
                    round(sum(judge_means) / len(judge_means), 4)
                    if judge_means
                    else None
                ),
            }
        )

    aggregate_variants.sort(
        key=lambda item: (
            item["mean_judge_score"] if item["mean_judge_score"] is not None else -1,
            item["mean_keyword_coverage"],
        ),
        reverse=True,
    )

    return {
        "schema": "prompt_ab_v1",
        "goal": "benchmark offline de prompts para LLM/RAG",
        "golden": relative_path(golden_path),
        "llm_provider": metadata.get("provider"),
        "llm_model": metadata.get("model_name"),
        "timeout_seconds": t,
        "top_k": top_k,
        "prompt_variants": [
            {
                "name": variant.name,
                "description": variant.description,
                "system_prompt": variant.system_prompt,
            }
            for variant in PROMPT_VARIANTS
        ],
        "items": rows,
        "aggregate": {
            "n_items": len(rows),
            "variants": aggregate_variants,
            "ranking_rule": (
                "prioriza mean_judge_score quando disponível; desempata por "
                "mean_keyword_coverage"
            ),
        },
        "include_judge": include_judge,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN, help="Caminho do golden.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=None)
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help="Também roda LLM-as-judge sobre cada variante.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Arquivo JSON de saída.")
    parser.add_argument(
        "--history-output",
        default=DEFAULT_HISTORY_OUT,
        help="JSONL append-only com resumo das execuções",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()
    try:
        result = run_prompt_ab_test(
            golden_path=args.golden,
            top_k=args.top_k,
            max_rows=args.max_rows,
            timeout_sec=args.timeout_sec,
            include_judge=args.with_judge,
        )
    except Exception:
        logger.exception("Falha no benchmark A/B de prompts")
        failure_metadata = build_run_metadata()
        write_json(
            args.out,
            {
                "schema": "prompt_ab_v1",
                **failure_metadata,
                "status": "failed",
                "golden": relative_path(args.golden),
                "top_k": args.top_k,
                "include_judge": args.with_judge,
            },
        )
        append_jsonl(
            args.history_output,
            {
                "schema": "prompt_ab_run_history_v1",
                **failure_metadata,
                "type": "prompt_ab",
                "status": "failed",
                "output_path": relative_path(args.out),
            },
        )
        raise SystemExit(1) from None

    metadata = build_run_metadata()
    result = {
        **result,
        **metadata,
        "status": "completed",
    }
    winner = (
        result["aggregate"]["variants"][0]
        if result["aggregate"]["variants"]
        else {}
    )
    persist_result(
        output_path=args.out,
        history_path=args.history_output,
        result_payload=result,
        history_payload={
            "schema": "prompt_ab_run_history_v1",
            **metadata,
            "type": "prompt_ab",
            "status": "completed",
            "output_path": relative_path(args.out),
            "golden": relative_path(args.golden),
            "top_k": args.top_k,
            "n_items": result["aggregate"]["n_items"],
            "include_judge": args.with_judge,
            "winner": winner.get("variant"),
            "winner_mean_keyword_coverage": winner.get("mean_keyword_coverage"),
            "winner_mean_judge_score": winner.get("mean_judge_score"),
        },
    )
    logger.info("Resultado do prompt A/B salvo em %s", Path(args.out))


if __name__ == "__main__":
    main()
