"""LLM-as-judge sobre o golden set usando o endpoint real do agente.

Fluxo:
1. Carrega itens do golden set.
2. Chama o endpoint real /llm/chat para gerar a resposta candidata.
3. Envia pergunta, resposta esperada e resposta candidata para o juiz LLM.
4. Salva scores, resposta candidata, tools usadas, trace e metadados de execução.

Critérios fixos, escala 1-5:
  - adequacao_negocio
  - correcao_conteudo
  - clareza_utilidade
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from agent.llm_gateway.factory import build_llm_client
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
DEFAULT_OUT = str(RESULTS_DIR / "llm_judge_scores.json")
DEFAULT_HISTORY_OUT = str(RUNS_DIR / "llm_judge_runs.jsonl")
DEFAULT_TIMEOUT_SEC = 300
DEFAULT_AGENT_ENDPOINT = "http://localhost:8000/llm/chat"
DEFAULT_ANSWER_STYLE = "short"

CRITERIA_KEYS = (
    "adequacao_negocio",
    "correcao_conteudo",
    "clareza_utilidade",
)

JUDGE_SYSTEM = (
    "És um avaliador académico (banca). Avalia a RESPOSTA_CANDIDATA face "
    "à pergunta e à referência.\n\n"
    "Critérios (escala inteira 1 a 5 para cada um):\n"
    "1) adequacao_negocio — Quão bem a resposta reflete o contexto de "
    "negócio do projeto (churn bancário, variáveis, pipeline, MLOps, "
    "serving, observabilidade alinhados ao datathon FIAP).\n"
    "2) correcao_conteudo — Quão consistente está com a RESPOSTA_REFERÊNCIA "
    "e com factos plausíveis (sem contradições graves).\n"
    "3) clareza_utilidade — Clareza, objetividade e utilidade para equipa "
    "técnica/negócio.\n\n"
    "Responde APENAS com um objeto JSON válido, sem markdown, neste formato "
    "exato:\n"
    '{"adequacao_negocio": <int 1-5>, "correcao_conteudo": <int 1-5>, '
    '"clareza_utilidade": <int 1-5>, "comentario": '
    '"<uma frase curta em português>"}'
)


def _utc_now_iso() -> str:
    """Retorna timestamp UTC em ISO-8601."""

    return datetime.now(UTC).isoformat()


def _timeout_seconds(override: int | None = None) -> int:
    """Resolve timeout via argumento, env ou default."""

    if override is not None:
        return max(30, override)

    raw = os.environ.get("RAGAS_TIMEOUT_SEC", "").strip()
    if raw.isdigit():
        return max(30, int(raw))

    return DEFAULT_TIMEOUT_SEC


def _normalize_golden_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normaliza campos esperados do golden set."""

    metadata = item.get("metadata") or {}
    category = item.get("category") or metadata.get("category")

    return {
        **item,
        "query": str(item.get("query") or item.get("question") or "").strip(),
        "expected_answer": str(item.get("expected_answer") or "").strip(),
        "category": category,
        "contexts": [
            str(context).strip()
            for context in item.get("contexts", [])
        ],
        "expected_tools": list(item.get("expected_tools", [])),
        "metadata": metadata,
    }


def load_golden_items(path: str | Path) -> list[dict[str, Any]]:
    """Carrega os itens do golden set."""

    full = Path(path)
    with open(full, encoding="utf-8") as file:
        data = json.load(file)

    items = [
        _normalize_golden_item(item)
        for item in data.get("items", [])
    ]

    if not items:
        raise ValueError(f"Nenhum item em {path}")

    missing_query = [
        item.get("id", "<sem-id>")
        for item in items
        if not item["query"]
    ]
    if missing_query:
        raise ValueError(f"Itens sem pergunta no golden set: {missing_query}")

    return items


def provider_chat(system: str, user: str) -> str:
    """Chama o provider LLM ativo."""

    client = build_llm_client()
    return client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )


def call_llm_chat_endpoint(
    *,
    endpoint_url: str,
    question: str,
    answer_style: str,
    include_trace: bool,
    timeout_sec: int,
) -> dict[str, Any]:
    """Chama o endpoint real /llm/chat do serving."""

    payload = {
        "message": question,
        "answer_style": answer_style,
        "include_trace": include_trace,
    }

    response = requests.post(
        endpoint_url,
        json=payload,
        timeout=timeout_sec,
    )
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict):
        raise TypeError("Resposta do endpoint /llm/chat não é um objeto JSON.")

    return data


def _extract_json_object(text: str) -> dict[str, Any]:
    """Faz parse do JSON do modelo; tolera bloco ```json ... ```."""

    clean_text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    if fence:
        clean_text = fence.group(1).strip()

    parsed = json.loads(clean_text)
    if not isinstance(parsed, dict):
        raise TypeError("Resposta do juiz não é um objeto JSON.")

    return parsed


extract_json_object = _extract_json_object


def _clamp_score(value: Any) -> int:
    """Converte score para inteiro entre 1 e 5."""

    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 1

    return max(1, min(5, score))


def judge_one(
    *,
    query: str,
    reference: str,
    candidate: str,
    timeout_sec: int,
) -> dict[str, Any]:
    """Executa uma chamada ao juiz e devolve scores + comentário."""

    user = (
        f"PERGUNTA:\n{query}\n\n"
        f"RESPOSTA_REFERÊNCIA (ground truth resumida):\n{reference}\n\n"
        f"RESPOSTA_CANDIDATA (avaliar):\n{candidate}\n"
    )

    _ = timeout_sec
    raw = provider_chat(JUDGE_SYSTEM, user)
    data = _extract_json_object(raw)

    output: dict[str, Any] = {}
    for key in CRITERIA_KEYS:
        output[key] = _clamp_score(data.get(key))

    output["comentario"] = str(data.get("comentario", "")).strip()[:500]
    return output


def build_empty_failure_row(context: dict[str, Any]) -> dict[str, Any]:
    """Monta uma linha de falha auditável para um item."""

    return {
        "id": context["item_id"],
        "category": context["category"],
        "metadata": context["metadata"],
        "expected_tools": context["expected_tools"],
        "curated_context_count": context["curated_context_count"],
        "query": context["query"],
        "expected_answer": context["reference"],
        "candidate_answer": "",
        "used_tools": [],
        "trace": [],
        "scores": {key: 1 for key in CRITERIA_KEYS},
        "mean_criteria": 1.0,
        "comentario_juiz": (
            "Falha ao obter resposta candidata pelo endpoint real."
        ),
        "endpoint_error": context["error"],
        "started_at": context["started_at"],
        "finished_at": context["finished_at"],
        "duration_seconds": context["duration_seconds"],
    }


def process_item(
    *,
    item: dict[str, Any],
    endpoint_url: str,
    answer_style: str,
    timeout: int,
) -> tuple[dict[str, Any], float]:
    """Processa um item do golden set."""

    item_id = str(item.get("id", ""))
    query = str(item["query"]).strip()
    reference = str(item["expected_answer"]).strip()

    started_at = _utc_now_iso()
    start_perf = time.perf_counter()

    try:
        payload = call_llm_chat_endpoint(
            endpoint_url=endpoint_url,
            question=query,
            answer_style=answer_style,
            include_trace=True,
            timeout_sec=timeout,
        )

        finished_at = _utc_now_iso()
        duration = round(time.perf_counter() - start_perf, 4)

        candidate = str(payload.get("answer", "")).strip()
        used_tools = payload.get("used_tools", [])
        trace = payload.get("trace", [])

        scores = judge_one(
            query=query,
            reference=reference,
            candidate=candidate,
            timeout_sec=timeout,
        )

        mean_score = sum(scores[k] for k in CRITERIA_KEYS) / len(CRITERIA_KEYS)

        row = {
            "id": item_id,
            "category": item.get("category"),
            "metadata": item.get("metadata", {}),
            "expected_tools": item.get("expected_tools", []),
            "curated_context_count": len(item.get("contexts", [])),
            "query": query,
            "expected_answer": reference,
            "candidate_answer": candidate,
            "used_tools": used_tools,
            "trace": trace,
            "scores": {k: scores[k] for k in CRITERIA_KEYS},
            "mean_criteria": round(mean_score, 4),
            "comentario_juiz": scores.get("comentario", ""),
            "endpoint_url": endpoint_url,
            "answer_style": answer_style,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration,
        }

    except Exception as error:  # noqa: BLE001
        finished_at = _utc_now_iso()
        duration = round(time.perf_counter() - start_perf, 4)

        row = build_empty_failure_row(
            {
                "item_id": item_id,
                "category": item.get("category"),
                "metadata": item.get("metadata", {}),
                "expected_tools": item.get("expected_tools", []),
                "curated_context_count": len(item.get("contexts", [])),
                "query": query,
                "reference": reference,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": duration,
                "error": str(error)[:500],
            }
        )

        mean_score = 1.0

    return row, mean_score


def run_llm_judge(
    golden_path: str | Path = DEFAULT_GOLDEN,
    *,
    max_rows: int | None = None,
    timeout_sec: int | None = None,
    endpoint_url: str = DEFAULT_AGENT_ENDPOINT,
    answer_style: str = DEFAULT_ANSWER_STYLE,
) -> dict[str, Any]:
    """Gera respostas via endpoint real + pontuações do juiz por item."""

    timeout = _timeout_seconds(timeout_sec)
    metadata = build_llm_client().metadata()

    run_started_at = _utc_now_iso()
    run_started_perf = time.perf_counter()

    items = load_golden_items(golden_path)
    limit = max_rows if max_rows is not None else len(items)

    rows_out: list[dict[str, Any]] = []
    sums = {key: 0.0 for key in CRITERIA_KEYS}
    n_scored = 0

    for item in items[:limit]:
        row, mean_3 = process_item(
            item=item,
            endpoint_url=endpoint_url,
            answer_style=answer_style,
            timeout=timeout,
        )

        rows_out.append(row)

        for key in CRITERIA_KEYS:
            sums[key] += row["scores"][key]

        n_scored += 1

        logger.info(
            "Juiz %s | média=%.2f | scores=%s",
            row["id"],
            mean_3,
            row["scores"],
        )

    aggregate = {
        key: round(sums[key] / n_scored, 4)
        for key in CRITERIA_KEYS
    } if n_scored else {}

    overall = (
        round(sum(aggregate.values()) / len(CRITERIA_KEYS), 4)
        if aggregate
        else 0.0
    )

    run_finished_at = _utc_now_iso()
    run_duration_seconds = round(time.perf_counter() - run_started_perf, 4)

    return {
        "schema": "llm_judge_v1",
        "scale": "1-5 por critério",
        "criteria": {
            "adequacao_negocio": (
                "Alinhamento ao domínio churn/MLOps/dados do projeto "
                "(obrigatório negócio)"
            ),
            "correcao_conteudo": (
                "Consistência com referência e plausibilidade"
            ),
            "clareza_utilidade": "Clareza e utilidade",
        },
        "golden": relative_path(golden_path),
        "llm_provider": metadata.get("provider"),
        "llm_model": metadata.get("model_name"),
        "endpoint_url": endpoint_url,
        "answer_style": answer_style,
        "items": rows_out,
        "aggregate_mean_per_criterion": aggregate,
        "overall_mean": overall,
        "n_items": n_scored,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "run_duration_seconds": run_duration_seconds,
    }


def main() -> None:
    """CLI do LLM-as-judge."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "LLM-as-judge usando o endpoint real /llm/chat do agente."
        ),
    )
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("-o", "--output", default=DEFAULT_OUT)
    parser.add_argument("--history-output", default=DEFAULT_HISTORY_OUT)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument(
        "--endpoint-url",
        default=DEFAULT_AGENT_ENDPOINT,
        help="Endpoint real do agente /llm/chat.",
    )
    parser.add_argument(
        "--answer-style",
        default=DEFAULT_ANSWER_STYLE,
        choices=["short", "default", "detailed"],
        help="Estilo de resposta enviado ao endpoint.",
    )

    args = parser.parse_args()

    try:
        payload = run_llm_judge(
            golden_path=args.golden,
            max_rows=args.max_rows,
            timeout_sec=args.timeout,
            endpoint_url=args.endpoint_url,
            answer_style=args.answer_style,
        )
    except Exception:
        logger.exception("Falha no LLM-as-judge")

        failure_metadata = build_run_metadata()
        write_json(
            args.output,
            {
                "schema": "llm_judge_scores_v1",
                **failure_metadata,
                "status": "failed",
                "golden": relative_path(args.golden),
                "endpoint_url": args.endpoint_url,
                "answer_style": args.answer_style,
            },
        )
        append_jsonl(
            args.history_output,
            {
                "schema": "llm_judge_run_history_v1",
                **failure_metadata,
                "type": "llm_judge",
                "status": "failed",
                "output_path": relative_path(args.output),
            },
        )
        raise SystemExit(1) from None

    metadata = build_run_metadata()
    payload = {
        **payload,
        "schema": "llm_judge_scores_v1",
        **metadata,
        "status": "completed",
    }

    persist_result_with_history(
        output_path=args.output,
        history_path=args.history_output,
        result_payload=payload,
        history_payload={
            "schema": "llm_judge_run_history_v1",
            **metadata,
            "type": "llm_judge",
            "status": "completed",
            "output_path": relative_path(args.output),
            "golden": relative_path(args.golden),
            "endpoint_url": args.endpoint_url,
            "answer_style": args.answer_style,
            "overall_mean": payload["overall_mean"],
            "run_started_at": payload["run_started_at"],
            "run_finished_at": payload["run_finished_at"],
            "run_duration_seconds": payload["run_duration_seconds"],
            **payload["aggregate_mean_per_criterion"],
        },
    )

    logger.info("Médias por critério: %s", payload["aggregate_mean_per_criterion"])
    logger.info(
        "Média global: %s | Salvo em %s",
        payload["overall_mean"],
        Path(args.output).resolve(),
    )


if __name__ == "__main__":
    main()
