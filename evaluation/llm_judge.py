"""LLM-as-judge sobre o golden set (Datathon): ≥3 critérios, incluindo negócio.

Fluxo mínimo: para cada item, gera resposta com o mesmo RAG que o RAGAS,
depois um juiz (Ollama) pontua 1–5 em três dimensões e devolve JSON agregado.

Critérios (fixos, escala 1–5):
  - adequacao_negocio: alinhamento ao domínio churn bancário / MLOps / dados do projeto
  - correcao_conteudo: consistência com a resposta de referência e
    plausibilidade factual
  - clareza_utilidade: clareza, objetividade e utilidade para o time

Requer Ollama (LLM_BASE_URL / OLLAMA_MODEL), mesmo padrão do ragas_eval.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from agent.rag_pipeline import retrieve_contexts

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN = "configs/evaluation/golden_set.yaml"
DEFAULT_OUT = "evaluation/results/llm_judge_scores.json"
DEFAULT_TIMEOUT_SEC = 300


def _timeout_seconds(override: int | None = None) -> int:
    if override is not None:
        return max(30, override)
    raw = os.environ.get("RAGAS_TIMEOUT_SEC", "").strip()
    if raw.isdigit():
        return max(30, int(raw))
    return DEFAULT_TIMEOUT_SEC


def load_golden_items(path: str | Path) -> list[dict[str, Any]]:
    full = Path(path)
    with open(full, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    items = data.get("items") or []
    if not items:
        raise ValueError(f"Nenhum item em {path}")
    return items


def ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout: int = 120,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama indisponível: {exc}") from exc
    content = (parsed.get("message") or {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama retornou resposta vazia.")
    return content.strip()


def generate_rag_answer(
    question: str,
    contexts: list[str],
    *,
    base_url: str,
    model: str,
    timeout_sec: int,
) -> str:
    system = (
        "Você é um assistente técnico do projeto de churn/MLOps. "
        "Responda em português, de forma objetiva, usando preferencialmente "
        "apenas as informações dos contextos fornecidos."
    )
    empty_ctx = "(nenhum contexto recuperado)"
    ctx_block = "\n\n---\n\n".join(contexts) if contexts else empty_ctx
    user = f"Contextos:\n{ctx_block}\n\nPergunta: {question}"
    return ollama_chat(base_url, model, system, user, timeout=timeout_sec)


CRITERIA_KEYS = (
    "adequacao_negocio",
    "correcao_conteudo",
    "clareza_utilidade",
)

JUDGE_SYSTEM = (
    "És um avaliador académico (banca). Avalia a RESPOSTA_CANDIDATA face à pergunta "
    "e à referência.\n\n"
    "Critérios (escala inteira 1 a 5 para cada um):\n"
    "1) adequacao_negocio — Quão bem a resposta reflete o contexto de negócio do "
    "projeto (churn bancário, variáveis, pipeline, MLOps, serving, observabilidade "
    "alinhados ao datathon FIAP).\n"
    "2) correcao_conteudo — Quão consistente está com a RESPOSTA_REFERÊNCIA e com "
    "factos plausíveis (sem contradições graves).\n"
    "3) clareza_utilidade — Clareza, objetividade e utilidade para equipa "
    "técnica/negócio.\n"
    "\n"
    "Responde APENAS com um objeto JSON válido, sem markdown, neste formato exato:\n"
    '{"adequacao_negocio": <int 1-5>, "correcao_conteudo": <int 1-5>, '
    '"clareza_utilidade": <int 1-5>, "comentario": "<uma frase curta em português>"}'
)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON do modelo; tolera bloco ```json ... ```."""

    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


extract_json_object = _extract_json_object


def _clamp_score(v: Any) -> int:
    try:
        x = int(float(v))
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, x))


def judge_one(  # noqa: PLR0913
    *,
    base_url: str,
    model: str,
    query: str,
    reference: str,
    candidate: str,
    timeout_sec: int,
) -> dict[str, Any]:
    """Uma chamada ao juiz; devolve dict com scores + comentario."""

    user = (
        f"PERGUNTA:\n{query}\n\n"
        f"RESPOSTA_REFERÊNCIA (ground truth resumida):\n{reference}\n\n"
        f"RESPOSTA_CANDIDATA (avaliar):\n{candidate}\n"
    )
    raw = ollama_chat(base_url, model, JUDGE_SYSTEM, user, timeout=timeout_sec)
    data = _extract_json_object(raw)
    out: dict[str, Any] = {}
    for k in CRITERIA_KEYS:
        out[k] = _clamp_score(data.get(k))
    out["comentario"] = str(data.get("comentario", "")).strip()[:500]
    return out


def run_llm_judge(  # noqa: PLR0914
    golden_path: str | Path = DEFAULT_GOLDEN,
    *,
    top_k: int = 3,
    max_rows: int | None = None,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Gera respostas RAG + pontuações do juiz por item."""

    t = _timeout_seconds(timeout_sec)
    base = (os.environ.get("LLM_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    model = (os.environ.get("OLLAMA_MODEL") or "").strip() or "qwen2.5:3b"

    items = load_golden_items(golden_path)
    limit = max_rows if max_rows is not None else len(items)
    rows_out: list[dict[str, Any]] = []
    sums = {k: 0.0 for k in CRITERIA_KEYS}
    n_scored = 0

    for item in items[:limit]:
        item_id = str(item.get("id", ""))
        q = str(item["query"]).strip()
        reference = str(item["expected_answer"]).strip()
        retrieved = retrieve_contexts(q, top_k=top_k)
        candidate = generate_rag_answer(
            q, retrieved, base_url=base, model=model, timeout_sec=t
        )
        scores = judge_one(
            base_url=base,
            model=model,
            query=q,
            reference=reference,
            candidate=candidate,
            timeout_sec=t,
        )
        mean_3 = sum(scores[k] for k in CRITERIA_KEYS) / len(CRITERIA_KEYS)
        row = {
            "id": item_id,
            "category": item.get("category"),
            "query": q,
            "scores": {k: scores[k] for k in CRITERIA_KEYS},
            "mean_criteria": round(mean_3, 4),
            "comentario_juiz": scores.get("comentario", ""),
        }
        rows_out.append(row)
        for k in CRITERIA_KEYS:
            sums[k] += scores[k]
        n_scored += 1
        logger.info(
            "Juiz %s | média=%.2f | %s",
            item_id,
            mean_3,
            scores,
        )

    agg = {k: round(sums[k] / n_scored, 4) for k in CRITERIA_KEYS} if n_scored else {}
    overall = round(sum(agg.values()) / len(CRITERIA_KEYS), 4) if agg else 0.0

    return {
        "schema": "llm_judge_v1",
        "scale": "1-5 por critério",
        "criteria": {
            "adequacao_negocio": (
                "Alinhamento ao domínio churn/MLOps/dados do projeto "
                "(obrigatório negócio)"
            ),
            "correcao_conteudo": "Consistência com referência e plausibilidade",
            "clareza_utilidade": "Clareza e utilidade",
        },
        "golden": str(Path(golden_path).resolve()),
        "ollama_base": base,
        "ollama_model": model,
        "top_k": top_k,
        "items": rows_out,
        "aggregate_mean_per_criterion": agg,
        "overall_mean": overall,
        "n_items": n_scored,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "LLM-as-judge (≥3 critérios, incl. negócio) sobre o golden set."
        ),
    )
    parser.add_argument("--golden", default=DEFAULT_GOLDEN, help="YAML do golden set")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limitar linhas (teste rápido)",
    )
    parser.add_argument("-o", "--output", default=DEFAULT_OUT, help="JSON de saída")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=f"Segundos (default env RAGAS_TIMEOUT_SEC ou {_timeout_seconds(None)})",
    )
    args = parser.parse_args()

    try:
        payload = run_llm_judge(
            golden_path=args.golden,
            top_k=args.top_k,
            max_rows=args.max_rows,
            timeout_sec=args.timeout,
        )
    except Exception:
        logger.exception("Falha no LLM-as-judge")
        raise SystemExit(1) from None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Médias por critério: %s", payload["aggregate_mean_per_criterion"])
    logger.info(
        "Média global: %s | Salvo em %s",
        payload["overall_mean"],
        out_path.resolve(),
    )


if __name__ == "__main__":
    main()
