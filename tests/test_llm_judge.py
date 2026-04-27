from __future__ import annotations

from unittest.mock import patch

from src.evaluation.llm_agent.llm_judge import (
    CRITERIA_KEYS,
    extract_json_object,
    judge_one,
    process_item,
)

_EXPECTED_ADEQUACAO = 4
_SCORE_MAX = 5
_ENDPOINT_URL = "http://localhost:8000/llm/chat"


def test_extract_json_with_fence() -> None:
    raw = (
        "```json\n"
        '{"adequacao_negocio": 4, '
        '"correcao_conteudo": 5, '
        '"clareza_utilidade": 3, '
        '"comentario": "ok"}'
        "\n```"
    )

    data = extract_json_object(raw)

    assert data["adequacao_negocio"] == _EXPECTED_ADEQUACAO


def test_judge_one_parses_scores() -> None:
    def fake_chat(*_args: object, **_kwargs: object) -> str:
        return (
            '{"adequacao_negocio": 4, '
            '"correcao_conteudo": 3, '
            '"clareza_utilidade": 5, '
            '"comentario": "teste"}'
        )

    with patch("src.evaluation.llm_agent.llm_judge.provider_chat", fake_chat):
        output = judge_one(
            query="q?",
            reference="ref",
            candidate="cand",
            timeout_sec=60,
        )

    for key in CRITERIA_KEYS:
        assert key in output
        assert 1 <= output[key] <= _SCORE_MAX

    assert output["comentario"] == "teste"


def test_process_item_uses_endpoint_and_judge() -> None:
    _SCORE_ADEQUACAO_ENDPOINT = 5
    _SCORE_CORRECAO_ENDPOINT = 5
    _SCORE_CLAREZA_ENDPOINT = 4
    item = _golden_item()
    endpoint_payload = {
        "answer": (
            "NumOfProducts representa a quantidade de produtos do cliente."
        ),
        "used_tools": ["rag_search"],
        "trace": [{"action": "rag_search"}],
    }
    scores = {
        "adequacao_negocio": _SCORE_ADEQUACAO_ENDPOINT,
        "correcao_conteudo": _SCORE_CORRECAO_ENDPOINT,
        "clareza_utilidade": _SCORE_CLAREZA_ENDPOINT,
        "comentario": "boa resposta",
    }

    with (
        patch(
            "src.evaluation.llm_agent.llm_judge.call_llm_chat_endpoint",
            return_value=endpoint_payload,
        ),
        patch(
            "src.evaluation.llm_agent.llm_judge.judge_one",
            return_value=scores,
        ),
    ):
        row, mean_score = process_item(
            item=item,
            endpoint_url=_ENDPOINT_URL,
            answer_style="short",
            timeout=60,
        )

    assert row["id"] == "gs-003"
    assert row["candidate_answer"] == endpoint_payload["answer"]
    assert row["used_tools"] == ["rag_search"]
    assert row["trace"] == [{"action": "rag_search"}]
    assert row["scores"]["adequacao_negocio"] == _SCORE_ADEQUACAO_ENDPOINT
    assert row["scores"]["correcao_conteudo"] == _SCORE_CORRECAO_ENDPOINT
    assert row["scores"]["clareza_utilidade"] == _SCORE_CLAREZA_ENDPOINT
    assert row["comentario_juiz"] == "boa resposta"
    assert row["endpoint_url"] == _ENDPOINT_URL
    assert row["answer_style"] == "short"
    assert mean_score == 14 / 3


def test_process_item_returns_failure_row_when_endpoint_fails() -> None:
    item = _golden_item(contexts=[])

    with patch(
        "src.evaluation.llm_agent.llm_judge.call_llm_chat_endpoint",
        side_effect=RuntimeError("endpoint fora do ar"),
    ):
        row, mean_score = process_item(
            item=item,
            endpoint_url=_ENDPOINT_URL,
            answer_style="short",
            timeout=60,
        )

    assert row["id"] == "gs-003"
    assert not row["candidate_answer"]
    assert row["used_tools"] == []
    assert row["trace"] == []
    assert row["endpoint_error"] == "endpoint fora do ar"
    assert row["mean_criteria"] == 1.0
    assert mean_score == 1.0


def _golden_item(contexts: list[str] | None = None) -> dict[str, object]:
    return {
        "id": "gs-003",
        "query": "O que representa NumOfProducts?",
        "expected_answer": "Número de produtos bancários ativos.",
        "contexts": ["contexto"] if contexts is None else contexts,
        "expected_tools": ["rag_search"],
        "metadata": {"category": "negocio_churn"},
        "category": "negocio_churn",
    }
