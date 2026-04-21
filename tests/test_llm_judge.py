from __future__ import annotations

from unittest.mock import patch

from evaluation.llm_judge import CRITERIA_KEYS, _extract_json_object, judge_one


def test_extract_json_with_fence() -> None:
    raw = '```json\n{"adequacao_negocio": 4, "correcao_conteudo": 5, "clareza_utilidade": 3, "comentario": "ok"}\n```'
    d = _extract_json_object(raw)
    assert d["adequacao_negocio"] == 4


def test_judge_one_parses_scores() -> None:
    def fake_chat(*_a: object, **_k: object) -> str:
        return (
            '{"adequacao_negocio": 4, "correcao_conteudo": 3, "clareza_utilidade": 5, '
            '"comentario": "teste"}'
        )

    with patch("evaluation.llm_judge.ollama_chat", fake_chat):
        out = judge_one(
            base_url="http://x",
            model="m",
            query="q?",
            reference="ref",
            candidate="cand",
            timeout_sec=60,
        )
    for k in CRITERIA_KEYS:
        assert k in out
        assert 1 <= out[k] <= 5
    assert out["comentario"] == "teste"

