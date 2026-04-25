from __future__ import annotations

from src.evaluation.llm_agent.ab_test_prompts import (
    PROMPT_VARIANTS,
    compute_keyword_coverage,
    extract_reference_terms,
    run_prompt_ab_test,
)

EXPECTED_VARIANTS = 3


def test_extract_reference_terms_keeps_routes_and_tools() -> None:
    reference = (
        "Prefixo /llm: health, status e chat; tools como rag_search e "
        "predict_churn."
    )
    terms = extract_reference_terms(reference)

    assert "/llm" in terms
    assert "rag_search" in terms
    assert "predict_churn" in terms


def test_compute_keyword_coverage_reports_hits_and_missing_terms() -> None:
    metrics = compute_keyword_coverage(
        answer="As rotas são /llm/health, /llm/status e /llm/chat.",
        reference="Prefixo /llm: health, status e chat.",
    )

    assert metrics["keyword_coverage"] > 0
    assert "/llm" in metrics["matched_terms"]


def test_run_prompt_ab_test_returns_three_variants(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.evaluation.llm_agent.ab_test_prompts.load_golden_items",
        lambda _path: [
            {
                "id": "gs-test",
                "category": "rag_llm",
                "query": "Quais rotas HTTP o projeto expõe para o assistente LLM?",
                "expected_answer": (
                    "Prefixo /llm: health, status e chat para o assistente."
                ),
            }
        ],
    )
    monkeypatch.setattr(
        "src.evaluation.llm_agent.ab_test_prompts.retrieve_contexts",
        lambda *_args, **_kwargs: [
            "README.md\nRotas /llm/health, /llm/status e /llm/chat."
        ],
    )
    monkeypatch.setattr(
        "src.evaluation.llm_agent.ab_test_prompts.build_llm_client",
        lambda *_args, **_kwargs: type(
            "StubProvider",
            (),
            {
                "metadata": staticmethod(
                    lambda: {"provider": "stub", "model_name": "modelo-teste"}
                )
            },
        )(),
    )

    def fake_generate_answer(
        _question: str,
        _contexts: list[str],
        variant: object,
    ) -> str:
        assert variant in PROMPT_VARIANTS
        return (
            "As rotas são /llm/health, /llm/status e /llm/chat."
            if variant.name != "baseline"
            else "O projeto possui rotas do prefixo /llm."
        )

    monkeypatch.setattr(
        "src.evaluation.llm_agent.ab_test_prompts.generate_prompt_answer",
        fake_generate_answer,
    )

    result = run_prompt_ab_test(max_rows=1)

    assert result["schema"] == "prompt_ab_v1"
    assert len(result["prompt_variants"]) == EXPECTED_VARIANTS
    assert result["aggregate"]["n_items"] == 1
    assert len(result["items"][0]["variants"]) == EXPECTED_VARIANTS
    assert result["aggregate"]["variants"][0]["mean_keyword_coverage"] >= 0
