from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.evaluation.llm_agent import ragas_eval

_MIN_GOLDEN_ITEMS = 20
_FAKE_MAX_TOKENS = 123


def test_load_golden_items_from_default_path() -> None:
    path = Path("data/golden-set.json")
    items = ragas_eval.load_golden_items(path)
    assert len(items) >= _MIN_GOLDEN_ITEMS
    assert all("query" in i and "expected_answer" in i for i in items)


def test_claude_ragas_factory_removes_conflicting_top_p(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}
    fake_llm = SimpleNamespace(model_args={"temperature": 0, "top_p": 0.1})

    class FakeAnthropic:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def fake_llm_factory(*_args: object, **kwargs: object) -> object:
        captured_kwargs.update(kwargs)
        return fake_llm

    monkeypatch.setattr(
        "src.evaluation.llm_agent.ragas_eval.import_module",
        lambda _name: SimpleNamespace(Anthropic=FakeAnthropic),
    )
    monkeypatch.setattr(
        "src.evaluation.llm_agent.ragas_eval.resolve_llm_api_key",
        lambda *_args, **_kwargs: "fake-key",
    )
    monkeypatch.setattr(
        "src.evaluation.llm_agent.ragas_eval.resolve_llm_base_url",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "src.evaluation.llm_agent.ragas_eval.resolve_llm_max_tokens",
        lambda *_args, **_kwargs: _FAKE_MAX_TOKENS,
    )
    monkeypatch.setattr(
        "src.evaluation.llm_agent.ragas_eval.llm_factory",
        fake_llm_factory,
    )

    ragas_eval._instructor_llm_from_provider("claude", "claude-test", 30)

    assert captured_kwargs["provider"] == "anthropic"
    assert captured_kwargs["max_tokens"] == _FAKE_MAX_TOKENS
    assert captured_kwargs["temperature"] == 0
    assert fake_llm.model_args == {"temperature": 0}
