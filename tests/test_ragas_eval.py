from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.evaluation.llm_agent import ragas_eval

_MIN_GOLDEN_ITEMS = 20
_FAKE_MAX_TOKENS = 123
_EXPECTED_VECTOR_SIZE = 2
_EXPECTED_DOCUMENT_COUNT = 2


def test_load_golden_items_from_default_path() -> None:
    path = Path("data/golden-set.json")
    items = ragas_eval.load_golden_items(path)
    assert len(items) >= _MIN_GOLDEN_ITEMS
    assert all("query" in i and "expected_answer" in i for i in items)


class FakeTextEmbedding:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir

    def passage_embed(self, texts: list[str]) -> list[np.ndarray]:
        _ = self.model_name
        return [np.asarray([float(len(text)), 1.0], dtype=np.float32) for text in texts]

    def query_embed(self, texts: list[str]) -> list[np.ndarray]:
        _ = self.cache_dir
        return [np.asarray([1.0, float(len(text))], dtype=np.float32) for text in texts]


def test_fastembed_langchain_embeddings_adapter_uses_fastembed(monkeypatch) -> None:
    monkeypatch.setattr(ragas_eval, "TextEmbedding", FakeTextEmbedding)

    embedder = ragas_eval.FastEmbedLangChainEmbeddings(
        model_name="fake-model",
        cache_dir="artifacts/rag/fastembed_model_cache",
    )

    documents = embedder.embed_documents(["abc", "de"])
    query = embedder.embed_query("abcd")

    assert len(documents) == _EXPECTED_DOCUMENT_COUNT
    assert len(query) == _EXPECTED_VECTOR_SIZE
    assert np.isclose(np.linalg.norm(documents[0]), 1.0)
    assert np.isclose(np.linalg.norm(query), 1.0)
    assert embedder.model_name == "fake-model"
    assert embedder.cache_dir == "artifacts/rag/fastembed_model_cache"


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
