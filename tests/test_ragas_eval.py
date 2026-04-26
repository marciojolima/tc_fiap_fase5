from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.evaluation.llm_agent import ragas_eval

_MIN_GOLDEN_ITEMS = 20
_FAKE_MAX_TOKENS = 123
_EXPECTED_VECTOR_SIZE = 2
_EXPECTED_DOCUMENT_COUNT = 2
_EXPECTED_TRACE_STEPS = 2
_FAKE_TIMEOUT = 30


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


class FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def test_generate_serving_chat_answer_calls_llm_chat_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> FakeHttpResponse:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHttpResponse(
            {
                "answer": "Resposta pelo serving.",
                "used_tools": ["rag_search"],
                "trace": [
                    {
                        "action": "rag_search",
                        "observation": json.dumps(
                            {
                                "tool_name": "rag_search",
                                "retrieved_contexts": ["ctx 1", "ctx 2"],
                                "evidence": ["evidencia resumida"],
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
            }
        )

    monkeypatch.setattr(ragas_eval, "urlopen", fake_urlopen)

    result = ragas_eval.generate_serving_chat_answer(
        question="Explique o RAG.",
        serving_base_url="http://serving:8000",
        timeout_sec=_FAKE_TIMEOUT,
    )

    assert captured["url"] == "http://serving:8000/llm/chat"
    assert captured["timeout"] == float(_FAKE_TIMEOUT)
    assert captured["payload"] == {
        "message": "Explique o RAG.",
        "include_trace": True,
        "answer_style": "medium",
    }
    assert result.answer == "Resposta pelo serving."
    assert result.retrieved_contexts == ["ctx 1", "ctx 2"]
    assert result.used_tools == ["rag_search"]
    assert result.trace_steps == 1


def test_extract_rag_contexts_falls_back_to_evidence() -> None:
    trace = [
        {
            "action": "rag_search",
            "observation": json.dumps(
                {
                    "tool_name": "rag_search",
                    "evidence": ["evidencia 1", "evidencia 2"],
                },
                ensure_ascii=False,
            ),
        }
    ]

    assert ragas_eval._extract_rag_contexts_from_trace(trace) == [
        "evidencia 1",
        "evidencia 2",
    ]


def test_build_dataset_uses_serving_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        ragas_eval,
        "generate_serving_chat_answer",
        lambda **_kwargs: ragas_eval.ServingChatEvaluation(
            answer="Resposta servida.",
            retrieved_contexts=["contexto servido"],
            used_tools=["rag_search"],
            trace_steps=_EXPECTED_TRACE_STEPS,
        ),
    )

    dataset = ragas_eval.build_dataset(
        [
            {
                "id": "gs-test",
                "query": "Pergunta?",
                "expected_answer": "Referência.",
                "contexts": ["contexto curado"],
                "category": "rag_llm",
            }
        ],
        max_rows=None,
        serving_base_url="http://127.0.0.1:8000",
        timeout_sec=_FAKE_TIMEOUT,
    )
    row = dataset[0]

    assert row["response"] == "Resposta servida."
    assert row["retrieved_contexts"] == ["contexto servido"]
    assert row["reference_contexts"] == ["contexto curado"]
    assert row["serving_used_tools"] == ["rag_search"]
    assert row["serving_trace_steps"] == _EXPECTED_TRACE_STEPS


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
