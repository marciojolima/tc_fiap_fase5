from __future__ import annotations

from pathlib import Path

import numpy as np

from agent import rag_pipeline


class FakeSentenceTransformer:
    def __init__(self, _model_name: str):
        self.model_name = _model_name

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        del batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings
        _ = self.model_name
        rows: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            rows.append(
                [
                    float(lowered.count("drift")),
                    float(lowered.count("rota") + lowered.count("/llm")),
                    float(lowered.count("churn")),
                ]
            )
        return np.asarray(rows, dtype=np.float32)


def _patch_rag_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rag_pipeline, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(rag_pipeline, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(
        rag_pipeline,
        "_RAG_STATE",
        rag_pipeline.RAGRuntimeState(
            index=None,
            encoder=None,
            embedding_model_name="",
        ),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "load_global_config",
        lambda: {
            "rag": {
                "top_k": 2,
                "chunk_size": 120,
                "chunk_overlap": 20,
                "embedding_model_name": "fake-model",
                "cache_dir": "artifacts/rag/cache",
                "history_path": "artifacts/rag/index_build_history.jsonl",
                "lexical_rerank_weight": 0.20,
                "vector_candidate_multiplier": 3,
            }
        },
    )


def test_discover_rag_source_paths_collects_docs_readme_and_hardcoded_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_rag_environment(monkeypatch, tmp_path)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "README.md").write_text("Projeto com RAG.\n", encoding="utf-8")
    (tmp_path / "docs" / "guide.md").write_text("Guia do RAG.\n", encoding="utf-8")
    (tmp_path / "docs" / "empty.md").write_text("", encoding="utf-8")
    (tmp_path / "data" / "processed" / "feature_columns.json").write_text(
        '{"features": ["Age", "Balance"]}',
        encoding="utf-8",
    )

    discovered = rag_pipeline.discover_rag_source_paths()
    relative_paths = {str(path.relative_to(tmp_path)) for path in discovered}

    assert "README.md" in relative_paths
    assert "docs/guide.md" in relative_paths
    assert "data/processed/feature_columns.json" in relative_paths
    assert "docs/empty.md" not in relative_paths


def test_initialize_rag_index_uses_cache_and_answers_queries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_rag_environment(monkeypatch, tmp_path)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "monitoring" / "drift").mkdir(parents=True, exist_ok=True)
    (tmp_path / "README.md").write_text(
        "As rotas /llm/health, /llm/status e /llm/chat fazem parte da API.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "drift.md").write_text(
        "O monitoramento de drift usa PSI e artefatos de drift.\n",
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "monitoring" / "drift" / "drift_status.json").write_text(
        '{"status": "warning", "message": "drift monitorado"}',
        encoding="utf-8",
    )

    first_stats = rag_pipeline.initialize_rag_index(force_rebuild=True)
    assert first_stats["build_source"] == "fresh"
    assert first_stats["chunk_count"] >= 1

    monkeypatch.setattr(
        rag_pipeline,
        "_RAG_STATE",
        rag_pipeline.RAGRuntimeState(
            index=None,
            encoder=None,
            embedding_model_name="",
        ),
    )

    second_stats = rag_pipeline.initialize_rag_index()
    assert second_stats["build_source"] == "cache"

    contexts = rag_pipeline.retrieve_contexts("Como o projeto monitora drift?", top_k=1)
    assert contexts
    assert "drift" in contexts[0].lower()
