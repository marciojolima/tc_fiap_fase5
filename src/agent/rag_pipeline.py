"""Simple RAG retrieval over project documentation/data metadata."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from common.config_loader import ROOT_DIR

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+", re.IGNORECASE)


@dataclass(frozen=True)
class RAGChunk:
    """A lightweight text chunk used by retrieval."""

    source: str
    text: str


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _chunk_text(raw_text: str, source: str, chunk_size: int = 700) -> list[RAGChunk]:
    chunks: list[RAGChunk] = []
    current = []
    current_size = 0
    for line in raw_text.splitlines():
        current.append(line)
        current_size += len(line)
        if current_size >= chunk_size:
            chunks.append(RAGChunk(source=source, text="\n".join(current).strip()))
            current = []
            current_size = 0
    if current:
        chunks.append(RAGChunk(source=source, text="\n".join(current).strip()))
    return [chunk for chunk in chunks if chunk.text]


def load_rag_documents() -> list[RAGChunk]:
    """Load key repository docs as retrieval chunks."""

    candidates = [
        ROOT_DIR / "README.md",
        ROOT_DIR / "STATUS_ATUAL_PROJETO.md",
        ROOT_DIR / "docs" / "DRIFT_MONITORING.md",
        ROOT_DIR / "docs" / "MODEL_CARD.md",
        ROOT_DIR / "data" / "processed" / "feature_columns.json",
        ROOT_DIR / "data" / "processed" / "schema_report.json",
    ]
    chunks: list[RAGChunk] = []
    for path in candidates:
        if not path.exists():
            continue
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(_chunk_text(raw_text, source=str(path.relative_to(ROOT_DIR))))
    return chunks


def retrieve_contexts(query: str, top_k: int = 3) -> list[str]:
    """Return top-k relevant contexts with file provenance."""

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    contexts: list[tuple[float, RAGChunk]] = []
    for chunk in load_rag_documents():
        chunk_tokens = _tokenize(chunk.text)
        if not chunk_tokens:
            continue
        overlap = sum(1 for token in query_tokens if token in chunk_tokens)
        # Small normalization to avoid favoring very large chunks.
        score = overlap / math.sqrt(len(chunk_tokens))
        if score <= 0:
            continue
        contexts.append((score, chunk))

    contexts.sort(key=lambda item: item[0], reverse=True)
    selected = contexts[:top_k]
    return [
        f"[Fonte: {chunk.source}]\n{chunk.text[:900]}"
        for _, chunk in selected
    ]
