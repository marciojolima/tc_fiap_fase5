"""RAG runtime com descoberta automatica, embeddings em memoria e cache local."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, time
from typing import Any

import joblib
import numpy as np

try:
    from fastembed import TextEmbedding
except ModuleNotFoundError:
    TextEmbedding = None  # type: ignore[assignment]

from common.config_loader import ROOT_DIR, load_global_config
from common.logger import get_logger
from monitoring.metrics import report_rag_index_stats, report_rag_query

logger = get_logger("agent.rag_pipeline")

DEFAULT_EMBEDDING_BACKEND = "fastembed"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CACHE_DIR = "artifacts/rag/cache"
DEFAULT_EMBEDDING_CACHE_DIR = "artifacts/rag/fastembed_model_cache"
DEFAULT_HISTORY_PATH = "artifacts/rag/index_build_history.jsonl"
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
DEFAULT_LEXICAL_RERANK_WEIGHT = 0.15
DEFAULT_VECTOR_CANDIDATE_MULTIPLIER = 4
CACHE_SCHEMA_VERSION = "fastembed_rag_index_v1"

JSON_CONTEXT_FILES = (
    "data/processed/feature_columns.json",
    "data/processed/schema_report.json",
    "data/feature_store/export_metadata.json",
    "artifacts/evaluation/model/drift/drift_status.json",
    "artifacts/evaluation/model/drift/drift_metrics.json",
    "artifacts/evaluation/model/retraining/retrain_request.json",
    "artifacts/evaluation/model/retraining/retrain_run.json",
    "artifacts/evaluation/model/retraining/promotion_decision.json",
)

SOURCE_GLOBS = (
    "docs/**/*.md",
    "README.md",
)

WHITESPACE_REPLACEMENTS = ("\r\n", "\r")


@dataclass(frozen=True)
class RAGChunk:
    """Trecho indexado com metadados suficientes para auditoria."""

    source: str
    source_type: str
    chunk_id: int
    text: str
    char_count: int


@dataclass(frozen=True)
class SourceManifestItem:
    """Manifesto de um arquivo fonte observado pelo RAG."""

    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class RAGIndex:
    """Indice vetorial pronto para consulta."""

    chunks: list[RAGChunk]
    embeddings: np.ndarray
    source_manifest: list[SourceManifestItem]
    source_mode: str
    stats: dict[str, Any]


@dataclass(frozen=True)
class RAGRuntimeState:
    """Estado atual do RAG carregado em memoria."""

    index: RAGIndex | None
    encoder: Any | None
    embedding_backend: str
    embedding_model_name: str


_RAG_STATE = RAGRuntimeState(
    index=None,
    encoder=None,
    embedding_backend=DEFAULT_EMBEDDING_BACKEND,
    embedding_model_name="",
)
_RAG_LOCK = threading.Lock()


def _get_runtime_state() -> RAGRuntimeState:
    return _RAG_STATE


def _replace_runtime_state(state: RAGRuntimeState) -> None:
    globals()["_RAG_STATE"] = state


def _rag_config() -> dict[str, Any]:
    config = load_global_config().get("rag", {})
    return {
        "top_k": int(config.get("top_k", DEFAULT_TOP_K)),
        "chunk_size": int(config.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        "chunk_overlap": int(config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
        "embedding_backend": str(
            config.get("embedding_backend", DEFAULT_EMBEDDING_BACKEND)
        ),
        "embedding_model_name": str(
            config.get("embedding_model_name", DEFAULT_EMBEDDING_MODEL)
        ),
        "cache_dir": str(config.get("cache_dir", DEFAULT_CACHE_DIR)),
        "embedding_cache_dir": str(
            config.get("embedding_cache_dir", DEFAULT_EMBEDDING_CACHE_DIR)
        ),
        "history_path": str(config.get("history_path", DEFAULT_HISTORY_PATH)),
        "lexical_rerank_weight": float(
            config.get("lexical_rerank_weight", DEFAULT_LEXICAL_RERANK_WEIGHT)
        ),
        "vector_candidate_multiplier": int(
            config.get(
                "vector_candidate_multiplier",
                DEFAULT_VECTOR_CANDIDATE_MULTIPLIER,
            )
        ),
    }


def _cache_dir() -> Path:
    return ROOT_DIR / _rag_config()["cache_dir"]


def _embedding_cache_dir() -> Path:
    return ROOT_DIR / _rag_config()["embedding_cache_dir"]


def _history_path() -> Path:
    return ROOT_DIR / _rag_config()["history_path"]


def _cache_manifest_path() -> Path:
    return _cache_dir() / "manifest.json"


def _cache_index_path() -> Path:
    return _cache_dir() / "index.joblib"


def _normalize_text(raw_text: str) -> str:
    normalized = raw_text
    for old in WHITESPACE_REPLACEMENTS:
        normalized = normalized.replace(old, "\n")

    lines = [line.strip() for line in normalized.split("\n")]
    collapsed_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if previous_blank:
                continue
            previous_blank = True
            collapsed_lines.append("")
            continue
        previous_blank = False
        collapsed_lines.append(" ".join(line.split()))

    return "\n".join(collapsed_lines).strip()


def _read_text_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return _normalize_text(raw)
        return json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)
    return _normalize_text(raw)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_rag_source_paths() -> list[Path]:
    """Descobre automaticamente o corpus markdown e adiciona JSON relevantes."""

    discovered: set[Path] = set()

    for pattern in SOURCE_GLOBS:
        matched = sorted(ROOT_DIR.glob(pattern))
        for path in matched:
            if path.is_file():
                discovered.add(path)

    for relative_path in JSON_CONTEXT_FILES:
        candidate = ROOT_DIR / relative_path
        if candidate.is_file():
            discovered.add(candidate)

    non_empty_sources: list[Path] = []
    for path in sorted(discovered):
        if path.stat().st_size <= 0:
            continue
        non_empty_sources.append(path)
    return non_empty_sources


def _build_source_manifest(paths: list[Path]) -> list[SourceManifestItem]:
    manifest: list[SourceManifestItem] = []
    for path in paths:
        stat = path.stat()
        manifest.append(
            SourceManifestItem(
                path=str(path.relative_to(ROOT_DIR)),
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=_hash_file(path),
            )
        )
    return manifest


def _chunk_text(
    raw_text: str,
    source: str,
    *,
    source_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[RAGChunk]:
    if not raw_text.strip():
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in raw_text.split("\n\n")
        if paragraph.strip()
    ]
    chunks: list[RAGChunk] = []
    current = ""
    chunk_id = 0

    def flush() -> None:
        nonlocal current, chunk_id
        cleaned = current.strip()
        if not cleaned:
            current = ""
            return
        chunks.append(
            RAGChunk(
                source=source,
                source_type=source_type,
                chunk_id=chunk_id,
                text=cleaned,
                char_count=len(cleaned),
            )
        )
        chunk_id += 1
        if chunk_overlap > 0 and cleaned:
            current = cleaned[-chunk_overlap:]
        else:
            current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            flush()

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        step = max(1, chunk_size - chunk_overlap)
        start = 0
        while start < len(paragraph):
            window = paragraph[start : start + chunk_size].strip()
            if window:
                chunks.append(
                    RAGChunk(
                        source=source,
                        source_type=source_type,
                        chunk_id=chunk_id,
                        text=window,
                        char_count=len(window),
                    )
                )
                chunk_id += 1
            start += step
        current = ""

    if current:
        flush()
    return chunks


def _load_and_chunk_sources(paths: list[Path]) -> tuple[list[RAGChunk], int]:
    cfg = _rag_config()
    chunks: list[RAGChunk] = []
    total_source_bytes = 0
    for path in paths:
        text = _read_text_file(path)
        if not text:
            continue
        total_source_bytes += path.stat().st_size
        chunks.extend(
            _chunk_text(
                text,
                str(path.relative_to(ROOT_DIR)),
                source_type=path.suffix.lower().lstrip(".") or "text",
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
            )
        )
    return chunks, total_source_bytes


def _estimate_chunks_memory_bytes(chunks: list[RAGChunk]) -> int:
    total = 0
    for chunk in chunks:
        total += len(chunk.text.encode("utf-8"))
        total += len(chunk.source.encode("utf-8"))
        total += len(chunk.source_type.encode("utf-8"))
        total += 24
    return total


def _get_process_rss_bytes() -> int:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        return 0
    if rss <= 0:
        return 0
    # Linux retorna KB; macOS retorna bytes. O ambiente do projeto e Linux.
    return int(rss * 1024)


def _load_fastembed_encoder(model_name: str, cache_dir: Path) -> Any:
    if TextEmbedding is None:
        raise RuntimeError(
            "Dependencia 'fastembed' nao instalada. "
            "Instale o extra serving ou rode 'poetry install --extras serving'."
        )

    return TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))


def _load_encoder(*, backend: str, model_name: str) -> Any:
    state = _get_runtime_state()
    if (
        state.encoder is not None
        and state.embedding_backend == backend
        and state.embedding_model_name == model_name
    ):
        return state.encoder

    embedding_cache_dir = _embedding_cache_dir()
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Carregando modelo de embeddings do RAG | backend=%s | model=%s | cache_dir=%s",
        backend,
        model_name,
        embedding_cache_dir,
    )
    if backend != "fastembed":
        raise ValueError(f"Backend de embeddings '{backend}' nao suportado pelo RAG.")

    encoder = _load_fastembed_encoder(model_name, embedding_cache_dir)
    _replace_runtime_state(
        RAGRuntimeState(
            index=state.index,
            encoder=encoder,
            embedding_backend=backend,
            embedding_model_name=model_name,
        )
    )
    return encoder


def _normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return matrix / norms


def _encode_texts(encoder: Any, texts: list[str], *, query: bool = False) -> np.ndarray:
    if query and hasattr(encoder, "query_embed"):
        embeddings = list(encoder.query_embed(texts))
    elif not query and hasattr(encoder, "passage_embed"):
        embeddings = list(encoder.passage_embed(texts))
    else:
        embeddings = list(encoder.embed(texts))
    return _normalize_embeddings(embeddings)


def _serialize_index(index: RAGIndex) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "embedding_backend": index.stats.get(
            "embedding_backend",
            DEFAULT_EMBEDDING_BACKEND,
        ),
        "embedding_model_name": index.stats.get(
            "embedding_model_name",
            DEFAULT_EMBEDDING_MODEL,
        ),
        "chunks": [asdict(chunk) for chunk in index.chunks],
        "embeddings": index.embeddings.astype(np.float32),
        "source_manifest": [asdict(item) for item in index.source_manifest],
        "source_mode": index.source_mode,
        "stats": index.stats,
    }


def _deserialize_index(payload: dict[str, Any]) -> RAGIndex:
    chunks = [RAGChunk(**chunk) for chunk in payload["chunks"]]
    source_manifest = [
        SourceManifestItem(**item) for item in payload["source_manifest"]
    ]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    return RAGIndex(
        chunks=chunks,
        embeddings=embeddings,
        source_manifest=source_manifest,
        source_mode=str(payload.get("source_mode", "cache")),
        stats=dict(payload.get("stats", {})),
    )


def _save_cache(index: RAGIndex) -> None:
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_manifest_path().write_text(
        json.dumps(
            [asdict(item) for item in index.source_manifest],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    joblib.dump(_serialize_index(index), _cache_index_path(), compress=3)


def _load_cache(  # noqa: PLR0911
    manifest: list[SourceManifestItem],
    *,
    backend: str,
    model_name: str,
) -> RAGIndex | None:
    manifest_path = _cache_manifest_path()
    index_path = _cache_index_path()
    if not manifest_path.exists() or not index_path.exists():
        return None

    try:
        cached_manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_manifest = [SourceManifestItem(**item) for item in cached_manifest_raw]
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Manifesto de cache do RAG invalido; reconstruindo indice.")
        return None

    if cached_manifest != manifest:
        return None

    try:
        payload = joblib.load(index_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Falha ao carregar cache do RAG; reconstruindo indice. erro=%s",
            exc,
        )
        return None
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        logger.info("Schema do cache do RAG mudou; reconstruindo indice.")
        return None
    if (
        payload.get("embedding_backend") != backend
        or payload.get("embedding_model_name") != model_name
    ):
        logger.info("Modelo/backend do cache do RAG mudou; reconstruindo indice.")
        return None
    return _deserialize_index(payload)


def _append_history(event: dict[str, Any]) -> None:
    history_path = _history_path()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _set_runtime_index(
    index: RAGIndex,
    encoder: Any,
    backend: str,
    model_name: str,
) -> None:
    _replace_runtime_state(
        RAGRuntimeState(
            index=index,
            encoder=encoder,
            embedding_backend=backend,
            embedding_model_name=model_name,
        )
    )


def _collect_source_manifest_and_encoder(
    *,
    backend: str,
    model_name: str,
    force_rebuild: bool,
) -> tuple[
    list[Path],
    list[SourceManifestItem],
    RAGIndex | None,
    Any,
    dict[str, float],
]:
    stage_durations: dict[str, float] = {}

    discover_start = perf_counter()
    source_paths = discover_rag_source_paths()
    source_manifest = _build_source_manifest(source_paths)
    stage_durations["source_discovery"] = perf_counter() - discover_start

    cache_index: RAGIndex | None = None
    if not force_rebuild:
        cache_start = perf_counter()
        cache_index = _load_cache(
            source_manifest,
            backend=backend,
            model_name=model_name,
        )
        stage_durations["cache_lookup"] = perf_counter() - cache_start

    encoder_start = perf_counter()
    encoder = _load_encoder(backend=backend, model_name=model_name)
    stage_durations["encoder_load"] = perf_counter() - encoder_start

    return source_paths, source_manifest, cache_index, encoder, stage_durations


def _build_index_from_sources(  # noqa: PLR0913
    *,
    source_paths: list[Path],
    source_manifest: list[SourceManifestItem],
    encoder: Any,
    backend: str,
    model_name: str,
    stage_durations: dict[str, float],
) -> RAGIndex:
    load_sources_start = perf_counter()
    chunks, source_bytes = _load_and_chunk_sources(source_paths)
    stage_durations["source_load_and_chunk"] = perf_counter() - load_sources_start

    if not chunks:
        raise RuntimeError("Nenhum chunk valido foi gerado para o RAG.")

    embed_start = perf_counter()
    embeddings = _encode_texts(encoder, [chunk.text for chunk in chunks])
    stage_durations["embedding"] = perf_counter() - embed_start

    index = RAGIndex(
        chunks=chunks,
        embeddings=embeddings,
        source_manifest=source_manifest,
        source_mode="fresh",
        stats={
            "source_bytes": source_bytes,
            "embedding_backend": backend,
            "embedding_model_name": model_name,
        },
    )

    cache_write_start = perf_counter()
    _save_cache(index)
    stage_durations["cache_write"] = perf_counter() - cache_write_start
    return index


def _finalize_index_stats(  # noqa: PLR0913
    *,
    source_manifest: list[SourceManifestItem],
    index: RAGIndex,
    build_source: str,
    backend: str,
    model_name: str,
    stage_durations: dict[str, float],
    rss_before: int,
    overall_start: float,
) -> dict[str, Any]:
    chunks_memory_bytes = _estimate_chunks_memory_bytes(index.chunks)
    embeddings_bytes = int(index.embeddings.nbytes)
    source_bytes = int(
        sum(item.size_bytes for item in source_manifest)
        if build_source == "cache"
        else index.stats.get("source_bytes", 0)
    )
    index_estimated_memory_bytes = chunks_memory_bytes + embeddings_bytes
    process_rss_delta_bytes = max(0, _get_process_rss_bytes() - rss_before)
    total_duration_seconds = perf_counter() - overall_start

    return {
        "initialized_at_epoch": time(),
        "build_source": build_source,
        "embedding_backend": backend,
        "embedding_model_name": model_name,
        "file_count": len(source_manifest),
        "chunk_count": len(index.chunks),
        "source_bytes": source_bytes,
        "chunks_memory_bytes": chunks_memory_bytes,
        "embeddings_bytes": embeddings_bytes,
        "index_estimated_memory_bytes": index_estimated_memory_bytes,
        "process_rss_delta_bytes": process_rss_delta_bytes,
        "total_duration_seconds": total_duration_seconds,
        "stage_durations_seconds": stage_durations,
        "cache_hit": build_source == "cache",
    }


def initialize_rag_index(  # noqa: PLR0914, PLR0915
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Inicializa o RAG no startup ou sob demanda, com cache persistido."""

    with _RAG_LOCK:
        cfg = _rag_config()
        backend = cfg["embedding_backend"]
        model_name = cfg["embedding_model_name"]
        state = _get_runtime_state()

        if not force_rebuild and state.index is not None:
            return dict(state.index.stats)

        overall_start = perf_counter()
        rss_before = _get_process_rss_bytes()
        (
            source_paths,
            source_manifest,
            cache_index,
            encoder,
            stage_durations,
        ) = _collect_source_manifest_and_encoder(
            backend=backend,
            model_name=model_name,
            force_rebuild=force_rebuild,
        )

        if cache_index is not None:
            build_source = "cache"
            index = cache_index
        else:
            build_source = "fresh"
            index = _build_index_from_sources(
                source_paths=source_paths,
                source_manifest=source_manifest,
                encoder=encoder,
                backend=backend,
                model_name=model_name,
                stage_durations=stage_durations,
            )

        memory_allocation_start = perf_counter()
        _set_runtime_index(index, encoder, backend, model_name)
        stage_durations["memory_ready"] = perf_counter() - memory_allocation_start

        stats = _finalize_index_stats(
            source_manifest=source_manifest,
            index=index,
            build_source=build_source,
            backend=backend,
            model_name=model_name,
            stage_durations=stage_durations,
            rss_before=rss_before,
            overall_start=overall_start,
        )
        hydrated_index = RAGIndex(
            chunks=index.chunks,
            embeddings=index.embeddings,
            source_manifest=source_manifest,
            source_mode=build_source,
            stats=stats,
        )
        _set_runtime_index(hydrated_index, encoder, backend, model_name)
        report_rag_index_stats(stats)
        _append_history(stats)

        logger.info(
            "RAG pronto | origem=%s | arquivos=%d | chunks=%d | tempo=%.3fs | "
            "indice_mem=%d bytes | rss_delta=%d bytes",
            build_source,
            stats["file_count"],
            stats["chunk_count"],
            stats["total_duration_seconds"],
            stats["index_estimated_memory_bytes"],
            stats["process_rss_delta_bytes"],
        )
        return stats


def _ensure_ready() -> RAGIndex:
    state = _get_runtime_state()
    if state.index is None:
        initialize_rag_index()
        state = _get_runtime_state()
    if state.index is None:
        raise RuntimeError("RAG nao inicializado.")
    return state.index


def _lexical_overlap_score(query_terms: list[str], chunk_text: str) -> float:
    if not query_terms:
        return 0.0
    chunk_terms = set(term for term in chunk_text.lower().split() if term)
    if not chunk_terms:
        return 0.0
    overlap = sum(1 for term in query_terms if term in chunk_terms)
    return overlap / max(1, len(set(query_terms)))


def _format_chunk(chunk: RAGChunk) -> str:
    return (
        "[Fonte: "
        f"{chunk.source} | tipo: {chunk.source_type} | chunk: {chunk.chunk_id}]\n"
        f"{chunk.text}"
    )


def _rank_candidates(  # noqa: PLR0913
    *,
    cleaned_query: str,
    index: RAGIndex,
    vector_scores: np.ndarray,
    limit: int,
    lexical_weight: float,
    candidate_multiplier: int,
) -> list[int]:
    candidate_count = min(
        len(index.chunks),
        max(limit, limit * candidate_multiplier),
    )
    candidate_idx = np.argpartition(vector_scores, -candidate_count)[-candidate_count:]
    candidate_idx = candidate_idx[np.argsort(vector_scores[candidate_idx])[::-1]]

    query_terms = [term for term in cleaned_query.lower().split() if term]
    scored_candidates: list[tuple[float, int]] = []
    for idx in candidate_idx:
        lexical_score = _lexical_overlap_score(query_terms, index.chunks[int(idx)].text)
        combined_score = (1.0 - lexical_weight) * float(vector_scores[int(idx)]) + (
            lexical_weight * lexical_score
        )
        scored_candidates.append((combined_score, int(idx)))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in scored_candidates[:limit]]


def retrieve_contexts(  # noqa: PLR0914
    query: str,
    top_k: int | None = None,
) -> list[str]:
    """Busca vetorial com rerank lexical leve sobre o indice em memoria."""

    started_at = perf_counter()
    cleaned_query = _normalize_text(query)
    if not cleaned_query:
        return []

    cfg = _rag_config()
    index = _ensure_ready()
    state = _get_runtime_state()
    encoder = state.encoder
    if encoder is None:
        raise RuntimeError("Modelo de embeddings do RAG nao esta carregado.")

    query_embedding = _encode_texts(encoder, [cleaned_query], query=True)[0]
    vector_scores = index.embeddings @ query_embedding

    limit = top_k or cfg["top_k"]
    selected_indices = _rank_candidates(
        cleaned_query=cleaned_query,
        index=index,
        vector_scores=vector_scores,
        limit=limit,
        lexical_weight=cfg["lexical_rerank_weight"],
        candidate_multiplier=cfg["vector_candidate_multiplier"],
    )
    contexts = [_format_chunk(index.chunks[idx]) for idx in selected_indices]
    report_rag_query(
        duration_seconds=perf_counter() - started_at,
        top_k=limit,
        returned_contexts=len(contexts),
    )
    return contexts


def get_rag_runtime_summary() -> dict[str, Any]:
    """Resumo util para status e debug operacional."""

    state = _get_runtime_state()
    if state.index is None:
        return {
            "ready": False,
            "embedding_backend": state.embedding_backend,
            "embedding_model_name": state.embedding_model_name,
            "cache_path": str(_cache_index_path().relative_to(ROOT_DIR)),
            "embedding_cache_path": str(
                _embedding_cache_dir().relative_to(ROOT_DIR)
            ),
        }

    summary = dict(state.index.stats)
    summary.update(
        {
            "ready": True,
            "cache_path": str(_cache_index_path().relative_to(ROOT_DIR)),
            "embedding_cache_path": str(
                _embedding_cache_dir().relative_to(ROOT_DIR)
            ),
            "history_path": str(_history_path().relative_to(ROOT_DIR)),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inicializa ou reconstrói o índice RAG persistido em joblib.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Reconstrói embeddings e cache mesmo quando o manifesto não mudou.",
    )
    args = parser.parse_args()
    stats = initialize_rag_index(force_rebuild=args.force_rebuild)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
