"""LightRAG-backed chunk retrieval shaped like the old vector API.

Drop-in ``retrieve_chunks`` for the KG branch: embeds the query the way
LightRAG does for queries (``embedding_func([query], context="query")``),
searches the chunk vector store, loads the full chunk records from
``text_chunks``, and optionally reranks with ``rerank_model_func``. Results
keep the OLD field names (``RetrievedDocV2``) so existing clients keep working.

LightRAG call signatures mirrored (lightrag 1.5.7, see ``.venv``):
- query embedding: ``await lg.embedding_func([query], context="query")`` -> ``[vec]``
- vector search:   ``await lg.chunks_vdb.query(query, top_k=n, query_embedding=vec)``
                   -> ``[{"id", "distance", "content", ...}]``
- full records:    ``await lg.text_chunks.get_by_ids(ids)`` -> ``[record | None]``
- rerank:          ``await rerank_func(query=..., documents=[text, ...], top_n=k)``
                   -> ``[{"index", "relevance_score"}, ...]``
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# Stored chunk keys echoed back to the client only when the record carries them.
# ``asset_key`` comes from the ingest-time backfill (images/tables); the API
# layer resolves it to a presigned ``artifact_url`` before responding.
_PASSTHROUGH_KEYS = ("page", "modality", "caption", "table_body", "artifact_url", "asset_key")

# Chunks shorter than this are ingest-era junk (fixed-window tail chunks like
# ":" / "sẻ:" that slipped into the DB before the chunker filter). They never
# reach rerank or the client. Chunk text is multilingual, so chars — not
# tokens — are the cheap proxy here.
MIN_CHUNK_CHARS = 20


def _rerank_min_score() -> float:
    """Optional floor on BGE relevance; 0 keeps every ranked result."""
    raw = os.environ.get("RERANK_MIN_SCORE")
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        logger.warning("RERANK_MIN_SCORE=%r is not a float; ignoring", raw)
        return 0.0


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalise_records(ids: list[str], records: Any) -> dict[str, dict]:
    """Normalise ``text_chunks.get_by_ids`` output to a dict keyed by chunk id.

    Handles the shapes the KV backends return: a list of dicts (or ``None``
    for misses), a bare dict, or a list of ``(id, record)`` tuples.
    """
    by_id: dict[str, dict] = {}
    if isinstance(records, dict):
        for key, value in records.items():
            if isinstance(value, dict):
                by_id[key] = value
        return by_id
    for index, value in enumerate(records or []):
        if isinstance(value, dict):
            chunk_id = value.get("_id") or value.get("id")
            if not chunk_id and index < len(ids):
                chunk_id = ids[index]
            if chunk_id:
                by_id[chunk_id] = value
        elif (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[1], dict)
        ):
            by_id[value[0]] = value[1]
    return by_id


def _rerank_scores(ranked: Any, count: int) -> list[tuple[int, float]]:
    """Validate LightRAG-format rerank output -> ``[(index, score)]`` desc."""
    scored: list[tuple[int, float]] = []
    for item in ranked or []:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if not 0 <= index < count:
            continue
        score = _as_float(item.get("relevance_score"))
        if score is None:
            continue
        scored.append((index, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def _bounded_env_int(name: str, default: int) -> int:
    """Read a bounded positive integer; bad values stay visible in logs."""
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r is not an integer; using %s", name, raw, default)
        value = default
    clamped = min(max(value, 1), 200)
    if value != clamped:
        logger.warning("%s=%s outside 1..200; using %s", name, value, clamped)
    return clamped


def _candidate_count() -> int:
    """Vector candidates fetched before reranking; rerank owns the final cut."""
    return _bounded_env_int("RETRIEVAL_CANDIDATES", 20)


def _fallback_top_k() -> int:
    """Final count only for no-rerank and rerank-failure paths."""
    return _bounded_env_int("RERANK_TOP_K", 5)


async def retrieve_chunks(
    rag: Any,
    query: str,
    *,
    candidate_count: int | None = None,
) -> tuple[list[dict], dict]:
    """Retrieve chunks from ``rag.lightrag`` shaped like ``RetrievedDocV2``.

    Returns ``(documents, meta)``. Raises ``RuntimeError`` on any storage
    failure (the API maps it to 503); successful rerank output is returned
    uncut. Rerank failure or disabled rerank falls back to vector order and
    cuts it with ``RERANK_TOP_K``.
    """
    start = time.monotonic()

    if candidate_count is None:
        candidate_count = _candidate_count()

    if not query or not query.strip():
        return [], {
            "latency_ms": int((time.monotonic() - start) * 1000),
            "candidate_count": 0,
            "returned": 0,
            "reranked": False,
            "fusion": "vector",
        }

    lg = getattr(rag, "lightrag", None)
    if lg is None:
        raise RuntimeError("RAG instance is not initialised (no lightrag)")

    def _meta(returned: int, candidate_count: int, fusion: str, reranked: bool) -> dict:
        return {
            "latency_ms": int((time.monotonic() - start) * 1000),
            "returned": returned,
            "candidate_count": candidate_count,
            "reranked": reranked,
            "fusion": fusion,
        }

    rerank_func = getattr(lg, "rerank_model_func", None)
    try:
        query_embedding = None
        embedding_func = getattr(lg, "embedding_func", None)
        if embedding_func is not None:
            embeddings = await embedding_func([query], context="query")
            query_embedding = embeddings[0]
        results = await lg.chunks_vdb.query(
            query, top_k=candidate_count, query_embedding=query_embedding
        )
    except Exception as exc:
        raise RuntimeError(f"chunk retrieval failed: {exc}") from exc

    if not results:
        return [], _meta(0, 0, "vector+rerank" if rerank_func is not None else "vector", False)

    ids = [r["id"] for r in results if r.get("id")]
    sims = {r["id"]: _as_float(r.get("distance")) for r in results if r.get("id")}

    try:
        records = await lg.text_chunks.get_by_ids(ids)
    except Exception as exc:
        raise RuntimeError(f"chunk fetch failed: {exc}") from exc

    by_id = _normalise_records(ids, records)
    pool: list[tuple[str, dict, float | None]] = []
    for chunk_id in ids:
        record = by_id.get(chunk_id)
        if record and len(record.get("content") or "") >= MIN_CHUNK_CHARS:
            pool.append((chunk_id, record, sims.get(chunk_id)))

    reranked = False
    if rerank_func is not None and pool:
        texts = [entry[1].get("content", "") for entry in pool]
        min_score = _rerank_min_score()
        try:
            ranked = await rerank_func(query=query, documents=texts, top_n=None)
            reranked = True
            scored = _rerank_scores(ranked, len(pool))
            if min_score:
                scored = [pair for pair in scored if pair[1] >= min_score]
            if scored:
                pool = [
                    (pool[index][0], pool[index][1], score)
                    for index, score in scored
                ]
            else:
                pool = []
        except Exception as exc:
            logger.warning("rerank failed, keeping vector order: %s", exc)
            reranked = False
            pool = pool[:_fallback_top_k()]
    else:
        pool = pool[:_fallback_top_k()]

    documents = []
    for chunk_id, record, score in pool:
        source_path = record.get("source_path") or record.get("file_path")
        doc_id = record.get("full_doc_id") or record.get("doc_id")
        document: dict[str, Any] = {
            "text": record.get("content", ""),
            "score": score,
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "source_path": source_path,
                "file_path": record.get("file_path"),
            },
            "reference_id": chunk_id,
            "doc": {
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "source_path": source_path,
            },
        }
        for key in _PASSTHROUGH_KEYS:
            if key in record:
                document[key] = record[key]
        documents.append(document)

    return documents, _meta(
        len(documents),
        len(ids),
        "vector+rerank" if rerank_func is not None else "vector",
        reranked,
    )
