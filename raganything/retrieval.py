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
import time
from typing import Any

logger = logging.getLogger(__name__)

# Stored chunk keys echoed back to the client only when the record carries them.
_PASSTHROUGH_KEYS = ("page", "modality", "caption", "table_body", "artifact_url")


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


async def retrieve_chunks(
    rag: Any,
    query: str,
    top_k: int = 10,
    *,
    min_candidates: int | None = None,
) -> tuple[list[dict], dict]:
    """Retrieve chunks from ``rag.lightrag`` shaped like ``RetrievedDocV2``.

    Returns ``(documents, meta)``. Raises ``RuntimeError`` on any storage
    failure (the API maps it to 503); rerank failures never raise and fall
    back to vector order.
    """
    start = time.monotonic()

    if not query or not query.strip():
        return [], {
            "latency_ms": int((time.monotonic() - start) * 1000),
            "requested_top_k": top_k,
            "returned": 0,
            "candidates": 0,
            "fusion": "vector",
        }

    lg = getattr(rag, "lightrag", None)
    if lg is None:
        raise RuntimeError("RAG instance is not initialised (no lightrag)")

    def _meta(returned: int, candidate_count: int, fusion: str) -> dict:
        return {
            "latency_ms": int((time.monotonic() - start) * 1000),
            "requested_top_k": top_k,
            "returned": returned,
            "candidates": candidate_count,
            "fusion": fusion,
        }

    rerank_func = getattr(lg, "rerank_model_func", None)
    candidates = max(top_k * 4, top_k) if rerank_func is not None else top_k
    if min_candidates is not None:
        candidates = max(candidates, min_candidates)
    fusion = "vector+rerank" if rerank_func is not None else "vector"

    try:
        query_embedding = None
        embedding_func = getattr(lg, "embedding_func", None)
        if embedding_func is not None:
            embeddings = await embedding_func([query], context="query")
            query_embedding = embeddings[0]
        results = await lg.chunks_vdb.query(
            query, top_k=candidates, query_embedding=query_embedding
        )
    except Exception as exc:
        raise RuntimeError(f"chunk retrieval failed: {exc}") from exc

    if not results:
        return [], _meta(0, 0, fusion)

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
        if record:
            pool.append((chunk_id, record, sims.get(chunk_id)))

    if rerank_func is not None and pool:
        texts = [entry[1].get("content", "") for entry in pool]
        try:
            ranked = await rerank_func(query=query, documents=texts, top_n=top_k)
            scored = _rerank_scores(ranked, len(pool))
            if scored:
                pool = [
                    (pool[index][0], pool[index][1], score)
                    for index, score in scored
                ][:top_k]
            else:
                pool = pool[:top_k]
        except Exception as exc:
            logger.warning("rerank failed, keeping vector order: %s", exc)
            pool = pool[:top_k]
    else:
        pool = pool[:top_k]

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

    return documents, _meta(len(documents), len(ids), fusion)
