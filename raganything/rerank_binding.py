"""BGE rerank binding for LightRAG (KG branch).

Port of the legacy BGE reranker client from the vector branch: POST
``{base}/rerank`` with ``{query, documents, top_k}`` returning
``{scores, ranked_indices}``, translated to the index-based format LightRAG's
``apply_rerank_if_enabled`` expects (``[{"index", "relevance_score"}]``).
Text-only documents; exceptions propagate so LightRAG falls back to the
original chunk order. ``RERANK_BASE_URL=""`` disables reranking entirely.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import httpx

DEFAULT_BASE_URL = "http://localhost:8010"
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT = 60


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def build_bge_rerank_func(
    base_url: str | None = None,
    top_k: int | None = None,
    timeout: float | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Callable[..., Any] | None:
    """Build the ``rerank_model_func`` for ``LightRAG``.

    Returns None when ``RERANK_BASE_URL`` is set to an empty string (rerank
    disabled by configuration). ``top_n=None`` falls back to
    ``RERANK_TOP_K``.
    """
    base = base_url if base_url is not None else os.environ.get("RERANK_BASE_URL", DEFAULT_BASE_URL)
    if not base.strip():
        return None
    base = base.rstrip("/")
    default_top_k = top_k if top_k is not None else int(
        _env("RERANK_TOP_K", str(DEFAULT_TOP_K)) or DEFAULT_TOP_K
    )
    timeout_s = timeout if timeout is not None else float(
        _env("RERANK_TIMEOUT", str(DEFAULT_TIMEOUT)) or DEFAULT_TIMEOUT
    )

    async def rerank_model_func(
        query: str, documents: list[str], top_n: int | None = None, **_
    ) -> list[dict[str, Any]]:
        payload = {
            "query": query,
            "documents": documents,
            "top_k": top_n if top_n is not None else default_top_k,
        }
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            resp = await client.post(f"{base}/rerank", json=payload)
            resp.raise_for_status()
            body = resp.json()

        scores = body.get("scores") or []
        indices = body.get("ranked_indices")
        if indices is None:
            indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        results = []
        for i in indices:
            idx = int(i)
            if 0 <= idx < len(scores):
                results.append({"index": idx, "relevance_score": float(scores[idx])})
        return results

    return rerank_model_func
