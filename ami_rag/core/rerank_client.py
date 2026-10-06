"""Rerank clients: legacy BGE (text-only) + vLLM Nemotron VL (multimodal).

- ``build_rerank_model_func``: legacy BGE reranker (POST {base}/rerank trả
  ``{scores, ranked_indices}``) - documents text-only.
- ``build_vllm_rerank_func``: vLLM serving ``nvidia/llama-nemotron-rerank-vl-1b-v2``
  (POST {base}/rerank trả ``{results: [{index, relevance_score}]}``) - documents
  dạng ``str`` hoặc ``{"content": [text / image_url parts]}`` (multimodal).
- ``build_rerank_documents``: chunk payload (Qdrant) -> documents cho rerank;
  chunk image/table/equation có ``asset_key`` kèm ảnh (data URI từ asset store).
- ``build_rerank_func``: dispatcher theo ``RERANK_BACKEND``.
"""

import asyncio
import logging

import httpx

from ami_rag.core.openai_embedder import data_url_from_bytes
from ami_rag.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Số request fetch ảnh đồng thời khi build documents cho rerank
ASSET_FETCH_CONCURRENCY = 4

# Modalities được kèm ảnh khi rerank multimodal
MULTIMODAL_RERANK_TYPES = ("image", "table", "equation")


def build_rerank_model_func(settings: Settings | None = None):
    """Legacy BGE reranker (documents text-only).

    Wraps the external rerank service (POST {base}/rerank returning
    {scores, ranked_indices}) and translates the response into the index-based
    format: [{"index": i, "relevance_score": s}, ...].
    Exceptions propagate: callers catch them and fall back to original chunks.
    """
    settings = settings or get_settings()
    base_url = settings.RERANK_BASE_URL.rstrip("/")

    async def rerank_model_func(query: str, documents: list[str], top_n: int | None = None, **_):
        payload = {
            "query": query,
            "documents": documents,
            "top_k": top_n if top_n is not None else settings.RERANK_TOP_K,
        }
        async with httpx.AsyncClient(timeout=settings.RERANK_TIMEOUT) as client:
            resp = await client.post(f"{base_url}/rerank", json=payload)
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


def build_vllm_rerank_func(settings: Settings | None = None):
    """vLLM reranker (Nemotron VL) - documents text hoặc multimodal.

    Payload: {model, query, documents, top_n}; documents là ``str`` hoặc
    ``{"content": [{"type": "text"|"image_url", ...}]}`` (data URI ảnh).
    Response: {results: [{index, relevance_score, ...}]}.
    """
    settings = settings or get_settings()
    base_url = settings.RERANK_BASE_URL.rstrip("/")
    model = settings.RERANK_MODEL

    async def rerank_func(query: str, documents: list[str | dict], top_n: int | None = None, **_):
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n if top_n is not None else settings.RERANK_TOP_K,
        }
        async with httpx.AsyncClient(timeout=settings.RERANK_TIMEOUT) as client:
            resp = await client.post(f"{base_url}/rerank", json=payload)
            resp.raise_for_status()
            body = resp.json()

        results = []
        for item in body.get("results") or []:
            idx = item.get("index")
            if isinstance(idx, int):
                results.append(
                    {"index": idx, "relevance_score": float(item.get("relevance_score") or 0.0)}
                )
        return results

    return rerank_func


async def build_rerank_documents(
    chunks: list[dict],
    asset_store=None,
    *,
    multimodal: bool = True,
) -> list[str | dict]:
    """Chunk payload (Qdrant) -> documents cho rerank.

    - text chunk (hoặc multimodal tắt / thiếu asset store / thiếu asset_key): str.
    - chunk image/table/equation có asset_key: fetch ảnh từ MinIO (bounded
      concurrency, lỗi -> rơi về text) rồi trả
      ``{"content": [text part, image_url data URI part]}``.
    """
    if not chunks:
        return []

    data_uris: dict[int, str] = {}
    if multimodal and asset_store is not None and hasattr(asset_store, "get_bytes"):
        wanted = [
            (i, c["asset_key"])
            for i, c in enumerate(chunks)
            if (c.get("modality") or "text") in MULTIMODAL_RERANK_TYPES
            and c.get("asset_key")
        ]
        if wanted:
            sem = asyncio.Semaphore(ASSET_FETCH_CONCURRENCY)

            async def _fetch(i: int, key: str):
                async with sem:
                    try:
                        data = await asyncio.to_thread(asset_store.get_bytes, key)
                    except Exception as exc:
                        logger.warning("fetch asset %s cho rerank lỗi: %s", key, exc)
                        data = None
                    return i, data

            for i, data in await asyncio.gather(*(_fetch(i, k) for i, k in wanted)):
                if data:
                    data_uris[i] = data_url_from_bytes(data)

    documents: list[str | dict] = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("content") or ""
        uri = data_uris.get(i)
        if uri:
            documents.append(
                {
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ]
                }
            )
        else:
            documents.append(text)
    return documents


def build_rerank_func(settings: Settings | None = None):
    """Dispatcher theo RERANK_BACKEND: legacy (BGE) | vllm (Nemotron VL)."""
    from ami_rag.settings import resolve_rerank_backend

    settings = settings or get_settings()
    if resolve_rerank_backend(settings) == "vllm":
        return build_vllm_rerank_func(settings)
    return build_rerank_model_func(settings)
