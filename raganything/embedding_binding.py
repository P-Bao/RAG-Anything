"""Nemotron VL embedding binding via the nemotron-vl-vllm gateway (KG branch).

Gateway contract (machine B, :8080): ``POST /v1/embeddings`` with
``{"input": [str items], "input_type": "query"|"document"}``, OpenAI-format
response. LightRAG injects ``context="query"|"document"`` on the wrapped func,
mapped 1:1 to ``input_type`` so asymmetric query/passage instructions are used
correctly. LightRAG's ``EmbeddingFunc`` only forwards ``context`` when built
with ``supports_asymmetric=True`` (see ``raganything.kg_factory``); without it
the kwarg is popped and queries are embedded as documents.

The returned arrays must be ``np.ndarray``: LightRAG's ``EmbeddingFunc``
validates ``result.size``, so a plain list fails every call.
"""

from __future__ import annotations

import asyncio
import os

import numpy as np
from typing import Any, Callable

from openai import AsyncOpenAI

DEFAULT_URL = "http://localhost:8080"
DEFAULT_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
DEFAULT_DIM = 2048
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_CONCURRENCY = 4


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def build_embedding_func(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    embedding_dim: int | None = None,
    batch_size: int | None = None,
    max_concurrency: int | None = None,
    client: AsyncOpenAI | None = None,
) -> Callable[..., Any]:
    """Build an async embedding func for ``EmbeddingFunc(func=...)``.

    KG branch is text-only: items are plain strings (chunks, entity and
    relation descriptions, queries). Batches requests and orders results
    by response ``index``.
    """
    base = base_url or _env("NEMOTRON_EMBED_URL", DEFAULT_URL)
    model_name = model or _env("NEMOTRON_EMBED_MODEL", DEFAULT_MODEL)
    dim = embedding_dim or DEFAULT_DIM
    batch = batch_size or _int_env("NEMOTRON_EMBED_BATCH_SIZE", DEFAULT_BATCH_SIZE)
    sem = asyncio.Semaphore(
        max_concurrency or _int_env("NEMOTRON_EMBED_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
    )
    client = client or AsyncOpenAI(
        base_url=f"{base.rstrip('/')}/v1",
        api_key=api_key or _env("NEMOTRON_EMBED_API_KEY", "EMPTY"),
    )

    async def _embed_batch(texts: list[str], input_type: str) -> list[list[float]]:
        async with sem:
            response = await client.embeddings.create(
                model=model_name,
                input=texts,
                extra_body={"input_type": input_type},
            )
        vectors = [None] * len(texts)
        for item in response.data:
            if not 0 <= item.index < len(texts):
                raise RuntimeError(f"gateway returned out-of-range index {item.index}")
            vectors[item.index] = item.embedding
        if any(v is None for v in vectors):
            raise RuntimeError(
                f"gateway returned {sum(v is None for v in vectors)} missing "
                f"embeddings for {len(texts)} inputs"
            )
        return vectors  # type: ignore[return-value]

    async def embed_func(
        texts: list[str],
        context: str = "document",
        **kwargs: Any,
    ) -> Any:
        input_type = "query" if context == "query" else "document"
        batches = [texts[i : i + batch] for i in range(0, len(texts), batch)]
        results = await asyncio.gather(*(_embed_batch(b, input_type) for b in batches))
        # LightRAG's EmbeddingFunc validates ``result.size``: a plain list
        # raises AttributeError there ("'list' object has no attribute
        # 'size'") and every vdb flush fails, so documents never index.
        return np.asarray([v for r in results for v in r], dtype=float)

    embed_func.model = model_name  # type: ignore[attr-defined]
    embed_func.dim = dim  # type: ignore[attr-defined]
    return embed_func
