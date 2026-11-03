"""Factory: build a wired RAGAnything for the KG branch from env config.

Wiring:
- llm: Qwen3.5 binding (``thinking`` per role — ON only for query)
- embedding: nemotron gateway, context-aware query/document input_type
- image caption: Gemini binding; ``None`` without CAPTION_LLM_API_KEY, in
  which case MinIO caption reuse must cover images (index script) or image
  describe falls back to the text-only llm and fails loudly on images
- rerank: legacy BGE service, query-time chunk ordering only;
  ``RERANK_BASE_URL=""`` disables it
- ``vision_model_func`` is never set here: VLM-enhanced query stays disabled
  by design (KG branch is text-only apart from image captioning)
"""

from __future__ import annotations

from typing import Any, Callable

from lightrag.utils import EmbeddingFunc

from raganything.caption_binding import build_caption_func
from raganything.embedding_binding import DEFAULT_DIM, build_embedding_func
from raganything.llm_binding import build_llm_model_func, role_llm_configs
from raganything.rerank_binding import build_bge_rerank_func


def build_kg_raganything(
    llm_model_func: Callable | None = None,
    embedding_func: Callable | None = None,
    caption_func: Callable | None = None,
    rerank_model_func: Callable | None = None,
    embed_dim: int = DEFAULT_DIM,
    lightrag_kwargs: dict[str, Any] | None = None,
    **raganything_kwargs: Any,
):
    """Build a ``RAGAnything`` with all KG-branch bindings wired.

    Injection params exist for tests; production uses the env-driven
    defaults. Extra ``lightrag_kwargs`` are merged after the role config and
    rerank so callers can still override (e.g. ``working_dir``).
    """
    from raganything import RAGAnything

    merged_lightrag_kwargs = {
        "role_llm_configs": role_llm_configs(),
        "rerank_model_func": (
            rerank_model_func
            if rerank_model_func is not None
            else build_bge_rerank_func()
        ),
    }
    if lightrag_kwargs:
        merged_lightrag_kwargs.update(lightrag_kwargs)

    return RAGAnything(
        llm_model_func=llm_model_func or build_llm_model_func(),
        embedding_func=embedding_func
        or EmbeddingFunc(
            embedding_dim=embed_dim,
            func=build_embedding_func(),
            # LightRAG pops the `context` kwarg unless this is set, which would
            # silently embed queries as documents (nemotron is asymmetric).
            supports_asymmetric=True,
        ),
        modal_caption_func=(
            caption_func if caption_func is not None else build_caption_func()
        ),
        lightrag_kwargs=merged_lightrag_kwargs,
        **raganything_kwargs,
    )
