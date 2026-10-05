import base64
import os
from collections.abc import Callable

from ami_rag.settings import Settings, get_settings

_rag_instance = None
_raganything_instance = None
_asset_store_instance = None
_remote_embedder_instance = None


def _inject_storage_env(settings: Settings) -> None:
    """LightRAG's Mongo/Qdrant storages read these env vars; Settings is the source of
    truth, so overwrite (not setdefault) to keep them consistent with the registry."""
    os.environ["MONGO_URI"] = settings.MONGO_URI
    os.environ["MONGO_DATABASE"] = settings.RAG_DB
    os.environ["QDRANT_URL"] = settings.QDRANT_URL
    if settings.QDRANT_API_KEY:
        os.environ["QDRANT_API_KEY"] = settings.QDRANT_API_KEY


def _build_llm_func(settings: Settings) -> Callable:
    # LightRAG calls `llm_model_func(prompt, system_prompt=..., ...)`, but the lightrag
    # `*_complete_if_cache` helpers take `model` as their FIRST positional argument, so a
    # `partial(..., model=...)` binds the prompt to `model` ("multiple values for 'model'").
    from lightrag.llm.openai import openai_complete_if_cache

    async def qwen_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            settings.QWEN_LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=settings.QWEN_LLM_BASE_URL,
            api_key=settings.QWEN_LLM_API_KEY or None,
            **kwargs,
        )

    return qwen_llm


def _image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:3] == b"GIF":
        return "image/gif"
    return "image/jpeg"


def _openai_style_messages(
    prompt: str,
    system_prompt: str | None,
    history_messages: list[dict] | None,
    image_data: str | None,
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    if image_data:
        mime = _image_mime(base64.b64decode(image_data))
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image_data}"},
            },
        ]
    else:
        content = prompt
    messages.append({"role": "user", "content": content})
    return messages


def _build_vision_func(settings: Settings) -> Callable:
    """Vision (VLM) callable for RAGAnything image captioning.

    RAGAnything calls it as `(prompt, system_prompt=..., image_data=<base64>)` or, for
    multimodal queries, `("", messages=[OpenAI-style messages with image_url parts])`.
    """
    from lightrag.llm.openai import openai_complete_if_cache

    model = settings.QWEN_VLM_MODEL or settings.QWEN_LLM_MODEL

    async def qwen_vision(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages is None and image_data:
            messages = _openai_style_messages(prompt, system_prompt, history_messages, image_data)
        extra = {"messages": messages} if messages is not None else {}
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=settings.QWEN_LLM_BASE_URL,
            api_key=settings.QWEN_LLM_API_KEY or None,
            **extra,
        )

    return qwen_vision


def _build_embedding_func(settings: Settings):
    """Embedding qua RemoteEmbedder (server máy B), bọc theo contract EmbeddingFunc
    của LightRAG (dùng tới khi LightRAG bị gỡ ở giai đoạn chuyển pipeline)."""
    from lightrag.utils import EmbeddingFunc

    async def remote_embed(texts: list[str]):
        import numpy as np

        vectors = await get_remote_embedder().embed_documents(list(texts))
        return np.array(vectors)

    return EmbeddingFunc(
        embedding_dim=settings.EMBED_DIM,
        max_token_size=8192,
        func=remote_embed,
    )


def build_rag(
    settings: Settings | None = None,
    *,
    llm_model_func: Callable | None = None,
    embedding_func=None,
):
    """Build a LightRAG instance wired to the AMI infrastructure.

    Storage mapping: Mongo `RAG_DB` for KV/doc-status/graph (MongoKVStorage,
    MongoDocStatusStorage, MongoGraphStorage), Qdrant for vectors (QdrantVectorDBStorage,
    workspace-partitioned). LLM = qwen-selfhost (OpenAI-compatible); embedding qua
    RemoteEmbedder (server máy B). Test paths can inject llm_model_func/embedding_func.
    """
    from lightrag import LightRAG

    from ami_rag.core.rerank_client import build_rerank_model_func

    settings = settings or get_settings()
    _inject_storage_env(settings)

    return LightRAG(
        working_dir=settings.WORKING_DIR,
        workspace=settings.WORKSPACE,
        kv_storage="MongoKVStorage",
        vector_storage="QdrantVectorDBStorage",
        graph_storage="MongoGraphStorage",
        doc_status_storage="MongoDocStatusStorage",
        llm_model_func=llm_model_func or _build_llm_func(settings),
        embedding_func=embedding_func or _build_embedding_func(settings),
        rerank_model_func=build_rerank_model_func(settings),
        chunk_token_size=settings.CHUNK_SIZE,
        chunk_overlap_token_size=settings.CHUNK_OVERLAP,
        entity_extract_max_gleaning=settings.MAX_GLEANING,
        addon_params={"language": settings.SUMMARY_LANGUAGE},
    )


def build_raganything(
    rag, settings: Settings | None = None, vision_model_func: Callable | None = None
):
    """Wrap an existing LightRAG in RAGAnything (parser + multimodal processors)."""
    from raganything import RAGAnything, RAGAnythingConfig

    settings = settings or get_settings()
    return RAGAnything(
        lightrag=rag,
        vision_model_func=vision_model_func or _build_vision_func(settings),
        config=RAGAnythingConfig(
            working_dir=settings.WORKING_DIR,
            parser=settings.PARSER,
            parse_method=settings.PARSE_METHOD,
            parser_output_dir=settings.PARSER_OUTPUT_DIR,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        ),
    )


def build_asset_store(settings: Settings | None = None):
    from ami_rag.storage.assets import MinioAssetStore

    settings = settings or get_settings()
    return MinioAssetStore(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        bucket=settings.MINIO_BUCKET,
        prefix=settings.ASSET_PREFIX,
        secure=settings.MINIO_SECURE,
        presign_expires=settings.MINIO_PRESIGN_EXPIRES,
    )


async def get_rag():
    """Async singleton: build once and initialize storages."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = build_rag()
        await _rag_instance.initialize_storages()
    return _rag_instance


async def get_raganything():
    """Async singleton: RAGAnything wrapping the shared LightRAG from get_rag()."""
    global _raganything_instance
    if _raganything_instance is None:
        _raganything_instance = build_raganything(await get_rag())
    return _raganything_instance


def get_asset_store():
    """Cached MinioAssetStore built from settings."""
    global _asset_store_instance
    if _asset_store_instance is None:
        _asset_store_instance = build_asset_store()
    return _asset_store_instance


def get_remote_embedder():
    """Cached RemoteEmbedder built from settings (handshake chạy lười ở lần embed đầu)."""
    global _remote_embedder_instance
    if _remote_embedder_instance is None:
        from ami_rag.core.remote_embedder import build_remote_embedder

        _remote_embedder_instance = build_remote_embedder(get_settings())
    return _remote_embedder_instance


async def close_rag() -> None:
    global _rag_instance, _raganything_instance, _asset_store_instance
    global _remote_embedder_instance
    _raganything_instance = None
    _asset_store_instance = None
    if _rag_instance is not None:
        await _rag_instance.finalize_storages()
        _rag_instance = None
    if _remote_embedder_instance is not None:
        await _remote_embedder_instance.close()
        _remote_embedder_instance = None
