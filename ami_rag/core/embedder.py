"""Embedder protocol + phân loại lỗi cho embedding server chạy trên máy GPU.

Máy A (pipeline/CLI) KHÔNG chứa torch/transformers - mọi embedding đi qua
``RemoteEmbedder`` gọi sang embedding server (máy B) qua HTTP.
"""

import re
from typing import Protocol


class EmbedderError(RuntimeError):
    """Lỗi cơ sở từ embedding server."""


class EmbedServerUnreachable(EmbedderError):
    """Không với tới server: mất kết nối, timeout, hoặc 5xx sau khi hết retry."""


class EmbedModelMismatch(EmbedderError):
    """model_name/dim của server lệch với config/handshake, hoặc server bị
    đổi model giữa chừng (response không khớp)."""


class EmbedServerOOM(EmbedderError):
    """Server hết GPU memory (507) - không giảm được batch."""


class EmbedInputTooLong(EmbedderError):
    """Input vượt max_input_tokens / payload (413)."""


class EmbedAuthError(EmbedderError):
    """Token bị từ chối (401/403)."""


class EmbedCircuitOpen(EmbedderError):
    """Circuit breaker mở: N lần liên tiếp không với tới server, dừng cả lô sớm."""


class Embedder(Protocol):
    """Interface embedding - backend có thể thay (hiện tại: RemoteEmbedder)."""

    model_name: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed danh sách văn bản (document embedding, có cache)."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed một câu hỏi (query embedding, không cache)."""
        ...


def embed_model_slug(model_name: str) -> str:
    """'Qwen/Qwen3-VL-Embedding-2B' -> 'qwen3-vl-embedding-2b'."""
    slug = model_name.rsplit("/", 1)[-1].strip().lower()
    return re.sub(r"[^a-z0-9._-]+", "-", slug).strip("-")


def collection_name(prefix: str, embed_model: str, chunker_version: str) -> str:
    """Quy ước tên collection vector: {prefix}__{embed_model_slug}__{chunker_version}.

    Không bao giờ trộn vector của hai embed model trong một collection.
    """
    return f"{prefix}__{embed_model_slug(embed_model)}__{chunker_version}"
