"""Chunker wrapper that drops the trailing micro-chunks of fixed-window
chunking (KG branch).

LightRAG's sliding window strides ``chunk_token_size - chunk_overlap`` and
never merges the final partial window: a document whose token count is not a
multiple of the stride emits a 1-token tail chunk (":" / "sẻ:" at the end of
crawled PTIT articles). Those tail chunks index fine but float above the 0.2
cosine threshold on junk queries, which is how punctuation leaked into the
retrieval response.

``build_chunking_func`` returns a callable with the legacy 6-arg contract
(``lightrag.chunking_func``) that delegates to ``chunking_by_token_size`` on a
worker thread and drops chunks below ``min_tokens``/``min_chars``. When the
filter would empty the whole list, the original chunks are returned so a tiny
document is never dropped by its own ingest.
"""

from __future__ import annotations

import asyncio
from typing import Any

from lightrag.chunker import chunking_by_token_size

MIN_CHUNK_TOKENS = 15
MIN_CHUNK_CHARS = 20


def build_chunking_func(
    min_tokens: int = MIN_CHUNK_TOKENS, min_chars: int = MIN_CHUNK_CHARS
):
    """Legacy 6-arg ``chunking_func`` that filters micro tail chunks."""

    def _chunk_and_filter(
        tokenizer,
        content: str,
        split_by_character,
        split_by_character_only: bool,
        chunk_overlap_token_size: int,
        chunk_token_size: int,
    ) -> list[dict[str, Any]]:
        chunks = chunking_by_token_size(
            tokenizer,
            content,
            split_by_character,
            split_by_character_only,
            chunk_overlap_token_size,
            chunk_token_size,
        )
        kept = [
            chunk
            for chunk in chunks
            if chunk["tokens"] >= min_tokens and len(chunk["content"]) >= min_chars
        ]
        if kept and len(kept) < len(chunks):
            for order, chunk in enumerate(kept):
                chunk["chunk_order_index"] = order
        return kept or chunks

    async def chunking_func(
        tokenizer,
        content: str,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        chunk_overlap_token_size: int = 100,
        chunk_token_size: int = 1200,
        **kwargs,
    ) -> list[dict[str, Any]]:
        # CPU-bound (full-document tokenize): hop to a thread per the
        # lightrag custom-chunker contract.
        return await asyncio.to_thread(
            _chunk_and_filter,
            tokenizer,
            content,
            split_by_character,
            split_by_character_only,
            chunk_overlap_token_size,
            chunk_token_size,
        )

    return chunking_func
