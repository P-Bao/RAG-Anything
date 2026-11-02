"""Tests for kg_factory wiring (bindings into RAGAnything)."""

import asyncio
import tempfile
from types import SimpleNamespace

import pytest

from lightrag.utils import EmbeddingFunc

from raganything.kg_factory import build_kg_raganything


async def _fake_llm(prompt, **kwargs):
    return "ok"


async def _fake_embed(texts, context="document"):
    return [[0.0] * 8 for _ in texts]


def _fake_caption(prompt, **kwargs):
    return "caption"


def _init(rag):
    return asyncio.run(rag._ensure_lightrag_initialized())


def test_factory_wires_bindings_and_disables_vlm_query():
    with tempfile.TemporaryDirectory() as tmp:
        rag = build_kg_raganything(
            llm_model_func=_fake_llm,
            embedding_func=EmbeddingFunc(embedding_dim=8, func=_fake_embed),
            caption_func=_fake_caption,
            lightrag_kwargs={"working_dir": tmp},
        )
        assert _init(rag)["success"] is True

        # VLM-enhanced query can never auto-enable: no vision_model_func.
        assert rag.vision_model_func is None
        # Image processor caption chain: dedicated caption func wins.
        assert rag.modal_processors["image"].modal_caption_func is _fake_caption
        # Text-only modal processors use the Qwen llm func.
        assert rag.modal_processors["table"].modal_caption_func is rag.llm_model_func
        # Role config reached LightRAG: thinking ON for query only.
        assert rag.lightrag.role_llm_configs == {
            "query": {"kwargs": {"thinking": True, "temperature": 0.6}}
        }


def test_factory_caption_none_without_key(monkeypatch):
    monkeypatch.delenv("CAPTION_LLM_API_KEY", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        rag = build_kg_raganything(
            llm_model_func=_fake_llm,
            embedding_func=EmbeddingFunc(embedding_dim=8, func=_fake_embed),
            lightrag_kwargs={"working_dir": tmp},
        )
        # caption falls back to the text-only llm func at processor level;
        # images without MinIO reuse will fail loudly on describe.
        assert rag.modal_caption_func is None
