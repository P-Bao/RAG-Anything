"""Test rerank clients (legacy BGE + vLLM Nemotron) với server giả - không gọi mạng thật.

Phủ: parse response hai backend, payload (model/top_n), dispatcher theo
RERANK_BACKEND, build_rerank_documents multimodal (ảnh từ asset store, fallback text).
"""
import json

import httpx

from ami_rag.core.rerank_client import (
    build_rerank_documents,
    build_rerank_func,
    build_rerank_model_func,
    build_vllm_rerank_func,
)
from ami_rag.settings import Settings

NEMOTRON = "nvidia/llama-nemotron-rerank-vl-1b-v2"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 6


def _settings(**kwargs) -> Settings:
    defaults = {
        "RERANK_BASE_URL": "http://testserver",
        "RERANK_TIMEOUT": 5,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class FakeAssetStore:
    """Asset store giả: key 'ok' trả PNG bytes, key 'boom' raise, còn lại None."""

    def __init__(self):
        self.fetched: list[str] = []

    def get_bytes(self, key):
        self.fetched.append(key)
        if key == "ok":
            return PNG_BYTES
        if key == "boom":
            raise RuntimeError("minio down")
        return None


class MockRerankServer:
    """Server giả: legacy trả {scores, ranked_indices}; vllm trả {results}."""

    def __init__(self, mode: str = "vllm"):
        self.mode = mode
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.requests.append(body)
        if self.mode == "legacy":
            scores = [0.9, 0.5, 0.1]
            ranked = [0, 1, 2][: body.get("top_k") or 3]
            return httpx.Response(
                200, json={"scores": scores, "ranked_indices": ranked}
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": -1.25},
                    {"index": 0, "relevance_score": 2.5},
                ]
            },
        )


def _patch_async_client(monkeypatch, server: MockRerankServer) -> None:
    """httpx.AsyncClient trong rerank funcs -> client với MockTransport."""
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        return original(transport=httpx.MockTransport(server.handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


async def test_legacy_func_maps_scores_and_indices(monkeypatch):
    server = MockRerankServer(mode="legacy")
    _patch_async_client(monkeypatch, server)
    func = build_rerank_model_func(_settings())

    results = await func("q", ["a", "b", "c"], top_n=2)
    assert results == [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.5},
    ]
    assert server.requests[0]["top_k"] == 2
    assert server.requests[0]["documents"] == ["a", "b", "c"]


async def test_vllm_func_maps_results(monkeypatch):
    server = MockRerankServer(mode="vllm")
    _patch_async_client(monkeypatch, server)
    func = build_vllm_rerank_func(_settings())

    results = await func("q", ["a", "b", "c"], top_n=2)
    assert results == [
        {"index": 2, "relevance_score": -1.25},
        {"index": 0, "relevance_score": 2.5},
    ]
    assert server.requests[0]["model"] == NEMOTRON
    assert server.requests[0]["top_n"] == 2
    assert server.requests[0]["documents"] == ["a", "b", "c"]


async def test_dispatcher_by_backend(monkeypatch):
    server = MockRerankServer(mode="vllm")
    _patch_async_client(monkeypatch, server)

    # RERANK_MODEL nemotron -> vllm func (payload có model + top_n)
    func = build_rerank_func(_settings(RERANK_MODEL=NEMOTRON))
    await func("q", ["a"], top_n=1)
    assert server.requests[0]["model"] == NEMOTRON
    assert server.requests[0]["top_n"] == 1

    # RERANK_MODEL BGE -> legacy func (payload top_k, không model)
    server.requests.clear()
    func = build_rerank_func(_settings(RERANK_MODEL="BAAI/bge-reranker-v2-m3"))
    await func("q", ["a"], top_n=1)
    assert server.requests[0] == {"query": "q", "documents": ["a"], "top_k": 1}


async def test_dispatcher_default_settings_is_vllm(monkeypatch):
    # Settings default (RERANK_MODEL nemotron) -> dispatcher chọn vllm func
    import ami_rag.core.rerank_client as rc

    called = {}

    def fake_build_vllm(settings):
        async def f(query, documents, top_n=None, **_):
            called["backend"] = "vllm"
            return []

        return f

    monkeypatch.setattr(rc, "build_vllm_rerank_func", fake_build_vllm)
    func = rc.build_rerank_func(_settings())
    await func("q", ["a"])
    assert called["backend"] == "vllm"


async def test_build_rerank_documents_text_only():
    chunks = [
        {"content": "chunk 1", "modality": "text"},
        {"content": "chunk 2"},
    ]
    docs = await build_rerank_documents(chunks, None, multimodal=True)
    assert docs == ["chunk 1", "chunk 2"]


async def test_build_rerank_documents_multimodal_with_assets():
    store = FakeAssetStore()
    chunks = [
        {"content": "text chunk", "modality": "text"},
        {"content": "image chunk", "modality": "image", "asset_key": "ok"},
        {"content": "table chunk", "modality": "table", "asset_key": "missing"},
        {"content": "broken chunk", "modality": "image", "asset_key": "boom"},
    ]
    docs = await build_rerank_documents(chunks, store, multimodal=True)
    assert docs[0] == "text chunk"
    assert isinstance(docs[1], dict)
    parts = docs[1]["content"]
    assert parts[0] == {"type": "text", "text": "image chunk"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # thiếu asset / fetch lỗi -> fallback text
    assert docs[2] == "table chunk"
    assert docs[3] == "broken chunk"
    # chunk text không được fetch asset
    assert store.fetched == ["ok", "missing", "boom"]


async def test_build_rerank_documents_multimodal_disabled():
    store = FakeAssetStore()
    chunks = [
        {"content": "image chunk", "modality": "image", "asset_key": "ok"},
    ]
    docs = await build_rerank_documents(chunks, store, multimodal=False)
    assert docs == ["image chunk"]
    assert store.fetched == []


async def test_build_rerank_documents_equation_included():
    store = FakeAssetStore()
    chunks = [
        {"content": "eq", "modality": "equation", "asset_key": "ok"},
    ]
    docs = await build_rerank_documents(chunks, store, multimodal=True)
    assert isinstance(docs[0], dict)
