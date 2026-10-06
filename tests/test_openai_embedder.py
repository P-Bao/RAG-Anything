"""Test OpenAIEmbedder (vLLM Nemotron VL embed) với server giả - không gọi mạng thật.

Phủ: handshake (/v1/models model id + probe dim), retry đúng lỗi tạm thời /
không retry 4xx, payload messages (role query/document, image_url data URI),
kiểm tra response (đổi model giữa chừng), circuit breaker, cache.
"""
import base64
import json

import httpx
import pytest

from ami_rag.core.embedder import (
    EmbedAuthError,
    EmbedCircuitOpen,
    EmbedderError,
    EmbedModelMismatch,
    EmbedServerUnreachable,
)
from ami_rag.core.embedding_cache import EmbeddingCache
from ami_rag.core.openai_embedder import OpenAIEmbedder, _data_url

MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 6
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


def _vec(i: int, dim: int = 8) -> list[float]:
    return [float((i + j) % 7) / 7 for j in range(dim)]


def _embed_body(model_name: str = MODEL, dim: int = 8) -> dict:
    return {
        "data": [{"index": 0, "embedding": _vec(0, dim)}],
        "model": model_name,
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


class FakeServer:
    """Server giả vLLM (models + embeddings); cấu hình được hành vi lỗi."""

    def __init__(self, dim: int = 8, model_name: str = MODEL):
        self.requests: list[dict] = []
        self.dim = dim
        self.model_name = model_name
        self.fail_next: list[int | Exception] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": self.model_name}]},
            )
        if path == "/v1/embeddings":
            body = json.loads(request.content.decode())
            # probe của verify(): không ghi, không tiêu fail_next
            if body["messages"][0]["content"][0].get("text") == "probe":
                return httpx.Response(
                    200, json=_embed_body(model_name=self.model_name, dim=self.dim)
                )
            self.requests.append(body)
            if self.fail_next:
                first = self.fail_next.pop(0)
                if isinstance(first, Exception):
                    raise first
                return httpx.Response(first, json={"detail": f"fake {first}"})
            return httpx.Response(
                200, json=_embed_body(model_name=self.model_name, dim=self.dim)
            )
        return httpx.Response(404, json={"detail": "not found"})


def _make_embedder(server: FakeServer, **kwargs) -> OpenAIEmbedder:
    defaults = {
        "base_url": "http://testserver",
        "model": MODEL,
        "expected_dim": 8,
        "timeout": 5,
        "retries": 2,
        "backoff_base": 0.0,
    }
    defaults.update(kwargs)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler),
        headers={"Authorization": "Bearer t"} if kwargs.get("token") else {},
    )
    return OpenAIEmbedder(http_client=client, **defaults)


async def test_handshake_ok_sets_model_and_dim():
    server = FakeServer()
    embedder = _make_embedder(server)
    info = await embedder.verify()
    assert info["data"][0]["id"] == MODEL
    assert embedder.dim == 8
    # probe của verify() không được ghi vào requests
    assert len(server.requests) == 0


async def test_handshake_model_mismatch_stops():
    server = FakeServer(model_name="other-model")
    embedder = _make_embedder(server)
    with pytest.raises(EmbedModelMismatch):
        await embedder.verify()


async def test_handshake_dim_mismatch_stops():
    server = FakeServer(dim=1024)
    embedder = _make_embedder(server)
    with pytest.raises(EmbedModelMismatch):
        await embedder.verify()


async def test_unreachable_on_models():
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    embedder = OpenAIEmbedder(
        base_url="http://testserver",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(dead)),
        retries=1,
        backoff_base=0.0,
    )
    with pytest.raises(EmbedServerUnreachable):
        await embedder.verify()


async def test_embed_documents_one_request_per_item_preserves_order():
    server = FakeServer()
    embedder = _make_embedder(server)
    texts = [f"text {i}" for i in range(5)]
    vectors = await embedder.embed_documents(texts)
    assert len(vectors) == 5
    assert len(server.requests) == 5  # 1 request / item
    sent = [r["messages"][0]["content"][0]["text"] for r in server.requests]
    assert sent == texts
    for r in server.requests:
        assert r["model"] == MODEL
        assert r["messages"][0]["role"] == "document"


async def test_retry_on_5xx_then_success():
    server = FakeServer()
    server.fail_next = [500, 503]
    embedder = _make_embedder(server)
    vectors = await embedder.embed_documents(["hello"])
    assert len(vectors) == 1
    assert len(server.requests) == 3  # 2 fail + 1 success


async def test_no_retry_on_4xx():
    server = FakeServer()
    server.fail_next = [400]
    embedder = _make_embedder(server)
    with pytest.raises(EmbedderError):
        await embedder.embed_documents(["hello"])
    assert len(server.requests) == 1  # không retry


async def test_auth_error_no_retry():
    server = FakeServer()
    server.fail_next = [401]
    embedder = _make_embedder(server)
    with pytest.raises(EmbedAuthError):
        await embedder.embed_documents(["hello"])
    assert len(server.requests) == 1


async def test_timeout_retried_then_unreachable():
    server = FakeServer()
    server.fail_next = [
        httpx.TimeoutException("timed out"),
        httpx.TimeoutException("timed out"),
        httpx.TimeoutException("timed out"),
    ]
    embedder = _make_embedder(server, retries=2)
    with pytest.raises(EmbedServerUnreachable):
        await embedder.embed_documents(["hello"])
    assert len(server.requests) == 3  # retries=2 -> 3 lần thử


async def test_circuit_breaker_stops_batch_early():
    server = FakeServer()

    def dead_embed(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return server.handler(request)
        body = json.loads(request.content.decode())
        if body["messages"][0]["content"][0].get("text") == "probe":
            return httpx.Response(200, json=_embed_body(dim=8))
        raise httpx.ConnectError("refused")

    embedder = OpenAIEmbedder(
        base_url="http://testserver",
        model=MODEL,
        expected_dim=8,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(dead_embed)),
        max_concurrency=1,
        retries=0,
        backoff_base=0.0,
    )
    await embedder.verify()
    with pytest.raises(EmbedCircuitOpen):
        await embedder.embed_documents([f"t{i}" for i in range(10)])
    # 5 lần liên tiếp mất kết nối -> dừng cả lô sớm, không thử 10 item


async def test_response_model_change_detected_mid_stream():
    server = FakeServer()

    def switching(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/embeddings" and len(server.requests) >= 1:
            # Request sau probe: server đã bị đổi model
            return httpx.Response(200, json=_embed_body(model_name="other-model"))
        return server.handler(request)

    embedder = OpenAIEmbedder(
        base_url="http://testserver",
        model=MODEL,
        expected_dim=8,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(switching)),
        retries=0,
        backoff_base=0.0,
    )
    await embedder.verify()
    with pytest.raises(EmbedModelMismatch):
        await embedder.embed_documents(["a", "b"])


async def test_cache_only_sends_misses(tmp_path):
    server = FakeServer()
    cache = EmbeddingCache(tmp_path / "cache.db")
    embedder = _make_embedder(server, cache=cache)
    await embedder.embed_documents(["doc a", "doc b"])
    assert len(server.requests) == 2  # 2 item (probe không được ghi)

    # Lần 2: cả hai trúng cache -> 0 request mới (verify đã cached)
    server.requests.clear()
    await embedder.embed_documents(["doc a", "doc b"])
    assert len(server.requests) == 0

    # Lần 3: 1 mới 1 cũ -> chỉ gửi miss
    server.requests.clear()
    await embedder.embed_documents(["doc a", "doc c"])
    assert len(server.requests) == 1
    assert server.requests[0]["messages"][0]["content"][0]["text"] == "doc c"


async def test_embed_query_uses_query_role():
    server = FakeServer()
    embedder = _make_embedder(server)
    await embedder.embed_query("câu hỏi")
    # requests[0] là query (probe của verify không được ghi)
    query_req = server.requests[0]
    assert query_req["messages"][0]["role"] == "query"
    assert query_req["messages"][0]["content"][0]["text"] == "câu hỏi"


async def test_embed_multimodal_sends_image_url_data_uri():
    server = FakeServer()
    embedder = _make_embedder(server)
    items = [
        "plain text",
        {"text": "caption", "image_b64": PNG_B64},
    ]
    vectors = await embedder.embed_documents(items)
    assert len(vectors) == 2
    sent_text = server.requests[0]["messages"][0]["content"]
    assert sent_text == [{"type": "text", "text": "plain text"}]
    sent_img = server.requests[1]["messages"][0]["content"]
    assert sent_img[0]["type"] == "image_url"
    assert sent_img[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert sent_img[1] == {"type": "text", "text": "caption"}


def test_data_url_mime_detection():
    assert _data_url(PNG_B64).startswith("data:image/png;base64,")
    jpeg_b64 = base64.b64encode(b"\xff\xd8\xff\xe0rest").decode("ascii")
    assert _data_url(jpeg_b64).startswith("data:image/jpeg;base64,")
    # base64 rác -> fallback jpeg, không raise
    assert _data_url("!!!!").startswith("data:image/jpeg;base64,")
