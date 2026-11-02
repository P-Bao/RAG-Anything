"""Tests for raganything.rerank_binding (legacy BGE service)."""

import json

import httpx
import pytest

from raganything.rerank_binding import build_bge_rerank_func


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://bge")


def test_translates_scores_and_ranked_indices():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read()
        return httpx.Response(
            200,
            json={"scores": [0.1, 0.9, 0.5], "ranked_indices": [1, 2, 0]},
        )

    func = build_bge_rerank_func(base_url="http://bge", transport=httpx.MockTransport(handler))
    import asyncio

    out = asyncio.run(func("q", ["a", "b", "c"], top_n=3))
    assert out == [
        {"index": 1, "relevance_score": 0.9},
        {"index": 2, "relevance_score": 0.5},
        {"index": 0, "relevance_score": 0.1},
    ]
    import json

    assert json.loads(captured["payload"]) == {
        "query": "q",
        "documents": ["a", "b", "c"],
        "top_k": 3,
    }


def test_missing_ranked_indices_falls_back_to_score_sort():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [0.2, 0.8]})

    func = build_bge_rerank_func(base_url="http://bge", transport=httpx.MockTransport(handler))
    import asyncio

    out = asyncio.run(func("q", ["a", "b"], top_n=None))
    assert [r["index"] for r in out] == [1, 0]


def test_ignores_top_n_returns_full_ranking():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"scores": [0.1, 0.5, 0.9], "ranked_indices": [2, 1, 0]}
        )

    func = build_bge_rerank_func(base_url="http://bge", transport=httpx.MockTransport(handler))
    import asyncio

    out = asyncio.run(func("q", ["a", "b", "c"], top_n=1))
    assert [r["index"] for r in out] == [2, 1, 0]


def test_top_k_equals_document_count_not_top_n():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [0.1] * 4})

    func = build_bge_rerank_func(base_url="http://bge", transport=httpx.MockTransport(handler))
    import asyncio

    captured = {}

    def spy(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.read()))
        return httpx.Response(200, json={"scores": [0.1] * 4, "ranked_indices": [0, 1, 2, 3]})

    func = build_bge_rerank_func(base_url="http://bge", transport=httpx.MockTransport(spy))
    out = asyncio.run(func("q", ["a", "b", "c", "d"], top_n=2))
    assert captured["top_k"] == 4
    assert len(out) == 4


def test_out_of_range_indices_skipped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [0.4], "ranked_indices": [0, 9, -1]})

    func = build_bge_rerank_func(base_url="http://bge", transport=httpx.MockTransport(handler))
    import asyncio

    out = asyncio.run(func("q", ["a"]))
    assert out == [{"index": 0, "relevance_score": 0.4}]


def test_empty_base_url_disables_rerank(monkeypatch):
    monkeypatch.setenv("RERANK_BASE_URL", "")
    assert build_bge_rerank_func() is None


def test_http_error_propagates():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    func = build_bge_rerank_func(base_url="http://bge", transport=httpx.MockTransport(handler))
    import asyncio

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(func("q", ["a"]))
