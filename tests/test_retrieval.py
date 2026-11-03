"""Tests for raganything.retrieval (LightRAG-backed chunk retrieval)."""

import asyncio

import pytest

from raganything.retrieval import retrieve_chunks


class FakeVDB:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    async def query(self, query, top_k, query_embedding=None):
        self.calls.append(
            {"query": query, "top_k": top_k, "query_embedding": query_embedding}
        )
        if self.error is not None:
            raise self.error
        return self.results


class FakeKV:
    def __init__(self, records):
        self.records = records
        self.calls = []

    async def get_by_ids(self, ids):
        self.calls.append(list(ids))
        return [self.records.get(chunk_id) for chunk_id in ids]


class RecordingEmbed:
    def __init__(self, vector=(0.1, 0.2)):
        self.vector = list(vector)
        self.calls = []

    async def __call__(self, texts, context="document", **kwargs):
        self.calls.append({"texts": list(texts), "context": context})
        return [list(self.vector) for _ in texts]


class FakeLightRAG:
    def __init__(self, results=None, records=None, embedding_func=None, rerank_model_func=None, error=None):
        self.chunks_vdb = FakeVDB(results, error=error)
        self.text_chunks = FakeKV(records or {})
        self.embedding_func = embedding_func
        self.rerank_model_func = rerank_model_func


class FakeRAG:
    def __init__(self, lightrag):
        self.lightrag = lightrag


def _records():
    return {
        "c1": {"content": "alpha", "full_doc_id": "d1", "file_path": "f1"},
        "c2": {"content": "beta", "full_doc_id": "d2", "file_path": "f2"},
        "c3": {"content": "gamma", "full_doc_id": "d3", "file_path": "f3"},
    }


def _results():
    return [
        {"id": "c1", "distance": 0.9},
        {"id": "c2", "distance": 0.8},
        {"id": "c3", "distance": 0.7},
    ]


def test_happy_path_rerank_reorders_and_scores():
    async def rerank(query, documents, top_n=None, **kwargs):
        return [
            {"index": 2, "relevance_score": 0.99},
            {"index": 1, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.1},
        ]

    lg = FakeLightRAG(
        results=_results(),
        records=_records(),
        embedding_func=RecordingEmbed(),
        rerank_model_func=rerank,
    )
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi", 2))

    assert [d["reference_id"] for d in docs] == ["c3", "c2"]
    assert [d["score"] for d in docs] == [0.99, 0.5]
    assert docs[0]["text"] == "gamma"
    assert docs[0]["metadata"] == {
        "doc_id": "d3",
        "chunk_id": "c3",
        "source_path": "f3",
        "file_path": "f3",
    }
    assert docs[0]["doc"] == {
        "document_id": "d3",
        "chunk_id": "c3",
        "source_path": "f3",
    }
    assert meta == {
        "latency_ms": meta["latency_ms"],
        "requested_top_k": 2,
        "returned": 2,
        "candidates": 3,
        "fusion": "vector+rerank",
    }
    assert isinstance(meta["latency_ms"], int)
    # rerank available -> candidate pool widened
    assert lg.chunks_vdb.calls[0]["top_k"] == 8


def test_no_rerank_keeps_vector_order():
    lg = FakeLightRAG(
        results=_results(), records=_records(), embedding_func=RecordingEmbed()
    )
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi", 2))

    assert [d["reference_id"] for d in docs] == ["c1", "c2"]
    assert [d["score"] for d in docs] == [0.9, 0.8]
    assert meta["fusion"] == "vector"
    assert meta["candidates"] == 3
    assert lg.chunks_vdb.calls[0]["top_k"] == 2


def test_rerank_failure_falls_back_to_vector_order():
    async def rerank(query, documents, top_n=None, **kwargs):
        raise RuntimeError("bge down")

    lg = FakeLightRAG(
        results=_results(),
        records=_records(),
        embedding_func=RecordingEmbed(),
        rerank_model_func=rerank,
    )
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi", 2))

    assert [d["reference_id"] for d in docs] == ["c1", "c2"]
    assert [d["score"] for d in docs] == [0.9, 0.8]
    assert meta["returned"] == 2


def test_empty_query_returns_empty():
    lg = FakeLightRAG(results=_results(), records=_records())
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "   ", 5))

    assert docs == []
    assert meta["returned"] == 0
    assert meta["candidates"] == 0


def test_storage_error_raises_runtime_error():
    lg = FakeLightRAG(error=RuntimeError("vdb exploded"))
    with pytest.raises(RuntimeError):
        asyncio.run(retrieve_chunks(FakeRAG(lg), "hi", 5))


def test_query_embedded_with_query_context():
    embed = RecordingEmbed(vector=(0.7, 0.8))
    lg = FakeLightRAG(
        results=[{"id": "c1", "distance": 0.9}],
        records=_records(),
        embedding_func=embed,
    )
    asyncio.run(retrieve_chunks(FakeRAG(lg), "hello", 1))

    assert embed.calls == [{"texts": ["hello"], "context": "query"}]
    assert lg.chunks_vdb.calls[0]["query_embedding"] == [0.7, 0.8]
