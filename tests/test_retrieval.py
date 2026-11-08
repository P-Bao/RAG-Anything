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
        "c1": {"content": "alpha chunk with enough text", "full_doc_id": "d1", "file_path": "f1"},
        "c2": {"content": "beta chunk with enough text", "full_doc_id": "d2", "file_path": "f2"},
        "c3": {"content": "gamma chunk with enough text", "full_doc_id": "d3", "file_path": "f3"},
    }


def _results():
    return [
        {"id": "c1", "distance": 0.9},
        {"id": "c2", "distance": 0.8},
        {"id": "c3", "distance": 0.7},
    ]


def test_rerank_output_is_returned_uncut(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "20")
    monkeypatch.setenv("RERANK_TOP_K", "2")

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
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi"))

    assert [d["reference_id"] for d in docs] == ["c3", "c2", "c1"]
    assert [d["score"] for d in docs] == [0.99, 0.5, 0.1]
    assert docs[0]["text"] == "gamma chunk with enough text"
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
        "candidate_count": 3,
        "returned": 3,
        "reranked": True,
        "fusion": "vector+rerank",
    }
    assert isinstance(meta["latency_ms"], int)
    assert lg.chunks_vdb.calls[0]["top_k"] == 20


def test_rerank_receives_explicit_top_n():
    calls = []

    async def rerank(query, documents, top_n=None, **kwargs):
        calls.append({"top_n": top_n, "documents": documents})
        return []

    lg = FakeLightRAG(
        results=_results(),
        records=_records(),
        embedding_func=RecordingEmbed(),
        rerank_model_func=rerank,
    )
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi"))

    assert docs == []
    assert calls == [{"top_n": None, "documents": [r["content"] for r in _records().values()]}]
    assert meta["reranked"] is True


def test_invalid_candidate_count_is_logged(monkeypatch, caplog):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "bogus")
    lg = FakeLightRAG(
        results=_results(), records=_records(), embedding_func=RecordingEmbed()
    )

    with caplog.at_level("WARNING"):
        asyncio.run(retrieve_chunks(FakeRAG(lg), "hi"))

    assert lg.chunks_vdb.calls[0]["top_k"] == 20
    assert "RETRIEVAL_CANDIDATES" in caplog.text
def test_rerank_return_count_controls_final_count(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "20")
    monkeypatch.setenv("RERANK_TOP_K", "5")

    async def rerank(query, documents, top_n=None, **kwargs):
        return [
            {"index": 2, "relevance_score": 0.99},
            {"index": 1, "relevance_score": 0.5},
        ]

    lg = FakeLightRAG(
        results=_results(),
        records=_records(),
        embedding_func=RecordingEmbed(),
        rerank_model_func=rerank,
    )
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi"))

    assert [d["reference_id"] for d in docs] == ["c3", "c2"]
    assert meta["returned"] == 2
    assert meta["reranked"] is True


def test_no_rerank_falls_back_to_rerank_top_k(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "20")
    monkeypatch.setenv("RERANK_TOP_K", "2")
    lg = FakeLightRAG(
        results=_results(), records=_records(), embedding_func=RecordingEmbed()
    )
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi"))

    assert [d["reference_id"] for d in docs] == ["c1", "c2"]
    assert [d["score"] for d in docs] == [0.9, 0.8]
    assert meta["fusion"] == "vector"
    assert meta["candidate_count"] == 3
    assert meta["reranked"] is False
    assert lg.chunks_vdb.calls[0]["top_k"] == 20


def test_rerank_failure_falls_back_to_vector_order(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "20")
    monkeypatch.setenv("RERANK_TOP_K", "2")

    async def rerank(query, documents, top_n=None, **kwargs):
        raise RuntimeError("bge down")

    lg = FakeLightRAG(
        results=_results(),
        records=_records(),
        embedding_func=RecordingEmbed(),
        rerank_model_func=rerank,
    )
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi"))

    assert [d["reference_id"] for d in docs] == ["c1", "c2"]
    assert [d["score"] for d in docs] == [0.9, 0.8]
    assert meta["returned"] == 2
    assert meta["reranked"] is False


def test_short_junk_chunks_are_dropped(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "5")
    monkeypatch.setenv("RERANK_TOP_K", "5")
    """Ingest-era micro tail chunks (":", "sẻ:") never reach the client."""
    records = {
        **_records(),
        "c4": {"content": ":", "full_doc_id": "d1", "file_path": "f1"},
        "c5": {"content": "ia sẻ:", "full_doc_id": "d2", "file_path": "f2"},
    }
    results = _results() + [
        {"id": "c4", "distance": 0.95},
        {"id": "c5", "distance": 0.94},
    ]
    lg = FakeLightRAG(results=results, records=records, embedding_func=RecordingEmbed())
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi"))

    assert [d["reference_id"] for d in docs] == ["c1", "c2", "c3"]
    assert meta["candidate_count"] == 5  # junk counted as candidates...
    assert meta["returned"] == 3  # ...but filtered before the client
    # filter sits after the KV fetch, before rerank: junk fetched, never ranked


def test_rerank_min_score_drops_low_relevance(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "20")

    async def rerank(query, documents, top_n=None, **kwargs):
        return [
            {"index": 0, "relevance_score": 0.02},
            {"index": 1, "relevance_score": 0.6},
        ]

    monkeypatch.setenv("RERANK_MIN_SCORE", "0.2")
    lg = FakeLightRAG(
        results=_results()[:2],
        records=_records(),
        embedding_func=RecordingEmbed(),
        rerank_model_func=rerank,
    )
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi"))

    assert [d["reference_id"] for d in docs] == ["c2"]
    assert [d["score"] for d in docs] == [0.6]
    assert meta["returned"] == 1


def test_rerank_min_score_empty_result_returns_nothing(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "20")

    """A floor that cuts everything returns [] instead of unranked fallback."""

    async def rerank(query, documents, top_n=None, **kwargs):
        return [{"index": 0, "relevance_score": 0.01}]

    monkeypatch.setenv("RERANK_MIN_SCORE", "0.2")
    lg = FakeLightRAG(
        results=_results(),
        records=_records(),
        embedding_func=RecordingEmbed(),
        rerank_model_func=rerank,
    )
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi"))

    assert docs == []
    assert meta["returned"] == 0


def test_rerank_min_score_invalid_value_is_ignored(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "20")

    async def rerank(query, documents, top_n=None, **kwargs):
        return [{"index": 1, "relevance_score": 0.05}]

    monkeypatch.setenv("RERANK_MIN_SCORE", "not-a-float")
    lg = FakeLightRAG(
        results=_results()[:2],
        records=_records(),
        embedding_func=RecordingEmbed(),
        rerank_model_func=rerank,
    )
    docs, _meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "hi"))

    assert [d["reference_id"] for d in docs] == ["c2"]


def test_empty_query_returns_empty(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "5")
    lg = FakeLightRAG(results=_results(), records=_records())
    docs, meta = asyncio.run(retrieve_chunks(FakeRAG(lg), "   "))

    assert docs == []
    assert meta["returned"] == 0
    assert meta["candidate_count"] == 0
    assert meta["reranked"] is False


def test_storage_error_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "5")
    lg = FakeLightRAG(error=RuntimeError("vdb exploded"))
    with pytest.raises(RuntimeError):
        asyncio.run(retrieve_chunks(FakeRAG(lg), "hi"))


def test_query_embedded_with_query_context():
    embed = RecordingEmbed(vector=(0.7, 0.8))
    lg = FakeLightRAG(
        results=[{"id": "c1", "distance": 0.9}],
        records=_records(),
        embedding_func=embed,
    )
    asyncio.run(retrieve_chunks(FakeRAG(lg), "hello", candidate_count=1))

    assert embed.calls == [{"texts": ["hello"], "context": "query"}]
    assert lg.chunks_vdb.calls[0]["query_embedding"] == [0.7, 0.8]
