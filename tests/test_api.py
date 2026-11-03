"""Tests for raganything.api (KG server on the old vector API port 8009)."""

import time

import pytest
from fastapi.testclient import TestClient

from raganything.api import JobQueue, create_app


class _Vdb:
    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n


class _LightRAG:
    def __init__(self):
        self.entities_vdb = _Vdb(3)
        self.relationships_vdb = _Vdb(2)
        self.chunks_vdb = _Vdb(5)
        self.queries: list[dict] = []

    async def aquery(self, query, **kwargs):
        self.queries.append({"query": query, **kwargs})
        return f"answer for {query}"


class _Rag:
    def __init__(self):
        self.lightrag = _LightRAG()

    async def aquery(self, query, **kwargs):
        return await self.lightrag.aquery(query, **kwargs)


class _State:
    def __init__(self):
        self.failed = []

    def counts(self):
        return {"indexed": 2, "failed": 1}

    def mark_failed(self, doc_id, error, stage="ingest"):
        self.failed.append((doc_id, error))

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    rag = _Rag()
    state = _State()

    async def fake_retrieve(rag_, query, top_k, **kwargs):
        return (
            [
                {
                    "text": "chunk text",
                    "score": 0.9,
                    "metadata": {"doc_id": "d1", "chunk_id": "c1", "source_path": "d1_p.pdf"},
                    "reference_id": "c1",
                    "doc": {"document_id": "d1", "chunk_id": "c1"},
                }
            ],
            {
                "latency_ms": 5,
                "requested_top_k": top_k,
                "returned": 1,
                "candidates": 4,
                "fusion": "vector+rerank",
            },
        )

    async def fake_ingest(rag_, **kwargs):
        on_doc = kwargs.get("on_doc")
        if on_doc:
            on_doc("indexed", "d1", {"captions": 2})
        return {"source": kwargs.get("source"), "total": 1, "indexed": 1, "skipped": 0,
                "failed": 0, "errors": [], "duration_s": 0.01, "totals": {"entities": 1}}

    monkeypatch.setattr("raganything.api.retrieve_chunks", fake_retrieve)
    monkeypatch.setattr("raganything.api.ingest", fake_ingest)
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.delenv("MONGO_URI", raising=False)

    app = create_app(rag=rag, state=state, queue=JobQueue())
    with TestClient(app) as c:
        yield c


def test_root_and_health(client):
    assert client.get("/").json()["service"] == "ami-rag-kg"
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_reports_kg_counts(client):
    body = client.get("/readyz").json()
    assert body["status"] == "ok"
    assert body["kg"] == {"entities": 3, "relations": 2, "chunks": 5}


def test_rag_query_shape(client):
    resp = client.post("/v2/rag/", json={"query": "hỏi gì đó", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "hỏi gì đó"
    assert body["documents"][0]["reference_id"] == "c1"
    assert body["references"] == [{"document_id": "d1", "chunk_id": "c1"}]
    assert body["meta"]["requested_top_k"] == 3
    assert body["meta"]["fusion"] == "vector+rerank"
    assert "answer" not in body


def test_rag_query_reads_last_user_message(client):
    resp = client.post(
        "/v2/rag/",
        json={
            "messages": [
                {"role": "user", "content": "câu đầu"},
                {"role": "assistant", "content": "trả lời"},
                {"role": "user", "content": "câu cuối"},
            ]
        },
    )
    assert resp.json()["query"] == "câu cuối"


def test_rag_query_without_question_is_400(client):
    assert client.post("/v2/rag/", json={}).status_code == 400


def test_rag_query_generate_adds_answer(client):
    resp = client.post("/v2/rag/", json={"query": "q", "generate": True})
    assert resp.json()["answer"] == "answer for q"


def test_references_can_be_disabled(client):
    resp = client.post("/v2/rag/", json={"query": "q", "include_references": False})
    assert resp.json()["references"] == []


def test_api_key_enforced_when_set(client, monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "secret")
    assert client.post("/v2/rag/", json={"query": "q"}).status_code == 401
    assert (
        client.post(
            "/v2/rag/", json={"query": "q"}, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    ok = client.post(
        "/v2/rag/", json={"query": "q"}, headers={"Authorization": "Bearer secret"}
    )
    assert ok.status_code == 200
    # health stays open
    assert client.get("/healthz").status_code == 200


def _wait_done(client, job_id: str, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/v2/rag/index/{job_id}").json()
        if job["state"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job not finished: {job}")


def test_index_enqueues_and_reports_progress(client):
    resp = client.post("/v2/rag/index", json={"source": "minio", "limit": 1})
    assert resp.status_code == 202
    job = _wait_done(client, resp.json()["job_id"])
    assert job["state"] == "done"
    assert job["summary"]["indexed"] == 1
    assert job["progress"] == [
        {"event": "indexed", "document_id": "d1", "captions": 2}
    ]


def test_index_unknown_job_is_404(client):
    assert client.get("/v2/rag/index/nope").status_code == 404


class _FakeJobs:
    """Duck-typed JobStore: the Redis path (no REDIS_URL in tests)."""

    def __init__(self):
        self.enqueued = []
        self.closed = False

    def enqueue(self, *, source, document_ids=None, limit=0, dry_run=False) -> str:
        self.enqueued.append(
            {
                "source": source,
                "document_ids": document_ids,
                "limit": limit,
                "dry_run": dry_run,
            }
        )
        return "job123"

    def get(self, job_id):
        return {"job_id": job_id, "state": "running", "progress": []} if job_id == "job123" else None

    def list(self, limit=20):
        return [{"job_id": "job123", "state": "queued"}]

    def close(self):
        self.closed = True


def test_redis_queue_enqueues_and_lists():
    from fastapi.testclient import TestClient

    from raganything.api import create_app

    jobs = _FakeJobs()
    app = create_app(rag=_Rag(), state=_State(), jobs=jobs)
    with TestClient(app) as c:
        resp = c.post("/v2/rag/index", json={"source": "minio", "limit": 0, "dry_run": True})
        assert resp.status_code == 202
        assert resp.json() == {"job_id": "job123", "status": "queued", "backend": "redis"}
        assert jobs.enqueued == [
            {"source": "minio", "document_ids": None, "limit": 0, "dry_run": True}
        ]

        assert c.get("/v2/rag/index/job123").json()["state"] == "running"
        assert c.get("/v2/rag/index/nope").status_code == 404
        listed = c.get("/v2/rag/index").json()
        assert listed == {"backend": "redis", "jobs": [{"job_id": "job123", "state": "queued"}]}
        assert c.get("/v2/rag/status").json()["queue"] == {"backend": "redis"}
    assert jobs.closed is True


def test_redis_jobs_helper_follows_env(monkeypatch):
    """REDIS_URL decides the backend: set -> Redis JobStore, unset -> in-process."""
    import asyncio

    from raganything import api as api_mod

    monkeypatch.delenv("REDIS_URL", raising=False)
    assert api_mod._redis_jobs() is None

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    store = api_mod._redis_jobs()
    assert type(store).__name__ == "JobStore"
    asyncio.run(store.close())  # from_url is lazy: no connection was opened


def test_in_process_queue_lists_recent_and_reports_backend(client):
    body = client.get("/v2/rag/index").json()
    assert body["backend"] == "in-process"
    client.post("/v2/rag/index", json={"source": "minio", "limit": 1})
    assert client.get("/v2/rag/status").json()["queue"] == {"backend": "in-process"}


def test_index_mongo_without_uri_is_400(client):
    resp = client.post("/v2/rag/index", json={"source": "mongo"})
    assert resp.status_code == 400


def test_kg_totals_reads_real_storage_shape():
    """Real LightRAG storages expose no __len__; NanoVectorDB does, behind an
    async property (client_storage -> raw storage dict)."""
    import asyncio

    from raganything.ingest import kg_totals

    class _NanoClient:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return self._rows

    class _Storage:
        """Real shape: _get_client() async method + client_storage async property."""

        def __init__(self, rows):
            self._rows = rows

        async def _get_client(self):
            return _NanoClient(self._rows)

        @property
        async def client_storage(self):
            return {"embedding_dim": 2048, "data": [{}] * 99}

    class _LightRAG:
        entities_vdb = _Storage(7)
        relationships_vdb = _Storage(0)
        chunks_vdb = _Storage(3)

    totals = asyncio.run(kg_totals(_LightRAG()))
    assert totals == {"entities": 7, "relations": 0, "chunks": 3}


def test_status_endpoint(client):
    body = client.get("/v2/rag/status").json()
    assert body["ready"] is True
    assert body["kg"]["chunks"] == 5
    assert body["doc_status_counts"] == {"indexed": 2, "failed": 1}
    assert body["queue"] == {"backend": "in-process"}
    assert body["in_process_queue"]["queued"] == 0
