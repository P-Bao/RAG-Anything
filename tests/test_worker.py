"""Tests for raganything.worker (resident kg-rag ingest worker). Fakes only."""

import asyncio

import pytest

import raganything.worker as worker


def make_job(job_id="j1", **over):
    job = {
        "job_id": job_id,
        "source": "minio",
        "document_ids": ["d1"],
        "file_paths": {"d1": "uploads/report.pdf"},
        "limit": 3,
        "dry_run": False,
    }
    job.update(over)
    return job


class FakeJobs:
    """In-memory JobStore (async, like the Redis one): returns queued jobs, then
    None; records calls."""

    def __init__(self, queued=None, reclaim=0, journal=None):
        self._queued = list(queued or [])
        self._reclaim = reclaim
        self.journal = journal if journal is not None else []
        self.claimed = []
        self.acked = []
        self.done = []
        self.failed = []
        self.progress = []

    async def reclaim_stale(self, min_idle_ms=None):
        self.journal.append("reclaim_stale")
        return [{"job_id": f"stale-{i}"} for i in range(self._reclaim)]

    async def claim(self, block_ms=5000):
        self.journal.append("claim")
        if self._queued:
            job = self._queued.pop(0)
            self.claimed.append(job)
            return job
        return None

    async def ack(self, job):
        self.journal.append("ack")
        self.acked.append(job)

    async def append_progress(self, job_id, entry):
        self.journal.append("progress")
        self.progress.append((job_id, entry))

    async def mark_done(self, job_id, summary):
        self.journal.append("mark_done")
        self.done.append((job_id, summary))

    async def mark_failed(self, job_id, error):
        self.journal.append("mark_failed")
        self.failed.append((job_id, error))

    async def close(self):
        self.journal.append("close")


class StopJobs(FakeJobs):
    """FakeJobs that raises CancelledError once the queue drains (test 7)."""

    async def claim(self, block_ms=5000):
        self.journal.append("claim")
        if self._queued:
            job = self._queued.pop(0)
            self.claimed.append(job)
            return job
        raise asyncio.CancelledError()


def test_process_job_happy_path(monkeypatch):
    calls = []
    summary = {"source": "minio", "total": 1, "indexed": 1}

    async def fake_ingest(rag, **kwargs):
        calls.append(kwargs)
        kwargs["on_doc"]("indexed", "d1", {"captions": 2})
        return summary

    monkeypatch.setattr(worker, "ingest", fake_ingest)
    jobs = FakeJobs()
    job = make_job()

    out = asyncio.run(worker.process_job(object(), jobs, job, state="STATE"))

    assert out is summary
    assert calls[0]["source"] == "minio"
    assert calls[0]["document_ids"] == ["d1"]
    assert calls[0]["file_paths"] == {"d1": "uploads/report.pdf"}
    assert calls[0]["limit"] == 3
    assert calls[0]["dry_run"] is False
    assert calls[0]["state"] == "STATE"
    assert jobs.progress == [
        ("j1", {"event": "indexed", "document_id": "d1", "captions": 2})
    ]
    assert jobs.done == [("j1", summary)]
    assert jobs.failed == []


def test_process_job_failure_marks_failed_and_returns_empty(monkeypatch):
    async def fake_ingest(rag, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(worker, "ingest", fake_ingest)
    jobs = FakeJobs()

    out = asyncio.run(worker.process_job(object(), jobs, make_job()))

    assert out == {}
    assert jobs.failed == [("j1", "ValueError: boom")]
    assert jobs.done == []


def test_run_worker_once_processes_one_and_returns(monkeypatch):
    journal = []

    async def fake_ingest(rag, **kwargs):
        journal.append("ingest")
        return {"indexed": 1}

    monkeypatch.setattr(worker, "ingest", fake_ingest)
    jobs = FakeJobs([make_job("j1")], journal=journal)

    asyncio.run(worker.run_worker(rag=object(), jobs=jobs, once=True))

    assert journal == ["reclaim_stale", "claim", "ingest", "mark_done", "ack"]
    assert jobs.acked == [make_job("j1")]
    assert jobs.done == [("j1", {"indexed": 1})]


def test_run_worker_reclaims_before_first_claim(monkeypatch):
    async def fake_ingest(rag, **kwargs):
        return {}

    monkeypatch.setattr(worker, "ingest", fake_ingest)
    jobs = FakeJobs(reclaim=2)

    asyncio.run(worker.run_worker(rag=object(), jobs=jobs, once=True))

    assert jobs.journal[:2] == ["reclaim_stale", "claim"]


def test_run_worker_once_empty_queue_returns(monkeypatch):
    async def fake_ingest(rag, **kwargs):  # pragma: no cover - never called
        raise AssertionError("ingest must not run on an empty queue")

    monkeypatch.setattr(worker, "ingest", fake_ingest)
    jobs = FakeJobs()

    asyncio.run(worker.run_worker(rag=object(), jobs=jobs, once=True))

    assert jobs.claimed == []
    assert jobs.journal == ["reclaim_stale", "claim"]


def test_main_returns_one_when_init_fails(monkeypatch):
    class FakeRag:
        async def _ensure_lightrag_initialized(self):
            return {"success": False, "error": "boom"}

    monkeypatch.setattr(worker, "build_kg_raganything", lambda **kwargs: FakeRag())

    assert worker.main() == 1


def test_worker_survives_a_failing_job(monkeypatch):
    calls = []

    async def fake_ingest(rag, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("first boom")
        kwargs["on_doc"]("indexed", "d2", {"captions": 1})
        return {"indexed": 1}

    monkeypatch.setattr(worker, "ingest", fake_ingest)
    jobs = StopJobs([make_job("j1"), make_job("j2")])

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker.run_worker(rag=object(), jobs=jobs))

    assert len(jobs.claimed) == 2
    assert jobs.failed == [("j1", "RuntimeError: first boom")]
    assert jobs.done == [("j2", {"indexed": 1})]
    assert jobs.acked == jobs.claimed
    assert jobs.progress == [
        ("j2", {"event": "indexed", "document_id": "d2", "captions": 1})
    ]
