"""Tests for raganything.jobs (Redis Stream ingest queue; duck-typed fake)."""

import pytest

import raganything.jobs as jobs
from raganything.jobs import (
    DEFAULT_GROUP,
    DEFAULT_STREAM,
    INDEX_KEY,
    JOB_TTL_SECONDS,
    JobStore,
)


class ResponseError(Exception):
    """Stands in for redis.exceptions.ResponseError in the fake."""


class FakeRedis:
    """Minimal async Redis covering only what JobStore calls."""

    def __init__(self):
        self.streams = {}  # name -> [[entry_id, fields]]
        self.groups = {}  # name -> {group: {"delivered": int, "pending": [...]}}
        self.hashes = {}
        self.expires = {}
        self.zsets = {}  # key -> {member: (score, seq)}
        self.acked = []
        self.xautoclaim_calls = []
        self.closed = False
        self._seq = 0
        self._zseq = 0

    async def xadd(self, name, fields, id="*"):
        self._seq += 1
        entry_id = f"{self._seq}-0"
        self.streams.setdefault(name, []).append([entry_id, dict(fields)])
        return entry_id

    async def xgroup_create(self, name, groupname, id="$", mkstream=False):
        if mkstream:
            self.streams.setdefault(name, [])
        groups = self.groups.setdefault(name, {})
        if groupname in groups:
            raise ResponseError("BUSYGROUP Consumer Group name already exists")
        groups[groupname] = {"delivered": 0, "pending": []}

    async def xreadgroup(
        self, groupname, consumername, streams, count=None, block=None, noack=False
    ):
        name = next(iter(streams))
        group = self.groups[name][groupname]
        entries = self.streams.get(name, [])
        n = count or 1
        new = entries[group["delivered"] : group["delivered"] + n]
        if not new:
            return None
        group["delivered"] += len(new)
        out = []
        for entry_id, fields in new:
            if not noack:
                group["pending"].append((entry_id, consumername, fields))
            out.append((entry_id, dict(fields)))
        return [[name, out]]  # redis-py shape: list of [stream, entries]

    async def xack(self, name, groupname, *ids):
        group = self.groups[name][groupname]
        removed = 0
        for entry_id in ids:
            before = len(group["pending"])
            group["pending"] = [p for p in group["pending"] if p[0] != entry_id]
            removed += before - len(group["pending"])
            self.acked.append(entry_id)
        return removed

    async def xautoclaim(
        self,
        name,
        groupname,
        consumername,
        min_idle_time,
        start_id="0-0",
        count=None,
        justid=False,
    ):
        self.xautoclaim_calls.append(min_idle_time)
        group = self.groups[name][groupname]
        messages = [(eid, dict(fields)) for eid, _cons, fields in group["pending"]]
        return ("0-0", messages, [])

    async def hset(self, key, mapping=None, **kwargs):
        h = self.hashes.setdefault(key, {})
        for k, v in (mapping or {}).items():
            h[k] = str(v)
        for k, v in kwargs.items():
            h[k] = str(v)
        return len(mapping or {})

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hincrby(self, key, field, amount=1):
        h = self.hashes.setdefault(key, {})
        h[field] = str(int(h.get(field, 0)) + amount)
        return int(h[field])

    async def expire(self, key, ttl):
        self.expires[key] = ttl
        return True

    async def zadd(self, key, mapping):
        z = self.zsets.setdefault(key, {})
        for member, score in mapping.items():
            self._zseq += 1
            z[member] = (score, self._zseq)
        return len(mapping)

    async def zrevrange(self, key, start, end):
        z = self.zsets.get(key, {})
        members = sorted(z, key=lambda m: z[m], reverse=True)
        return members[start:] if end == -1 else members[start : end + 1]

    async def aclose(self):
        self.closed = True


# --------------------------------------------------------------------------
# enqueue / get
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_enqueue_writes_stream_hash_and_index():
    fake = FakeRedis()
    store = JobStore(client=fake)

    job_id = await store.enqueue(
        source="mongo",
        document_ids=["d1"],
        file_paths={"d1": "uploads/report.pdf"},
        limit=5,
        dry_run=True,
    )

    assert len(job_id) == 12
    h = fake.hashes[f"kg:job:{job_id}"]
    assert h["state"] == "queued"
    assert h["source"] == "mongo"
    assert h["file_paths"] == '{"d1": "uploads/report.pdf"}'
    assert h["limit"] == "5"
    assert h["dry_run"] == "1"
    assert h["attempts"] == "0"
    assert job_id in fake.zsets[INDEX_KEY]
    assert fake.expires[f"kg:job:{job_id}"] == JOB_TTL_SECONDS
    assert fake.streams[DEFAULT_STREAM][0][1]["job_id"] == job_id


@pytest.mark.asyncio
async def test_get_unknown_and_type_coercion():
    fake = FakeRedis()
    store = JobStore(client=fake)

    assert await store.get("nope") is None

    job_id = await store.enqueue(
        source="minio",
        document_ids=["a", "b"],
        file_paths={"a": "a.pdf", "b": "b.pdf"},
        limit=3,
        dry_run=True,
    )
    job = await store.get(job_id)
    assert isinstance(job["limit"], int) and job["limit"] == 3
    assert job["dry_run"] is True
    assert isinstance(job["attempts"], int) and job["attempts"] == 0
    assert job["document_ids"] == ["a", "b"]
    assert job["file_paths"] == {"a": "a.pdf", "b": "b.pdf"}
    assert job["state"] == "queued"


@pytest.mark.asyncio
async def test_get_no_document_ids_and_dry_run_false():
    fake = FakeRedis()
    store = JobStore(client=fake)
    job_id = await store.enqueue(source="mongo")
    job = await store.get(job_id)
    assert job["document_ids"] is None
    assert job["file_paths"] is None
    assert job["dry_run"] is False


# --------------------------------------------------------------------------
# claim / ack
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_claim_sets_running_bumps_attempts_then_empty():
    fake = FakeRedis()
    store = JobStore(client=fake)
    job_id = await store.enqueue(source="mongo")

    job = await store.claim(block_ms=10)
    assert job["job_id"] == job_id
    assert job["_entry_id"] == "1-0"
    assert job["state"] == "running"
    assert job["attempts"] == 1
    assert job["started_at"]

    assert await store.claim(block_ms=10) is None


@pytest.mark.asyncio
async def test_ack_uses_entry_id_and_drains_pending():
    fake = FakeRedis()
    store = JobStore(client=fake)
    await store.enqueue(source="mongo")
    job = await store.claim(block_ms=10)

    await store.ack(job)

    assert fake.acked == [job["_entry_id"]]
    assert fake.groups[DEFAULT_STREAM][DEFAULT_GROUP]["pending"] == []


# --------------------------------------------------------------------------
# progress / terminal states
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_append_progress_caps(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_PROGRESS", 3)
    fake = FakeRedis()
    store = JobStore(client=fake)
    job_id = await store.enqueue(source="mongo")

    for i in range(5):
        await store.append_progress(job_id, {"event": "x", "i": i})

    job = await store.get(job_id)
    assert len(job["progress"]) == 3
    assert [e["i"] for e in job["progress"]] == [2, 3, 4]


@pytest.mark.asyncio
async def test_mark_done_sets_summary_and_finished_at():
    fake = FakeRedis()
    store = JobStore(client=fake)
    job_id = await store.enqueue(source="mongo")

    await store.mark_done(job_id, {"indexed": 2})

    job = await store.get(job_id)
    assert job["state"] == "done"
    assert job["summary"] == {"indexed": 2}
    assert job["finished_at"]


@pytest.mark.asyncio
async def test_mark_failed_truncates_error():
    fake = FakeRedis()
    store = JobStore(client=fake)
    job_id = await store.enqueue(source="mongo")

    await store.mark_failed(job_id, "e" * 3000)

    job = await store.get(job_id)
    assert job["state"] == "failed"
    assert len(job["error"]) == 2000
    assert job["finished_at"]


# --------------------------------------------------------------------------
# reclaim
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reclaim_stale_bumps_attempts_and_logs():
    fake = FakeRedis()
    store = JobStore(client=fake, stale_seconds=600)
    job_id = await store.enqueue(source="mongo")
    await store.claim(block_ms=10)  # delivers one entry into pending

    reclaimed = await store.reclaim_stale()

    assert fake.xautoclaim_calls == [600000]
    assert len(reclaimed) == 1
    assert reclaimed[0]["job_id"] == job_id
    assert reclaimed[0]["state"] == "running"
    assert reclaimed[0]["attempts"] == 2
    assert reclaimed[0]["_entry_id"] == "1-0"
    assert {"event": "reclaimed"} in reclaimed[0]["progress"]

    await store.reclaim_stale(min_idle_ms=1000)
    assert fake.xautoclaim_calls[-1] == 1000


# --------------------------------------------------------------------------
# list / availability
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_newest_first_skips_deleted():
    fake = FakeRedis()
    store = JobStore(client=fake)
    a = await store.enqueue(source="mongo")
    b = await store.enqueue(source="minio")
    c = await store.enqueue(source="mongo")

    del fake.hashes[f"kg:job:{b}"]  # simulate an expired / deleted hash

    listed = await store.list(limit=10)
    assert [j["job_id"] for j in listed] == [c, a]


def test_available_and_missing_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert jobs.available() is False

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    assert jobs.available() is True

    with pytest.raises(RuntimeError):
        JobStore(url="")


def test_client_has_no_read_timeout_for_blocking_reads(monkeypatch):
    """XREADGROUP BLOCK 5000 vs redis-py's default 5s socket timeout raised
    TimeoutError on every idle poll; the read timeout must be disabled."""
    captured = {}

    def fake_from_url(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeRedis()

    monkeypatch.setattr(jobs.aioredis.Redis, "from_url", fake_from_url)
    store = JobStore(url="redis://redis:6379/0")

    store._client_or_raise()

    assert captured["url"] == "redis://redis:6379/0"
    assert captured["decode_responses"] is True
    assert captured["socket_timeout"] is None
    assert captured["socket_connect_timeout"] == 5


@pytest.mark.asyncio
async def test_close_prefers_aclose():
    fake = FakeRedis()
    store = JobStore(client=fake)
    await store.close()
    assert fake.closed is True
