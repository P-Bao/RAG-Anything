"""Redis Stream ingest job queue for the KG server.

Redis Stream + consumer group gives at-least-once delivery: a job stays in the
group's pending list until the worker ``XACK``s it, so a crashed worker's
in-flight jobs can be reclaimed with ``XAUTOCLAIM`` instead of silently lost.
The file-based LightRAG storage is last-write-wins and must never be written by
two processes, so exactly one worker consumes the group; the group/pending
machinery only recovers a single worker that died mid-job.

The API enqueues (``XADD`` + a status hash + an index zset); the resident worker
claims, processes, reports progress and acks. State lives in
``kg:job:{job_id}`` hashes (TTL'd) and the ``kg:jobs`` zset keeps an index.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import socket
import time
import uuid

from redis import asyncio as aioredis

logger = logging.getLogger(__name__)

DEFAULT_STREAM = "kg:ingest"
DEFAULT_GROUP = "kg-workers"
JOB_TTL_SECONDS = 7 * 24 * 3600
STALE_SECONDS = 600
MAX_PROGRESS = 500
HASH_PREFIX = "kg:job:"
INDEX_KEY = "kg:jobs"


def available() -> bool:
    """True when ``REDIS_URL`` is set, so the API picks Redis over in-process."""
    return bool(os.environ.get("REDIS_URL"))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_optional(value):
    return None if value in (None, "", "null") else value


def _load_json(value, default):
    if value in (None, "", "null"):
        return default
    return json.loads(value)


class JobStore:
    def __init__(
        self,
        url=None,
        *,
        stream=None,
        group=None,
        consumer=None,
        client=None,
        ttl_seconds=JOB_TTL_SECONDS,
        stale_seconds=None,
    ):
        self.url = url if url is not None else os.environ.get("REDIS_URL", "")
        self.stream = stream or os.environ.get("KG_JOB_STREAM") or DEFAULT_STREAM
        self.group = group or os.environ.get("KG_JOB_GROUP") or DEFAULT_GROUP
        self.consumer = (
            consumer
            or os.environ.get("KG_WORKER_NAME")
            or f"kg-{socket.gethostname()}-{os.getpid()}"
        )
        if stale_seconds is None:
            env_stale = os.environ.get("KG_WORKER_STALE_SECONDS")
            stale_seconds = int(env_stale) if env_stale else STALE_SECONDS
        self.stale_seconds = stale_seconds
        self.ttl_seconds = ttl_seconds
        self._client = client
        self._group_ready = False
        if self._client is None and not self.url:
            raise RuntimeError("REDIS_URL is not set")

    # -- internals ---------------------------------------------------------
    def _key(self, job_id: str) -> str:
        return f"{HASH_PREFIX}{job_id}"

    def _client_or_raise(self):
        if self._client is None:
            # lazy: tests inject a client, and available() gates construction
            self._client = aioredis.Redis.from_url(
                self.url,
                decode_responses=True,
                # XREADGROUP BLOCK waits server-side for the whole block window.
                # redis-py defaults socket_timeout to 5s, which races a 5s block
                # and raises TimeoutError on every idle poll, so reads must not
                # time out; keepalive still detects a dead peer.
                socket_timeout=None,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
        return self._client

    async def _ensure_group(self, client) -> None:
        if self._group_ready:
            return
        try:
            await client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except Exception as exc:  # BUSYGROUP means another worker created it
            if "BUSYGROUP" not in str(exc):
                raise
        self._group_ready = True

    @staticmethod
    def _to_job(raw: dict) -> dict:
        return {
            "job_id": raw.get("job_id"),
            "source": raw.get("source"),
            "document_ids": _load_json(raw.get("document_ids"), None),
            "file_paths": _load_json(raw.get("file_paths"), None),
            "limit": _as_int(raw.get("limit")),
            "dry_run": raw.get("dry_run") == "1",
            "state": raw.get("state"),
            "progress": _load_json(raw.get("progress"), []) or [],
            "summary": _load_json(raw.get("summary"), None),
            "error": _as_optional(raw.get("error")),
            "attempts": _as_int(raw.get("attempts")),
            "created_at": _as_optional(raw.get("created_at")),
            "started_at": _as_optional(raw.get("started_at")),
            "finished_at": _as_optional(raw.get("finished_at")),
        }

    # -- API ---------------------------------------------------------------
    async def enqueue(
        self,
        *,
        source: str,
        document_ids: list[str] | None = None,
        file_paths: dict[str, str] | None = None,
        limit: int = 0,
        dry_run: bool = False,
    ) -> str:
        client = self._client_or_raise()
        job_id = uuid.uuid4().hex[:12]
        doc_json = json.dumps(list(document_ids)) if document_ids else ""
        paths_json = json.dumps(file_paths) if file_paths else ""
        dry = "1" if dry_run else "0"
        await client.xadd(
            self.stream,
            {
                "job_id": job_id,
                "source": source,
                "document_ids": doc_json,
                "file_paths": paths_json,
                "limit": str(limit),
                "dry_run": dry,
            },
        )
        await client.hset(
            self._key(job_id),
            mapping={
                "job_id": job_id,
                "source": source,
                "document_ids": doc_json,
                "file_paths": paths_json,
                "limit": str(limit),
                "dry_run": dry,
                "state": "queued",
                "progress": "[]",
                "summary": "",
                "error": "",
                "attempts": "0",
                "created_at": _now(),
                "started_at": "",
                "finished_at": "",
            },
        )
        await client.expire(self._key(job_id), self.ttl_seconds)
        await client.zadd(INDEX_KEY, {job_id: time.time()})
        return job_id

    async def claim(self, block_ms: int = 5000) -> dict | None:
        client = self._client_or_raise()
        await self._ensure_group(client)
        resp = await client.xreadgroup(
            self.group, self.consumer, {self.stream: ">"}, count=1, block=block_ms
        )
        if not resp:
            return None
        _stream, messages = resp[0]
        entry_id, fields = messages[0]
        job_id = fields.get("job_id")
        if not job_id:
            return None
        await client.hset(
            self._key(job_id), mapping={"state": "running", "started_at": _now()}
        )
        await client.hincrby(self._key(job_id), "attempts", 1)
        job = await self.get(job_id)
        if job is None:
            return None
        job["_entry_id"] = str(entry_id)
        return job

    async def reclaim_stale(self, min_idle_ms: int | None = None) -> list[dict]:
        client = self._client_or_raise()
        await self._ensure_group(client)
        idle = self.stale_seconds * 1000 if min_idle_ms is None else min_idle_ms
        # XAUTOCLAIM returns (cursor, messages[, deleted_ids]) across versions.
        result = await client.xautoclaim(
            self.stream, self.group, self.consumer, idle, "0-0", count=100
        )
        messages = result[1] if len(result) > 1 else []
        jobs = []
        for entry_id, fields in messages:
            if not fields:
                continue  # entry trimmed away; nothing to recover
            job_id = fields.get("job_id")
            if not job_id:
                continue
            logger.warning("reclaiming stale ingest job %s (entry %s)", job_id, entry_id)
            await client.hset(
                self._key(job_id), mapping={"state": "running", "started_at": _now()}
            )
            await client.hincrby(self._key(job_id), "attempts", 1)
            await self.append_progress(job_id, {"event": "reclaimed"})
            job = await self.get(job_id)
            if job is not None:
                job["_entry_id"] = str(entry_id)
                jobs.append(job)
        return jobs

    async def ack(self, job: dict) -> None:
        entry_id = job.get("_entry_id")
        if not entry_id:
            return
        client = self._client_or_raise()
        await client.xack(self.stream, self.group, entry_id)

    async def append_progress(self, job_id: str, entry: dict) -> None:
        client = self._client_or_raise()
        raw = await client.hgetall(self._key(job_id))
        progress = _load_json(raw.get("progress"), []) or []
        progress.append(entry)
        # cap the log so a long job cannot grow the hash without bound
        progress = progress[-MAX_PROGRESS:]
        await client.hset(
            self._key(job_id), mapping={"progress": json.dumps(progress)}
        )

    async def mark_done(self, job_id: str, summary: dict) -> None:
        client = self._client_or_raise()
        await client.hset(
            self._key(job_id),
            mapping={
                "state": "done",
                "summary": json.dumps(summary),
                "finished_at": _now(),
            },
        )

    async def mark_failed(self, job_id: str, error: str) -> None:
        client = self._client_or_raise()
        await client.hset(
            self._key(job_id),
            mapping={
                "state": "failed",
                "error": str(error)[:2000],
                "finished_at": _now(),
            },
        )

    async def get(self, job_id: str) -> dict | None:
        client = self._client_or_raise()
        raw = await client.hgetall(self._key(job_id))
        if not raw:
            return None
        return self._to_job(raw)

    async def list(self, limit: int = 20) -> list[dict]:
        client = self._client_or_raise()
        ids = await client.zrevrange(INDEX_KEY, 0, limit - 1)
        out = []
        for job_id in ids:
            job = await self.get(job_id)
            if job is not None:
                out.append(job)
        return out

    async def close(self) -> None:
        client = self._client
        if client is None:
            return
        self._client = None
        closer = getattr(client, "aclose", None) or getattr(client, "close", None)
        if closer is None:
            return
        result = closer()
        if inspect.isawaitable(result):
            await result
