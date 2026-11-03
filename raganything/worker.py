"""Resident ingest worker for the ``kg-rag`` container.

Consumes jobs from the Redis Stream ``JobStore`` and runs them one at a time
(LightRAG file storage must never be written concurrently) until the container
is stopped. ``main`` is the entry point for ``python -m raganything.worker``;
``restart: unless-stopped`` retries it when LightRAG init fails (fail fast).

Progress is streamed to the same ``JobStore`` the HTTP server enqueues on, so
``GET /v2/rag/index/{id}`` sees a job running in another container.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from raganything.doc_state import DocStatusStore
from raganything.ingest import ingest
from raganything.jobs import JobStore
from raganything.kg_factory import build_kg_raganything

logger = logging.getLogger(__name__)


async def _drain(pending: set[asyncio.Task]) -> None:
    """Wait for the fire-and-forget progress writes before the terminal write."""
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def process_job(rag: Any, jobs: Any, job: dict, *, state: Any = None) -> dict:
    """Run one ingest job, reporting progress and marking terminal state.

    Never raises for a bad job: the worker loop must survive it. Returns the
    ingest summary on success, ``{}`` on failure.
    """
    job_id = str(job.get("job_id") or "")
    started = time.monotonic()
    # ``on_doc`` is called synchronously by ingest, so progress writes are
    # scheduled and awaited once per document (drained before the final write).
    pending: set[asyncio.Task] = set()

    def on_doc(event: str, doc_id: str, info: dict) -> None:
        task = asyncio.ensure_future(
            jobs.append_progress(job_id, {"event": event, "document_id": doc_id, **info})
        )
        pending.add(task)
        task.add_done_callback(pending.discard)

    try:
        summary = await ingest(
            rag,
            source=job["source"],
            document_ids=job["document_ids"],
            limit=job["limit"],
            dry_run=job["dry_run"],
            state=state,
            on_doc=on_doc,
        )
    except Exception as exc:  # a bad job must never kill the worker loop
        logger.exception("ingest job %s failed", job_id)
        await _drain(pending)
        await jobs.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
        return {}

    await _drain(pending)
    await jobs.mark_done(job_id, summary)
    logger.info(
        "job %s done in %.1fs: indexed=%s skipped=%s failed=%s",
        job_id,
        time.monotonic() - started,
        summary.get("indexed"),
        summary.get("skipped"),
        summary.get("failed"),
    )
    return summary


async def run_worker(
    *, rag: Any, jobs: Any, state: Any = None, poll_ms: int = 5000, once: bool = False
) -> None:
    """Claim and process jobs forever (or once, for tests).

    Reclaims pending entries left by a killed worker before the first claim.
    A ``CancelledError`` is re-raised so graceful shutdown finishes the current
    job first; any other loop error (e.g. a Redis hiccup) is logged and retried
    instead of crashing the container.
    """
    reclaimed = await jobs.reclaim_stale()
    if reclaimed:
        logger.warning("reclaimed %d stale job(s) from a previous worker", len(reclaimed))

    while True:
        try:
            job = await jobs.claim(block_ms=poll_ms)
            if job is None:
                if once:
                    return
                await asyncio.sleep(0)  # claim blocks; stay cooperative
                continue
            logger.info(
                "claimed job %s: source=%s limit=%s dry_run=%s docs=%s",
                job.get("job_id"),
                job.get("source"),
                job.get("limit"),
                job.get("dry_run"),
                job.get("document_ids"),
            )
            await process_job(rag, jobs, job, state=state)
            await jobs.ack(job)
            if once:
                return
        except asyncio.CancelledError:
            # Graceful shutdown: the current job already finished; just stop.
            raise
        except Exception:  # Redis hiccup or unexpected error: keep alive
            logger.exception("worker loop error; retrying")
            await asyncio.sleep(poll_ms / 1000)


def main() -> int:
    """Container entry point: build the KG rag, then run the resident worker."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    working_dir = os.environ.get("KG_WORKING_DIR") or "./rag_storage"
    poll_ms = int(os.environ.get("KG_WORKER_POLL_MS") or 5000)

    async def _serve() -> int:
        try:
            jobs = JobStore()  # checked first: no REDIS_URL, no worker
        except RuntimeError as exc:
            logger.error("cannot start worker: %s", exc)
            return 1

        rag = build_kg_raganything(lightrag_kwargs={"working_dir": working_dir})
        init = await rag._ensure_lightrag_initialized()
        if not (init or {}).get("success"):
            logger.error("LightRAG init failed: %s", (init or {}).get("error"))
            return 1  # fail fast; restart: unless-stopped retries

        state = DocStatusStore()
        try:
            logger.info("worker ready, waiting for jobs (working_dir=%s)", working_dir)
            await run_worker(rag=rag, jobs=jobs, state=state, poll_ms=poll_ms)
        finally:
            await jobs.close()
            state.close()
        return 0

    return asyncio.run(_serve())


if __name__ == "__main__":
    raise SystemExit(main())
