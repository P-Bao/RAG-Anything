"""HTTP server for the KG branch (drop-in shape of the old vector API :8009).

Endpoints (this session — admin/metrics/redis/qdrant/otel intentionally out):
  GET  /                    service banner
  GET  /healthz             liveness
  GET  /readyz              readiness: LightRAG + storage initialised
  POST /v2/rag/             hybrid chunk retrieval (LightRAG vector + BGE rerank);
                            ``generate=true`` adds the LLM answer
  POST /v2/rag/index        202 + job id; ingest documents from MinIO or Mongo
  GET  /v2/rag/index/{id}   job status/progress
  GET  /v2/rag/status       KG counts + queue + per-document status counts

Auth mirrors the old server: ``Authorization: Bearer <RAG_API_KEY>`` on every
``/v2/rag/*`` route; empty ``RAG_API_KEY`` disables auth. Ingest jobs run on a
single in-process worker — LightRAG file storage must never be written
concurrently — while queries run concurrently in request handlers.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from raganything.ingest import ingest
from raganything.retrieval import retrieve_chunks

logger = logging.getLogger(__name__)

SERVICE_NAME = "ami-rag-kg"
SERVICE_VERSION = "0.1.0"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Job queue (one index at a time)
# ---------------------------------------------------------------------------
class JobQueue:
    """FIFO of ingest jobs processed by a single worker task."""

    def __init__(self, history: int = 100):
        self._queue: asyncio.Queue[tuple[str, dict, Callable]] = asyncio.Queue()
        self._jobs: dict[str, dict] = {}
        self._order: list[str] = []
        self._history = history
        self._worker: asyncio.Task | None = None

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    def submit(self, job_id: str, job: dict, factory: Callable) -> None:
        self._jobs[job_id] = job
        self._order.append(job_id)
        while len(self._order) > self._history:
            self._jobs.pop(self._order.pop(0), None)
        self._queue.put_nowait((job_id, job, factory))

    async def _run(self) -> None:
        while True:
            _job_id, job, factory = await self._queue.get()

            def on_doc(event: str, doc_id: str, info: dict) -> None:
                job["progress"].append({"event": event, "document_id": doc_id, **info})

            job["state"] = "running"
            job["started_at"] = _now()
            try:
                job["summary"] = await factory(on_doc)
            except asyncio.CancelledError:
                job["state"] = "cancelled"
                raise
            except Exception as exc:  # a failed job must not kill the worker
                job["state"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("ingest job %s failed", job["job_id"])
            else:
                job["state"] = "done"
            finally:
                job["finished_at"] = _now()
                self._queue.task_done()

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def stats(self) -> dict[str, Any]:
        return {"queued": self._queue.qsize(), "history": len(self._jobs)}


# ---------------------------------------------------------------------------
# Schemas (field names mirror the old vector API)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class RAGRequest(BaseModel):
    query: str | None = None
    messages: list[ChatMessage] | None = None
    top_k: int | None = Field(default=None, ge=1, le=200)
    include_references: bool = True
    generate: bool = False
    mode: str = "mix"


class IndexRequest(BaseModel):
    source: Literal["minio", "mongo"] = "mongo"
    document_ids: list[str] | None = None
    limit: int = Field(default=0, ge=0)
    dry_run: bool = False


def _resolve_query(req: RAGRequest) -> str:
    if req.query and req.query.strip():
        return req.query.strip()
    for message in reversed(req.messages or []):
        if (message.role or "").lower() == "user" and (message.content or "").strip():
            return message.content.strip()
    return ""


def _references(documents: list[dict]) -> list[dict]:
    seen: set[str] = set()
    refs: list[dict] = []
    for doc in documents:
        ref_id = str(doc.get("reference_id") or doc.get("metadata", {}).get("chunk_id") or "")
        if not ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        refs.append(doc.get("doc") or doc.get("metadata") or {})
    return refs


def _kg_totals(rag: Any) -> dict[str, int] | None:
    if rag is None:
        return None
    lg = rag.lightrag
    return {
        "entities": len(lg.entities_vdb),
        "relations": len(lg.relationships_vdb),
        "chunks": len(lg.chunks_vdb),
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def create_app(
    *,
    rag: Any = None,
    state: Any = None,
    queue: JobQueue | None = None,
    working_dir: str | None = None,
) -> FastAPI:
    """Build the app. ``rag``/``state``/``queue`` are injection points for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.doc_state is None:
            from raganything.doc_state import DocStatusStore

            app.state.doc_state = DocStatusStore()
        if app.state.rag is None:
            try:
                from raganything.kg_factory import build_kg_raganything

                rag_ = build_kg_raganything(
                    lightrag_kwargs={
                        "working_dir": working_dir
                        or _env("KG_WORKING_DIR", "./rag_storage")
                    }
                )
                init = await rag_._ensure_lightrag_initialized()
                if init.get("success"):
                    app.state.rag = rag_
                else:
                    app.state.init_error = str(init.get("error"))
            except Exception as exc:
                app.state.init_error = f"{type(exc).__name__}: {exc}"
                logger.exception("LightRAG init failed")
        app.state.ready = app.state.rag is not None
        app.state.queue.start()
        try:
            yield
        finally:
            await app.state.queue.stop()
            if app.state.doc_state is not None:
                app.state.doc_state.close()

    app = FastAPI(title="AMI KG RAG API", version=SERVICE_VERSION, lifespan=lifespan)
    app.state.rag = rag
    app.state.ready = False
    app.state.init_error = None
    app.state.doc_state = state
    app.state.queue = queue or JobQueue()

    def verify_api_key(authorization: str | None = Header(default=None)) -> None:
        """Bearer auth identical to the old vector API; empty key = open."""
        expected = _env("RAG_API_KEY")
        if not expected:
            return
        token = ""
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if not hmac.compare_digest(token, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")

    # -- open routes --------------------------------------------------------
    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": SERVICE_NAME, "version": SERVICE_VERSION}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        if not app.state.ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "error" if app.state.init_error else "initializing",
                    "error": app.state.init_error,
                },
            )
        return {"status": "ok", "kg": _kg_totals(app.state.rag)}

    # -- /v2/rag/* (bearer auth) -------------------------------------------
    @app.post("/v2/rag/", dependencies=[Depends(verify_api_key)])
    async def rag_query(req: RAGRequest) -> dict[str, Any]:
        query = _resolve_query(req)
        if not query:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "query or messages with a user message required",
            )
        if app.state.rag is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "KG not ready")
        top_k = req.top_k or int(_env("RERANK_TOP_K", "5") or 5)
        try:
            documents, meta = await retrieve_chunks(app.state.rag, query, top_k)
        except RuntimeError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        response: dict[str, Any] = {
            "query": query,
            "documents": documents,
            "references": _references(documents) if req.include_references else [],
            "meta": meta,
        }
        if req.generate:
            response["answer"] = await app.state.rag.aquery(
                query, mode=req.mode, chunk_top_k=top_k
            )
        return response

    @app.post(
        "/v2/rag/index",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_api_key)],
    )
    async def rag_index(req: IndexRequest) -> dict[str, Any]:
        if app.state.rag is None and not req.dry_run:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "KG not ready")
        if req.source == "mongo" and not _env("MONGO_URI"):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "source=mongo requires MONGO_URI"
            )
        job_id = uuid.uuid4().hex[:12]

        async def factory(on_doc: Callable) -> dict:
            return await ingest(
                app.state.rag,
                source=req.source,
                document_ids=req.document_ids,
                limit=req.limit,
                dry_run=req.dry_run,
                state=app.state.doc_state,
                on_doc=on_doc,
            )

        app.state.queue.submit(
            job_id,
            {
                "job_id": job_id,
                "kind": "index",
                "params": req.model_dump(),
                "state": "queued",
                "progress": [],
                "summary": None,
                "error": None,
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
            },
            factory,
        )
        return {"job_id": job_id, "status": "queued"}

    @app.get("/v2/rag/index/{job_id}", dependencies=[Depends(verify_api_key)])
    async def rag_index_status(job_id: str) -> dict[str, Any]:
        job = app.state.queue.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job id")
        return job

    @app.get("/v2/rag/status", dependencies=[Depends(verify_api_key)])
    async def rag_status() -> dict[str, Any]:
        counts = app.state.doc_state.counts() if app.state.doc_state else {}
        return {
            "workspace": _env("KG_WORKING_DIR", "./rag_storage"),
            "ready": bool(app.state.ready),
            "kg": _kg_totals(app.state.rag),
            "queue": app.state.queue.stats(),
            "doc_status_counts": counts,
        }

    return app


app = create_app()


def run() -> None:
    """Entrypoint: ``python -c "from raganything.api import run; run()"``."""
    import uvicorn

    uvicorn.run(
        app,
        host=_env("API_HOST", "0.0.0.0"),
        port=int(_env("API_PORT", "8009") or 8009),
    )
