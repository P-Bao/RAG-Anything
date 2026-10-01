import asyncio
import logging
import os
import tempfile
import time
import uuid
from functools import partial
from pathlib import Path

from opentelemetry.trace import Status, StatusCode

from ami_rag.observability import (
    INGEST_ASSET_UPLOAD_FAILURES_TOTAL,
    INGEST_DOCUMENTS,
    INGEST_DURATION_SECONDS,
    INGEST_EVENTS_TOTAL,
    INGEST_IN_FLIGHT,
    INGEST_ITEMS_TOTAL,
    INGEST_PAGES_TOTAL,
    INGEST_PARSE_FAILURES_TOTAL,
    INGEST_STREAM_LAG,
    INGEST_STREAM_PENDING,
    observe_ingest_stage,
    tracer,
)
from ami_rag.queue.events import EVENT_DELETED, RagEvent
from ami_rag.queue.streams import RagStreamQueue
from ami_rag.settings import Settings, get_settings
from ami_rag.sources import (
    SOURCE_MINIO_PARSE,
    resolve_file_path,
    select_source,
    source_hash,
)

logger = logging.getLogger(__name__)

_COUNT_TYPES = ("text", "image", "table", "equation")
_CRAWL_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_STATS_INTERVAL_SECONDS = 15
_COUNTS_INTERVAL_SECONDS = 60


def count_content_list(content_list: list[dict]) -> tuple[dict, int]:
    """Items per modality (unknown types fold into `other`) and page count (max page_idx + 1)."""
    counts = {t: 0 for t in _COUNT_TYPES}
    other = 0
    max_page = -1
    for item in content_list:
        item_type = item.get("type")
        if item_type in counts:
            counts[item_type] += 1
        else:
            other += 1
        page_idx = item.get("page_idx")
        if isinstance(page_idx, int) and page_idx > max_page:
            max_page = page_idx
    if other:
        counts["other"] = other
    return counts, max_page + 1


class IngestWorker:
    """Consume RAG ingest events from Redis Streams and drive RAGAnything.

    created/updated -> select_source: minio_parse (pdf/docx: MinIO file parsed by the
    multimodal parser, assets uploaded to MinIO) or mongo_text (Mongo `content`) ->
    insert_content_list with doc_id=mongo_id; unchanged source_hash is skipped.
    deleted -> adelete_by_doc_id + MinIO assets + registry row. Updates are
    delete-reinsert.
    """

    def __init__(
        self,
        rag_anything,
        docs_repo,
        state_repo,
        queue: RagStreamQueue,
        asset_store,
        settings: Settings | None = None,
        image_worker=None,
        logger_=None,
    ):
        self.rag_anything = rag_anything
        self.docs_repo = docs_repo
        self.state_repo = state_repo
        self.queue = queue
        self.asset_store = asset_store
        self.settings = settings or get_settings()
        self.image_worker = image_worker
        self.log = logger_ or logger
        self._seen_statuses: set[str] = set()

    async def _purge_index(self, doc_id: str) -> None:
        with observe_ingest_stage("delete"):
            await self.rag_anything.lightrag.adelete_by_doc_id(doc_id)
            await asyncio.to_thread(self.asset_store.delete_doc_assets, doc_id)

    async def _upload_assets(self, doc_id: str, content_list: list[dict]) -> list[str]:
        try:
            with observe_ingest_stage("upload_assets"):
                return await asyncio.to_thread(
                    self.asset_store.upload_content_list_assets, doc_id, content_list
                )
        except Exception:
            INGEST_ASSET_UPLOAD_FAILURES_TOTAL.inc()
            raise

    async def _drop_parse_cache(self, local: Path) -> None:
        """Delete this parse's RAGAnything cache entry.

        The cache key embeds the (temp) file path, so it can never hit again and
        would only accumulate full content lists in Mongo. Best effort.
        """
        try:
            key = self.rag_anything._generate_cache_key(local, self.settings.PARSE_METHOD)
            await self.rag_anything.parse_cache.delete([key])
            await self.rag_anything.parse_cache.index_done_callback()
        except Exception as exc:
            self.log.debug("parse cache cleanup skipped for %s: %s", local.name, exc)

    async def _crawl_image_items(self, doc: dict, doc_id: str, tmp: str) -> list[dict]:
        """Download crawl images (CrawlImageWorker -> MinIO) and fetch them locally as
        image items; failures are skipped (best effort)."""
        items: list[dict] = []
        for url in self.image_worker.collect_image_urls(doc):
            key = await self.image_worker.ensure_in_minio(doc_id, url)
            if not key or Path(key).suffix.lower() not in _CRAWL_IMAGE_EXTS:
                continue
            try:
                local = await asyncio.to_thread(self.asset_store.fetch, key, Path(tmp))
            except Exception as exc:
                self.log.warning("crawl image %s for %s not fetched: %s", url, doc_id, exc)
                continue
            items.append(
                {"type": "image", "img_path": str(local), "image_caption": [], "page_idx": 0}
            )
        return items

    async def handle_event(self, event: RagEvent) -> dict | None:
        """Process one event. Returns ingest info when the document was (re)indexed,
        None when nothing was indexed (deleted, missing, empty, unchanged)."""
        doc_id = event.document_id

        if event.event == EVENT_DELETED:
            await self._purge_index(doc_id)
            await asyncio.to_thread(self.state_repo.delete, doc_id)
            self.log.info("deleted document %s from RAG index", doc_id)
            return None

        doc = await asyncio.to_thread(self.docs_repo.find_by_id, doc_id)
        if doc is None:
            self.log.warning("document %s not found in Mongo, skipping", doc_id)
            return None

        source = select_source(doc)
        file_path = resolve_file_path(doc)
        if source == SOURCE_MINIO_PARSE:
            if not doc.get("file_path"):
                raise ValueError(f"document {doc_id} has no file_path in MinIO to parse")
        elif not (doc.get("content") or "").strip():
            self.log.warning("document %s has empty content, skipping", doc_id)
            return None

        h = source_hash(doc, source)
        prev = await asyncio.to_thread(self.state_repo.get, doc_id)
        if prev and prev.get("status") == "processed" and prev.get("source_hash") == h:
            self.log.debug("document %s unchanged, skipping", doc_id)
            return None

        if prev:
            await self._purge_index(doc_id)

        assets: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            if source == SOURCE_MINIO_PARSE:
                with observe_ingest_stage("download"):
                    local = await asyncio.to_thread(
                        self.asset_store.fetch, doc["file_path"], Path(tmp)
                    )
                try:
                    with observe_ingest_stage("parse"):
                        content_list, _ = await self.rag_anything.parse_document(
                            str(local), output_dir=tmp, parse_method=self.settings.PARSE_METHOD
                        )
                except Exception:
                    INGEST_PARSE_FAILURES_TOTAL.labels(
                        document_type=doc.get("document_type") or "unknown"
                    ).inc()
                    raise
                await self._drop_parse_cache(local)
                assets = await self._upload_assets(doc_id, content_list)
            else:
                content_list = [{"type": "text", "text": doc["content"], "page_idx": 0}]
                if getattr(self.settings, "CRAWL_IMAGES_ENABLED", False) and self.image_worker:
                    content_list += await self._crawl_image_items(doc, doc_id, tmp)
                    assets = await self._upload_assets(doc_id, content_list)

            if not content_list:
                self.log.warning("document %s produced an empty content list, skipping", doc_id)
                return None

            with observe_ingest_stage("upload_assets"):
                await asyncio.to_thread(self.asset_store.save_content_list, doc_id, content_list)
            # Insert inside the temp dir: image processing reads local `img_path` files.
            with observe_ingest_stage("insert"):
                await self.rag_anything.insert_content_list(
                    content_list, file_path=file_path, doc_id=doc_id
                )

        counts, page_count = count_content_list(content_list)
        for modality, n in counts.items():
            if n:
                INGEST_ITEMS_TOTAL.labels(modality=modality).inc(n)
        if page_count:
            INGEST_PAGES_TOTAL.inc(page_count)
        self.log.info(
            "indexed document %s (type=%s, source=%s, file_path=%s, counts=%s)",
            doc_id,
            doc.get("document_type"),
            source,
            file_path,
            counts,
        )
        return {
            "source": source,
            "source_hash": h,
            "file_path": file_path or "",
            "assets": assets,
            "counts": counts,
            "page_count": page_count,
        }

    async def _process_message(self, message_id: str, event: RagEvent) -> None:
        attempts = await asyncio.to_thread(
            self.state_repo.begin_attempt, event.document_id, event.content_hash or ""
        )
        INGEST_IN_FLIGHT.inc()
        started = time.perf_counter()
        source = "none"
        with tracer.start_as_current_span(
            "ingest_document",
            attributes={"document_id": event.document_id, "event": event.event},
        ) as span:
            try:
                try:
                    result = await self.handle_event(event)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    self.log.error(
                        "failed to process event %s for document %s: %s",
                        event.event,
                        event.document_id,
                        exc,
                    )
                    if attempts >= self.settings.WORKER_MAX_DELIVERY:
                        INGEST_EVENTS_TOTAL.labels(event=event.event, result="failed").inc()
                        await asyncio.to_thread(
                            self.state_repo.mark_failed,
                            event.document_id,
                            str(exc),
                            event.content_hash,
                        )
                        await self.queue.ack(message_id)
                        self.log.error(
                            "document %s marked failed after %s attempts",
                            event.document_id,
                            attempts,
                        )
                    else:
                        INGEST_EVENTS_TOTAL.labels(event=event.event, result="retry").inc()
                    return
                if result:
                    source = result.get("source") or "none"
                    span.set_attribute("source", source)
                    INGEST_EVENTS_TOTAL.labels(event=event.event, result="processed").inc()
                    await self.mark_processed(event.document_id, result)
                else:
                    INGEST_EVENTS_TOTAL.labels(event=event.event, result="skipped").inc()
                    await asyncio.to_thread(self.state_repo.release_attempt, event.document_id)
                await self.queue.ack(message_id)
            finally:
                INGEST_DURATION_SECONDS.labels(source=source).observe(time.perf_counter() - started)
                INGEST_IN_FLIGHT.dec()

    async def mark_processed(self, doc_id: str, result: dict) -> None:
        """Persist the outcome returned by handle_event into the registry."""
        await asyncio.to_thread(
            partial(
                self.state_repo.mark_processed,
                doc_id,
                result["source_hash"],
                source=result["source"],
                file_path=result["file_path"],
                parser=self.settings.PARSER,
                assets=result["assets"],
                counts=result["counts"],
                page_count=result["page_count"],
            )
        )

    async def _update_queue_gauges(self) -> None:
        try:
            pending, lag = await self.queue.stats()
            INGEST_STREAM_PENDING.set(pending)
            INGEST_STREAM_LAG.set(lag)
        except Exception as exc:
            self.log.warning("queue gauge update failed: %s", exc)

    async def _update_document_gauges(self) -> None:
        try:
            counts = await asyncio.to_thread(self.state_repo.counts)
            for status in self._seen_statuses - set(counts):
                INGEST_DOCUMENTS.labels(status=status).set(0)
            for status, n in counts.items():
                INGEST_DOCUMENTS.labels(status=status).set(n)
            self._seen_statuses = set(counts)
        except Exception as exc:
            self.log.warning("document gauge update failed: %s", exc)

    async def run_forever(self) -> None:
        await self.queue.ensure_group()
        self.log.info(
            "ingest worker started (stream=%s group=%s consumer=%s)",
            self.settings.RAG_STREAM,
            self.settings.RAG_CONSUMER_GROUP,
            self.queue._consumer,
        )
        next_stats = next_counts = 0.0
        while True:
            now = time.monotonic()
            if now >= next_stats:
                next_stats = now + _STATS_INTERVAL_SECONDS
                await self._update_queue_gauges()
            if now >= next_counts:
                next_counts = now + _COUNTS_INTERVAL_SECONDS
                await self._update_document_gauges()
            try:
                batch = await self.queue.read_batch(
                    count=self.settings.WORKER_BATCH,
                    block_ms=self.settings.WORKER_POLL_BLOCK_MS,
                )
            except Exception as exc:
                self.log.error("queue read failed: %s, retrying", exc)
                await asyncio.sleep(5)
                continue
            for message_id, event in batch:
                await self._process_message(message_id, event)


def _build_default_deps(settings: Settings):
    """Returns (docs_repo, state_repo, queue, asset_store, image_worker)."""
    from ami_rag.core.factory import get_asset_store
    from ami_rag.queue.streams import RagStreamQueue
    from ami_rag.storage.mongo_docs import MongoDocumentRepo
    from ami_rag.storage.rag_documents import RagDocumentsRepo

    consumer = settings.RAG_CONSUMER_NAME or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    docs_repo = MongoDocumentRepo(
        mongo_uri=settings.MONGO_URI,
        db_name=settings.ORG_DB,
        collection_name=settings.DOC_COLLECTION,
    )
    state_repo = RagDocumentsRepo(
        mongo_uri=settings.MONGO_URI,
        db_name=settings.RAG_DB,
        collection_name=settings.RAG_DOCUMENTS_COLLECTION,
    )
    image_worker = None
    if settings.CRAWL_IMAGES_ENABLED:
        from ami_rag.workers.crawl_images import CrawlImageWorker

        image_worker = CrawlImageWorker(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            bucket=settings.MINIO_BUCKET,
            prefix=settings.CRAWL_IMAGES_PREFIX,
            secure=settings.MINIO_SECURE,
        )
    redis_client = aioredis_from_url(settings.REDIS_URL)
    queue = RagStreamQueue(
        redis_client=redis_client,
        stream=settings.RAG_STREAM,
        group=settings.RAG_CONSUMER_GROUP,
        consumer=consumer,
        retry_idle_ms=settings.WORKER_RETRY_IDLE_MS,
    )
    return docs_repo, state_repo, queue, get_asset_store(), image_worker


def aioredis_from_url(url: str):
    import redis.asyncio as aioredis

    return aioredis.from_url(url, decode_responses=True)


async def run_worker(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    # Standalone worker only: the embedded worker shares the API's /metrics.
    from prometheus_client import start_http_server

    start_http_server(settings.WORKER_METRICS_PORT)
    from ami_rag.core.factory import get_raganything

    docs_repo, state_repo, queue, asset_store, image_worker = _build_default_deps(settings)
    worker = IngestWorker(
        rag_anything=await get_raganything(),
        docs_repo=docs_repo,
        state_repo=state_repo,
        queue=queue,
        asset_store=asset_store,
        settings=settings,
        image_worker=image_worker,
    )
    try:
        await worker.run_forever()
    finally:
        from ami_rag.core.factory import close_rag

        await close_rag()


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())
