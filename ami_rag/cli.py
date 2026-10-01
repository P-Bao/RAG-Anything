import argparse
import asyncio
from itertools import islice

from ami_rag.settings import get_settings
from ami_rag.sources import select_source, source_hash


def _build_docs_repo(settings):
    from ami_rag.storage.mongo_docs import MongoDocumentRepo

    return MongoDocumentRepo(
        mongo_uri=settings.MONGO_URI,
        db_name=settings.ORG_DB,
        collection_name=settings.DOC_COLLECTION,
    )


def _build_state_repo(settings):
    from ami_rag.storage.rag_documents import RagDocumentsRepo

    return RagDocumentsRepo(
        mongo_uri=settings.MONGO_URI,
        db_name=settings.RAG_DB,
        collection_name=settings.RAG_DOCUMENTS_COLLECTION,
    )


def _build_queue(settings):
    import os
    import uuid

    from ami_rag.queue.streams import RagStreamQueue
    from ami_rag.workers.ingest_worker import aioredis_from_url

    return RagStreamQueue(
        redis_client=aioredis_from_url(settings.REDIS_URL),
        stream=settings.RAG_STREAM,
        group=settings.RAG_CONSUMER_GROUP,
        consumer=settings.RAG_CONSUMER_NAME or f"cli-{os.getpid()}-{uuid.uuid4().hex[:8]}",
    )


def _parse_types(raw: str | None) -> list[str] | None:
    types = [t.strip() for t in (raw or "").split(",") if t.strip()]
    return types or None


def _select_documents(docs_repo, args) -> list[dict]:
    types = _parse_types(args.type)
    if args.doc_ids:
        documents = [d for d in (docs_repo.find_by_id(i) for i in args.doc_ids) if d]
        if types:
            documents = [d for d in documents if d.get("document_type") in types]
        return documents[: args.limit] if args.limit else documents
    cursor = docs_repo.iter_all(batch_size=args.batch_size, document_types=types)
    return list(islice(cursor, args.limit) if args.limit else cursor)


def _print_dry_run(documents: list[dict], processed_ids: set[str]) -> None:
    print(f"{'doc_id':<26}{'type':<8}{'source':<13}{'hash':<14}processed")
    for doc in documents:
        source = select_source(doc)
        print(
            f"{doc['_id']!s:<26}{doc.get('document_type') or '-'!s:<8}"
            f"{source:<13}{source_hash(doc, source)[:12]:<14}"
            f"{'yes' if str(doc['_id']) in processed_ids else 'no'}"
        )
    print(f"{len(documents)} document(s); dry-run, nothing written")


def _event_for(doc: dict):
    from ami_rag.queue.events import EVENT_CREATED, RagEvent

    return RagEvent(
        event=EVENT_CREATED,
        document_id=str(doc["_id"]),
        content_hash=doc.get("content_hash"),
        document_type=doc.get("document_type"),
        org_id=str(doc["organization_unit_id"]) if doc.get("organization_unit_id") else None,
    )


async def _reindex_direct(settings, docs_repo, state_repo, documents: list[dict]) -> None:
    from ami_rag.core.factory import close_rag, get_asset_store, get_raganything
    from ami_rag.workers.ingest_worker import IngestWorker

    worker = IngestWorker(
        rag_anything=await get_raganything(),
        docs_repo=docs_repo,
        state_repo=state_repo,
        queue=None,
        asset_store=get_asset_store(),
        settings=settings,
    )
    indexed = skipped = failed = 0
    try:
        for doc in documents:
            doc_id = str(doc["_id"])
            event = _event_for(doc)
            try:
                result = await worker.handle_event(event)
            except Exception as exc:
                failed += 1
                print(f"FAILED {doc_id}: {exc}")
                await asyncio.to_thread(
                    state_repo.mark_failed, doc_id, str(exc), event.content_hash
                )
                continue
            if result:
                await worker.mark_processed(doc_id, result)
                indexed += 1
                print(f"indexed {doc_id} counts={result['counts']}")
            else:
                skipped += 1
    finally:
        await close_rag()
    print(f"direct reindex done: indexed={indexed} skipped={skipped} failed={failed}")


async def cmd_reindex(args) -> None:
    settings = get_settings()
    docs_repo = _build_docs_repo(settings)
    state_repo = _build_state_repo(settings)
    documents = await asyncio.to_thread(_select_documents, docs_repo, args)

    if args.dry_run:
        processed_ids = await asyncio.to_thread(state_repo.get_processed_ids)
        _print_dry_run(documents, processed_ids)
        return

    if args.force:
        ids = [str(d["_id"]) for d in documents]
        staled = await asyncio.to_thread(state_repo.mark_stale, ids)
        print(f"marked {staled} processed document(s) stale (--force)")

    if args.direct:
        await _reindex_direct(settings, docs_repo, state_repo, documents)
        return

    queue = _build_queue(settings)
    for doc in documents:
        await queue.publish(_event_for(doc))
    print(f"published {len(documents)} ingest events to {settings.RAG_STREAM}")


async def cmd_status(_args) -> None:
    import httpx

    settings = get_settings()
    docs_repo = _build_docs_repo(settings)
    state_repo = _build_state_repo(settings)
    queue = _build_queue(settings)
    total = await asyncio.to_thread(docs_repo.count)
    state_counts = await asyncio.to_thread(state_repo.counts)
    failed_ids = await asyncio.to_thread(state_repo.get_failed_ids)
    pending = await queue.pending_count()
    print(f"workspace: {settings.WORKSPACE}")
    print(f"documents (active) in org_db: {total}")
    print(f"rag documents: {state_counts}")
    print(f"queue pending: {pending}")
    print(f"failed document ids: {failed_ids or 'none'}")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.RERANK_BASE_URL.rstrip('/')}/health")
            print(f"rerank service: {'ok' if resp.status_code == 200 else resp.status_code}")
    except Exception:
        print("rerank service: unreachable")


async def cmd_purge(args) -> None:
    from ami_rag.core.factory import close_rag, get_asset_store, get_rag

    settings = get_settings()
    state_repo = _build_state_repo(settings)
    rag = await get_rag()
    try:
        await rag.adelete_by_doc_id(args.doc_id)
        removed = await asyncio.to_thread(get_asset_store().delete_doc_assets, args.doc_id)
        await asyncio.to_thread(state_repo.delete, args.doc_id)
        print(f"purged {args.doc_id} from the RAG index ({removed} asset object(s) removed)")
    finally:
        await close_rag()


def main() -> None:
    parser = argparse.ArgumentParser(prog="ami-rag", description="AMI RAG maintenance CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_reindex = sub.add_parser("reindex", help="publish ingest events for existing documents")
    p_reindex.add_argument("--all", action="store_true", help="reindex every active document")
    p_reindex.add_argument("--doc-ids", nargs="*", default=[], help="specific document ids")
    p_reindex.add_argument(
        "--type", default=None, help="comma-separated document types (pdf,docx,text,crawl)"
    )
    p_reindex.add_argument("--limit", type=int, default=None)
    p_reindex.add_argument("--batch-size", type=int, default=100)
    p_reindex.add_argument(
        "--force", action="store_true", help="re-ingest even if already processed"
    )
    p_reindex.add_argument(
        "--dry-run", action="store_true", help="list what would be ingested and exit"
    )
    p_reindex.add_argument(
        "--direct", action="store_true", help="ingest in-process instead of publishing events"
    )

    sub.add_parser("status", help="show rag document status, queue depth and failed docs")

    p_purge = sub.add_parser(
        "purge-doc", help="delete one document from the RAG index, assets and registry"
    )
    p_purge.add_argument("--doc-id", required=True)

    args = parser.parse_args()
    if args.command == "reindex" and not (args.all or args.doc_ids):
        parser.error("reindex requires --all or --doc-ids")
    if args.command == "reindex":
        asyncio.run(cmd_reindex(args))
    elif args.command == "status":
        asyncio.run(cmd_status(args))
    elif args.command == "purge-doc":
        asyncio.run(cmd_purge(args))
