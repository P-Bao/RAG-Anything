import argparse
import asyncio
from itertools import islice

from ami_rag.settings import get_settings
from ami_rag.sources import source_hash


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


def _split_processed(documents: list[dict], processed_hashes: dict[str, str]):
    """(to_ingest, already_processed): processed with an unchanged source hash is skipped."""
    todo, done = [], []
    for doc in documents:
        unchanged = processed_hashes.get(str(doc["_id"])) == source_hash(doc)
        (done if unchanged else todo).append(doc)
    return todo, done


def _print_table(rows: dict[str, list[int]], headers: list[str]) -> None:
    print(f"{'type':<10}" + "".join(f"{h:>12}" for h in headers))
    for key, values in sorted(rows.items()):
        print(f"{key:<10}" + "".join(f"{v:>12}" for v in values))
    totals = [sum(v[i] for v in rows.values()) for i in range(len(headers))]
    print(f"{'ALL':<10}" + "".join(f"{v:>12}" for v in totals))


def _print_dry_run(documents: list[dict], processed_hashes: dict[str, str]) -> None:
    _, done = _split_processed(documents, processed_hashes)
    done_ids = {str(d["_id"]) for d in done}
    rows: dict[str, list[int]] = {}
    for doc in documents:
        row = rows.setdefault(doc.get("document_type") or "-", [0, 0, 0])
        skipped = str(doc["_id"]) in done_ids
        row[0] += 1
        row[1] += skipped
        row[2] += not skipped
    _print_table(rows, ["total", "skip(done)", "to_ingest"])
    print("dry-run, nothing written (processed docs are skipped unless --force)")


def _event_for(doc: dict):
    from ami_rag.queue.events import EVENT_CREATED, RagEvent

    return RagEvent(
        event=EVENT_CREATED,
        document_id=str(doc["_id"]),
        content_hash=doc.get("content_hash") or source_hash(doc),
        document_type=doc.get("document_type"),
        org_id=str(doc["organization_unit_id"]) if doc.get("organization_unit_id") else None,
    )


def _print_error_summary(by_code: dict[str, list[tuple[str, str]]]) -> None:
    """by_code: {error_code: [(doc_id, message), ...]} -> grouped table with hints."""
    from ami_rag.index_check import ERROR_HINTS

    if not by_code:
        return
    print("\nerrors by cause:")
    print(f"  {'code':<22}{'docs':>6}  example")
    for code, items in sorted(by_code.items(), key=lambda kv: -len(kv[1])):
        doc_id, message = items[0]
        print(f"  {code:<22}{len(items):>6}  {doc_id}: {message[:140]}")
        print(f"  {'':<22}{'':>6}  -> {ERROR_HINTS.get(code, ERROR_HINTS['OTHER'])}")


def _print_fatal(exc, remaining: int) -> None:
    from ami_rag.index_check import ERROR_HINTS

    print(f"\nSTOPPED: {exc}")
    print(f"  {ERROR_HINTS.get(exc.code, '')}")
    print(
        f"  {remaining} document(s) were NOT attempted and keep their current status "
        "(nothing was marked failed because of this error)."
    )


async def _reindex_direct(settings, docs_repo, state_repo, documents: list[dict]) -> int:
    from ami_rag.core.factory import close_rag, get_asset_store, get_raganything
    from ami_rag.index_check import (
        FATAL_ERROR_CODES,
        FatalIngestError,
        classify_error,
        preflight_providers,
        short_error,
    )
    from ami_rag.workers.ingest_worker import IngestWorker

    rag_anything = await get_raganything()
    worker = IngestWorker(
        rag_anything=rag_anything,
        docs_repo=docs_repo,
        state_repo=state_repo,
        queue=None,
        asset_store=get_asset_store(),
        settings=settings,
    )
    indexed = skipped = failed = 0
    errors: dict[str, list[tuple[str, str]]] = {}
    rc = 0
    try:
        try:
            await preflight_providers(rag_anything.lightrag)
        except FatalIngestError as exc:
            _print_fatal(exc, len(documents))
            return 2
        for pos, doc in enumerate(documents):
            doc_id = str(doc["_id"])
            event = _event_for(doc)
            try:
                result = await worker.handle_event(event)
            except Exception as exc:
                code = classify_error(exc)
                if code in FATAL_ERROR_CODES:
                    _print_fatal(
                        FatalIngestError(code, f"{doc_id}: {short_error(exc)}"),
                        len(documents) - pos,
                    )
                    rc = 2
                    break
                failed += 1
                errors.setdefault(code, []).append((doc_id, short_error(exc)))
                print(f"FAILED {doc_id} [{code}]: {short_error(exc)}")
                await asyncio.to_thread(
                    state_repo.mark_failed,
                    doc_id,
                    str(exc),
                    event.content_hash,
                    error_code=code,
                    error_stage="ingest",
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
    _print_error_summary(errors)
    return rc or (1 if failed else 0)


async def _inspect_many(rag, doc_ids: list[str], concurrency: int = 8) -> list:
    from ami_rag.index_check import inspect_document

    sem = asyncio.Semaphore(concurrency)

    async def one(doc_id: str):
        async with sem:
            return await inspect_document(rag, doc_id)

    return list(await asyncio.gather(*(one(i) for i in doc_ids)))


async def _repair_one(rag, st, require: bool, retries: int = 2):
    """Repair one doc. Returns (kind, payload):
    ok -> after-stats | full -> message | fatal -> FatalIngestError | fail -> (code, message)
    """
    from ami_rag.index_check import (
        ERR_INCOMPLETE,
        FATAL_ERROR_CODES,
        FatalIngestError,
        IngestIncompleteError,
        classify_error,
        repair_document,
        short_error,
    )

    for attempt in range(retries + 1):
        try:
            after = await repair_document(rag, st.doc_id, st)
        except IngestIncompleteError as exc:
            return "full", short_error(exc)
        except Exception as exc:
            code = classify_error(exc)
            if code in FATAL_ERROR_CODES:
                return "fatal", FatalIngestError(code, f"{st.doc_id}: {short_error(exc)}")
            if attempt < retries:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            return "fail", (code, short_error(exc))
        problems = after.problems(require)
        if problems:
            return "fail", (ERR_INCOMPLETE, "; ".join(problems))
        return "ok", after
    return "fail", ("OTHER", "unreachable")


async def _reindex_repair(args, settings, state_repo, documents: list[dict]) -> int:
    """Re-run only the missing index parts (chunk vectors, entities) of processed AND failed
    docs from the chunk text already in Mongo. Docs with no chunks / no chunk text, and docs
    never registered, fall back to a full re-ingest (published to the stream).

    Stops the whole command on quota/auth errors (they would fail every remaining doc).
    Returns the process exit code: 0 all good, 1 some docs still failed, 2 stopped early.
    """
    from ami_rag.core.factory import close_rag, get_rag
    from ami_rag.index_check import FatalIngestError, preflight_providers

    require = settings.INGEST_REQUIRE_ENTITIES
    repairable = await asyncio.to_thread(state_repo.get_repairable_hashes)
    failed_ids = set(await asyncio.to_thread(state_repo.get_failed_ids))
    candidates = [d for d in documents if str(d["_id"]) in repairable]
    unregistered = [d for d in documents if str(d["_id"]) not in repairable]
    by_id = {str(d["_id"]): d for d in candidates}
    print(
        f"repair: {len(candidates)}/{len(documents)} selected doc(s) are processed/failed "
        f"({sum(1 for i in by_id if i in failed_ids)} failed); "
        f"{len(unregistered)} never ingested -> full ingest"
    )
    rag = await get_rag()
    rc = 0
    try:
        stats = await _inspect_many(rag, list(by_id))
        bad = [st for st in stats if not st.ok(require)]
        healed = [st for st in stats if st.ok(require) and st.doc_id in failed_ids]
        for st in bad:
            print(f"  {st.line(require)} type={by_id[st.doc_id].get('document_type')}")
        print(
            f"repair: {len(stats) - len(bad)} complete, {len(bad)} incomplete "
            f"({len(healed)} complete but still marked failed)"
        )
        if args.dry_run:
            print("dry-run, nothing written")
            return 0

        for st in healed:
            await asyncio.to_thread(state_repo.mark_repaired, st.doc_id, st.as_dict())
        if bad:
            try:
                await preflight_providers(rag)
            except FatalIngestError as exc:
                _print_fatal(exc, len(bad))
                return 2

        repaired = failed = 0
        full_reingest: list[dict] = list(unregistered)
        errors: dict[str, list[tuple[str, str]]] = {}
        transient_streak = 0
        width = max(1, settings.REPAIR_CONCURRENCY)
        for start in range(0, len(bad), width):
            batch = bad[start : start + width]
            results = await asyncio.gather(*(_repair_one(rag, st, require) for st in batch))
            fatal = None
            for st, (kind, payload) in zip(batch, results):
                if kind == "ok":
                    repaired += 1
                    transient_streak = 0
                    print(f"  repaired {payload.line(require)}")
                    await asyncio.to_thread(state_repo.mark_repaired, st.doc_id, payload.as_dict())
                elif kind == "full":
                    print(f"  NEEDS FULL REINDEX {st.doc_id}: {payload}")
                    full_reingest.append(by_id[st.doc_id])
                elif kind == "fatal":
                    fatal = fatal or payload
                else:
                    code, message = payload
                    failed += 1
                    transient_streak = (
                        transient_streak + 1 if code not in ("INDEX_INCOMPLETE",) else 0
                    )
                    errors.setdefault(code, []).append((st.doc_id, message))
                    print(f"  REPAIR FAILED {st.doc_id} [{code}]: {message}")
                    await asyncio.to_thread(
                        state_repo.mark_failed,
                        st.doc_id,
                        message,
                        None,
                        error_code=code,
                        error_stage="repair",
                    )
            if fatal is not None:
                done = start + len(batch)
                _print_fatal(fatal, len(bad) - done)
                rc = 2
                break
            if transient_streak >= 5:
                print(
                    f"\nSTOPPED: {transient_streak} consecutive documents failed "
                    f"with provider errors; {len(bad) - start - len(batch)} not attempted."
                )
                rc = 2
                break

        print(f"repair done: repaired={repaired} failed={failed} full_reindex={len(full_reingest)}")
        _print_error_summary(errors)
        if full_reingest and rc != 2:
            ids = [str(d["_id"]) for d in full_reingest]
            await asyncio.to_thread(state_repo.mark_stale, ids)
            queue = _build_queue(settings)
            for doc in full_reingest:
                await queue.publish(_event_for(doc))
            print(f"published {len(full_reingest)} full ingest events to {settings.RAG_STREAM}")
        if rc == 0 and failed:
            rc = 1
        if rc:
            print(
                f"exit code {rc}: re-run `ami-rag reindex --all --repair` after fixing the cause above"
            )
        return rc
    finally:
        await close_rag()


async def cmd_reindex(args) -> int:
    """Returns the process exit code (0 ok, 1 some docs failed, 2 stopped early)."""
    settings = get_settings()
    docs_repo = _build_docs_repo(settings)
    state_repo = _build_state_repo(settings)
    documents = await asyncio.to_thread(_select_documents, docs_repo, args)

    if args.repair:
        return await _reindex_repair(args, settings, state_repo, documents)

    processed_hashes = await asyncio.to_thread(state_repo.get_processed_hashes)
    if args.dry_run:
        _print_dry_run(documents, processed_hashes if not args.force else {})
        return 0

    if args.force:
        ids = [str(d["_id"]) for d in documents]
        staled = await asyncio.to_thread(state_repo.mark_stale, ids)
        print(f"marked {staled} processed document(s) stale (--force)")
    else:
        documents, done = _split_processed(documents, processed_hashes)
        print(f"skipping {len(done)} already processed document(s); {len(documents)} to ingest")

    if args.direct:
        return await _reindex_direct(settings, docs_repo, state_repo, documents)

    queue = _build_queue(settings)
    for doc in documents:
        await queue.publish(_event_for(doc))
    print(f"published {len(documents)} ingest events to {settings.RAG_STREAM}")
    return 0


async def cmd_verify(args) -> bool:
    """Count what really exists in the index per document. Returns True when all are OK."""
    from ami_rag.core.factory import close_rag, get_rag

    settings = get_settings()
    docs_repo = _build_docs_repo(settings)
    state_repo = _build_state_repo(settings)
    documents = await asyncio.to_thread(_select_documents, docs_repo, args)
    hashes = await asyncio.to_thread(state_repo.get_processed_hashes)
    by_id = {str(d["_id"]): d for d in documents}
    require = settings.INGEST_REQUIRE_ENTITIES

    rag = await get_rag()
    try:
        stats = await _inspect_many(rag, list(by_id))
    finally:
        await close_rag()

    rows: dict[str, list[int]] = {}
    bad = 0
    failed_info = await asyncio.to_thread(state_repo.failed_by_code, 10**9)
    failed_code = {i: code for code, e in failed_info.items() for i in e["ids"]}
    for st in stats:
        doc_type = by_id[st.doc_id].get("document_type") or "-"
        row = rows.setdefault(doc_type, [0, 0, 0, 0, 0])
        row[0] += 1
        row[1] += st.ok(require)
        row[2] += not st.ok(require)
        row[3] += st.doc_id in hashes
        row[4] += st.chunks
        if not st.ok(require):
            bad += 1
        if args.all or not st.ok(require):
            if st.doc_id in hashes:
                registry = "processed"
            elif st.doc_id in failed_code:
                registry = f"failed({failed_code[st.doc_id]})"
            else:
                registry = "never-ingested"
            print(f"{st.line(require)} type={doc_type} registry={registry}")
    _print_table(rows, ["total", "ok", "incomplete", "registry_ok", "chunks"])
    print(
        f"index ready: {len(stats) - bad}/{len(stats)} docs "
        f"(chunk vectors present{' + entities' if require else ''})"
    )
    selected = set(by_id)
    _print_error_summary(
        {
            code: [(i, e["error"]) for i in e["ids"] if i in selected]
            for code, e in failed_info.items()
            if any(i in selected for i in e["ids"])
        }
    )
    if bad:
        print("fix: ami-rag reindex --all --repair [--dry-run]")
    return bad == 0


async def cmd_status(_args) -> None:
    import httpx

    settings = get_settings()
    docs_repo = _build_docs_repo(settings)
    state_repo = _build_state_repo(settings)
    queue = _build_queue(settings)
    total = await asyncio.to_thread(docs_repo.count)
    state_counts = await asyncio.to_thread(state_repo.counts)
    failed_info = await asyncio.to_thread(state_repo.failed_by_code)
    pending = await queue.pending_count()
    print(f"workspace: {settings.WORKSPACE}")
    print(f"documents (active) in org_db: {total}")
    print(f"rag documents: {state_counts}")
    print(f"queue pending: {pending}")
    if failed_info:
        from ami_rag.index_check import ERROR_HINTS

        print("failed documents by cause:")
        for code, entry in sorted(failed_info.items(), key=lambda kv: -kv[1]["count"]):
            print(f"  {code}: {entry['count']} doc(s), e.g. {entry['ids']}")
            print(f"    {ERROR_HINTS.get(code, ERROR_HINTS['OTHER'])}")
    else:
        print("failed document ids: none")
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
        "--type",
        default=None,
        help="comma-separated document types (pdf,docx,text,crawl)",
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
        "--direct",
        action="store_true",
        help="ingest in-process instead of publishing events",
    )

    p_reindex.add_argument(
        "--repair",
        action="store_true",
        help="processed docs only: re-run missing chunk vectors/entities from stored chunk text",
    )

    p_verify = sub.add_parser(
        "verify", help="count chunks/vectors/entities that really exist per document"
    )
    p_verify.add_argument("--doc-ids", nargs="*", default=[], help="specific document ids")
    p_verify.add_argument("--type", default=None, help="comma-separated document types")
    p_verify.add_argument("--limit", type=int, default=None)
    p_verify.add_argument("--batch-size", type=int, default=100)
    p_verify.add_argument(
        "--all", action="store_true", help="print every document, not only incomplete"
    )

    sub.add_parser("status", help="show rag document status, queue depth and failed docs")

    p_purge = sub.add_parser(
        "purge-doc", help="delete one document from the RAG index, assets and registry"
    )
    p_purge.add_argument("--doc-id", required=True)

    args = parser.parse_args()
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpcore", "pymongo"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.command == "reindex" and not (args.all or args.doc_ids):
        parser.error("reindex requires --all or --doc-ids")
    if args.command == "reindex":
        rc = asyncio.run(cmd_reindex(args))
        if rc:
            raise SystemExit(rc)
    elif args.command == "verify":
        if not asyncio.run(cmd_verify(args)):
            raise SystemExit(1)
    elif args.command == "status":
        asyncio.run(cmd_status(args))
    elif args.command == "purge-doc":
        asyncio.run(cmd_purge(args))
