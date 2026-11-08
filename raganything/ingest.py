"""Batch ingest for the KG branch (shared by the CLI script and the HTTP server).

Two sources:
- ``minio``: documents already parsed by ami-multimodal-retrieval (artifacts
  under ``{ASSET_PREFIX}/{doc_id}/``), captions reused from ``descriptions.json``.
- ``mongo``: the production upload registry (``status=active``); documents
  missing artifacts are parsed on the spot (MinerU + backend_data OCR +
  describe) and then indexed.

Both share one assemble + ``ainsert`` path. Every document reports progress
through an ``on_doc(event, doc_id, info)`` callback so the CLI can print and
the server can stream job progress. Indexing must stay single-threaded per
storage directory (file-based KV/vector/graph writes are last-write-wins).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from typing import Any, Callable

from raganything.caption_reuse import (
    CaptionReuseStore,
    extract_descriptions,
    join_caption,
)
from raganything.parse_pipeline import ParseArtifactStore, parse_and_store

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def format_item(item: dict, description: str) -> str:
    """Port of their _format_modal_chunk templates (KG branch, text-only).

    Returns "" when an item carries no usable content: a bare ``[Hình ảnh]``
    or ``[Bảng]/Chú thích: Không có`` line is noise that would be embedded
    and entity-extracted as a junk chunk (their _build_chunks dropped those).
    """
    type_ = str(item.get("type") or "text").lower()
    asset_key = str(item.get("asset_key") or "").strip()
    if type_ == "image":
        captions = join_caption(item.get("image_caption") or item.get("img_caption"))
        footnotes = join_caption(item.get("image_footnote") or item.get("img_footnote"))
        if not (description or captions or footnotes):
            return ""
        parts = ["[Hình ảnh]"]
        if asset_key:
            # asset reference stays in the text so backfill_chunk_metadata can
            # map the description to its MinIO asset after LightRAG chunking
            parts.append(f"Ảnh: {asset_key}")
        if description:
            parts.append(description)
        if captions:
            parts.append(f"Tóm tắt: {captions}")
        if footnotes:
            parts.append(footnotes)
        return "\n".join(parts).strip()
    if type_ == "table":
        caption = join_caption(item.get("table_caption"))
        body = join_caption(item.get("table_body") or item.get("text") or "")
        if not (body or caption or description):
            return ""
        parts = ["[Bảng]"]
        if asset_key:
            parts.append(f"Ảnh: {asset_key}")
        parts.append(f"Chú thích: {caption or 'Không có'}")
        if description.strip():
            parts.append(f"Mô tả: {description}")
        parts.append(body)
        return "\n".join(parts).strip()
    if type_ == "equation":
        text = join_caption(item.get("text") or item.get("latex") or "")
        head = "[Công thức]"
        if description.strip():
            head += f"\nMô tả: {description}"
        return f"{head}\nCông thức: {text}".strip()
    if type_ == "generic":
        text = join_caption(item.get("text") or "")
        return (f"{description}\n{text}".strip() if description else text).strip()
    # text and other types: raw content only
    return join_caption(item.get("text") or "").strip()


def assemble_document_text(
    content_list: list[dict], descriptions: dict[int, str]
) -> tuple[str, dict[str, int]]:
    """One text per document, item order preserved; returns (text, counts)."""
    parts: list[str] = []
    counts: dict[str, int] = {}
    for index, item in enumerate(content_list):
        type_ = str(item.get("type") or "text").lower()
        counts[type_] = counts.get(type_, 0) + 1
        text = format_item(item, descriptions.get(index, ""))
        if text:
            parts.append(text)
    return "\n\n".join(parts), counts


def item_metadatas(content_list: list[dict], descriptions: dict[int, str]) -> list[dict]:
    """Chunk-record metadata per asset item (image/table), document order.

    ``marker`` matches the ``Ảnh: {asset_key}`` line ``format_item`` writes, so
    backfill_chunk_metadata can locate the chunk holding each asset. Items
    without an ``asset_key`` (text-only tables) carry nothing to point at and
    are skipped.
    """
    metas: list[dict] = []
    for index, item in enumerate(content_list):
        type_ = str(item.get("type") or "text").lower()
        if type_ not in ("image", "table"):
            continue
        asset_key = str(item.get("asset_key") or "").strip()
        if not asset_key:
            continue
        meta: dict[str, Any] = {
            "modality": type_,
            "asset_key": asset_key,
            "marker": f"Ảnh: {asset_key}",
        }
        page = item.get("page_idx")
        if isinstance(page, int):
            meta["page"] = page
        caption = (
            descriptions.get(index, "").strip()
            or join_caption(item.get("image_caption") or item.get("img_caption")).strip()
            or join_caption(item.get("table_caption")).strip()
        )
        if caption:
            meta["caption"] = caption
        metas.append(meta)
    return metas


async def backfill_chunk_metadata(
    lg: Any, track_id: Any, doc_id: str, metas: list[dict]
) -> None:
    """Merge asset metadata into the inserted doc's text_chunks records.

    LightRAG hardcodes chunk records to five fields (``_build_inserting_chunks``
    in lightrag 1.5.7), so the metadata rides in the text (``Ảnh: {asset_key}``
    markers) and is lifted back into the KV here. Merge-only: chunks shared
    between documents (identical text -> identical chunk hash) keep the other
    document's keys, and a chunk holding several assets keeps the first one in
    document order (the rest stay discoverable via the markers in the text).
    Never raises: losing metadata must not fail an otherwise indexed document.
    """
    if not metas:
        return
    try:
        rows = await lg.doc_status.get_docs_by_track_id(str(track_id))
        doc_rows = [v for k, v in (rows or {}).items() if str(k).startswith("doc-")]
        if not doc_rows:
            return
        chunk_ids = _row_field(doc_rows[0], "chunks_list") or []
        if not chunk_ids:
            return
        records = await lg.text_chunks.get_by_ids(list(chunk_ids))
        updates: dict[str, dict] = {}
        for chunk_id, record in zip(chunk_ids, records or []):
            if not isinstance(record, dict):
                continue
            content = record.get("content") or ""
            for meta in metas:
                if meta["marker"] in content and chunk_id not in updates:
                    updates[chunk_id] = {
                        **record,
                        **{k: v for k, v in meta.items() if k != "marker"},
                    }
        if updates:
            await lg.text_chunks.upsert(updates)
    except Exception as exc:
        logger.warning("backfill metadata %s lỗi (bỏ qua): %s", doc_id, exc)


def artifact_text(
    store: CaptionReuseStore, doc_id: str
) -> tuple[str, dict, int, list[dict]] | None:
    """(text, item counts, reused captions, asset metas); None when absent."""
    content_list = store.load_content_list(doc_id)
    if content_list is None:
        return None
    descriptions = store.get_descriptions(doc_id)
    text, counts = assemble_document_text(content_list, descriptions)
    return (
        text,
        counts,
        len(descriptions),
        item_metadatas(content_list, descriptions),
    )


async def list_candidate_docs(store: CaptionReuseStore, limit: int = 0) -> list[str]:
    """Doc ids with a content_list artifact, most recently modified first.

    ``limit`` 0 or None means every doc (same convention as the mongo path).
    """
    objects = store._client.list_objects(
        store.bucket, prefix=f"{store.prefix}/", recursive=True
    )
    docs: list[tuple[object, str]] = []
    for obj in objects:
        rest = obj.object_name[len(store.prefix) + 1 :]
        if rest.endswith("content_list.json"):
            docs.append((obj.last_modified, rest.split("/")[0]))
    docs.sort(key=lambda pair: pair[0], reverse=True)
    selected = docs[:limit] if limit else docs
    return [doc_id for _, doc_id in selected]


def _summary(source: str, dry_run: bool, started: float) -> dict[str, Any]:
    return {
        "source": source,
        "dry_run": dry_run,
        "total": 0,
        "indexed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
        "duration_s": 0.0,
        "started_at": started,
    }


async def _resolve_attr(owner: Any, name: str) -> Any:
    """Read an attribute that may be a value, a method, or an async property."""
    raw = getattr(owner, name, None)
    if raw is None:
        return None
    if asyncio.iscoroutine(raw):  # async property (e.g. client_storage)
        return await raw
    if callable(raw):
        result = raw()
        if asyncio.iscoroutine(result):
            return await result
        return result
    return raw


async def _vdb_size(vdb: Any) -> int | None:
    """Row count of a LightRAG vector store.

    The storage wrappers have no ``__len__``. ``_get_client()`` yields the
    NanoVectorDB instance whose ``__len__`` is the row count; the
    ``client_storage`` async property is a fallback and returns the raw
    storage dict (rows under ``data``), not the client.
    """
    try:
        return len(vdb)
    except TypeError:
        pass
    for attr in ("_get_client", "client_storage"):
        try:
            inner = await _resolve_attr(vdb, attr)
        except Exception:
            continue
        if inner is None:
            continue
        try:
            return len(inner)
        except TypeError:
            rows = inner.get("data") if hasattr(inner, "get") else None
            if rows is not None:
                return len(rows)
    return None


async def kg_totals(rag: Any) -> dict[str, int | None] | None:
    """Entity/relation/chunk counts of the local KG, or None without a rag."""
    if rag is None:
        return None
    lg = getattr(rag, "lightrag", rag)  # accepts a RAGAnything or a LightRAG
    return {
        "entities": await _vdb_size(lg.entities_vdb),
        "relations": await _vdb_size(lg.relationships_vdb),
        "chunks": await _vdb_size(lg.chunks_vdb),
    }


def _notify(on_doc: Callable[..., None] | None, event: str, doc_id: str, **info) -> None:
    if on_doc is None:
        return
    try:
        on_doc(event, doc_id, info)
    except Exception:  # progress reporting must never break ingest
        pass


def _row_field(row: Any, name: str) -> Any:
    """Read a doc_status row field from a dict (JSON backend) or a
    DocProcessingStatus dataclass."""
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _citation_path(doc: dict, fallback: str) -> str:
    """Best available citation name for a Mongo document."""
    for key in ("original_file_name", "file_path", "title"):
        value = str(doc.get(key) or "").strip()
        if value:
            return value
    return fallback


async def _insert_outcome(lg: Any, track_id: Any, text: str) -> tuple[str, str | None]:
    """Classify one ainsert attempt by the doc_status rows of its track.

    ainsert returns normally even when LightRAG failed the document or
    skipped it as a duplicate: a duplicate leaves only a ``dup-*`` record
    (the ``doc-`` row keeps its stale status from an older run). Rows are
    read back via ``get_docs_by_track_id`` so only this attempt's rows
    count. Returns ``("processed"|"failed"|"skipped", reason)``.
    """
    from lightrag.utils import compute_mdhash_id

    rows = await lg.doc_status.get_docs_by_track_id(str(track_id))
    doc_rows = [v for k, v in (rows or {}).items() if str(k).startswith("doc-")]
    dup_rows = [v for k, v in (rows or {}).items() if str(k).startswith("dup-")]
    if doc_rows:
        row = doc_rows[0]
        status = _row_field(row, "status")
        status = str(getattr(status, "value", status) or "")
        if status == "processed":
            return "processed", None
        reason = str(_row_field(row, "error_msg") or "")
        return "failed", (reason or f"lightrag doc_status={status}")
    if dup_rows:
        reason = str(_row_field(dup_rows[0], "error_msg") or "")
        return "skipped", (reason or "duplicate of an already stored document")
    # No rows for this track: content stored earlier (status file wiped,
    # race re-enqueue) or the insert silently did nothing.
    stored = await lg.full_docs.get_by_id(compute_mdhash_id(text, prefix="doc-"))
    if stored:
        return "skipped", "already stored (no doc_status row)"
    return "failed", "no doc_status row after insert"


async def index_artifacts(
    rag: Any,
    store: CaptionReuseStore,
    doc_ids: list[str],
    *,
    dry_run: bool = False,
    on_doc: Callable[..., None] | None = None,
    file_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Index docs that already have MinIO artifacts (reuse captions, no LLM)."""
    started = time.monotonic()
    summary = _summary("minio", dry_run, started)
    summary["total"] = len(doc_ids)
    for doc_id in doc_ids:
        cached = artifact_text(store, doc_id)
        if cached is None:
            summary["skipped"] += 1
            summary["errors"].append(
                {"document_id": doc_id, "error": "no content_list artifact", "skipped": True}
            )
            _notify(on_doc, "skip", doc_id, reason="no content_list artifact")
            continue
        text, counts, reused, metas = cached
        if not text.strip():
            summary["skipped"] += 1
            summary["errors"].append(
                {"document_id": doc_id, "error": "empty text after assembly", "skipped": True}
            )
            _notify(on_doc, "skip", doc_id, reason="empty text after assembly")
            continue
        if dry_run:
            summary["skipped"] += 1
            _notify(
                on_doc, "dry-run", doc_id, chars=len(text), items=counts, captions=reused
            )
            continue
        # Progress before the insert, not only after: a doc takes minutes and
        # an empty progress list looks like a hung job.
        _notify(on_doc, "indexing", doc_id, chars=len(text), items=counts)
        source_path = (file_paths or {}).get(doc_id) or doc_id
        track_id = await rag.lightrag.ainsert(text, file_paths=source_path)
        outcome, reason = await _insert_outcome(rag.lightrag, track_id, text)
        if outcome == "processed":
            await backfill_chunk_metadata(rag.lightrag, track_id, doc_id, metas)
            summary["indexed"] += 1
            _notify(on_doc, "indexed", doc_id, captions=reused, items=counts)
            continue
        if outcome == "skipped":
            # LightRAG skipped the insert (duplicate of stored content):
            # not an error, but the reason still belongs in the summary.
            summary["skipped"] += 1
            summary["errors"].append(
                {"document_id": doc_id, "error": reason, "skipped": True}
            )
            _notify(on_doc, "skip", doc_id, reason=reason)
            continue
        summary["failed"] += 1
        summary["errors"].append({"document_id": doc_id, "error": reason})
        _notify(on_doc, "failed", doc_id, error=reason)
    summary["duration_s"] = round(time.monotonic() - started, 3)
    summary["totals"] = await kg_totals(rag)
    return summary


async def ingest_minio(
    rag: Any,
    store: CaptionReuseStore | None = None,
    doc_ids: list[str] | None = None,
    *,
    limit: int = 0,
    dry_run: bool = False,
    on_doc: Callable[..., None] | None = None,
    file_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Index MinIO artifacts; ``doc_ids`` None scans the artifact prefix.

    ``limit`` 0 means every doc with artifacts; ignored when ``doc_ids`` given.
    """
    store = store or CaptionReuseStore()
    ids = list(doc_ids) if doc_ids else await list_candidate_docs(store, limit)
    if not ids:
        started = time.monotonic()
        summary = _summary("minio", dry_run, started)
        summary["totals"] = await kg_totals(rag)
        return summary
    return await index_artifacts(
        rag,
        store,
        ids,
        dry_run=dry_run,
        on_doc=on_doc,
        file_paths=file_paths,
    )


async def ingest_mongo(
    rag: Any,
    *,
    document_ids: list[str] | None = None,
    limit: int = 0,
    dry_run: bool = False,
    state: Any = None,
    on_doc: Callable[..., None] | None = None,
    file_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Index the Mongo registry; parse PDFs that have no artifacts yet.

    ``state`` (optional DocStatusStore) records per-document status. Dry-run
    never parses and never writes to MinIO.
    """
    from raganything.caption_binding import build_caption_func
    from raganything.mongo_registry import MongoDocumentRepo, select_source

    started = time.monotonic()
    summary = _summary("mongo", dry_run, started)

    repo = MongoDocumentRepo()
    try:
        if document_ids:
            docs = [
                doc
                for doc in (
                    await asyncio.to_thread(repo.find_by_id, doc_id)
                    for doc_id in document_ids
                )
                if doc
            ]
        else:
            docs = await asyncio.to_thread(repo.list_active)
            if limit:
                docs = docs[:limit]
    finally:
        repo.close()

    summary["total"] = len(docs)
    if not docs:
        summary["duration_s"] = round(time.monotonic() - started, 3)
        summary["totals"] = await kg_totals(rag)
        return summary

    store = CaptionReuseStore()
    parse_store = ParseArtifactStore()
    caption_func = build_caption_func()

    for doc in docs:
        doc_id = str(doc.get("_id") or "")
        source = select_source(doc)
        try:
            outcome = await _ingest_mongo_doc(
                rag,
                store,
                parse_store,
                caption_func,
                doc,
                doc_id,
                source,
                dry_run=dry_run,
                state=state,
                on_doc=on_doc,
                file_paths=file_paths,
            )
        except Exception as exc:  # one bad doc must not stop the batch
            summary["failed"] += 1
            summary["errors"].append({"document_id": doc_id, "error": str(exc)})
            if state is not None and not dry_run:
                state.mark_failed(doc_id, str(exc))
            _notify(on_doc, "failed", doc_id, error=str(exc))
            continue
        summary[outcome] += 1
        if outcome == "failed":
            summary["errors"].append(
                {"document_id": doc_id, "error": "lightrag doc_status failed"}
            )

    summary["duration_s"] = round(time.monotonic() - started, 3)
    summary["totals"] = await kg_totals(rag)
    return summary


async def _ingest_mongo_doc(
    rag: Any,
    store: CaptionReuseStore,
    parse_store: ParseArtifactStore,
    caption_func: Any,
    doc: dict,
    doc_id: str,
    source: str,
    *,
    dry_run: bool,
    state: Any,
    on_doc: Callable[..., None] | None,
    file_paths: dict[str, str] | None = None,
) -> str:
    """Returns "indexed" | "skipped" | "failed"."""
    if source == "mongo_text":
        text = (doc.get("content") or "").strip()
        if not text:
            _notify(on_doc, "skip", doc_id, reason="mongo_text without content")
            return "skipped"
        if dry_run:
            _notify(on_doc, "dry-run", doc_id, chars=len(text), parse=False)
            return "skipped"
        if state is not None:
            state.ensure_pending(
                doc_id,
                content_hash="",
                source="mongo_text",
                document_type=doc.get("document_type") or "",
                title=doc.get("title") or "",
                organization_unit_id=doc.get("organization_unit_id"),
                owner_id=doc.get("owner_id"),
            )
        _notify(on_doc, "indexing", doc_id, source=source, chars=len(text), items={"text": 1})
        citation = (file_paths or {}).get(doc_id) or _citation_path(doc, doc_id)
        track_id = await rag.lightrag.ainsert(text, file_paths=citation)
        outcome, reason = await _insert_outcome(rag.lightrag, track_id, text)
        if outcome == "failed":
            _notify(on_doc, "failed", doc_id, error=reason)
            if state is not None:
                state.mark_failed(doc_id, reason)
            return "failed"
        if outcome == "skipped":
            _notify(on_doc, "skip", doc_id, reason=reason)
            return "skipped"
        if state is not None:
            state.mark_indexed(doc_id, counts={"text": 1}, parser="mongo_text")
        _notify(on_doc, "indexed", doc_id, source="mongo_text", chars=len(text))
        return "indexed"

    file_path = doc.get("file_path")
    if not file_path:
        _notify(on_doc, "skip", doc_id, reason="minio_parse without file_path")
        return "skipped"

    cached = artifact_text(store, doc_id)
    if cached is None and dry_run:
        _notify(on_doc, "dry-run", doc_id, parse=True, file_path=file_path)
        return "skipped"

    if state is not None and not dry_run:
        state.ensure_pending(
            doc_id,
            file_path=file_path,
            source="minio_parse",
            document_type=doc.get("document_type") or "",
            title=doc.get("title") or "",
            organization_unit_id=doc.get("organization_unit_id"),
            owner_id=doc.get("owner_id"),
        )

    if cached is None:
        _notify(on_doc, "parsing", doc_id, file_path=file_path)
        if state is not None:
            state.mark_stage(doc_id, "parse")
        with tempfile.TemporaryDirectory(prefix="kg-parse-") as tmp:
            local = await asyncio.to_thread(parse_store.fetch, file_path, tmp)
            content_list, descriptions = await parse_and_store(
                parse_store, doc_id, local, caption_func=caption_func
            )
        if state is not None:
            state.mark_stage(doc_id, "describe")
        descriptions_map = extract_descriptions(descriptions)
        text, counts = assemble_document_text(content_list, descriptions_map)
        metas = item_metadatas(content_list, descriptions_map)
        reused = sum(
            1 for d in descriptions if str(d.get("description") or "").strip()
        )
    else:
        text, counts, reused, metas = cached

    if not text.strip():
        _notify(on_doc, "skip", doc_id, reason="empty text after assembly")
        return "skipped"

    _notify(on_doc, "indexing", doc_id, source=source, chars=len(text), items=counts)
    citation = (file_paths or {}).get(doc_id) or _citation_path(doc, doc_id)
    track_id = await rag.lightrag.ainsert(text, file_paths=citation)
    outcome, reason = await _insert_outcome(rag.lightrag, track_id, text)
    if outcome == "failed":
        _notify(on_doc, "failed", doc_id, error=reason)
        if state is not None:
            state.mark_failed(doc_id, reason)
        return "failed"
    if outcome == "skipped":
        _notify(on_doc, "skip", doc_id, reason=reason)
        return "skipped"
    await backfill_chunk_metadata(rag.lightrag, track_id, doc_id, metas)
    if state is not None:
        state.mark_indexed(
            doc_id,
            counts=counts,
            parser=_env("PARSE_TEXT_SOURCE", "backend_data"),
        )
    _notify(on_doc, "indexed", doc_id, source="minio_parse", captions=reused, items=counts)
    return "indexed"


async def ingest(
    rag: Any,
    *,
    source: str = "mongo",
    document_ids: list[str] | None = None,
    limit: int = 0,
    dry_run: bool = False,
    state: Any = None,
    on_doc: Callable[..., None] | None = None,
    store: CaptionReuseStore | None = None,
    file_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Dispatch to the minio or mongo ingest path."""
    source = (source or "mongo").strip().lower()
    if source == "minio":
        return await ingest_minio(
            rag,
            store,
            document_ids,
            limit=limit,
            dry_run=dry_run,
            on_doc=on_doc,
            file_paths=file_paths,
        )
    if source == "mongo":
        return await ingest_mongo(
            rag,
            document_ids=document_ids,
            limit=limit,
            dry_run=dry_run,
            state=state,
            on_doc=on_doc,
            file_paths=file_paths,
        )
    raise ValueError(f"unknown source: {source!r} (expected 'minio' or 'mongo')")

