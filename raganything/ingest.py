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


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def format_item(item: dict, description: str) -> str:
    """Port of their _format_modal_chunk templates (KG branch, text-only).

    Returns "" when an item carries no usable content: a bare ``[Hình ảnh]``
    or ``[Bảng]/Chú thích: Không có`` line is noise that would be embedded
    and entity-extracted as a junk chunk (their _build_chunks dropped those).
    """
    type_ = str(item.get("type") or "text").lower()
    if type_ == "image":
        captions = join_caption(item.get("image_caption") or item.get("img_caption"))
        footnotes = join_caption(item.get("image_footnote") or item.get("img_footnote"))
        if not (description or captions or footnotes):
            return ""
        parts = ["[Hình ảnh]"]
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
        parts = ["[Bảng]", f"Chú thích: {caption or 'Không có'}"]
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


def artifact_text(store: CaptionReuseStore, doc_id: str) -> tuple[str, dict, int] | None:
    """(text, item counts, reused captions) from MinIO artifacts; None when absent."""
    content_list = store.load_content_list(doc_id)
    if content_list is None:
        return None
    descriptions = store.get_descriptions(doc_id)
    text, counts = assemble_document_text(content_list, descriptions)
    return text, counts, len(descriptions)


async def list_candidate_docs(store: CaptionReuseStore, limit: int) -> list[str]:
    """Doc ids with a content_list artifact, most recently modified first."""
    objects = store._client.list_objects(
        store.bucket, prefix=f"{store.prefix}/", recursive=True
    )
    docs: list[tuple[object, str]] = []
    for obj in objects:
        rest = obj.object_name[len(store.prefix) + 1 :]
        if rest.endswith("content_list.json"):
            docs.append((obj.last_modified, rest.split("/")[0]))
    docs.sort(key=lambda pair: pair[0], reverse=True)
    return [doc_id for _, doc_id in docs[:limit]]


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


async def index_artifacts(
    rag: Any,
    store: CaptionReuseStore,
    doc_ids: list[str],
    *,
    dry_run: bool = False,
    on_doc: Callable[..., None] | None = None,
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
        text, counts, reused = cached
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
        await rag.lightrag.ainsert(text)
        summary["indexed"] += 1
        _notify(on_doc, "indexed", doc_id, captions=reused, items=counts)
    summary["duration_s"] = round(time.monotonic() - started, 3)
    summary["totals"] = await kg_totals(rag)
    return summary


async def ingest_minio(
    rag: Any,
    store: CaptionReuseStore | None = None,
    doc_ids: list[str] | None = None,
    *,
    limit: int = 2,
    dry_run: bool = False,
    on_doc: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Index MinIO artifacts; ``doc_ids`` None scans the artifact prefix."""
    store = store or CaptionReuseStore()
    ids = list(doc_ids) if doc_ids else await list_candidate_docs(store, limit or 2)
    if not ids:
        started = time.monotonic()
        summary = _summary("minio", dry_run, started)
        summary["totals"] = await kg_totals(rag)
        return summary
    return await index_artifacts(rag, store, ids, dry_run=dry_run, on_doc=on_doc)


async def ingest_mongo(
    rag: Any,
    *,
    document_ids: list[str] | None = None,
    limit: int = 0,
    dry_run: bool = False,
    state: Any = None,
    on_doc: Callable[..., None] | None = None,
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
            indexed = await _ingest_mongo_doc(
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
            )
        except Exception as exc:  # one bad doc must not stop the batch
            summary["failed"] += 1
            summary["errors"].append({"document_id": doc_id, "error": str(exc)})
            if state is not None and not dry_run:
                state.mark_failed(doc_id, str(exc))
            _notify(on_doc, "failed", doc_id, error=str(exc))
            continue
        if indexed:
            summary["indexed"] += 1
        else:
            summary["skipped"] += 1

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
) -> bool:
    """Returns True when the document text was inserted into LightRAG."""
    if source == "mongo_text":
        text = (doc.get("content") or "").strip()
        if not text:
            _notify(on_doc, "skip", doc_id, reason="mongo_text without content")
            return False
        if dry_run:
            _notify(on_doc, "dry-run", doc_id, chars=len(text), parse=False)
            return False
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
        await rag.lightrag.ainsert(text)
        if state is not None:
            state.mark_indexed(doc_id, counts={"text": 1}, parser="mongo_text")
        _notify(on_doc, "indexed", doc_id, source="mongo_text", chars=len(text))
        return True

    file_path = doc.get("file_path")
    if not file_path:
        _notify(on_doc, "skip", doc_id, reason="minio_parse without file_path")
        return False

    cached = artifact_text(store, doc_id)
    if cached is None and dry_run:
        _notify(on_doc, "dry-run", doc_id, parse=True, file_path=file_path)
        return False

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
        text, counts = assemble_document_text(
            content_list, extract_descriptions(descriptions)
        )
        reused = sum(
            1 for d in descriptions if str(d.get("description") or "").strip()
        )
    else:
        text, counts, reused = cached

    if not text.strip():
        _notify(on_doc, "skip", doc_id, reason="empty text after assembly")
        return False

    await rag.lightrag.ainsert(text)
    if state is not None:
        state.mark_indexed(
            doc_id,
            counts=counts,
            parser=_env("PARSE_TEXT_SOURCE", "backend_data"),
        )
    _notify(on_doc, "indexed", doc_id, source="minio_parse", captions=reused, items=counts)
    return True


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
) -> dict[str, Any]:
    """Dispatch to the minio or mongo ingest path."""
    source = (source or "mongo").strip().lower()
    if source == "minio":
        return await ingest_minio(
            rag,
            store,
            document_ids,
            limit=limit or 2,
            dry_run=dry_run,
            on_doc=on_doc,
        )
    if source == "mongo":
        return await ingest_mongo(
            rag,
            document_ids=document_ids,
            limit=limit,
            dry_run=dry_run,
            state=state,
            on_doc=on_doc,
        )
    raise ValueError(f"unknown source: {source!r} (expected 'minio' or 'mongo')")

