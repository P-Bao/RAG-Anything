"""Index documents into the KG branch (LightRAG 1.5.7 + Qwen3.5 + nemotron).

Two ingest sources (both reuse the same assemble + insert path in
``raganything.ingest``):
  index   - docs already parsed by ami-multimodal-retrieval (MinIO artifacts,
            captions reused from descriptions.json, 0 caption API calls)
  mongo   - the production upload registry; docs missing artifacts are parsed
            on the spot (MinerU + backend_data OCR + describe)

Subcommands:
  index   --doc ID [--doc ID ...] | --limit N [--dry-run]
  mongo   --doc ObjectId [--doc ...] | --limit N [--dry-run]
  reindex [--doc ID ...] [--dry-run] - wipe the KG, index only the allowlist
  status                     - entity/relation/chunk counts of the local KG
  query   QUERY [QUERY ...]  - aquery (mode mix, thinking ON via role config)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raganything.caption_reuse import CaptionReuseStore  # noqa: E402
from raganything.ingest import (  # noqa: E402
    artifact_text,
    assemble_document_text,
    format_item,
    ingest_minio,
    ingest_mongo,
    list_candidate_docs,
)
from raganything.kg_factory import build_kg_raganything  # noqa: E402

__all__ = [
    "artifact_text",
    "assemble_document_text",
    "format_item",
    "list_candidate_docs",
]

# The only documents that should exist in the KG: everything else currently
# indexed (crawled articles, stray artifacts) is reindex waste. Override the
# allowlist with repeated --doc.
REINDEX_DOC_IDS = [
    "6a8ee761694f1f325e516aa9",  # Sổ tay sinh viên 2026 (68 ảnh khuôn viên)
    "6a8f144188f0aede5ebc8828",  # Quyết định công tác cố vấn học tập 2022
    "6a8eec753b7bc85ec8357ed7",  # Thông báo xét tuyển bổ sung ThS tích hợp
    "6a8eec423b7bc85ec8357ed6",  # Quy định xét tuyển bổ sung tài năng
]

# All data-bearing storages except llm_response_cache: entity extraction cache
# stays valid (keyed by text) and saves LLM calls on the reinsert.
_WIPE_STORAGES = (
    "full_docs",
    "text_chunks",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
    "entities_vdb",
    "relationships_vdb",
    "chunks_vdb",
    "chunk_entity_relation_graph",
    "doc_status",
)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def _print_doc(event: str, doc_id: str, info: dict) -> None:
    if event == "indexed":
        detail = ", ".join(f"{k}={v}" for k, v in info.items() if v)
        print(f"indexed {doc_id}: {detail}" if detail else f"indexed {doc_id}")
    elif event == "skip":
        print(f"skip {doc_id}: {info.get('reason')}")
    elif event == "parsing":
        print(f"parse {doc_id}: {info.get('file_path')}")
    elif event == "failed":
        print(f"FAILED {doc_id}: {info.get('error')}")
    elif event == "dry-run":
        print(f"[dry-run] {doc_id}: {info}")


def _print_summary(summary: dict) -> None:
    totals = summary.get("totals") or {}
    tail = (
        f"KG totals: entities={totals.get('entities')} "
        f"relations={totals.get('relations')} chunks={totals.get('chunks')}"
        if totals
        else ""
    )
    print(
        f"{summary['source']}: total={summary['total']} indexed={summary['indexed']} "
        f"skipped={summary['skipped']} failed={summary['failed']} "
        f"in {summary['duration_s']:.0f}s {tail}".rstrip()
    )


async def _build_rag():
    rag = build_kg_raganything(
        lightrag_kwargs={"working_dir": _env("KG_WORKING_DIR", "./rag_storage")}
    )
    init = await rag._ensure_lightrag_initialized()
    if not init.get("success"):
        print(f"LightRAG init failed: {init.get('error')}")
        return None
    return rag


async def cmd_index(args: argparse.Namespace) -> int:
    store = CaptionReuseStore()
    if not args.doc:
        candidates = await list_candidate_docs(store, args.limit)
        if not candidates:
            print("no documents with content_list artifacts found in MinIO")
            return 1
    rag = await _build_rag()
    if rag is None:
        return 1
    summary = await ingest_minio(
        rag,
        store,
        args.doc,
        limit=args.limit,
        dry_run=args.dry_run,
        on_doc=_print_doc,
    )
    _print_summary(summary)
    return 0


async def cmd_reindex(args: argparse.Namespace) -> int:
    """Wipe the KG (index only, MinIO/Mongo untouched) and index the allowlist.

    The allowlist is authoritative: every previously indexed document is
    dropped with the storages. Run with the API server stopped — file-based
    KV/graph storage is last-write-wins.
    """
    from raganything.ingest import kg_totals

    store = CaptionReuseStore()
    doc_ids = args.doc or REINDEX_DOC_IDS
    rag = await _build_rag()
    if rag is None:
        return 1
    lg = rag.lightrag
    if args.dry_run:
        before = await kg_totals(lg)
        print(f"[dry-run] would wipe {len(_WIPE_STORAGES)} storages; totals: {before}")
        print(f"[dry-run] would index: {', '.join(doc_ids)}")
        for doc_id in doc_ids:
            if artifact_text(store, doc_id) is None:
                print(f"[dry-run] WARNING {doc_id}: no content_list artifact")
        return 0
    print(f"before: {await kg_totals(lg)}")
    wiped = 0
    for name in _WIPE_STORAGES:
        storage = getattr(lg, name, None)
        if storage is not None:
            await storage.drop()
            wiped += 1
    print(f"wiped {wiped}/{len(_WIPE_STORAGES)} storages")
    summary = await ingest_minio(rag, store, doc_ids, on_doc=_print_doc)
    _print_summary(summary)
    return 0


async def cmd_mongo(args: argparse.Namespace) -> int:
    from raganything.doc_state import DocStatusStore

    rag = None
    if not args.dry_run:
        rag = await _build_rag()
        if rag is None:
            return 1
    state = None if args.dry_run else DocStatusStore()
    summary = await ingest_mongo(
        rag,
        document_ids=args.doc,
        limit=args.limit,
        dry_run=args.dry_run,
        state=state,
        on_doc=_print_doc,
    )
    _print_summary(summary)
    return 0


async def cmd_status(_args: argparse.Namespace) -> int:
    # Counts come from storage files at init; dummy funcs are never called.
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    from raganything.ingest import kg_totals

    async def dummy_llm(prompt, **kw):
        return ""

    async def dummy_embed(texts, context="document"):
        return [[0.0] * 2048 for _ in texts]

    lg = LightRAG(
        working_dir=_env("KG_WORKING_DIR", "./rag_storage"),
        llm_model_func=dummy_llm,
        embedding_func=EmbeddingFunc(embedding_dim=2048, func=dummy_embed),
    )
    await lg.initialize_storages()
    try:
        totals = await kg_totals(lg)
    finally:
        await lg.finalize_storages()
    print(
        f"KG: entities={totals['entities']} "
        f"relations={totals['relations']} chunks={totals['chunks']}"
    )
    return 0


async def cmd_query(args: argparse.Namespace) -> int:
    rag = await _build_rag()
    if rag is None:
        return 1
    # Rerank binding returns the FULL ranking; chunk_top_k (RERANK_TOP_K)
    # is the only cut, applied by LightRAG after reranking.
    chunk_top_k = int(_env("RERANK_TOP_K", "5") or 5)
    for query in args.query:
        print(f"Q: {query}")
        print(await rag.aquery(query, mode=args.mode, chunk_top_k=chunk_top_k))
        print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index docs from MinIO artifacts")
    p_index.add_argument("--doc", action="append", help="doc id (repeatable)")
    p_index.add_argument(
        "--limit", type=int, default=2, help="max docs when scanning (0 = all)"
    )
    p_index.add_argument("--dry-run", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_mongo = sub.add_parser(
        "mongo", help="index Mongo registry docs (parse PDFs missing artifacts)"
    )
    p_mongo.add_argument("--doc", action="append", help="document ObjectId (repeatable)")
    p_mongo.add_argument("--limit", type=int, default=0, help="max docs (0 = all active)")
    p_mongo.add_argument("--dry-run", action="store_true")
    p_mongo.set_defaults(func=cmd_mongo)

    p_reindex = sub.add_parser(
        "reindex", help="wipe the KG and index only the pinned allowlist docs"
    )
    p_reindex.add_argument("--doc", action="append", help="override the allowlist")
    p_reindex.add_argument("--dry-run", action="store_true")
    p_reindex.set_defaults(func=cmd_reindex)

    p_status = sub.add_parser("status", help="show KG counts")
    p_status.set_defaults(func=cmd_status)

    p_query = sub.add_parser("query", help="query the KG")
    p_query.add_argument("query", nargs="+", help="question(s)")
    p_query.add_argument("--mode", default="mix", help="local|global|hybrid|naive|mix")
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
