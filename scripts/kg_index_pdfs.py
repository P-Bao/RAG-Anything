"""Index PDFs already parsed by ami-multimodal-retrieval into the KG branch.

Reads their MinIO artifacts (content_list.json + descriptions.json, read-only),
reuses the stored LLM captions (0 caption API calls), assembles one text per
document and inserts it into LightRAG: chunking + Qwen3.5 entity extraction
(thinking OFF) + nemotron embeddings build the knowledge graph.

Subcommands:
  index   --doc ID [--doc ID ...] | --limit N [--dry-run]
  mongo   --doc ObjectId [--doc ...] | --limit N [--dry-run]
          - docs from the Mongo registry (status=active, read-only);
            missing MinIO artifacts are parsed (MinerU + backend_data OCR,
            describe) then indexed; existing artifacts are reused
  status                     - entity/relation/chunk counts of the local KG
  query   QUERY [QUERY ...]  - aquery (mode mix, thinking ON via role config)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raganything.caption_reuse import CaptionReuseStore, extract_descriptions, join_caption  # noqa: E402
from raganything.kg_factory import build_kg_raganything  # noqa: E402


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


def artifact_text(store: CaptionReuseStore, doc_id: str) -> tuple[str, dict, int] | None:
    """(text, item counts, reused captions) from MinIO artifacts; None when absent."""
    content_list = store.load_content_list(doc_id)
    if content_list is None:
        return None
    descriptions = store.get_descriptions(doc_id)
    text, counts = assemble_document_text(content_list, descriptions)
    return text, counts, len(descriptions)


async def cmd_index(args: argparse.Namespace) -> int:
    store = CaptionReuseStore()
    if args.doc:
        doc_ids = args.doc
    else:
        doc_ids = await list_candidate_docs(store, args.limit or 2)
    if not doc_ids:
        print("no documents with content_list artifacts found in MinIO")
        return 1

    rag = build_kg_raganything(
        lightrag_kwargs={"working_dir": _env("KG_WORKING_DIR", "./rag_storage")}
    )
    init = await rag._ensure_lightrag_initialized()
    if not init.get("success"):
        print(f"LightRAG init failed: {init.get('error')}")
        return 1

    total = 0
    started = time.monotonic()
    for doc_id in doc_ids:
        doc_started = time.monotonic()
        cached = artifact_text(store, doc_id)
        if cached is None:
            print(f"skip {doc_id}: no content_list artifact")
            continue
        text, counts, reused = cached
        if not text.strip():
            print(f"skip {doc_id}: empty text after assembly")
            continue
        if args.dry_run:
            print(
                f"[dry-run] {doc_id}: {len(text)} chars, "
                f"{reused} reused captions, items {counts}"
            )
            continue
        await rag.lightrag.ainsert(text)
        total += 1
        print(
            f"indexed {doc_id}: {reused} reused captions, items {counts}, "
            f"{time.monotonic() - doc_started:.0f}s"
        )
    if not args.dry_run:
        lg = rag.lightrag
        print(
            f"KG totals: entities={len(lg.entities_vdb)} "
            f"relations={len(lg.relationships_vdb)} chunks={len(lg.chunks_vdb)} "
            f"in {time.monotonic() - started:.0f}s"
        )
    return 0


async def cmd_mongo(args: argparse.Namespace) -> int:
    """Index docs from the Mongo registry; parse PDFs missing artifacts first.

    Read-only on Mongo (no status writes): idempotence = artifact already in
    MinIO -> reuse, else parse + caption + upload (same layout as the sibling
    pipeline) and index. Dry-run never parses.
    """
    from raganything.caption_binding import build_caption_func
    from raganything.mongo_registry import MongoDocumentRepo, select_source
    from raganything.parse_pipeline import ParseArtifactStore, parse_and_store

    repo = MongoDocumentRepo()  # raises without MONGO_URI
    try:
        if args.doc:
            docs = [
                doc
                for doc in (
                    await asyncio.to_thread(repo.find_by_id, doc_id)
                    for doc_id in args.doc
                )
                if doc
            ]
        else:
            docs = await asyncio.to_thread(repo.list_active)
            if args.limit:
                docs = docs[: args.limit]
    finally:
        repo.close()
    if not docs:
        print("no active documents found in Mongo")
        return 1

    store = CaptionReuseStore()
    parse_store = ParseArtifactStore()
    caption_func = build_caption_func()

    rag = None
    if not args.dry_run:
        rag = build_kg_raganything(
            lightrag_kwargs={"working_dir": _env("KG_WORKING_DIR", "./rag_storage")}
        )
        init = await rag._ensure_lightrag_initialized()
        if not init.get("success"):
            print(f"LightRAG init failed: {init.get('error')}")
            return 1

    total = 0
    started = time.monotonic()
    for doc in docs:
        doc_id = str(doc.get("_id") or "")
        doc_started = time.monotonic()
        source = select_source(doc)

        if source == "mongo_text":
            text = (doc.get("content") or "").strip()
            if not text:
                print(f"skip {doc_id}: mongo_text without content")
                continue
            if args.dry_run:
                print(f"[dry-run] {doc_id}: mongo_text, {len(text)} chars")
                continue
            await rag.lightrag.ainsert(text)
            total += 1
            print(
                f"indexed {doc_id}: mongo_text {len(text)} chars, "
                f"{time.monotonic() - doc_started:.0f}s"
            )
            continue

        file_path = doc.get("file_path")
        if not file_path:
            print(f"skip {doc_id}: minio_parse without file_path")
            continue

        cached = artifact_text(store, doc_id)
        if cached is None and args.dry_run:
            print(f"[dry-run] {doc_id}: PARSE needed, file_path={file_path}")
            continue
        if cached is None:
            print(f"parse {doc_id}: {file_path}")
            with tempfile.TemporaryDirectory(prefix="kg-parse-") as tmp:
                try:
                    local = await asyncio.to_thread(parse_store.fetch, file_path, tmp)
                    content_list, descriptions = await parse_and_store(
                        parse_store, doc_id, local, caption_func=caption_func
                    )
                except Exception as exc:
                    print(f"FAILED {doc_id}: parse: {exc}")
                    continue
            text, counts = assemble_document_text(
                content_list, extract_descriptions(descriptions)
            )
            reused = sum(
                1 for d in descriptions if str(d.get("description") or "").strip()
            )
        else:
            text, counts, reused = cached
        if not text.strip():
            print(f"skip {doc_id}: empty text after assembly")
            continue
        if args.dry_run:
            print(
                f"[dry-run] {doc_id}: reuse artifact, {reused} captions, "
                f"{len(text)} chars, items {counts}"
            )
            continue
        await rag.lightrag.ainsert(text)
        total += 1
        print(
            f"indexed {doc_id}: {reused} captions, items {counts}, "
            f"{time.monotonic() - doc_started:.0f}s"
        )

    if rag is not None:
        lg = rag.lightrag
        print(
            f"KG totals: entities={len(lg.entities_vdb)} "
            f"relations={len(lg.relationships_vdb)} chunks={len(lg.chunks_vdb)} "
            f"in {time.monotonic() - started:.0f}s"
        )
    return 0


async def cmd_status(_args: argparse.Namespace) -> int:
    # Counts come from storage files at init; dummy funcs are never called.
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    async def dummy_llm(prompt, **kw):
        return ""

    async def dummy_embed(texts, context="document"):
        return [[0.0] * 2048 for _ in texts]

    lg = LightRAG(
        working_dir=_env("KG_WORKING_DIR", "./rag_storage"),
        llm_model_func=dummy_llm,
        embedding_func=EmbeddingFunc(embedding_dim=2048, func=dummy_embed),
    )
    print(
        f"KG: entities={len(lg.entities_vdb)} "
        f"relations={len(lg.relationships_vdb)} chunks={len(lg.chunks_vdb)}"
    )
    return 0


async def cmd_query(args: argparse.Namespace) -> int:
    rag = build_kg_raganything(
        lightrag_kwargs={"working_dir": _env("KG_WORKING_DIR", "./rag_storage")}
    )
    init = await rag._ensure_lightrag_initialized()
    if not init.get("success"):
        print(f"LightRAG init failed: {init.get('error')}")
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
    p_index.add_argument("--limit", type=int, default=2, help="max docs when scanning")
    p_index.add_argument("--dry-run", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_mongo = sub.add_parser(
        "mongo", help="index Mongo registry docs (parse PDFs missing artifacts)"
    )
    p_mongo.add_argument("--doc", action="append", help="document ObjectId (repeatable)")
    p_mongo.add_argument("--limit", type=int, default=0, help="max docs (0 = all active)")
    p_mongo.add_argument("--dry-run", action="store_true")
    p_mongo.set_defaults(func=cmd_mongo)

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
