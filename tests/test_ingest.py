"""Tests for raganything.ingest progress reporting (fakes only).

The worker shows ``progress`` to the operator, so a long insert must announce
itself before it runs — an empty list while a doc takes minutes looks hung.
"""

import asyncio

from raganything.ingest import backfill_chunk_metadata, index_artifacts


class _Store:
    """Duck-typed CaptionReuseStore (only what artifact_text uses)."""

    def __init__(self, docs: dict):
        self._docs = docs

    def load_content_list(self, doc_id):
        return self._docs.get(doc_id, (None, {}))[0]

    def get_descriptions(self, doc_id):
        return self._docs.get(doc_id, (None, {}))[1]


class _DocStatus:
    """Rows keyed by track id, like the real storage's get_docs_by_track_id."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    async def get_docs_by_track_id(self, track_id):
        return self.rows.get(track_id, {})


class _FullDocs:
    """Key-insensitive: real ids come from compute_mdhash_id(text)."""

    def __init__(self, docs=None):
        self.docs = dict(docs or {})

    async def get_by_id(self, key):
        return next(iter(self.docs.values()), None)


class _TextChunks:
    """In-memory text_chunks KV: get_by_ids/upsert like the JSON backend."""

    def __init__(self, records: dict | None = None):
        self.records = dict(records or {})
        self.upserts: list[dict] = []

    async def get_by_ids(self, ids):
        return [self.records.get(i) for i in ids]

    async def upsert(self, data):
        self.records.update(data)
        self.upserts.append(data)


_PROCESSED = {"doc-k": {"status": "processed", "error_msg": None}}


class _LightRAG:
    def __init__(self, doc_row=_PROCESSED, full_docs=None, chunks_list=None):
        self.inserted: list[str] = []
        self.entities_vdb = []
        self.relationships_vdb = []
        self.chunks_vdb = []
        self.doc_status = _DocStatus()
        self.full_docs = _FullDocs(full_docs)
        self.text_chunks = _TextChunks()
        if chunks_list:
            self.doc_status.rows["track-1"] = {
                "doc-k": {**_PROCESSED["doc-k"], "chunks_list": chunks_list}
            }
        elif doc_row:
            self.doc_status.rows["track-1"] = doc_row

    async def ainsert(self, text):
        self.inserted.append(text)
        return f"track-{len(self.inserted)}"


class _Rag:
    def __init__(self, doc_row=_PROCESSED, full_docs=None):
        self.lightrag = _LightRAG(doc_row, full_docs)


def _events():
    events: list[tuple[str, str]] = []
    return events, lambda event, doc_id, info: events.append((event, doc_id))


def test_index_announces_the_doc_before_inserting():
    rag = _Rag()
    events, on_doc = _events()

    summary = asyncio.run(
        index_artifacts(
            rag,
            _Store({"d1": ([{"type": "text", "text": "hola"}], {})}),
            ["d1"],
            on_doc=on_doc,
        )
    )

    assert [event for event, _ in events] == ["indexing", "indexed"]
    assert rag.lightrag.inserted == ["hola"]
    assert summary["indexed"] == 1
    assert summary["totals"] == {"entities": 0, "relations": 0, "chunks": 0}


def test_dry_run_lists_without_inserting():
    rag = _Rag()
    events, on_doc = _events()

    summary = asyncio.run(
        index_artifacts(
            rag,
            _Store({"d1": ([{"type": "text", "text": "hola"}], {})}),
            ["d1"],
            dry_run=True,
            on_doc=on_doc,
        )
    )

    assert [event for event, _ in events] == ["dry-run"]
    assert rag.lightrag.inserted == []
    assert summary["indexed"] == 0 and summary["skipped"] == 1


def test_missing_artifact_is_skipped_with_a_reason():
    rag = _Rag()
    events, on_doc = _events()

    summary = asyncio.run(index_artifacts(rag, _Store({}), ["missing"], on_doc=on_doc))

    assert [event for event, _ in events] == ["skip"]
    assert summary["errors"] == [
        {"document_id": "missing", "error": "no content_list artifact", "skipped": True}
    ]
    assert rag.lightrag.inserted == []


def test_failed_document_is_counted_as_failed_with_reason():
    """LightRAG marks a document failed inside doc_status without raising."""
    rag = _Rag(
        doc_row={
            "doc-k": {
                "status": "failed",
                "error_msg": "NanoVectorDBStorage[entities] flush failed",
            }
        }
    )
    events, on_doc = _events()

    summary = asyncio.run(
        index_artifacts(rag, _Store({"d1": ([{"type": "text", "text": "hola"}], {})}), ["d1"], on_doc=on_doc)
    )

    assert [event for event, _ in events] == ["indexing", "failed"]
    assert summary["indexed"] == 0 and summary["failed"] == 1
    assert summary["errors"] == [
        {"document_id": "d1", "error": "NanoVectorDBStorage[entities] flush failed"}
    ]


def test_processed_document_is_counted_as_indexed():
    rag = _Rag()
    events, on_doc = _events()
    summary = asyncio.run(
        index_artifacts(rag, _Store({"d1": ([{"type": "text", "text": "hola"}], {})}), ["d1"], on_doc=on_doc)
    )
    assert [event for event, _ in events] == ["indexing", "indexed"]
    assert summary["indexed"] == 1 and summary["failed"] == 0


def test_missing_doc_status_row_counts_as_failed():
    """No rows at all and nothing stored -> the insert silently did nothing."""
    rag = _Rag(doc_row={})
    events, on_doc = _events()
    summary = asyncio.run(
        index_artifacts(rag, _Store({"d1": ([{"type": "text", "text": "hola"}], {})}), ["d1"], on_doc=on_doc)
    )
    assert [event for event, _ in events] == ["indexing", "failed"]
    assert summary["failed"] == 1
    assert summary["errors"] == [
        {"document_id": "d1", "error": "no doc_status row after insert"}
    ]


def test_duplicate_document_is_skipped_with_reason():
    """A duplicate leaves only a dup-* record; the doc- row stays stale from
    an older run, so the outcome must come from this run's track rows."""
    rag = _Rag(
        doc_row={
            "dup-k": {
                "status": "failed",
                "error_msg": (
                    "File name already exists. Original doc_id: doc-x, "
                    "Status: processing"
                ),
            }
        }
    )
    events, on_doc = _events()

    summary = asyncio.run(
        index_artifacts(rag, _Store({"d1": ([{"type": "text", "text": "hola"}], {})}), ["d1"], on_doc=on_doc)
    )

    assert [event for event, _ in events] == ["indexing", "skip"]
    assert summary["indexed"] == 0 and summary["failed"] == 0 and summary["skipped"] == 1
    assert summary["errors"] == [
        {
            "document_id": "d1",
            "error": (
                "File name already exists. Original doc_id: doc-x, "
                "Status: processing"
            ),
            "skipped": True,
        }
    ]


def test_stored_content_without_doc_status_row_is_skipped():
    """Content already in full_docs (status file wiped, race re-enqueue):
    not a failure."""
    rag = _Rag(doc_row={}, full_docs={"doc-k": {"content": "hola"}})
    events, on_doc = _events()

    summary = asyncio.run(
        index_artifacts(rag, _Store({"d1": ([{"type": "text", "text": "hola"}], {})}), ["d1"], on_doc=on_doc)
    )

    assert [event for event, _ in events] == ["indexing", "skip"]
    assert summary["skipped"] == 1 and summary["failed"] == 0
    assert summary["errors"] == [
        {"document_id": "d1", "error": "already stored (no doc_status row)", "skipped": True}
    ]


def test_index_writes_asset_marker_into_chunk_text():
    """The ``Ảnh: {asset_key}`` marker must survive into the inserted text."""
    rag = _Rag()
    content_list = [
        {"type": "text", "text": "hola"},
        {
            "type": "image",
            "asset_key": "rag-assets/d1/abc123.png",
            "page_idx": 3,
            "image_caption": "cap",
        },
    ]
    asyncio.run(
        index_artifacts(rag, _Store({"d1": (content_list, {1: "photo of gate"})}), ["d1"])
    )

    text = rag.lightrag.inserted[0]
    assert "[Hình ảnh]\nẢnh: rag-assets/d1/abc123.png\nphoto of gate" in text


def test_backfill_merges_asset_metadata_into_chunk_records():
    """Marker in chunk content -> flat metadata keys merged into the record."""
    lg = _LightRAG(chunks_list=["chunk-1", "chunk-2"])
    lg.text_chunks.records = {
        "chunk-1": {
            "content": "para\n\n[Hình ảnh]\nẢnh: rag-assets/d1/abc123.png\ndesc",
            "full_doc_id": "doc-k",
            "tokens": 30,
        },
        "chunk-2": {"content": "no image here", "full_doc_id": "doc-k", "tokens": 5},
    }
    asyncio.run(
        backfill_chunk_metadata(
            lg,
            "track-1",
            "d1",
            [
                {
                    "modality": "image",
                    "asset_key": "rag-assets/d1/abc123.png",
                    "page": 3,
                    "caption": "photo of gate",
                    "marker": "Ảnh: rag-assets/d1/abc123.png",
                }
            ],
        )
    )

    upserted = lg.text_chunks.upserts[0]
    assert set(upserted) == {"chunk-1"}
    merged = upserted["chunk-1"]
    assert merged["modality"] == "image"
    assert merged["asset_key"] == "rag-assets/d1/abc123.png"
    assert merged["page"] == 3
    assert merged["caption"] == "photo of gate"
    assert merged["full_doc_id"] == "doc-k"  # original fields preserved
    assert "marker" not in merged
    assert lg.text_chunks.records["chunk-2"] == {
        "content": "no image here",
        "full_doc_id": "doc-k",
        "tokens": 5,
    }


def test_backfill_survives_storage_errors():
    """A broken KV must not fail an otherwise indexed document."""
    lg = _LightRAG(chunks_list=["chunk-1"])

    async def boom(ids):
        raise RuntimeError("kv down")

    lg.text_chunks.get_by_ids = boom
    asyncio.run(
        backfill_chunk_metadata(
            lg, "track-1", "d1", [{"modality": "image", "asset_key": "k", "marker": "Ảnh: k"}]
        )
    )
    assert lg.text_chunks.upserts == []


def test_backfill_without_chunks_list_is_a_noop():
    lg = _LightRAG()  # doc row without chunks_list
    asyncio.run(
        backfill_chunk_metadata(
            lg, "track-1", "d1", [{"modality": "image", "asset_key": "k", "marker": "Ảnh: k"}]
        )
    )
    assert lg.text_chunks.upserts == []
