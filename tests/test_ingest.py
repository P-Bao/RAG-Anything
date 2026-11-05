"""Tests for raganything.ingest progress reporting (fakes only).

The worker shows ``progress`` to the operator, so a long insert must announce
itself before it runs — an empty list while a doc takes minutes looks hung.
"""

import asyncio

from raganything.ingest import index_artifacts


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


_PROCESSED = {"doc-k": {"status": "processed", "error_msg": None}}


class _LightRAG:
    def __init__(self, doc_row=_PROCESSED, full_docs=None):
        self.inserted: list[str] = []
        self.entities_vdb = []
        self.relationships_vdb = []
        self.chunks_vdb = []
        self.doc_status = _DocStatus()
        self.full_docs = _FullDocs(full_docs)
        if doc_row:
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
