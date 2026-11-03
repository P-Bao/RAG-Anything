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


class _LightRAG:
    def __init__(self):
        self.inserted: list[str] = []
        self.entities_vdb = []
        self.relationships_vdb = []
        self.chunks_vdb = []

    async def ainsert(self, text):
        self.inserted.append(text)


class _Rag:
    def __init__(self):
        self.lightrag = _LightRAG()


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
