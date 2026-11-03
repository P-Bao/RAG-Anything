"""Tests for raganything.doc_state (KG ingest status; Mongo or JSON file)."""

import json

import pytest
from bson import ObjectId

from raganything.doc_state import (
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    DocStatusStore,
)


# --------------------------------------------------------------------------
# file fallback backend
# --------------------------------------------------------------------------
def test_file_fallback_round_trip(tmp_path, monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    path = tmp_path / "doc_status.json"
    store = DocStatusStore(path=str(path))

    store.ensure_pending("d1", file_path="a.pdf", content_hash="h1")
    doc = store.get("d1")
    assert doc["status"] == STATUS_PENDING
    assert doc["stage"] == STATUS_PENDING
    assert doc["attempts"] == 0
    assert doc["document_id"] == "d1"
    assert doc["file_path"] == "a.pdf"

    store.mark_stage("d1", "parse", page_count=3, parser="mineru")
    doc = store.get("d1")
    assert doc["status"] == STATUS_PROCESSING
    assert doc["stage"] == "parse"
    assert doc["page_count"] == 3
    assert doc["parser"] == "mineru"

    store.mark_indexed("d1", counts={"chunks": 7}, page_count=3)
    doc = store.get("d1")
    assert doc["status"] == STATUS_INDEXED
    assert doc["stage"] == STATUS_INDEXED
    assert doc["counts"] == {"chunks": 7}

    # file is valid JSON holding the record
    with open(path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["d1"]["status"] == STATUS_INDEXED


def test_ensure_pending_is_insert_only(tmp_path, monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    store = DocStatusStore(path=str(tmp_path / "s.json"))
    store.ensure_pending("d1", source="first")
    store.mark_failed("d1", "boom")
    store.ensure_pending("d1", source="second")
    doc = store.get("d1")
    # existing row untouched (status still failed, attempt kept)
    assert doc["status"] == STATUS_FAILED
    assert doc["attempts"] == 1
    assert doc["source"] == "first"


# --------------------------------------------------------------------------
# failure / retry bookkeeping
# --------------------------------------------------------------------------
def test_mark_failed_increments_truncates_and_lists(tmp_path, monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    store = DocStatusStore(path=str(tmp_path / "s.json"))
    store.ensure_pending("d1")
    store.mark_failed("d1", "e" * 3000, stage="embed")
    doc = store.get("d1")
    assert doc["status"] == STATUS_FAILED
    assert doc["stage"] == "embed"
    assert doc["attempts"] == 1
    assert len(doc["error"]) == 2000

    store.mark_failed("d1", "boom again")
    assert store.get("d1")["attempts"] == 2

    failed = store.list_failed()
    assert [d["document_id"] for d in failed] == ["d1"]

    store.clear_attempts("d1")
    doc = store.get("d1")
    assert doc["attempts"] == 0
    assert doc["status"] == STATUS_PENDING


# --------------------------------------------------------------------------
# counts / unknown id
# --------------------------------------------------------------------------
def test_counts_aggregates_statuses(tmp_path, monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    store = DocStatusStore(path=str(tmp_path / "s.json"))
    store.ensure_pending("p1")
    store.ensure_pending("p2")
    store.ensure_pending("i1")
    store.mark_indexed("i1")
    store.ensure_pending("f1")
    store.mark_failed("f1", "x")
    store.ensure_pending("s1")
    store.mark_stage("s1", "parse")

    assert store.counts() == {
        STATUS_PENDING: 2,
        STATUS_INDEXED: 1,
        STATUS_FAILED: 1,
        STATUS_PROCESSING: 1,
    }


def test_get_unknown_id_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    store = DocStatusStore(path=str(tmp_path / "s.json"))
    assert store.get("nope") is None


def test_path_from_kg_working_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.setenv("KG_WORKING_DIR", str(tmp_path))
    store = DocStatusStore()
    store.ensure_pending("d1")
    assert (tmp_path / "doc_status.json").exists()


# --------------------------------------------------------------------------
# Mongo backend (duck-typed fake client)
# --------------------------------------------------------------------------
class _FakeCollection:
    def __init__(self, name="kg_rag_documents", docs=None):
        self.name = name
        self.docs = docs or []
        self.update_calls = []
        self.find_one_calls = []
        self.find_calls = []
        self.count_calls = []

    def create_index(self, *args, **kwargs):
        pass

    def update_one(self, query, update, upsert=False):
        self.update_calls.append({"query": query, "update": update, "upsert": upsert})

    def find_one(self, query):
        self.find_one_calls.append(query)
        return next((d for d in self.docs if d.get("_id") == query.get("_id")), None)

    def find(self, query, **kwargs):
        self.find_calls.append({"query": query, **kwargs})
        return list(self.docs)

    def count_documents(self, query):
        self.count_calls.append(query)
        return sum(1 for d in self.docs if d.get("status") == query.get("status"))


class _FakeDB:
    def __init__(self, col, name):
        self._col = col
        self.name = name

    def __getitem__(self, name):
        return self._col


class _FakeClient:
    def __init__(self, col):
        self._col = col
        self.closed = False

    def __getitem__(self, db_name):
        return _FakeDB(self._col, db_name)

    def close(self):
        self.closed = True


def _mongo_store(col):
    return DocStatusStore(client=_FakeClient(col))


def test_mongo_ensure_pending_upserts():
    col = _FakeCollection()
    store = _mongo_store(col)
    store.ensure_pending("d1", file_path="a.pdf")
    call = col.update_calls[-1]
    assert call["upsert"] is True
    assert call["query"] == {"_id": "d1"}
    assert call["update"]["$setOnInsert"]["status"] == STATUS_PENDING
    assert call["update"]["$set"]["document_id"] == "d1"


def test_mongo_mark_indexed_sets_status():
    col = _FakeCollection()
    store = _mongo_store(col)
    store.mark_indexed("d1", counts={"chunks": 1})
    call = col.update_calls[-1]
    assert call["update"]["$set"]["status"] == STATUS_INDEXED
    assert call["update"]["$set"]["stage"] == STATUS_INDEXED
    assert call["update"]["$set"]["counts"] == {"chunks": 1}


def test_mongo_get_queries_by_objectid():
    oid = "507f1f77bcf86cd799439011"
    col = _FakeCollection(docs=[{"_id": ObjectId(oid), "document_id": oid}])
    store = _mongo_store(col)
    doc = store.get(oid)
    assert col.find_one_calls[0] == {"_id": ObjectId(oid)}
    assert doc["_id"] == oid
    assert doc["document_id"] == oid


def test_mongo_uses_kg_state_collection_not_multimodal(monkeypatch):
    monkeypatch.delenv("KG_STATE_COLLECTION", raising=False)
    col = _FakeCollection()
    store = _mongo_store(col)
    assert store.collection_name == "kg_rag_documents"
    assert store.collection_name != "multimodal_rag_documents"


def test_collection_name_env_override(monkeypatch):
    monkeypatch.setenv("KG_STATE_COLLECTION", "custom_state")
    col = _FakeCollection()
    store = _mongo_store(col)
    assert store.collection_name == "custom_state"
