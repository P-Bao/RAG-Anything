"""Tests for raganything.mongo_registry (read-only Mongo docs access)."""

import pytest
from bson import ObjectId

from raganything.mongo_registry import (
    SOURCE_MINIO_PARSE,
    SOURCE_MONGO_TEXT,
    MongoDocumentRepo,
    select_source,
)


# --------------------------------------------------------------------------
# select_source
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        ({"original_file_name": "x.pdf"}, SOURCE_MINIO_PARSE),
        ({"original_file_name": "y.txt"}, SOURCE_MONGO_TEXT),
        ({"original_file_name": ""}, SOURCE_MONGO_TEXT),
        ({"original_file_name": "   "}, SOURCE_MONGO_TEXT),
        ({}, SOURCE_MONGO_TEXT),
        ({"original_file_name": None}, SOURCE_MONGO_TEXT),
        ({"original_file_name": "REPORT.PDF"}, SOURCE_MINIO_PARSE),
        ({"original_file_name": "notes.TXT"}, SOURCE_MONGO_TEXT),
    ],
)
def test_select_source(doc, expected):
    assert select_source(doc) == expected


# --------------------------------------------------------------------------
# fake pymongo client (duck-typed client[db][col])
# --------------------------------------------------------------------------
class _FakeCollection:
    def __init__(self, docs=None):
        self.docs = docs or []
        self.find_calls = []
        self.count_calls = []
        self.find_one_calls = []

    def find(self, query, batch_size=None):
        self.find_calls.append({"query": query, "batch_size": batch_size})
        return list(self.docs)

    def count_documents(self, query):
        self.count_calls.append(query)
        return len(self.docs)

    def find_one(self, query):
        self.find_one_calls.append(query)
        return next((d for d in self.docs if d.get("_id") == query["_id"]), None)


class _FakeDB:
    def __init__(self, col):
        self._col = col

    def __getitem__(self, name):
        return self._col


class _FakeClient:
    def __init__(self, col):
        self._col = col
        self.closed = False

    def __getitem__(self, name):
        return _FakeDB(self._col)

    def close(self):
        self.closed = True


def _repo(col):
    return MongoDocumentRepo(client=_FakeClient(col))


# --------------------------------------------------------------------------
# list_active / count
# --------------------------------------------------------------------------
def test_list_active_default_query():
    col = _FakeCollection([{"_id": 1}, {"_id": 2}])
    repo = _repo(col)
    docs = repo.list_active()
    assert docs == [{"_id": 1}, {"_id": 2}]
    assert col.find_calls[0]["query"] == {"status": "active"}


def test_list_active_document_types_adds_in_filter():
    col = _FakeCollection([])
    repo = _repo(col)
    repo.list_active(document_types=["pdf"])
    assert col.find_calls[0]["query"] == {
        "status": "active",
        "document_type": {"$in": ["pdf"]},
    }


def test_count_uses_same_filter():
    col = _FakeCollection([{"_id": 1}])
    repo = _repo(col)
    assert repo.count() == 1
    assert col.count_calls == [{"status": "active"}]


# --------------------------------------------------------------------------
# find_by_id
# --------------------------------------------------------------------------
def test_find_by_id_invalid_objectid_returns_none():
    col = _FakeCollection([])
    repo = _repo(col)
    assert repo.find_by_id("not-an-objectid") is None
    assert col.find_one_calls == []


def test_find_by_id_valid_calls_find_one_with_objectid():
    oid = "507f1f77bcf86cd799439011"
    col = _FakeCollection([{"_id": ObjectId(oid)}])
    repo = _repo(col)
    result = repo.find_by_id(oid)
    assert result == {"_id": ObjectId(oid)}
    assert col.find_one_calls[0] == {"_id": ObjectId(oid)}


# --------------------------------------------------------------------------
# constructor / MONGO_URI
# --------------------------------------------------------------------------
def test_missing_mongo_uri_and_no_client_raises(monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    with pytest.raises(ValueError):
        MongoDocumentRepo()


def test_injected_client_ok_without_mongo_uri(monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    col = _FakeCollection([])
    repo = _repo(col)
    assert repo.count() == 0
