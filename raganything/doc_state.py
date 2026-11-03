"""Doc ingest status for the KG server (write side).

Separate collection from the multimodal pipeline: ``kg_rag_documents`` (not
``multimodal_rag_documents``). Two pipelines run over overlapping documents in
different working dirs / embedders; sharing one status row would make each one
stomp the other's ``status``/``stage``/``attempts``. Keeping the KG write side
in its own collection lets the KG worker own its row without fighting the
multimodal retry loop.

Backends hide behind one public API: a Mongo backend (``pymongo``, ``tz_aware``)
and a JSON-file backend when no ``MONGO_URI`` and no injected client are given.
Unlike ``MongoDocumentRepo`` this never raises on a missing ``MONGO_URI``; the
file fallback is the documented behaviour.

Status vocabulary mirrors ``ami_rag.storage.doc_status`` so clients used to that
API keep working: ``pending | processing | indexed | failed``.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from datetime import datetime, timezone

from bson import ObjectId
from pymongo import MongoClient

# Statuses (mirror ami_rag.storage.doc_status)
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"

# Stage marker for a fully indexed document
STAGE_INDEXED = "indexed"

_STATUSES = (STATUS_INDEXED, STATUS_FAILED, STATUS_PENDING, STATUS_PROCESSING)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize(doc: dict | None) -> dict | None:
    """Return a copy with datetimes as ISO strings and ObjectId as str."""
    if doc is None:
        return None
    out = {}
    for key, value in doc.items():
        if isinstance(value, datetime):
            out[key] = value.isoformat()
        elif isinstance(value, ObjectId):
            out[key] = str(value)
        else:
            out[key] = value
    return out


class _MongoBackend:
    def __init__(self, client, db_name: str, collection_name: str):
        self._client = client
        self._col = client[db_name][collection_name]
        try:
            self._col.create_index("status")
        except Exception:
            pass

    @staticmethod
    def _key(doc_id):
        s = str(doc_id)
        return ObjectId(s) if ObjectId.is_valid(s) else s

    def ensure_pending(self, doc_id, fields: dict) -> None:
        now = _now()
        set_on_insert = {
            "status": STATUS_PENDING,
            "stage": STATUS_PENDING,
            "attempts": 0,
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
        set_on_insert.update({k: v for k, v in fields.items() if v is not None})
        self._col.update_one(
            {"_id": self._key(doc_id)},
            {
                "$set": {"document_id": str(doc_id)},
                "$setOnInsert": set_on_insert,
            },
            upsert=True,
        )

    def mark_stage(self, doc_id, stage: str, fields: dict) -> None:
        updates = {
            "status": STATUS_PROCESSING,
            "stage": stage,
            "updated_at": _now(),
            "document_id": str(doc_id),
        }
        updates.update({k: v for k, v in fields.items() if v is not None})
        self._col.update_one({"_id": self._key(doc_id)}, {"$set": updates}, upsert=True)

    def mark_indexed(self, doc_id, counts, page_count, parser, source_path) -> None:
        updates = {
            "status": STATUS_INDEXED,
            "stage": STAGE_INDEXED,
            "page_count": page_count,
            "updated_at": _now(),
            "document_id": str(doc_id),
        }
        if counts is not None:
            updates["counts"] = counts
        if parser:
            updates["parser"] = parser
        if source_path:
            updates["source_path"] = source_path
        self._col.update_one({"_id": self._key(doc_id)}, {"$set": updates}, upsert=True)

    def mark_failed(self, doc_id, error, stage: str) -> None:
        now = _now()
        self._col.update_one(
            {"_id": self._key(doc_id)},
            {
                "$set": {
                    "status": STATUS_FAILED,
                    "stage": stage,
                    "error": str(error)[:2000],
                    "updated_at": now,
                    "document_id": str(doc_id),
                },
                "$inc": {"attempts": 1},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    def get(self, doc_id):
        return self._col.find_one({"_id": self._key(doc_id)})

    def counts(self) -> dict:
        out = {}
        for status in _STATUSES:
            n = self._col.count_documents({"status": status})
            if n:
                out[status] = n
        return out

    def list_failed(self, limit: int = 100) -> list:
        return list(
            self._col.find(
                {"status": STATUS_FAILED},
                sort=[("updated_at", -1)],
                limit=limit,
            )
        )

    def clear_attempts(self, doc_id) -> None:
        self._col.update_one(
            {"_id": self._key(doc_id)},
            {"$set": {"attempts": 0, "status": STATUS_PENDING, "updated_at": _now()}},
        )

    def close(self) -> None:
        self._client.close()


class _FileBackend:
    def __init__(self, path: str):
        self._path = path
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:
                self._data = json.load(f)
        except FileNotFoundError:
            self._data = {}

    def _save(self) -> None:
        directory = os.path.dirname(self._path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def _ensure(self, doc_id) -> dict:
        key = str(doc_id)
        rec = self._data.get(key)
        if rec is None:
            now = _now().isoformat()
            rec = {"_id": key, "document_id": key, "attempts": 0, "created_at": now}
            self._data[key] = rec
        return rec

    def ensure_pending(self, doc_id, fields: dict) -> None:
        key = str(doc_id)
        if key in self._data:
            return
        now = _now().isoformat()
        rec = {
            "_id": key,
            "document_id": key,
            "status": STATUS_PENDING,
            "stage": STATUS_PENDING,
            "attempts": 0,
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
        rec.update({k: v for k, v in fields.items() if v is not None})
        self._data[key] = rec
        self._save()

    def mark_stage(self, doc_id, stage: str, fields: dict) -> None:
        rec = self._ensure(doc_id)
        rec["status"] = STATUS_PROCESSING
        rec["stage"] = stage
        rec["updated_at"] = _now().isoformat()
        rec.update({k: v for k, v in fields.items() if v is not None})
        self._save()

    def mark_indexed(self, doc_id, counts, page_count, parser, source_path) -> None:
        rec = self._ensure(doc_id)
        rec["status"] = STATUS_INDEXED
        rec["stage"] = STAGE_INDEXED
        rec["page_count"] = page_count
        rec["updated_at"] = _now().isoformat()
        if counts is not None:
            rec["counts"] = counts
        if parser:
            rec["parser"] = parser
        if source_path:
            rec["source_path"] = source_path
        self._save()

    def mark_failed(self, doc_id, error, stage: str) -> None:
        rec = self._ensure(doc_id)
        rec["status"] = STATUS_FAILED
        rec["stage"] = stage
        rec["error"] = str(error)[:2000]
        rec["attempts"] = int(rec.get("attempts", 0)) + 1
        rec["updated_at"] = _now().isoformat()
        self._save()

    def get(self, doc_id):
        rec = self._data.get(str(doc_id))
        return copy.deepcopy(rec) if rec is not None else None

    def counts(self) -> dict:
        out = {}
        for rec in self._data.values():
            status = rec.get("status")
            if status:
                out[status] = out.get(status, 0) + 1
        return out

    def list_failed(self, limit: int = 100) -> list:
        rows = [r for r in self._data.values() if r.get("status") == STATUS_FAILED]
        rows.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
        return [copy.deepcopy(r) for r in rows[:limit]]

    def clear_attempts(self, doc_id) -> None:
        rec = self._ensure(doc_id)
        rec["attempts"] = 0
        rec["status"] = STATUS_PENDING
        rec["updated_at"] = _now().isoformat()
        self._save()

    def close(self) -> None:
        pass


class DocStatusStore:
    """KG-server document ingest status, backed by Mongo or a JSON file."""

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str | None = None,
        collection_name: str | None = None,
        client=None,
        path: str | None = None,
    ):
        mongo_uri = mongo_uri or os.environ.get("MONGO_URI")
        self.db_name = db_name or os.environ.get("ORG_DB") or "organization_db"
        self.collection_name = (
            collection_name
            or os.environ.get("KG_STATE_COLLECTION")
            or "kg_rag_documents"
        )
        if mongo_uri or client is not None:
            self._backend = _MongoBackend(
                client or MongoClient(mongo_uri, tz_aware=True),
                self.db_name,
                self.collection_name,
            )
        else:
            if path is None:
                base = os.environ.get("KG_WORKING_DIR") or "./rag_storage"
                path = os.path.join(base, "doc_status.json")
            self._backend = _FileBackend(path)
        self.path = path

    def ensure_pending(
        self,
        doc_id,
        *,
        file_path: str = "",
        content_hash: str = "",
        source: str = "",
        document_type: str = "",
        title: str = "",
        organization_unit_id=None,
        owner_id=None,
    ) -> None:
        self._backend.ensure_pending(
            doc_id,
            {
                "file_path": file_path,
                "content_hash": content_hash,
                "source": source,
                "document_type": document_type,
                "title": title,
                "organization_unit_id": organization_unit_id,
                "owner_id": owner_id,
            },
        )

    def mark_stage(self, doc_id, stage: str, **fields) -> None:
        self._backend.mark_stage(doc_id, stage, fields)

    def mark_indexed(
        self,
        doc_id,
        *,
        counts=None,
        page_count: int = 0,
        parser: str = "",
        source_path: str = "",
    ) -> None:
        self._backend.mark_indexed(doc_id, counts, page_count, parser, source_path)

    def mark_failed(self, doc_id, error, stage: str = "ingest") -> None:
        self._backend.mark_failed(doc_id, error, stage)

    def get(self, doc_id) -> dict | None:
        return _serialize(self._backend.get(doc_id))

    def counts(self) -> dict:
        return self._backend.counts()

    def list_failed(self, limit: int = 100) -> list[dict]:
        return [_serialize(d) for d in self._backend.list_failed(limit)]

    def clear_attempts(self, doc_id) -> None:
        self._backend.clear_attempts(doc_id)

    def close(self) -> None:
        self._backend.close()
