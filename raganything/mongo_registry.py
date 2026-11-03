"""Read-only Mongo registry for production document uploads.

Port of ``ami_rag.storage.mongo_docs.MongoDocumentRepo`` (ami-multimodal-retrieval):
read-only access to the backend docs collection (``organization_db.documents``).
The KG worker never writes to this collection; document lifecycle writes belong
to the backend docs service alone. ``select_source`` decides whether the content
comes from the Mongo ``content`` field or from re-parsing the MinIO upload.
"""

from __future__ import annotations

import os

from bson import ObjectId
from pymongo import MongoClient

SOURCE_MONGO_TEXT = "mongo_text"
SOURCE_MINIO_PARSE = "minio_parse"


def select_source(doc: dict) -> str:
    """Decide where the document content comes from.

    - `original_file_name` empty or `.txt` (text, crawl, seed data): the text stored
      in Mongo `content` is the source of truth.
    - otherwise (docx/doc/pdf uploads): the original file in MinIO is re-parsed by
      the multimodal parser; Mongo text/tables are ignored.
    """
    name = (doc.get("original_file_name") or "").strip().lower()
    if not name or name.endswith(".txt"):
        return SOURCE_MONGO_TEXT
    return SOURCE_MINIO_PARSE


class MongoDocumentRepo:
    """Read-only access to the backend docs Mongo collection."""

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str | None = None,
        collection_name: str | None = None,
        client=None,
    ):
        mongo_uri = mongo_uri or os.environ.get("MONGO_URI")
        if mongo_uri is None and client is None:
            raise ValueError("MONGO_URI is not set (env MONGO_URI or argument required)")
        self._client = client or MongoClient(mongo_uri, tz_aware=True)
        db_name = db_name or os.environ.get("ORG_DB") or "organization_db"
        collection_name = (
            collection_name or os.environ.get("DOC_COLLECTION") or "documents"
        )
        self._col = self._client[db_name][collection_name]

    def find_by_id(self, document_id: str) -> dict | None:
        if not ObjectId.is_valid(document_id):
            return None
        return self._col.find_one({"_id": ObjectId(document_id)})

    def list_active(
        self, batch_size: int = 100, document_types: list[str] | None = None
    ) -> list[dict]:
        query: dict = {"status": "active"}
        if document_types:
            query["document_type"] = {"$in": document_types}
        return list(self._col.find(query, batch_size=batch_size))

    def count(self) -> int:
        return self._col.count_documents({"status": "active"})

    def close(self) -> None:
        self._client.close()
