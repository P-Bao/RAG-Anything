import logging
from datetime import datetime, timezone

from pymongo import MongoClient, ReturnDocument

logger = logging.getLogger(__name__)

STATUS_PROCESSING = "processing"
STATUS_PROCESSED = "processed"
STATUS_FAILED = "failed"
STATUS_STALE = "stale"


class RagDocumentsRepo:
    """Worker-owned registry of ingested documents (organization_db.multimodal_rag_documents).

    One row per document (`_id` = Mongo ObjectId string): source fingerprint, status,
    attempts, MinIO assets and per-modality counts. Unchanged re-ingests are skipped
    via `source_hash`; failed docs stop consuming the delivery budget after
    WORKER_MAX_DELIVERY attempts. Lives in the shared `organization_db` under the `multimodal_` prefix; the backend collections are only read.
    """

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str = "multimodal_rag_documents",
    ):
        self._client: MongoClient = MongoClient(mongo_uri, tz_aware=True)
        self._col = self._client[db_name][collection_name]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Indexes for admin/status queries and `$lookup` joins; best effort (idempotent)."""
        try:
            self._col.create_index("status")
            self._col.create_index("organization_unit_id")
            self._col.create_index("document_oid")
        except Exception as exc:
            logger.warning("could not ensure %s indexes: %s", self._col.name, exc)

    def get(self, document_id: str) -> dict | None:
        return self._col.find_one({"_id": document_id})

    def begin_attempt(self, document_id: str, source_hash: str | None = None) -> int:
        now = datetime.now(timezone.utc)
        doc = self._col.find_one_and_update(
            {"_id": document_id},
            {
                "$set": {"updated_at": now, "last_hash": source_hash or ""},
                "$inc": {"attempts": 1},
                "$setOnInsert": {
                    "created_at": now,
                    "status": STATUS_PROCESSING,
                    "error": "",
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int((doc or {}).get("attempts", 1))

    def release_attempt(self, document_id: str) -> None:
        """Undo begin_attempt for an event that ingested nothing (skipped/missing/empty).

        Keeps `attempts` meaningful for the failure budget and drops the orphan
        `processing` row a never-ingested document would otherwise leave behind.
        """
        doc = self._col.find_one_and_update(
            {"_id": document_id},
            {"$inc": {"attempts": -1}},
            return_document=ReturnDocument.AFTER,
        )
        if doc and doc.get("status") == STATUS_PROCESSING and not doc.get("source_hash"):
            self._col.delete_one({"_id": document_id, "status": STATUS_PROCESSING})

    def mark_processed(
        self,
        document_id: str,
        source_hash: str | None = None,
        *,
        source: str = "",
        file_path: str = "",
        parser: str = "",
        assets: list[str] | None = None,
        counts: dict | None = None,
        page_count: int = 0,
        meta: dict | None = None,
    ) -> None:
        self._col.update_one(
            {"_id": document_id},
            {
                "$set": {
                    **(meta or {}),
                    "status": STATUS_PROCESSED,
                    "source_hash": source_hash or "",
                    "source": source,
                    "file_path": file_path,
                    "parser": parser,
                    "assets": assets or [],
                    "counts": counts or {},
                    "page_count": page_count,
                    "error": "",
                    "error_code": "",
                    "error_stage": "",
                    "attempts": 0,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    def mark_failed(
        self,
        document_id: str,
        error: str,
        source_hash: str | None = None,
        *,
        error_code: str = "OTHER",
        error_stage: str = "",
        retryable: bool = True,
    ) -> None:
        self._col.update_one(
            {"_id": document_id},
            {
                "$set": {
                    "status": STATUS_FAILED,
                    "error": error[:2000],
                    "error_code": error_code,
                    "error_stage": error_stage,
                    "error_retryable": retryable,
                    # fresh delivery budget for a later reprocess/reindex
                    "attempts": 0,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    def mark_repaired(self, document_id: str, index_stats: dict) -> None:
        """Index completed in place from stored chunks: failed/processed row -> processed."""
        self._col.update_one(
            {"_id": document_id},
            {
                "$set": {
                    "status": STATUS_PROCESSED,
                    "index_stats": index_stats,
                    "error": "",
                    "error_code": "",
                    "error_stage": "",
                    "error_retryable": False,
                    "attempts": 0,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    def get_repairable_hashes(self) -> dict[str, str]:
        """{document_id: source_hash} of processed AND failed rows (both can be repaired
        in place from the chunk text already stored in Mongo)."""
        return {
            doc["_id"]: doc.get("source_hash") or ""
            for doc in self._col.find(
                {"status": {"$in": [STATUS_PROCESSED, STATUS_FAILED]}},
                {"source_hash": 1},
            )
        }

    def failed_by_code(self, sample: int = 3) -> dict[str, dict]:
        """{error_code: {"count": n, "ids": [first ids], "error": sample message}}."""
        from ami_rag.index_check import classify_error

        out: dict[str, dict] = {}
        for doc in self._col.find(
            {"status": STATUS_FAILED}, {"error": 1, "error_code": 1, "error_stage": 1}
        ):
            code = doc.get("error_code") or classify_error(doc.get("error", ""))
            entry = out.setdefault(code, {"count": 0, "ids": [], "error": doc.get("error", "")})
            entry["count"] += 1
            if len(entry["ids"]) < sample:
                entry["ids"].append(doc["_id"])
        return out

    def mark_stale(self, document_ids: list[str] | None = None) -> int:
        """Force re-ingest: processed rows become `stale` so the hash check no longer skips."""
        query: dict = {"status": STATUS_PROCESSED}
        if document_ids is not None:
            query["_id"] = {"$in": document_ids}
        return self._col.update_many(query, {"$set": {"status": STATUS_STALE}}).modified_count

    def get_failed_ids(self) -> list[str]:
        return [doc["_id"] for doc in self._col.find({"status": STATUS_FAILED}, {"_id": 1})]

    def get_processed_ids(self) -> set[str]:
        return {doc["_id"] for doc in self._col.find({"status": STATUS_PROCESSED}, {"_id": 1})}

    def get_processed_hashes(self) -> dict[str, str]:
        """{document_id: source_hash} of processed rows, to skip unchanged docs on reindex."""
        return {
            doc["_id"]: doc.get("source_hash") or ""
            for doc in self._col.find({"status": STATUS_PROCESSED}, {"source_hash": 1})
        }

    def set_index_stats(self, document_id: str, stats: dict) -> None:
        self._col.update_one(
            {"_id": document_id},
            {"$set": {"index_stats": stats, "updated_at": datetime.now(timezone.utc)}},
        )

    def delete(self, document_id: str) -> None:
        self._col.delete_one({"_id": document_id})

    def counts(self) -> dict:
        pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        return {doc["_id"] or "unknown": doc["count"] for doc in self._col.aggregate(pipeline)}

    def close(self) -> None:
        self._client.close()
