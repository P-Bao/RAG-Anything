from pathlib import Path

import pytest

from ami_rag.queue.events import RagEvent


class FakeRAG:
    def __init__(self, query_result=None):
        self.query_result = query_result if query_result is not None else _default_success()
        self.calls = []

    async def aquery_data(self, query, mode="mix", **kwargs):
        self.calls.append(("query_data", query, mode, kwargs))
        return self.query_result

    def set_chunks(self, chunks):
        self.query_result["data"]["chunks"] = chunks

    async def ainsert(self, input, ids=None, file_paths=None, **kwargs):
        self.calls.append(("insert", input, ids, file_paths))

    async def adelete_by_doc_id(self, doc_id):
        self.calls.append(("delete", doc_id))


def _default_success():
    return {
        "status": "success",
        "data": {
            "entities": [
                {
                    "entity_name": "Học viện Công nghệ PTIT",
                    "entity_type": "Organization",
                    "description": "Trường đại học kỹ thuật.",
                    "source_id": "chunk-1",
                    "file_path": "documents/HV/a.pdf",
                    "created_at": "2026-01-01",
                    "reference_id": "1",
                }
            ],
            "relationships": [],
            "chunks": [
                {
                    "content": "Học phí năm 2026 là 12 triệu mỗi học kỳ.",
                    "file_path": "64b000000000000000000001_a.pdf",
                    "chunk_id": "chunk-1",
                    "reference_id": "1",
                },
                {
                    "content": "Tuyển sinh sử dụng phương thức xét tuyển kết hợp.",
                    "file_path": "64b000000000000000000003_b.pdf",
                    "chunk_id": "chunk-2",
                    "reference_id": "2",
                },
                {
                    "content": "Sơ đồ quy trình tuyển sinh.",
                    "file_path": "64b000000000000000000001_a.pdf",
                    "chunk_id": "chunk-3",
                    "reference_id": "1",
                    "modality": "image",
                    "asset_key": "rag-assets/64b000000000000000000001/i.png",
                    "page_idx": 2,
                    "caption": "Sơ đồ",
                },
                {
                    "content": "Bảng học phí theo ngành.",
                    "file_path": "64b000000000000000000001_a.pdf",
                    "chunk_id": "chunk-4",
                    "reference_id": "1",
                    "modality": "table",
                    "table_body": "| ngành | phí |\n|---|---|\n| CNTT | 12tr |",
                    "page_idx": 5,
                    "caption": "Học phí",
                },
            ],
            "references": [
                {"reference_id": "1", "file_path": "64b000000000000000000001_a.pdf"},
                {"reference_id": "2", "file_path": "64b000000000000000000003_b.pdf"},
            ],
        },
        "metadata": {
            "query_mode": "mix",
            "keywords": {"high_level": ["học phí"], "low_level": ["2026"]},
        },
    }


class FakeDocsRepo:
    def __init__(self, docs=None):
        self.docs = docs or {}

    def find_by_id(self, document_id):
        return self.docs.get(document_id)

    def find_by_file_path(self, file_path):
        for doc in self.docs.values():
            if doc.get("file_path") == file_path:
                return doc
        return None

    def find_by_source_url(self, source_url):
        for doc in self.docs.values():
            if (doc.get("metadata") or {}).get("source_url") == source_url:
                return doc
        return None

    def count(self):
        return len(self.docs)


class FakeRagDocumentsRepo:
    """In-memory stand-in for ami_rag.storage.rag_documents.RagDocumentsRepo."""

    def __init__(self):
        self.state = {}

    def get(self, document_id):
        return self.state.get(document_id)

    def begin_attempt(self, document_id, source_hash=None):
        entry = self.state.setdefault(document_id, {"attempts": 0, "status": "processing"})
        entry["attempts"] += 1
        return entry["attempts"]

    def release_attempt(self, document_id):
        entry = self.state.get(document_id)
        if entry is None:
            return
        entry["attempts"] = max(entry.get("attempts", 1) - 1, 0)
        if entry.get("status") == "processing" and not entry.get("source_hash"):
            self.state.pop(document_id, None)

    def mark_processed(
        self,
        document_id,
        source_hash=None,
        *,
        source="",
        file_path="",
        parser="",
        assets=None,
        counts=None,
        page_count=0,
        meta=None,
    ):
        self.state[document_id] = {
            "attempts": 0,
            "status": "processed",
            "source_hash": source_hash or "",
            "source": source,
            "file_path": file_path,
            "parser": parser,
            "assets": assets or [],
            "counts": counts or {},
            "page_count": page_count,
            **(meta or {}),
            "error": "",
        }

    def mark_failed(
        self,
        document_id,
        error,
        source_hash=None,
        *,
        error_code="OTHER",
        error_stage="",
        retryable=True,
    ):
        entry = self.state.setdefault(document_id, {})
        entry.update(
            {
                "attempts": 0,
                "status": "failed",
                "error": error,
                "error_code": error_code,
                "error_stage": error_stage,
                "error_retryable": retryable,
            }
        )

    def mark_repaired(self, document_id, index_stats):
        entry = self.state.setdefault(document_id, {})
        entry.update(
            {
                "attempts": 0,
                "status": "processed",
                "index_stats": index_stats,
                "error": "",
                "error_code": "",
            }
        )

    def mark_stale(self, document_ids=None):
        n = 0
        for k, v in self.state.items():
            if v.get("status") == "processed" and (document_ids is None or k in document_ids):
                v["status"] = "stale"
                n += 1
        return n

    def get_failed_ids(self):
        return [k for k, v in self.state.items() if v.get("status") == "failed"]

    def get_processed_ids(self):
        return {k for k, v in self.state.items() if v.get("status") == "processed"}

    def get_processed_hashes(self):
        return {
            k: v.get("source_hash", "")
            for k, v in self.state.items()
            if v.get("status") == "processed"
        }

    def get_repairable_hashes(self):
        return {
            k: v.get("source_hash", "")
            for k, v in self.state.items()
            if v.get("status") in ("processed", "failed")
        }

    def failed_by_code(self, sample=3):
        out = {}
        for k, v in self.state.items():
            if v.get("status") != "failed":
                continue
            entry = out.setdefault(
                v.get("error_code") or "OTHER",
                {"count": 0, "ids": [], "error": v.get("error", "")},
            )
            entry["count"] += 1
            if len(entry["ids"]) < sample:
                entry["ids"].append(k)
        return out

    def set_index_stats(self, document_id, stats):
        self.state.setdefault(document_id, {})["index_stats"] = stats

    def delete(self, document_id):
        self.state.pop(document_id, None)

    def counts(self):
        counts = {}
        for v in self.state.values():
            counts[v.get("status", "unknown")] = counts.get(v.get("status", "unknown"), 0) + 1
        return counts


# Alias kept for tests/fixtures that still use the old name (admin API tests).
FakeStateRepo = FakeRagDocumentsRepo


class FakeAssetStore:
    """In-memory stand-in for MinioAssetStore."""

    def __init__(self, fail_presign=False):
        self.calls = []
        self.fetched = []
        self.content_lists = {}
        self.deleted_assets = []
        self.fail_presign = fail_presign

    def fetch(self, object_name, dest_dir):
        self.calls.append(("fetch", object_name))
        dest = Path(dest_dir) / Path(object_name).name
        dest.write_bytes(b"%PDF-fake")
        self.fetched.append(object_name)
        return dest

    def upload_content_list_assets(self, doc_id, content_list):
        self.calls.append(("upload_assets", doc_id))
        keys = []
        for i, item in enumerate(content_list):
            if item.get("type") == "image" and item.get("img_path"):
                key = f"rag-assets/{doc_id}/img{i}.png"
                item["asset_key"] = key
                keys.append(key)
        return keys

    def save_content_list(self, doc_id, content_list):
        self.calls.append(("save_content_list", doc_id))
        self.content_lists[doc_id] = [dict(i) for i in content_list]
        return f"rag-assets/{doc_id}/content_list.json"

    def load_content_list(self, doc_id):
        return self.content_lists.get(doc_id)

    def presign(self, key):
        if not key or self.fail_presign:
            return None
        return f"https://minio/presigned/{key}"

    def delete_doc_assets(self, doc_id):
        self.calls.append(("delete_assets", doc_id))
        self.deleted_assets.append(doc_id)
        self.content_lists.pop(doc_id, None)
        return 1


class _FakeLightRAG:
    def __init__(self, calls):
        self._calls = calls

    async def adelete_by_doc_id(self, doc_id):
        self._calls.append(("delete", doc_id))


class _FakeParseCache:
    def __init__(self):
        self.deleted = []

    async def delete(self, ids):
        self.deleted.extend(ids)

    async def index_done_callback(self):
        pass


class FakeRAGAnything:
    """Stand-in for raganything.RAGAnything (parse + insert_content_list)."""

    def __init__(self, content_list=None, fail_insert=False):
        self.calls = []
        self.lightrag = _FakeLightRAG(self.calls)
        self.content_list = content_list
        self.fail_insert = fail_insert
        self.parse_cache = _FakeParseCache()

    def _generate_cache_key(self, file_path, parse_method=None, **kwargs):
        return f"cache::{Path(file_path).name}::{parse_method}"

    async def parse_document(self, file_path, output_dir=None, parse_method=None, **kwargs):
        self.calls.append(("parse", file_path, parse_method))
        if self.content_list is not None:
            return [dict(i) for i in self.content_list], "parsed-doc-id"
        img = Path(output_dir) / "img.png"
        img.write_bytes(b"\x89PNG-fake")
        return (
            [
                {"type": "text", "text": "Giới thiệu", "page_idx": 0},
                {
                    "type": "image",
                    "img_path": str(img),
                    "image_caption": ["Sơ đồ"],
                    "page_idx": 1,
                },
                {
                    "type": "table",
                    "table_body": "| a | b |\n|---|---|\n| 1 | 2 |",
                    "page_idx": 2,
                },
            ],
            "parsed-doc-id",
        )

    async def insert_content_list(
        self, content_list, file_path="unknown_document", doc_id=None, **kwargs
    ):
        if self.fail_insert:
            raise RuntimeError("llm down")
        self.calls.append(("insert", [dict(i) for i in content_list], file_path, doc_id))


class FakeQueue:
    def __init__(self):
        self.messages = []
        self.acked = []
        self.published = []
        self._consumer = "fake-consumer"

    async def ensure_group(self):
        pass

    async def read_batch(self, count=10, block_ms=5000):
        batch = self.messages[:count]
        self.messages = self.messages[count:]
        return batch

    async def ack(self, message_id):
        self.acked.append(message_id)

    async def publish(self, event: RagEvent):
        self.published.append(event)
        return "1-1"

    async def pending_count(self):
        return len(self.messages)


class FakeRerank:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    async def __call__(self, query, documents, top_n=None, **kwargs):
        self.calls.append((query, len(documents), top_n))
        if self.fail:
            raise ConnectionError("rerank down")
        n = min(top_n or len(documents), len(documents))
        return [{"index": i, "relevance_score": round(0.9 - i * 0.1, 2)} for i in range(n)]


@pytest.fixture
def fake_rag():
    return FakeRAG()


@pytest.fixture
def fake_docs_repo():
    return FakeDocsRepo(
        docs={
            "64b000000000000000000001": {
                "_id": "64b000000000000000000001",
                "title": "Doc A",
                "content": "nội dung tài liệu A",
                "content_hash": "hash-a",
                "document_type": "pdf",
                "original_file_name": "x.pdf",
                "file_path": "documents/HV/a.pdf",
                "organization_unit_id": "64b000000000000000000002",
                "metadata": {},
            },
            "64b000000000000000000004": {
                "_id": "64b000000000000000000004",
                "title": "Doc D",
                "content": "text docx cũ",
                "content_hash": "hash-d",
                "document_type": "docx",
                "original_file_name": "d.docx",
                "file_path": "documents/HV/d.docx",
                "organization_unit_id": "64b000000000000000000002",
                "metadata": {},
            },
            "64b000000000000000000005": {
                "_id": "64b000000000000000000005",
                "title": "Text E",
                "content": "nội dung text E",
                "content_hash": "hash-e",
                "document_type": "text",
                "original_file_name": None,
                "file_path": None,
                "organization_unit_id": "64b000000000000000000002",
                "metadata": {},
            },
            "64b000000000000000000006": {
                "_id": "64b000000000000000000006",
                "title": "Txt F",
                "content": "nội dung file txt F",
                "content_hash": "hash-f",
                "document_type": "text",
                "original_file_name": "a.txt",
                "file_path": "documents/HV/a.txt",
                "organization_unit_id": "64b000000000000000000002",
                "metadata": {},
            },
            "64b000000000000000000003": {
                "_id": "64b000000000000000000003",
                "title": "Crawl B",
                "content": "nội dung crawl B",
                "content_hash": "hash-b",
                "document_type": "crawl",
                "original_file_name": None,
                "file_path": None,
                "organization_unit_id": "64b000000000000000000002",
                "metadata": {"source_url": "https://example.com/b"},
            },
        }
    )


@pytest.fixture
def fake_state_repo():
    return FakeRagDocumentsRepo()


@pytest.fixture
def fake_rag_anything():
    return FakeRAGAnything()


@pytest.fixture
def fake_asset_store():
    return FakeAssetStore()


@pytest.fixture
def fake_queue():
    return FakeQueue()
