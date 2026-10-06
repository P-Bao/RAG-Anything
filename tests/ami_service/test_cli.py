"""Test CLI (status / retry / reindex) với fake runner + fake repos - 0 mạng thật.

Phủ: status mặc định 0 lời gọi embed, --doc chi tiết, --check-embed-server;
retry chọn doc failed / --max-attempts / preflight dừng sớm / một lỗi không dừng lô;
reindex --stale mặc định / --dry-run không ghi / preflight / lockfile.
"""
from datetime import datetime, timedelta, timezone

import pytest

from ami_rag.cli import cmd_reindex, cmd_retry, cmd_status
from ami_rag.core.embedder import EmbedModelMismatch, EmbedServerUnreachable
from ami_rag.core.pipeline import StageOutcome
from ami_rag.settings import Settings
from ami_rag.storage.doc_status import (
    LEGACY_PROCESSED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    DocStatusStore,
)
from tests.ami_service.fakes import FakeMongoCollection

MODEL = "Qwen/Qwen3-VL-Embedding-2B"


class FakeRunner:
    def __init__(
        self,
        fail_ids=None,
        chunks=10,
        cache_hits=3,
        preflight_error=None,
        asset_store=None,
    ):
        self.fail_ids = set(fail_ids or [])
        self.chunks = chunks
        self.cache_hits = cache_hits
        self.preflight_error = preflight_error
        self.preflight_calls = 0
        self.runs: list[tuple] = []
        self.asset_store = asset_store

    async def preflight_embed(self):
        self.preflight_calls += 1
        if self.preflight_error:
            raise self.preflight_error

    async def run(self, doc_id, *, from_stage=None, dry_run=False):
        self.runs.append((doc_id, from_stage, dry_run))
        return StageOutcome(
            doc_id=doc_id,
            stage="indexed",
            ok=doc_id not in self.fail_ids,
            error="" if doc_id not in self.fail_ids else f"{doc_id} boom",
            chunk_count=self.chunks,
            cache_hits=self.cache_hits,
        )


class FakeCLIAssetStore:
    """load_content_list trả content_list hoặc None (mô phỏng MinIO)."""

    def __init__(self, with_content_list: bool):
        self.with_content_list = with_content_list

    def load_content_list(self, doc_id):
        return [{"type": "text", "text": "x"}] if self.with_content_list else None


class FakeScanDocsRepo:
    """iter_all cho scan (mô phỏng MongoDocumentRepo)."""

    def __init__(self, docs: dict):
        self.docs = docs

    def find_by_id(self, doc_id):
        return self.docs.get(doc_id)

    def find_by_file_path(self, file_path):
        for doc in self.docs.values():
            if doc.get("file_path") == file_path:
                return doc
        return None

    def iter_all(self, batch_size=100, document_types=None):
        return iter(list(self.docs.values()))

    def count(self):
        return len(self.docs)


class FakeEmbedder:
    def __init__(self, model=MODEL, dim=2048, error=None):
        self.model = model
        self.dim = dim
        self.error = error
        self.verify_calls = 0

    async def verify(self):
        self.verify_calls += 1
        if self.error:
            raise self.error
        return {
            "model_name": self.model,
            "dim": self.dim,
            "server_version": "test",
            "query_instruction": "Q.",
            "document_instruction": "D.",
        }


def _make_settings(**overrides) -> Settings:
    defaults = {
        "WORKSPACE": "multimodal",
        "EMBED_MODEL": MODEL,
        "EMBED_DIM": 2048,
        "CHUNKER_VERSION": "v1",
        "CLI_STUCK_PROCESSING_MINUTES": 60,
        "EMBED_SERVER_URL": "http://localhost:8007",
        "EMBED_TIMEOUT": 5,
        "MONGO_URI": "mongodb://localhost:27017",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_store() -> DocStatusStore:
    s = DocStatusStore.__new__(DocStatusStore)
    s._col = FakeMongoCollection()
    return s


def _seed(store: DocStatusStore):
    """2 indexed (model hiện tại), 1 legacy processed, 1 failed."""
    store.ensure_pending("doc-1")
    store.ensure_pending("doc-2")
    store.ensure_pending("doc-3")
    store.ensure_pending("doc-4")
    store.mark_indexed("doc-1", chunk_count=5, embed_model=MODEL, embed_dim=2048, chunker_version="v1")
    store.mark_indexed("doc-2", chunk_count=4, embed_model=MODEL, embed_dim=2048, chunker_version="v1")
    # legacy: pipeline cũ (không embed_model) -> stale
    store._col.insert("doc-3", {"status": LEGACY_PROCESSED, "source_path": "old.pdf"})
    store.mark_failed("doc-4", "boom", "embed")


class _Args:
    def __init__(self, **kwargs):
        self.failed = False
        self.stale = False
        self.doc = None
        self.check_embed_server = False
        self.all_failed = False
        self.max_attempts = 3
        self.yes = True
        self.all = False
        self.scan = False
        self.dry_run = False
        self.from_stage = None
        self.embed_server_url = None
        self.embed_batch_size = None
        for key, value in kwargs.items():
            setattr(self, key, value)


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------
async def test_status_summary_counts_and_missing_pending(capsys):
    settings = _make_settings()
    store = _make_store()
    _seed(store)

    class DocsRepo:
        def count(self):
            return 6  # 5 bản ghi + 1 thiếu -> pending

    await cmd_status(_Args(), settings=settings, store=store, docs_repo=DocsRepo())
    out = capsys.readouterr().out
    assert "indexed" in out
    assert "failed" in out
    assert "stale" in out  # legacy doc-3
    assert "pending" in out
    assert f"embed_model={MODEL}" in out
    assert "multimodal__qwen3-vl-embedding-2b__v1" in out


async def test_status_default_makes_no_embed_calls(capsys, monkeypatch):
    """status (không --check-embed-server): 0 embed server call, 0 model call."""
    settings = _make_settings()
    store = _make_store()
    _seed(store)

    class DocsRepo:
        def count(self):
            return 4

    def forbidden(*_a, **_k):
        raise AssertionError("status không được gọi embed server")

    monkeypatch.setattr("ami_rag.cli._build_embedder", forbidden)
    await cmd_status(_Args(), settings=settings, store=store, docs_repo=DocsRepo())
    assert "unreachable" not in capsys.readouterr().out


async def test_status_stuck_processing_reported(capsys):
    settings = _make_settings()
    store = _make_store()
    store.ensure_pending("doc-stuck")
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    store._col.insert("doc-stuck", {
        "status": STATUS_PROCESSING,
        "stage": "embed",
        "updated_at": old,
        "attempts": 1,
    })

    class DocsRepo:
        def count(self):
            return 1

    await cmd_status(_Args(), settings=settings, store=store, docs_repo=DocsRepo())
    assert "TREO" in capsys.readouterr().out


async def test_status_doc_detail(capsys):
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    await cmd_status(
        _Args(doc="doc-4"), settings=settings, store=store,
        docs_repo=type("R", (), {"count": lambda self: 4})(),
    )
    out = capsys.readouterr().out
    assert "doc-4" in out
    assert "error (embed): boom" in out
    assert "stage:" in out


async def test_status_doc_path_resolved_via_docs_repo(capsys):
    settings = _make_settings()
    store = _make_store()
    _seed(store)

    class DocsRepo:
        def count(self):
            return 4

        def find_by_id(self, doc_id):
            return None

        def find_by_file_path(self, path):
            return {"_id": "doc-1", "file_path": path} if path == "a.pdf" else None

    await cmd_status(
        _Args(doc="a.pdf"), settings=settings, store=store, docs_repo=DocsRepo()
    )
    out = capsys.readouterr().out
    assert "doc_id:         doc-1" in out


async def test_status_check_embed_server_ok(capsys):
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    embedder = FakeEmbedder()

    class DocsRepo:
        def count(self):
            return 4

    await cmd_status(
        _Args(check_embed_server=True), settings=settings, store=store,
        docs_repo=DocsRepo(), embedder=embedder,
    )
    out = capsys.readouterr().out
    assert embedder.verify_calls == 1
    assert "khớp config" in out


async def test_status_check_embed_server_mismatch(capsys):
    settings = _make_settings()
    store = _make_store()
    embedder = FakeEmbedder(error=EmbedModelMismatch("embed server chạy model 'other-model'"))
    await cmd_status(
        _Args(check_embed_server=True), settings=settings, store=store,
        docs_repo=type("R", (), {"count": lambda self: 0})(), embedder=embedder,
    )
    assert "LỆCH hoặc lỗi" in capsys.readouterr().out


# --------------------------------------------------------------------------
# retry
# --------------------------------------------------------------------------
async def test_retry_all_failed_runs_only_failed():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner()
    rc = await cmd_retry(
        _Args(all_failed=True, yes=True), settings=settings, store=store,
        docs_repo=None, runner=runner,
    )
    assert rc == 0
    assert sorted(doc_id for doc_id, _, _ in runner.runs) == ["doc-4"]


async def test_retry_doc_specific_non_failed_skipped():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner()
    rc = await cmd_retry(
        _Args(doc=["doc-1"], yes=True), settings=settings, store=store,
        docs_repo=None, runner=runner,
    )
    assert rc == 0
    assert runner.runs == []


async def test_retry_max_attempts_skips():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    store.mark_failed("doc-4", "boom 2", "embed")  # attempts = 2
    runner = FakeRunner()
    await cmd_retry(
        _Args(all_failed=True, max_attempts=2, yes=True), settings=settings,
        store=store, docs_repo=None, runner=runner,
    )
    assert runner.runs == []


async def test_retry_preflight_failure_stops_early_no_fail_marked():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner(preflight_error=EmbedServerUnreachable("embed server unreachable"))
    rc = await cmd_retry(
        _Args(all_failed=True, yes=True), settings=settings, store=store,
        docs_repo=None, runner=runner,
    )
    assert rc == 2
    assert runner.runs == []
    # không đánh fail doc nào: status giữ nguyên failed, attempts giữ nguyên
    assert store.get("doc-4")["attempts"] == 1


async def test_retry_one_failure_doesnt_stop_batch():
    settings = _make_settings()
    store = _make_store()
    store.ensure_pending("doc-a")
    store.ensure_pending("doc-b")
    store.mark_failed("doc-a", "e1", "chunk")
    store.mark_failed("doc-b", "e2", "chunk")
    runner = FakeRunner(fail_ids={"doc-a"})
    rc = await cmd_retry(
        _Args(all_failed=True, yes=True), settings=settings, store=store,
        docs_repo=None, runner=runner,
    )
    assert rc == 1
    assert sorted(doc_id for doc_id, _, _ in runner.runs) == ["doc-a", "doc-b"]


async def test_retry_from_stage_passed_to_runner():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner()
    await cmd_retry(
        _Args(all_failed=True, from_stage="parse", yes=True), settings=settings,
        store=store, docs_repo=None, runner=runner,
    )
    assert runner.runs == [("doc-4", "parse", False)]


# --------------------------------------------------------------------------
# reindex
# --------------------------------------------------------------------------
async def test_reindex_default_stale_includes_legacy_and_model_mismatch():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    # doc-1 indexed bằng model CŨ -> stale so với config hiện tại
    store._col._docs["doc-1"]["embed_model"] = "old-embed-model"
    runner = FakeRunner()
    rc = await cmd_reindex(_Args(), settings=settings, store=store, runner=runner)
    assert rc == 0
    # doc-3 (legacy) + doc-1 (lệch embed_model) -> stale
    assert sorted(doc_id for doc_id, _, _ in runner.runs) == ["doc-1", "doc-3"]


async def test_reindex_dry_run_no_preflight_no_writes():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner(preflight_error=EmbedServerUnreachable("must not be called"))
    rc = await cmd_reindex(
        _Args(dry_run=True), settings=settings, store=store, runner=runner
    )
    assert rc == 0
    assert runner.preflight_calls == 0
    assert all(dry_run for _, _, dry_run in runner.runs)
    # store không đổi
    assert store.get("doc-4")["status"] == STATUS_FAILED


async def test_reindex_preflight_failure_stops_early():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner(preflight_error=EmbedServerUnreachable("embed server unreachable"))
    rc = await cmd_reindex(_Args(), settings=settings, store=store, runner=runner)
    assert rc == 2
    assert all(not dry_run for _, _, dry_run in runner.runs) and not runner.runs


async def test_reindex_all_runs_everything_with_preflight():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner()
    rc = await cmd_reindex(
        _Args(all=True, from_stage="chunk"), settings=settings, store=store, runner=runner
    )
    assert rc == 0
    assert runner.preflight_calls == 1
    assert sorted(doc_id for doc_id, from_stage, _ in runner.runs) == [
        "doc-1", "doc-2", "doc-3", "doc-4",
    ]
    assert all(from_stage == "chunk" for _, from_stage, _ in runner.runs)


async def test_reindex_dry_run_prints_chunk_and_cache_counts(capsys):
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner(chunks=7, cache_hits=2)
    await cmd_reindex(
        _Args(dry_run=True), settings=settings, store=store, runner=runner
    )
    out = capsys.readouterr().out
    # Mặc định --stale: chỉ doc-3 (legacy) là stale
    assert "1 doc" in out
    assert "7 chunk cần embed" in out
    assert "2 trúng cache" in out


async def test_reindex_lockfile_conflict_stops(monkeypatch):
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner()

    class HeldLockfile:
        def __init__(self, path):
            pass

        def acquire(self):
            from ami_rag.core.lockfile import LockHeld

            raise LockHeld("đang giữ")

        def release(self):
            pass

    monkeypatch.setattr("ami_rag.core.lockfile.Lockfile", HeldLockfile)
    rc = await cmd_reindex(_Args(), settings=settings, store=store, runner=runner)
    assert rc == 2
    assert runner.runs == []


async def test_reindex_from_stage_invalid_raises():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    with pytest.raises(ValueError):
        await cmd_reindex(
            _Args(from_stage="indexed"), settings=settings, store=store,
            runner=FakeRunner(),
        )


# --------------------------------------------------------------------------
# reindex --scan (Phase 5)
# --------------------------------------------------------------------------
async def test_reindex_scan_creates_pending_and_selects_candidates():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    # doc-3: legacy `processed` (stale); doc-9: chưa có bản ghi -> scan tạo pending
    docs = {
        "doc-3": {"_id": "doc-3", "file_path": "old.pdf", "content": "x"},
        "doc-9": {"_id": "doc-9", "file_path": "new.pdf", "content": "y"},
    }
    runner = FakeRunner(asset_store=FakeCLIAssetStore(with_content_list=False))
    rc = await cmd_reindex(
        _Args(scan=True), settings=settings, store=store,
        runner=runner, docs_repo=FakeScanDocsRepo(docs),
    )
    assert rc == 0
    # candidates = pending (doc-9) + stale (doc-3); indexed/failed loại trừ
    assert sorted(doc_id for doc_id, _, _ in runner.runs) == ["doc-3", "doc-9"]
    assert store.get("doc-9")["status"] == STATUS_PENDING
    # content_list thiếu trong MinIO -> from_stage None (full parse)
    assert all(fs is None for _, fs, _ in runner.runs)


async def test_reindex_scan_content_list_present_uses_chunk_stage():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    docs = {
        "doc-3": {"_id": "doc-3", "file_path": "old.pdf", "content": "x"},
        "doc-9": {"_id": "doc-9", "file_path": "new.pdf", "content": "y"},
    }
    runner = FakeRunner(asset_store=FakeCLIAssetStore(with_content_list=True))
    rc = await cmd_reindex(
        _Args(scan=True), settings=settings, store=store,
        runner=runner, docs_repo=FakeScanDocsRepo(docs),
    )
    assert rc == 0
    assert sorted(doc_id for doc_id, _, _ in runner.runs) == ["doc-3", "doc-9"]
    # content_list đã có -> from_stage "chunk" (không gọi lại parse/LLM)
    assert all(fs == "chunk" for _, fs, _ in runner.runs)


async def test_reindex_scan_does_not_select_indexed_or_failed():
    settings = _make_settings()
    store = _make_store()
    _seed(store)  # doc-1/doc-2 indexed, doc-3 legacy, doc-4 failed
    docs = {"doc-1": {"_id": "doc-1", "file_path": "a.pdf"}, "doc-4": {"_id": "doc-4", "content": "x"}}
    runner = FakeRunner(asset_store=FakeCLIAssetStore(with_content_list=True))
    rc = await cmd_reindex(
        _Args(scan=True), settings=settings, store=store,
        runner=runner, docs_repo=FakeScanDocsRepo(docs),
    )
    assert rc == 0
    # chỉ doc-3 (legacy stale); indexed + failed không reindex
    assert [doc_id for doc_id, _, _ in runner.runs] == ["doc-3"]
