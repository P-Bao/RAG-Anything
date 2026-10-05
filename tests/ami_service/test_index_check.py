import pytest

from ami_rag import cli
from ami_rag.index_check import (
    IndexStats,
    IngestIncompleteError,
    inspect_document,
    repair_document,
)
from ami_rag.queue.events import RagEvent
from ami_rag.sources import source_hash

from .conftest import FakeRAGAnything
from .test_worker import CRAWL_ID, _worker


class _KV:
    """get_by_id / get_by_ids over a dict (None for missing, like the Mongo/Qdrant impls)."""

    def __init__(self, rows=None):
        self.rows = dict(rows or {})
        self.upserts = []

    async def get_by_id(self, key):
        return self.rows.get(key)

    async def get_by_ids(self, keys):
        return [self.rows.get(k) for k in keys]

    async def upsert(self, data):
        self.upserts.append(data)
        self.rows.update(data)


class _Rag:
    def __init__(self, chunk_ids, vectors, entities=0, relations=0, texts=True):
        self.doc_status = _KV({"d1": {"chunks_list": chunk_ids}})
        self.chunks_vdb = _KV({c: {"id": c} for c in vectors})
        self.text_chunks = _KV(
            {c: {"_id": c, "content": f"text {c}", "full_doc_id": "d1"} for c in chunk_ids}
            if texts
            else {}
        )
        self.full_entities = _KV(
            {"d1": {"entity_names": [f"e{i}" for i in range(entities)]}} if entities else {}
        )
        self.full_relations = _KV(
            {"d1": {"relation_pairs": [["a", "b"]] * relations}} if relations else {}
        )
        self.inserted_done = 0

    async def _insert_done(self):
        self.inserted_done += 1


def test_stats_problems():
    assert IndexStats("d", chunks=3, vectors=3, entities=2).ok()
    assert IndexStats("d", chunks=3, vectors=1, entities=2).problems() == [
        "2/3 chunk vectors missing"
    ]
    assert IndexStats("d", chunks=3, vectors=3, entities=0).problems() == ["no entities"]
    assert IndexStats("d", chunks=3, vectors=3, entities=0).ok(require_entities=False)
    assert IndexStats("d").problems() == ["no chunks"]


async def test_inspect_counts_real_index():
    rag = _Rag(["c1", "c2", "c3"], vectors=["c1"], entities=2, relations=1)
    stats = await inspect_document(rag, "d1")
    assert (stats.chunks, stats.vectors, stats.entities, stats.relations) == (
        3,
        1,
        2,
        1,
    )
    assert stats.missing_vectors == 2
    assert "INCOMPLETE" in stats.line()


async def test_repair_reembeds_only_missing_vectors_and_skips_extraction():
    rag = _Rag(["c1", "c2"], vectors=["c1"], entities=3)
    after = await repair_document(rag, "d1")
    assert list(rag.chunks_vdb.upserts[0]) == ["c2"]
    assert "_id" not in rag.chunks_vdb.upserts[0]["c2"]
    assert after.vectors == 2 and after.ok()
    assert rag.inserted_done == 1


async def test_repair_needs_full_reindex_when_chunk_text_missing():
    rag = _Rag(["c1"], vectors=[], entities=1, texts=False)
    with pytest.raises(IngestIncompleteError, match="full reindex"):
        await repair_document(rag, "d1")


async def test_worker_fails_document_when_index_incomplete(
    fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    rag_anything = FakeRAGAnything()
    rag_anything.lightrag.chunks_vdb = _KV()
    rag_anything.lightrag.doc_status = _KV({CRAWL_ID: {"chunks_list": ["c1", "c2"]}})
    rag_anything.lightrag.full_entities = _KV()
    rag_anything.lightrag.full_relations = _KV()
    worker = _worker(rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store)
    with pytest.raises(IngestIncompleteError, match="2/2 chunk vectors missing"):
        await worker.handle_event(RagEvent(event="created", document_id=CRAWL_ID))


async def test_worker_records_index_stats_when_complete(
    fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    rag_anything = FakeRAGAnything()
    rag_anything.lightrag.chunks_vdb = _KV({"c1": {"id": "c1"}})
    rag_anything.lightrag.doc_status = _KV({CRAWL_ID: {"chunks_list": ["c1"]}})
    rag_anything.lightrag.full_entities = _KV({CRAWL_ID: {"entity_names": ["x"]}})
    rag_anything.lightrag.full_relations = _KV()
    worker = _worker(rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store)
    result = await worker.handle_event(RagEvent(event="created", document_id=CRAWL_ID))
    assert result["meta"]["index_stats"] == {
        "chunks": 1,
        "vectors": 1,
        "entities": 1,
        "relations": 0,
    }


def test_reindex_skips_processed_with_unchanged_hash(fake_docs_repo):
    docs = list(fake_docs_repo.docs.values())
    processed = docs[0]
    hashes = {str(processed["_id"]): source_hash(processed)}
    todo, done = cli._split_processed(docs, hashes)
    assert [d["_id"] for d in done] == [processed["_id"]]
    assert processed not in todo and len(todo) == len(docs) - 1

    changed = {str(processed["_id"]): "stale-hash"}
    todo, done = cli._split_processed(docs, changed)
    assert not done and len(todo) == len(docs)


# --- error classification / fatal stop / repair of failed docs -------------------------


def test_classify_error_codes():
    from ami_rag.index_check import (
        ERR_AUTH,
        ERR_OTHER,
        ERR_QUOTA,
        ERR_RATE_LIMIT,
        ERR_UPSTREAM,
        classify_error,
    )

    assert (
        classify_error("402 RESOURCE_EXHAUSTED. Your prepayment credits are depleted") == ERR_QUOTA
    )
    assert classify_error("429 RESOURCE_EXHAUSTED. Resource exhausted") == ERR_RATE_LIMIT
    assert classify_error("ServerError: 503 UNAVAILABLE high demand") == ERR_UPSTREAM
    assert classify_error("403 PERMISSION_DENIED api key") == ERR_AUTH
    assert classify_error(IngestIncompleteError("x")) == "INDEX_INCOMPLETE"
    # hex chunk ids containing digits must not be read as HTTP codes
    assert classify_error("chunk-a402b429c500: boom") == ERR_OTHER


class _Fake:
    def __init__(self, exc=None):
        self.exc = exc

    async def __call__(self, *_a, **_k):
        if self.exc:
            raise self.exc
        return [0.0]


async def test_preflight_raises_on_quota_but_not_transient():
    from ami_rag.index_check import FatalIngestError, preflight_providers

    class R:
        embedding_func = _Fake(RuntimeError("402 RESOURCE_EXHAUSTED prepayment credits"))
        llm_model_func = _Fake()

    with pytest.raises(FatalIngestError) as err:
        await preflight_providers(R())
    assert err.value.code == "QUOTA_EXHAUSTED"

    class T:
        embedding_func = _Fake(RuntimeError("503 UNAVAILABLE"))
        llm_model_func = _Fake(RuntimeError("429 Resource exhausted"))

    await preflight_providers(T())  # transient only: no raise


def test_failed_by_code_classifies_legacy_rows(fake_state_repo):
    fake_state_repo.mark_failed("a", "x", error_code="QUOTA_EXHAUSTED")
    fake_state_repo.mark_failed("b", "y", error_code="QUOTA_EXHAUSTED")
    summary = fake_state_repo.failed_by_code()
    assert summary["QUOTA_EXHAUSTED"]["count"] == 2


class _Args:
    dry_run = False


class _Settings:
    INGEST_REQUIRE_ENTITIES = True
    REPAIR_CONCURRENCY = 2
    RAG_STREAM = "rag:ingest"


def _patch_repair(monkeypatch, rag, repair_fn):
    import ami_rag.index_check as ic
    from ami_rag.core import factory

    async def get_rag():
        return rag

    async def close_rag():
        return None

    async def ok_preflight(_rag):
        return None

    monkeypatch.setattr(factory, "get_rag", get_rag)
    monkeypatch.setattr(factory, "close_rag", close_rag)
    monkeypatch.setattr(ic, "preflight_providers", ok_preflight)
    monkeypatch.setattr(ic, "repair_document", repair_fn)


async def test_repair_includes_failed_docs_and_promotes_to_processed(monkeypatch, fake_state_repo):
    fake_state_repo.mark_failed("d1", "402 prepayment credits", error_code="QUOTA_EXHAUSTED")
    rag = _Rag(["c1"], vectors=[], entities=0)

    async def repair(rag_, doc_id, st=None):
        return IndexStats(doc_id, chunks=1, vectors=1, entities=3)

    _patch_repair(monkeypatch, rag, repair)
    rc = await cli._reindex_repair(_Args(), _Settings(), fake_state_repo, [{"_id": "d1"}])
    assert rc == 0
    assert fake_state_repo.state["d1"]["status"] == "processed"
    assert fake_state_repo.state["d1"]["index_stats"]["entities"] == 3


async def test_repair_stops_on_quota_and_does_not_mark_remaining_failed(
    monkeypatch, fake_state_repo, capsys
):
    for i in ("d1", "d2", "d3", "d4"):
        fake_state_repo.state[i] = {"status": "processed", "source_hash": ""}
    rag = _Rag(["c1"], vectors=[], entities=0)
    rag.doc_status = _KV({i: {"chunks_list": ["c1"]} for i in ("d1", "d2", "d3", "d4")})
    rag.full_entities = _KV()
    rag.full_relations = _KV()
    calls = []

    async def repair(rag_, doc_id, st=None):
        calls.append(doc_id)
        raise RuntimeError("402 RESOURCE_EXHAUSTED. Your prepayment credits are depleted")

    _patch_repair(monkeypatch, rag, repair)
    docs = [{"_id": i} for i in ("d1", "d2", "d3", "d4")]
    rc = await cli._reindex_repair(_Args(), _Settings(), fake_state_repo, docs)
    out = capsys.readouterr().out
    assert rc == 2
    assert len(calls) == 2  # only the first batch (concurrency 2) was attempted
    assert all(fake_state_repo.state[i]["status"] == "processed" for i in ("d1", "d2", "d3", "d4"))
    assert "QUOTA_EXHAUSTED" in out and "NOT attempted" in out


async def test_repair_reports_transient_failures_by_cause(monkeypatch, fake_state_repo, capsys):
    fake_state_repo.state["d1"] = {"status": "processed", "source_hash": ""}
    rag = _Rag(["c1"], vectors=[], entities=0)

    async def repair(rag_, doc_id, st=None):
        raise RuntimeError("503 UNAVAILABLE high demand")

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(cli.asyncio, "sleep", no_sleep)
    _patch_repair(monkeypatch, rag, repair)
    rc = await cli._reindex_repair(_Args(), _Settings(), fake_state_repo, [{"_id": "d1"}])
    assert rc == 1
    entry = fake_state_repo.state["d1"]
    assert entry["status"] == "failed" and entry["error_code"] == "UPSTREAM_UNAVAILABLE"
    assert "errors by cause" in capsys.readouterr().out
