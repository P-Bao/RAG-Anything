import json
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

from ami_rag import observability as obs
from ami_rag.api.routes import rag as rag_routes

from .test_api import app, client  # noqa: F401  (fixtures)

P = "multimodal_rag_retrieval_"
BODY = {"messages": [{"role": "user", "content": "học phí"}]}


def _val(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(P + name, labels or None) or 0.0


async def test_search_records_request_metrics(client):  # noqa: F811
    before = {
        "req": _val("requests_total", mode="mix"),
        "dur": _val("duration_seconds_count", mode="mix"),
        "docs": _val("docs_returned_count"),
        "chunks": _val("chunks_retrieved_count"),
        "lightrag": _val("stage_duration_seconds_count", stage="raganything_query"),
        "rerank": _val("stage_duration_seconds_count", stage="rerank"),
        "resolve": _val("stage_duration_seconds_count", stage="resolve"),
    }
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    assert _val("requests_total", mode="mix") == before["req"] + 1
    assert _val("duration_seconds_count", mode="mix") == before["dur"] + 1
    assert _val("docs_returned_count") == before["docs"] + 1
    assert _val("chunks_retrieved_count") == before["chunks"] + 1
    assert _val("stage_duration_seconds_count", stage="raganything_query") == before["lightrag"] + 1
    assert _val("stage_duration_seconds_count", stage="rerank") == before["rerank"] + 1
    assert _val("stage_duration_seconds_count", stage="resolve") > before["resolve"]
    assert _val("requests_in_flight") == 0


async def test_stream_endpoint_counts_as_request(client):  # noqa: F811
    before = _val("requests_total", mode="mix")
    resp = await client.post("/v2/rag/stream", json=BODY)
    assert resp.status_code == 200
    assert _val("requests_total", mode="mix") == before + 1


async def test_v2_mode_label(client):  # noqa: F811
    before = _val("requests_total", mode="local")
    resp = await client.post("/v2/rag/", json={**BODY, "version": 2, "mode": "local"})
    assert resp.status_code == 200
    assert _val("requests_total", mode="local") == before + 1


async def test_rerank_fallback_on_error(client, app):  # noqa: F811
    from .conftest import FakeRerank

    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: FakeRerank(fail=True)
    before = _val("rerank_fallback_total", reason="error")
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    assert _val("rerank_fallback_total", reason="error") == before + 1


async def test_lightrag_non_success_counted(client, fake_rag):  # noqa: F811
    fake_rag.query_result = {"status": "failure", "message": "boom"}
    before = _val("lightrag_failures_total")
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    assert _val("lightrag_failures_total") == before + 1
    assert resp.json()["documents"] == []


async def test_exception_counts_error_and_releases_in_flight(client, fake_rag):  # noqa: F811
    async def boom(query, mode="mix", **kwargs):
        raise RuntimeError("qdrant down")

    fake_rag.aquery_data = boom
    before = _val("errors_total", mode="mix")
    with pytest.raises(RuntimeError):
        await client.post("/v2/rag/", json=BODY)
    assert _val("errors_total", mode="mix") == before + 1
    assert _val("requests_in_flight") == 0


async def test_metrics_endpoint_exposes_series(client):  # noqa: F811
    await client.post("/v2/rag/", json=BODY)
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert P + "requests_total" in resp.text
    assert P + "stage_duration_seconds_bucket" in resp.text


I = "multimodal_rag_ingest_"
PDF_ID = "64b000000000000000000001"
TEXT_ID = "64b000000000000000000005"


def _ival(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(I + name, labels or None) or 0.0


def _ingest_worker(rag_anything, docs_repo, state_repo, queue, asset_store, max_delivery=3):
    from ami_rag.workers.ingest_worker import IngestWorker

    class _S:
        WORKER_MAX_DELIVERY = max_delivery
        WORKER_BATCH = 10
        WORKER_POLL_BLOCK_MS = 10
        RAG_STREAM = "rag:ingest"
        RAG_CONSUMER_GROUP = "ami-rag"
        PARSER = "mineru"
        PARSE_METHOD = "auto"
        CRAWL_IMAGES_ENABLED = False

    return IngestWorker(
        rag_anything=rag_anything,
        docs_repo=docs_repo,
        state_repo=state_repo,
        queue=queue,
        asset_store=asset_store,
        settings=_S(),
    )


async def test_ingest_processed_records_metrics(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    from ami_rag.queue.events import RagEvent

    worker = _ingest_worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    stages = ("download", "parse", "upload_assets", "insert")
    before_stage = {s: _ival("stage_duration_seconds_count", stage=s) for s in stages}
    before_ev = _ival("events_total", event="created", result="processed")
    before_dur = _ival("duration_seconds_count", source="minio_parse")
    before_items = {m: _ival("items_total", modality=m) for m in ("text", "image", "table")}
    before_pages = _ival("pages_total")

    await worker._process_message("1-1", RagEvent(event="created", document_id=PDF_ID))

    assert _ival("events_total", event="created", result="processed") == before_ev + 1
    assert _ival("duration_seconds_count", source="minio_parse") == before_dur + 1
    for s in stages:
        assert _ival("stage_duration_seconds_count", stage=s) > before_stage[s]
    for m, n in before_items.items():
        assert _ival("items_total", modality=m) == n + 1
    assert _ival("pages_total") == before_pages + 3
    assert _ival("in_flight") == 0


async def test_ingest_skipped_event(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    from ami_rag.queue.events import RagEvent

    worker = _ingest_worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    before = _ival("events_total", event="created", result="skipped")
    await worker._process_message("1-1", RagEvent(event="created", document_id="missing-doc"))
    assert _ival("events_total", event="created", result="skipped") == before + 1
    assert _ival("in_flight") == 0


async def test_ingest_parse_failure_retry_then_failed(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    from ami_rag.queue.events import RagEvent

    async def boom(file_path, output_dir=None, parse_method=None, **kwargs):
        raise RuntimeError("mineru crashed")

    fake_rag_anything.parse_document = boom
    worker = _ingest_worker(
        fake_rag_anything,
        fake_docs_repo,
        fake_state_repo,
        fake_queue,
        fake_asset_store,
        max_delivery=2,
    )
    doc_type = fake_docs_repo.docs[PDF_ID].get("document_type") or "unknown"
    before_fail = _ival("parse_failures_total", document_type=doc_type)
    before_retry = _ival("events_total", event="created", result="retry")
    before_failed = _ival("events_total", event="created", result="failed")
    event = RagEvent(event="created", document_id=PDF_ID)

    await worker._process_message("1-1", event)
    assert _ival("events_total", event="created", result="retry") == before_retry + 1
    assert _ival("events_total", event="created", result="failed") == before_failed

    await worker._process_message("1-1", event)
    assert _ival("events_total", event="created", result="failed") == before_failed + 1
    assert _ival("parse_failures_total", document_type=doc_type) == before_fail + 2
    assert _ival("in_flight") == 0


async def test_ingest_asset_upload_failure_counted(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    from ami_rag.queue.events import RagEvent

    def boom(doc_id, content_list):
        raise RuntimeError("minio down")

    fake_asset_store.upload_content_list_assets = boom
    worker = _ingest_worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )
    before = _ival("asset_upload_failures_total")
    await worker._process_message("1-1", RagEvent(event="created", document_id=PDF_ID))
    assert _ival("asset_upload_failures_total") == before + 1


async def test_ingest_gauges_updated_and_errors_swallowed(
    fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    worker = _ingest_worker(
        fake_rag_anything, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
    )

    async def stats():
        return 4, 7

    fake_queue.stats = stats
    fake_state_repo.counts = lambda: {"processed": 2}
    await worker._update_queue_gauges()
    await worker._update_document_gauges()
    assert _ival("stream_pending") == 4
    assert _ival("stream_lag") == 7
    assert _ival("documents", status="processed") == 2

    fake_state_repo.counts = lambda: {"failed": 1}
    await worker._update_document_gauges()
    assert _ival("documents", status="processed") == 0
    assert _ival("documents", status="failed") == 1

    async def bad_stats():
        raise RuntimeError("redis down")

    fake_queue.stats = bad_stats
    await worker._update_queue_gauges()  # must not raise


def test_truncate_documents_json_keeps_valid_json():
    payload = [{"text": "x" * 5000, "score": 0.9, "metadata": {}} for _ in range(10)]
    out = obs._truncate_documents_json(payload, max_chars=6000)
    assert len(out) <= 6000
    assert len(json.loads(out)) == 10
    assert obs._truncate_documents_json(payload, max_chars=0) == json.dumps(
        payload, ensure_ascii=False
    )


def test_normalize_otlp_traces_url():
    for raw in ("http://tempo:4318", "http://tempo:4318/", "http://tempo:4318/v1/traces"):
        assert obs.normalize_otlp_traces_url(raw) == "http://tempo:4318/v1/traces"


def test_dashboard_metrics_exist_in_code():
    path = (
        Path(__file__).resolve().parents[2]
        / "monitoring/dashboards/multimodal-rag-retrieval-dashboard.json"
    )
    dash = json.loads(path.read_text())
    assert dash["spec"]["title"] == "Multimodal RAG Retrieval"
    text = path.read_text()
    import re

    # Families (not samples): labeled metrics that were never observed export no sample yet.
    exported = set()
    for fam in REGISTRY.collect():
        exported.add(fam.name)
        exported.update(fam.name + sfx for sfx in ("_total", "_bucket", "_count", "_sum"))
    used = set(re.findall(r"multimodal_rag_(?:retrieval|ingest)_[a-z_]+", text))
    assert used, "dashboard uses no metrics"
    missing = {u for u in used if u not in exported}
    assert not missing, f"dashboard references unknown metrics: {missing}"


def test_alert_rule_metrics_exist_in_code():
    import re

    path = Path(__file__).resolve().parents[2] / "monitoring/helm/prometheusrule.yaml"
    text = path.read_text()
    exported = set()
    for fam in REGISTRY.collect():
        exported.add(fam.name)
        exported.update(fam.name + sfx for sfx in ("_total", "_bucket", "_count", "_sum"))
    used = set(re.findall(r"multimodal_rag_(?:retrieval|ingest)_[a-z_]+", text))
    assert used, "alert rules use no metrics"
    missing = {u for u in used if u not in exported}
    assert not missing, f"alert rules reference unknown metrics: {missing}"
