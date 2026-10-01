import json

import httpx
import pytest
from bson import ObjectId

from ami_rag.api import main as api_main
from ami_rag.api.resolver import DocResolver
from ami_rag.api.routes import admin as admin_routes
from ami_rag.api.routes import rag as rag_routes
from ami_rag.settings import get_settings

from .conftest import FakeRerank


@pytest.fixture
def app(fake_rag, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store):
    app = api_main.create_app()
    fake_resolver = DocResolver(
        docs_repo=fake_docs_repo,
        minio_endpoint="localhost:9000",
        minio_access_key="x",
        minio_secret_key="x",
        minio_bucket="ami-data-documents",
    )
    fake_resolver._presign = lambda object_name: None
    app.dependency_overrides[rag_routes.get_rag] = lambda: fake_rag
    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: FakeRerank()
    app.dependency_overrides[rag_routes.get_resolver] = lambda: fake_resolver
    app.dependency_overrides[rag_routes.get_asset_store] = lambda: fake_asset_store
    app.dependency_overrides[admin_routes.get_rag] = lambda: fake_rag
    app.dependency_overrides[admin_routes.get_queue] = lambda: fake_queue
    app.dependency_overrides[admin_routes.get_state_repo] = lambda: fake_state_repo
    app.dependency_overrides[admin_routes.get_docs_repo] = lambda: fake_docs_repo
    return app


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_v1_exact_contract_shape(client):
    resp = await client.post(
        "/v2/rag/", json={"messages": [{"role": "user", "content": "học phí"}]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"query", "documents"}
    assert body["query"] == "học phí"
    assert len(body["documents"]) == 4
    for doc in body["documents"]:
        assert set(doc.keys()) == {"text", "score", "metadata"}
        assert set(doc["metadata"].keys()) == {
            "source",
            "page",
            "document_id",
            "chunk_index",
            "global_id",
        }
        assert doc["score"] is not None
    assert [d["metadata"]["page"] for d in body["documents"]] == [None, None, 2, 5]


async def test_v1_rerank_scores_descending(client, app):
    resp = await client.post(
        "/v2/rag/", json={"messages": [{"role": "user", "content": "học phí"}], "top_k": 5}
    )
    docs = resp.json()["documents"]
    scores = [d["score"] for d in docs]
    assert scores == sorted(scores, reverse=True)


async def test_v1_document_id_resolved_from_file_path(client):
    resp = await client.post(
        "/v2/rag/", json={"messages": [{"role": "user", "content": "học phí"}]}
    )
    docs = resp.json()["documents"]
    assert docs[0]["metadata"]["document_id"] == "64b000000000000000000001"


async def test_v1_default_mode_is_mix_sent_to_lightrag(client, fake_rag):
    await client.post("/v2/rag/", json={"messages": [{"role": "user", "content": "học phí"}]})
    _, _, mode, kwargs = fake_rag.calls[0]
    assert mode == "mix"
    assert kwargs["enable_rerank"] is False
    assert "chunk_top_k" in kwargs


async def test_v2_extended_shape(client):
    payload = {
        "messages": [{"role": "user", "content": "học phí"}],
        "version": 2,
        "include_kg": True,
    }
    resp = await client.post("/v2/rag/", json=payload)
    body = resp.json()
    assert set(body.keys()) == {"query", "documents", "mode", "references", "meta"}
    assert body["mode"] == "mix"
    assert len(body["references"]) == 2
    assert body["meta"]["keywords"]["high_level"] == ["học phí"]
    assert "latency_ms" in body["meta"]
    doc = body["documents"][0]
    assert doc["reference_id"] == "1"
    assert doc["doc"]["document_id"] == "64b000000000000000000001"
    assert doc["doc"]["title"] == "Doc A"
    assert doc["doc"]["source_url"] is None


async def test_v2_entities_included(client):
    payload = {
        "messages": [{"role": "user", "content": "học phí"}],
        "version": 2,
        "include_kg": True,
    }
    resp = await client.post("/v2/rag/", json=payload)
    entities = resp.json()["documents"][0]["entities"]
    assert entities[0]["name"] == "Học viện Công nghệ PTIT"


async def test_v2_entities_excluded_by_default(client):
    payload = {"messages": [{"role": "user", "content": "học phí"}], "version": 2}
    resp = await client.post("/v2/rag/", json=payload)
    assert resp.json()["documents"][0]["entities"] is None


async def test_rerank_failure_degrades_to_unscored_chunks(client, app):
    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: FakeRerank(fail=True)
    resp = await client.post(
        "/v2/rag/", json={"messages": [{"role": "user", "content": "học phí"}]}
    )
    docs = resp.json()["documents"]
    assert len(docs) == 4
    assert all(d["score"] is None for d in docs)


async def test_v2_filter_by_organization(client):
    payload = {
        "messages": [{"role": "user", "content": "học phí"}],
        "version": 2,
        "filters": {"organization_unit_id": "64b000000000000000000002"},
    }
    resp = await client.post("/v2/rag/", json=payload)
    assert resp.status_code == 200


async def test_v2_filter_excludes_everything(client):
    payload = {
        "messages": [{"role": "user", "content": "học phí"}],
        "version": 2,
        "filters": {"organization_unit_id": "org-khac"},
    }
    resp = await client.post("/v2/rag/", json=payload)
    assert resp.json()["documents"] == []


async def test_v2_custom_mode_forwarded(client, fake_rag):
    payload = {
        "messages": [{"role": "user", "content": "học phí"}],
        "version": 2,
        "mode": "naive",
    }
    await client.post("/v2/rag/", json=payload)
    _, _, mode, _ = fake_rag.calls[0]
    assert mode == "naive"


async def test_auth_required_when_key_configured(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "RAG_API_KEY", "secret-key")
    resp = await client.post("/v2/rag/", json={"messages": [{"role": "user", "content": "q"}]})
    assert resp.status_code == 401
    resp = await client.post(
        "/v2/rag/",
        json={"messages": [{"role": "user", "content": "q"}]},
        headers={"Authorization": "Bearer secret-key"},
    )
    assert resp.status_code == 200


async def test_v2_modality_fields(client):
    payload = {"messages": [{"role": "user", "content": "học phí"}], "version": 2}
    docs = (await client.post("/v2/rag/", json=payload)).json()["documents"]
    by_modality = {d["modality"]: d for d in docs}
    assert set(by_modality) == {"text", "image", "table"}
    text, image, table = by_modality["text"], by_modality["image"], by_modality["table"]
    assert text["artifact_url"] is None and text["table_body"] is None and text["page"] is None
    assert (
        image["artifact_url"] == "https://minio/presigned/rag-assets/64b000000000000000000001/i.png"
    )
    assert image["page"] == 2
    assert table["table_body"].startswith("| ngành")
    assert table["page"] == 5
    assert table["caption"] == "Học phí"


async def test_v2_filter_modality_before_rerank(client, app, fake_rag):
    rerank = FakeRerank()
    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: rerank
    payload = {
        "messages": [{"role": "user", "content": "học phí"}],
        "version": 2,
        "filters": {"modality": ["table"]},
    }
    docs = (await client.post("/v2/rag/", json=payload)).json()["documents"]
    assert [d["modality"] for d in docs] == ["table"]
    assert rerank.calls[0][1] == 1
    assert len(fake_rag.query_result["data"]["chunks"]) == 4


async def test_v2_presign_failure_gives_null_artifact_url(client, fake_asset_store):
    from ami_rag.observability import PRESIGN_FAILURES_TOTAL

    fake_asset_store.fail_presign = True
    before = PRESIGN_FAILURES_TOTAL._value.get()
    payload = {"messages": [{"role": "user", "content": "học phí"}], "version": 2}
    docs = (await client.post("/v2/rag/", json=payload)).json()["documents"]
    assert all(d["artifact_url"] is None for d in docs)
    assert PRESIGN_FAILURES_TOTAL._value.get() == before + 1


async def test_stream_includes_modality(client):
    resp = await client.post(
        "/v2/rag/stream",
        json={"messages": [{"role": "user", "content": "học phí"}], "version": 2},
    )
    lines = [json.loads(line) for line in resp.text.splitlines() if line]
    docs = [doc for line in lines for doc in line.get("documents", [])]
    assert {d["modality"] for d in docs} == {"text", "image", "table"}
    assert any(d["artifact_url"] for d in docs)


async def test_stream_emits_ndjson(client):
    resp = await client.post(
        "/v2/rag/stream", json={"messages": [{"role": "user", "content": "học phí"}]}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    body = resp.text
    assert '"status": "retrieving"}' in body
    assert '"documents"' in body
    assert '"status": "done"}' in body
    assert '"source"' not in body.split('"documents"')[1]


async def test_admin_pipeline_status(client):
    resp = await client.get("/admin/pipeline_status")
    assert resp.status_code == 200
    body = resp.json()
    assert "doc_status_counts" in body
    assert "queue_pending" in body


async def test_admin_reprocess_failed(client, fake_state_repo, fake_queue):
    fake_state_repo.mark_failed("64b000000000000000000001", "boom")
    resp = await client.post("/admin/reprocess_failed")
    assert resp.status_code == 200
    assert resp.json()["requeued"] == 1
    assert len(fake_queue.published) == 1


async def test_admin_reindex_by_ids(client, fake_docs_repo, fake_queue):
    resp = await client.post(
        "/admin/reindex",
        json={"document_ids": ["64b000000000000000000001", "64b000000000000000000003"]},
    )
    assert resp.json()["published"] == 2
    assert len(fake_queue.published) == 2


async def test_admin_requires_api_key(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "RAG_API_KEY", "secret-key")
    resp = await client.get("/admin/pipeline_status")
    assert resp.status_code == 401
    resp = await client.get("/admin/pipeline_status", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401
    resp = await client.get(
        "/admin/pipeline_status", headers={"Authorization": "Bearer secret-key"}
    )
    assert resp.status_code == 200


async def test_admin_document_status(client, fake_state_repo):
    fake_state_repo.mark_processed(
        "doc1",
        "h",
        source="minio_parse",
        parser="mineru",
        counts={"text": 1, "table": 1},
        page_count=3,
        meta={
            "organization_unit_id": ObjectId("64b000000000000000000002"),
            "owner_id": "sub-1",
            "document_type": "pdf",
            "title": "Sổ tay",
        },
    )
    resp = await client.get("/admin/documents/doc1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == "doc1"
    assert body["status"] == "processed"
    assert body["source"] == "minio_parse"
    assert body["counts"] == {"text": 1, "table": 1}
    assert body["page_count"] == 3
    assert body["parser"] == "mineru"
    assert body["error"] is None
    assert body["organization_unit_id"] == "64b000000000000000000002"
    assert body["owner_id"] == "sub-1"
    assert body["document_type"] == "pdf"
    assert body["title"] == "Sổ tay"
    assert set(body) == {
        "document_id",
        "status",
        "source",
        "counts",
        "page_count",
        "parser",
        "document_type",
        "title",
        "organization_unit_id",
        "owner_id",
        "error",
        "updated_at",
    }


async def test_admin_document_status_not_found(client):
    resp = await client.get("/admin/documents/missing")
    assert resp.status_code == 404


async def test_admin_document_content(client, fake_asset_store):
    fake_asset_store.content_lists["doc1"] = [
        {"type": "text", "text": "Giới thiệu", "page_idx": 0},
        {
            "type": "image",
            "asset_key": "rag-assets/doc1/i.png",
            "image_caption": ["Sơ đồ"],
            "page_idx": 1,
        },
        {
            "type": "table",
            "table_body": "| a | b |",
            "table_caption": ["Bảng 1"],
            "asset_key": "rag-assets/doc1/t.png",
            "page_idx": 2,
        },
    ]
    resp = await client.get("/admin/documents/doc1/content")
    assert resp.status_code == 200
    body = resp.json()
    assert [b["type"] for b in body["blocks"]] == ["text", "image", "table"]
    assert body["blocks"][1]["asset_url"] == "https://minio/presigned/rag-assets/doc1/i.png"
    assert len(body["tables"]) == 1
    assert body["tables"][0]["table_body"] == "| a | b |"
    assert body["markdown"] == (
        "Giới thiệu\n\n![Sơ đồ](https://minio/presigned/rag-assets/doc1/i.png)\n\n*Bảng 1*\n\n| a | b |"
    )


async def test_admin_document_content_not_found(client):
    resp = await client.get("/admin/documents/missing/content")
    assert resp.status_code == 404


def test_blocks_to_markdown_skips_image_without_url_and_renders_equation():
    blocks = admin_routes.build_content_blocks(
        [
            {"type": "image", "image_caption": ["x"]},
            {"type": "equation", "text": "E=mc^2", "page_idx": 0},
        ]
    )
    assert admin_routes.blocks_to_markdown(blocks) == "$$E=mc^2$$"


async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
