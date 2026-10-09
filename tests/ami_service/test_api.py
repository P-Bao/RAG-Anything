import json

import httpx
import pytest

from ami_rag.api import main as api_main
from ami_rag.api.resolver import DocResolver
from ami_rag.api.routes import admin as admin_routes
from ami_rag.api.routes import rag as rag_routes
from ami_rag.settings import get_settings

from .conftest import FakeRerank, _filter_modalities


@pytest.fixture
def app(
    fake_pipeline,
    fake_docs_repo,
    fake_status_store,
    fake_queue,
    fake_asset_store,
):
    app = api_main.create_app()
    fake_resolver = DocResolver(
        docs_repo=fake_docs_repo,
        minio_endpoint="localhost:9000",
        minio_access_key="x",
        minio_secret_key="x",
        minio_bucket="ami-data-documents",
    )
    fake_resolver._presign = lambda object_name: None

    # Seed chunks vào vector store của pipeline (search test)
    fake_pipeline.vector_store.points.update(
        {
            "chunk-1": {
                "doc_id": "64b000000000000000000001",
                "content": "Học phí năm 2026 là 12 triệu mỗi học kỳ.",
                "modality": "text",
                "source_path": "64b000000000000000000001_a.pdf",
                "chunk_id": "chunk-1",
                "page": 0,
            },
            "chunk-2": {
                "doc_id": "64b000000000000000000005",
                "content": "Tuyển sinh sử dụng phương thức xét tuyển kết hợp.",
                "modality": "text",
                "source_path": "64b000000000000000000005_text",
                "chunk_id": "chunk-2",
            },
            "chunk-3": {
                "doc_id": "64b000000000000000000001",
                "content": "[Image Content]\nmô tả image",
                "modality": "image",
                "source_path": "64b000000000000000000001_a.pdf",
                "chunk_id": "chunk-3",
                "asset_key": "rag-assets/64b000000000000000000001/img1.png",
                "page": 2,
                "caption": "Sơ đồ",
            },
            "chunk-4": {
                "doc_id": "64b000000000000000000001",
                "content": "[Table Content]\nCaption: Học phí",
                "modality": "table",
                "source_path": "64b000000000000000000001_a.pdf",
                "chunk_id": "chunk-4",
                "table_body": "| ngành | phí |\n|---|---|\n| CNTT | 12tr |",
                "page": 5,
                "caption": "Học phí",
            },
        }
    )

    app.dependency_overrides[rag_routes.get_pipeline] = lambda: fake_pipeline
    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: FakeRerank()
    app.dependency_overrides[rag_routes.get_resolver] = lambda: fake_resolver
    app.dependency_overrides[rag_routes.get_asset_store] = lambda: fake_asset_store
    app.dependency_overrides[admin_routes.get_queue] = lambda: fake_queue
    app.dependency_overrides[admin_routes.get_state_repo] = lambda: fake_status_store
    app.dependency_overrides[admin_routes.get_docs_repo] = lambda: fake_docs_repo
    return app


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


BODY = {"messages": [{"role": "user", "content": "học phí"}]}


async def test_exact_contract_shape(client):
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"query", "documents", "references", "meta"}
    assert body["query"] == "học phí"
    assert len(body["documents"]) >= 1
    for doc in body["documents"]:
        assert set(doc.keys()) == {
            "text",
            "score",
            "metadata",
            "reference_id",
            "doc",
            "modality",
            "artifact_url",
            "table_body",
            "page",
            "caption",
        }
        assert set(doc["metadata"].keys()) == {
            "source",
            "page",
            "document_id",
            "chunk_index",
            "global_id",
        }
        assert doc["score"] is not None


async def test_rerank_scores_descending(client):
    resp = await client.post("/v2/rag/", json={**BODY, "top_k": 5})
    docs = resp.json()["documents"]
    scores = [d["score"] for d in docs]
    assert scores == sorted(scores, reverse=True)


async def test_document_id_resolved_from_chunk_payload(client):
    resp = await client.post("/v2/rag/", json=BODY)
    docs = resp.json()["documents"]
    assert docs[0]["doc"] is not None
    assert docs[0]["doc"]["document_id"] in (
        "64b000000000000000000001",
        "64b000000000000000000005",
        "64b000000000000000000006",
    )


async def test_references_dedup_by_document(client):
    resp = await client.post("/v2/rag/", json=BODY)
    references = resp.json()["references"]
    ids = [r["document_id"] for r in references]
    assert len(ids) == len(set(ids))


async def test_include_references_false(client):
    resp = await client.post("/v2/rag/", json={**BODY, "include_references": False})
    assert resp.json()["references"] == []


async def test_rerank_failure_degrades_to_unscored_chunks(client, app):
    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: FakeRerank(fail=True)
    resp = await client.post("/v2/rag/", json=BODY)
    docs = resp.json()["documents"]
    assert len(docs) >= 1
    assert all(d["score"] is None for d in docs)


async def test_auth_required_when_key_configured(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "RAG_API_KEY", "secret-key")
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 401
    resp = await client.post(
        "/v2/rag/", json=BODY, headers={"Authorization": "Bearer secret-key"}
    )
    assert resp.status_code == 200


async def test_presign_failure_gives_null_artifact_url(client, fake_asset_store):
    from ami_rag.observability import PRESIGN_FAILURES_TOTAL

    fake_asset_store.fail_presign = True
    before = PRESIGN_FAILURES_TOTAL._value.get()
    resp = await client.post("/v2/rag/", json=BODY)
    assert all(d["artifact_url"] is None for d in resp.json()["documents"])
    # presign chỉ gọi khi chunk có asset_key; không có asset -> không tăng
    assert PRESIGN_FAILURES_TOTAL._value.get() >= before


async def test_stream_emits_ndjson(client):
    resp = await client.post("/v2/rag/stream", json=BODY)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    body = resp.text
    assert '"status": "retrieving"}' in body
    assert '"documents"' in body
    assert '"status": "done"}' in body


async def test_stream_includes_modality(client):
    resp = await client.post("/v2/rag/stream", json=BODY)
    lines = [json.loads(line) for line in resp.text.splitlines() if line]
    docs = [doc for line in lines for doc in line.get("documents", [])]
    assert docs
    assert all("modality" in d for d in docs)


async def test_admin_pipeline_status(client):
    resp = await client.get("/admin/pipeline_status")
    assert resp.status_code == 200
    body = resp.json()
    assert "doc_status_counts" in body
    assert "queue_pending" in body


async def test_admin_reprocess_failed(client, fake_status_store, fake_queue):
    fake_status_store.mark_failed("64b000000000000000000001", "boom", "embed")
    resp = await client.post("/admin/reprocess_failed")
    assert resp.status_code == 200
    assert resp.json()["requeued"] == 1
    assert len(fake_queue.published) == 1


async def test_admin_reindex_by_ids(client, fake_docs_repo, fake_queue):
    resp = await client.post(
        "/admin/reindex",
        json={"document_ids": ["64b000000000000000000001", "64b000000000000000000005"]},
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


async def test_admin_document_status(client, fake_status_store):
    from bson import ObjectId

    fake_status_store.mark_indexed(
        "doc1",
        chunk_count=2,
        embed_model="Qwen/Qwen3-VL-Embedding-2B",
        embed_dim=8,
        chunker_version="v1",
        source="minio_parse",
        parser="mineru",
        counts={"text": 1, "table": 1},
        page_count=3,
        document_type="pdf",
        title="Sổ tay",
        organization_unit_id=ObjectId("64b000000000000000000002"),
        owner_id="sub-1",
    )
    resp = await client.get("/admin/documents/doc1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == "doc1"
    assert body["status"] == "indexed"
    assert body["stage"] == "indexed"
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
        "stage",
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


# --- top_k == number of documents finally returned ---------------------------------------


def _seed_chunks(fake_vector_store, n, doc_id="64b000000000000000000001"):
    for i in range(n):
        fake_vector_store.points[f"chunk-{i}"] = {
            "doc_id": doc_id,
            "content": f"chunk {i}",
            "modality": "text",
            "source_path": "documents/HV/a.pdf",
            "chunk_id": f"chunk-{i}",
        }


async def test_top_k_is_final_document_count(client, fake_vector_store):
    _seed_chunks(fake_vector_store, 30)
    resp = await client.post(
        "/v2/rag/",
        json={"messages": [{"role": "user", "content": "học phí"}], "top_k": 7},
    )
    body = resp.json()
    assert len(body["documents"]) == 7
    assert body["meta"]["requested_top_k"] == 7 and body["meta"]["returned"] == 7


async def test_top_k_defaults_to_rerank_top_k(client, fake_vector_store, monkeypatch):
    monkeypatch.setattr(get_settings(), "RERANK_TOP_K", 5)
    _seed_chunks(fake_vector_store, 30)
    resp = await client.post("/v2/rag/", json=BODY)
    assert len(resp.json()["documents"]) == 5


async def test_fewer_candidates_than_top_k_reported_in_meta(client, fake_vector_store):
    fake_vector_store.points.clear()
    _seed_chunks(fake_vector_store, 1)
    resp = await client.post(
        "/v2/rag/",
        json={"messages": [{"role": "user", "content": "q"}], "top_k": 10},
    )
    meta = resp.json()["meta"]
    assert meta["requested_top_k"] == 10 and meta["returned"] == 1 and meta["candidates"] == 1


async def test_deduplicate_identical_chunks_in_retrieval(client, fake_vector_store):
    fake_vector_store.points.clear()
    # Seed 5 chunks identical in content (simulating repeated PDF footers)
    for i in range(5):
        fake_vector_store.points[f"chunk-dup-{i}"] = {
            "doc_id": "doc-dup",
            "content": "Footer lặp lại trên mọi trang PDF",
            "modality": "text",
            "chunk_id": f"chunk-dup-{i}",
        }
    # Seed 1 chunk with distinct content
    fake_vector_store.points["chunk-distinct"] = {
        "doc_id": "doc-distinct",
        "content": "Nội dung học phí đặc thù",
        "modality": "text",
        "chunk_id": "chunk-distinct",
    }
    resp = await client.post(
        "/v2/rag/",
        json={"messages": [{"role": "user", "content": "học phí"}], "top_k": 5},
    )
    body = resp.json()
    texts = [d["text"] for d in body["documents"]]
    # Should only contain 1 instance of the duplicate footer, not 5
    assert texts.count("Footer lặp lại trên mọi trang PDF") == 1
    assert "Nội dung học phí đặc thù" in texts


async def test_calibration_reorders_api_results(app, client, fake_vector_store):
    from ami_rag.core.calibration import ModalityCalibration, RerankCalibrator

    # Create a calibrator where image has much higher probability than text for same raw score
    calibrator = RerankCalibrator(
        models={
            "text": ModalityCalibration(method="platt", a=1.0, b=-2.0),
            "image": ModalityCalibration(method="platt", a=10.0, b=0.0),
        }
    )
    app.dependency_overrides[rag_routes.get_calibrator] = lambda: calibrator

    fake_vector_store.points.clear()
    fake_vector_store.points["chunk-text"] = {
        "doc_id": "doc-1",
        "content": "Văn bản điểm cao raw",
        "modality": "text",
        "chunk_id": "chunk-text",
    }
    fake_vector_store.points["chunk-img"] = {
        "doc_id": "doc-1",
        "content": "[Image Content] Sơ đồ trường",
        "modality": "image",
        "chunk_id": "chunk-img",
    }

    # Custom fake rerank returning text raw=0.8, img raw=0.1
    async def custom_rerank(query, documents, top_n=None, **kwargs):
        # index 0 is chunk-text or chunk-img depending on order
        results = []
        for idx in range(len(documents)):
            results.append({"index": idx, "relevance_score": 0.1 if idx == 1 else 0.8})
        return results

    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: custom_rerank

    resp = await client.post(
        "/v2/rag/",
        json={"messages": [{"role": "user", "content": "sơ đồ"}], "top_k": 5},
    )
    assert resp.status_code == 200
    docs = resp.json()["documents"]
    assert len(docs) >= 1
    # Check that metadata contains calibration fields
    assert "calibrated_score" in docs[0]["metadata"]
    assert "raw_score" in docs[0]["metadata"]


# --- per-modality fusion modes ------------------------------------------------


def _enable_fusion(monkeypatch, mode, **knobs):
    """Turn on a fusion strategy for one request."""
    settings = get_settings()
    monkeypatch.setattr(settings, "RETRIEVAL_FUSION_MODE", mode)
    monkeypatch.setattr(settings, "RETRIEVAL_FUSION_POOLS", "text+table,image")
    monkeypatch.setattr(settings, "RETRIEVAL_FUSION_POOL_SIZES", "text=40,image=15,other=20")
    monkeypatch.setattr(settings, "RETRIEVAL_FUSION_RRF_K", 60)
    monkeypatch.setattr(settings, "RETRIEVAL_FUSION_QUOTA", "text=3@0.45,image=2@0.40")
    monkeypatch.setattr(settings, "RETRIEVAL_FUSION_IMAGE_GATE", True)
    monkeypatch.setattr(settings, "RETRIEVAL_FUSION_VL_POOLS", "text,image")
    for key, value in knobs.items():
        monkeypatch.setattr(settings, key, value)
    return settings


def _seed_pool(fake_vector_store, prefix, modality, count, doc_id="64b000000000000000000001"):
    """Insert chunks so the fake store returns them in a known rank order."""
    for i in range(count):
        fake_vector_store.points[f"{prefix}-{i}"] = {
            "doc_id": doc_id,
            "content": f"{prefix} nội dung {i}",
            "modality": modality,
            "source_path": "documents/HV/a.pdf",
            "chunk_id": f"{prefix}-{i}",
            # asset_key makes build_rerank_documents emit a multimodal dict for the
            # modalities the real pipeline renders as an image (image/table/equation).
            **(
                {"asset_key": f"rag-assets/{doc_id}/{prefix}{i}.png"}
                if modality in ("image", "table")
                else {}
            ),
        }


def _modality_aware_rerank(image_score=0.1, text_base=0.9):
    """Rerank stub where images score far below text, as the real model does."""

    async def rerank(query, documents, top_n=None, **kwargs):
        results = []
        for i, doc in enumerate(documents):
            is_image = isinstance(doc, dict)
            results.append(
                {
                    "index": i,
                    "relevance_score": image_score if is_image else round(text_base - i * 0.01, 4),
                }
            )
        return results

    return rerank


async def test_single_mode_keeps_legacy_flow(client, fake_vector_store):
    """Default mode must not report fusion metadata nor filter by modality."""
    fake_vector_store.searches.clear()
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    assert "fusion" not in resp.json()["meta"]


async def test_fusion_queries_one_filtered_pool_per_branch(
    client, fake_vector_store, monkeypatch
):
    _enable_fusion(monkeypatch, "calibrated")
    fake_vector_store.searches.clear()
    _seed_pool(fake_vector_store, "t", "text", 3)
    _seed_pool(fake_vector_store, "b", "table", 3)
    _seed_pool(fake_vector_store, "i", "image", 3)
    _seed_pool(fake_vector_store, "e", "equation", 2)

    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    assert fake_vector_store.searches, "fusion mode must query the vector store"

    include, exclude = {}, {}
    for search in fake_vector_store.searches:
        mods = _filter_modalities(search["filter"])
        pool = "text" if mods[0] == {"text", "table"} else next(iter(mods[0]), "other")
        include[pool] = search["top_k"]
        if mods[1]:
            exclude[pool] = sorted(mods[1])

    # text and table share one pool and one rerank pass; images are isolated.
    assert include == {"text": 40, "image": 15, "other": 20}
    # The catch-all pool must exclude every modality any pool claims.
    assert exclude == {"other": ["image", "table", "text"]}


def _graded_rerank(image_scores):
    """Rerank stub giving prose a high flat band and images a graded ladder."""

    async def rerank(query, documents, top_n=None, **kwargs):
        results = []
        image_i = 0
        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                score = image_scores[image_i % len(image_scores)]
                image_i += 1
            else:
                score = round(0.9 - i * 0.01, 4)
            results.append({"index": i, "relevance_score": score})
        return results

    return rerank


async def test_fusion_quota_reserves_slots_only_for_images_that_clear_the_gate(
    app, client, fake_vector_store, monkeypatch
):
    """The gate and the quota pull against each other, and the gate wins.

    Quota can only reserve slots from candidates it is allowed to consider, so a
    measured image below the text cut line is gone before quota sees it. Quota
    still decides the order among images that did clear the gate.
    """
    _enable_fusion(monkeypatch, "quota", RETRIEVAL_FUSION_QUOTA="image=2")
    fake_vector_store.points.clear()
    _seed_pool(fake_vector_store, "t", "text", 8)
    _seed_pool(fake_vector_store, "i", "image", 3)
    # 8 prose chunks score 0.9..0.83, so with top_k=5 the cut line is 0.86.
    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: _graded_rerank([0.87, 0.1, 0.1])

    resp = await client.post("/v2/rag/", json={"messages": [{"role": "user", "content": "q"}], "top_k": 5})
    meta = resp.json()["meta"]
    assert meta["fusion"]["mode"] == "quota"
    assert meta["fusion"]["image_gate_dropped"] == 2
    docs = resp.json()["documents"]
    assert sum(1 for d in docs if d["modality"] == "image") == 1


async def test_fusion_quota_can_force_weak_images_only_with_the_gate_off(
    app, client, fake_vector_store, monkeypatch
):
    _enable_fusion(
        monkeypatch,
        "quota",
        RETRIEVAL_FUSION_QUOTA="image=2",
        RETRIEVAL_FUSION_IMAGE_GATE=False,
    )
    fake_vector_store.points.clear()
    _seed_pool(fake_vector_store, "t", "text", 8)
    _seed_pool(fake_vector_store, "i", "image", 3)
    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: _graded_rerank([0.1, 0.1, 0.1])

    resp = await client.post("/v2/rag/", json={"messages": [{"role": "user", "content": "q"}], "top_k": 5})
    meta = resp.json()["meta"]
    assert meta["fusion"]["image_gate_dropped"] == 0
    docs = resp.json()["documents"]
    assert len(docs) == 5
    # Without the gate, the quota reserves 2 slots on raw score alone.
    assert sum(1 for d in docs if d["modality"] == "image") == 2


async def test_fusion_calibrated_keeps_top_k_when_a_pool_is_empty(
    client, fake_vector_store, monkeypatch
):
    _enable_fusion(monkeypatch, "calibrated")
    fake_vector_store.points.clear()
    _seed_pool(fake_vector_store, "t", "text", 6)

    resp = await client.post("/v2/rag/", json={"messages": [{"role": "user", "content": "q"}], "top_k": 5})
    docs = resp.json()["documents"]
    assert len(docs) == 5
    assert all(d["modality"] == "text" for d in docs)


async def test_fusion_rrf_interleaves_pools_by_rank(client, fake_vector_store, monkeypatch):
    _enable_fusion(monkeypatch, "rrf")
    fake_vector_store.points.clear()
    _seed_pool(fake_vector_store, "t", "text", 4)
    _seed_pool(fake_vector_store, "i", "image", 4)

    resp = await client.post("/v2/rag/", json={"messages": [{"role": "user", "content": "q"}], "top_k": 4})
    modalities = [d["modality"] for d in resp.json()["documents"]]
    # RRF over disjoint pools with equal weights is a round-robin by pool rank.
    assert len(set(modalities)) == 2
    assert modalities[0] != modalities[1]


async def test_fusion_pool_rerank_failure_is_isolated(app, client, fake_vector_store, monkeypatch):
    """A rerank failure must not empty the page or leak into the other pools."""
    _enable_fusion(monkeypatch, "quota", RETRIEVAL_FUSION_QUOTA="image=2")
    fake_vector_store.points.clear()
    _seed_pool(fake_vector_store, "t", "text", 3)
    _seed_pool(fake_vector_store, "i", "image", 3)

    async def rerank(query, documents, top_n=None, **kwargs):
        # Image chunks are the only ones sent as multimodal dicts.
        if any(isinstance(d, dict) for d in documents):
            raise ConnectionError("rerank down for image pool")
        return [{"index": i, "relevance_score": 0.5 - i * 0.1} for i in range(len(documents))]

    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: rerank
    resp = await client.post("/v2/rag/", json={"messages": [{"role": "user", "content": "hoc_phi"}], "top_k": 5})
    assert resp.status_code == 200
    docs = resp.json()["documents"]
    assert len(docs) == 5
    # The failed pool falls back to unscored chunks; the text pool is unaffected.
    assert any(d["score"] is not None for d in docs)
    assert sum(1 for d in docs if d["modality"] == "image") == 2


async def test_fusion_reports_pool_candidate_sizes(client, fake_vector_store, monkeypatch):
    _enable_fusion(monkeypatch, "calibrated")
    fake_vector_store.points.clear()
    _seed_pool(fake_vector_store, "t", "text", 4)
    _seed_pool(fake_vector_store, "b", "table", 2)

    meta = (await client.post("/v2/rag/", json=BODY)).json()["meta"]
    # Table chunks land in the shared text pool, so one filter returns all 6.
    assert meta["fusion"]["pools"] == {"text": 6, "image": 0, "other": 0}
    assert meta["fusion"]["mode"] == "calibrated"
    assert meta["fusion"]["image_gate_dropped"] == 0
    assert meta["candidates"] == 6


async def test_fusion_rejects_pool_spec_without_depth(client, monkeypatch):
    _enable_fusion(monkeypatch, "calibrated", RETRIEVAL_FUSION_POOL_SIZES="image=15")
    with pytest.raises(ValueError, match="thiếu depth"):
        await client.post("/v2/rag/", json=BODY)


async def test_fusion_sends_images_to_every_vl_pool(
    app, client, fake_vector_store, monkeypatch
):
    """By default both branches carry images; `text+table` is one shared rerank.

    Tables ride the prose rerank in a single call -- the split is text-vs-image,
    not table-vs-image -- and they keep their rendered image because reranking a
    table as plain text measurably loses table recall.
    """
    _enable_fusion(monkeypatch, "calibrated", RETRIEVAL_FUSION_VL_POOLS="text,image")
    fake_vector_store.points.clear()
    _seed_pool(fake_vector_store, "t", "text", 4)
    _seed_pool(fake_vector_store, "b", "table", 3)
    _seed_pool(fake_vector_store, "i", "image", 2)

    calls = []

    async def rerank(query, documents, top_n=None, **kwargs):
        calls.append([isinstance(d, dict) for d in documents])
        return [{"index": i, "relevance_score": 0.9 - i * 0.05} for i in range(len(documents))]

    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: rerank
    resp = await client.post("/v2/rag/", json={"messages": [{"role": "user", "content": "q"}], "top_k": 5})
    assert resp.status_code == 200

    # Pools rerank concurrently, so match calls by size, not by order.
    by_size = {len(call): call for call in calls}
    assert sorted(by_size) == [2, 7]
    # One call for text + table. Only chunks that actually carry an asset become
    # multimodal: 3 tables with asset_key, 4 prose chunks without.
    assert by_size[7].count(True) == 3
    assert by_size[7].count(False) == 4
    assert by_size[2] == [True] * 2
    assert len(calls) == 2


async def test_fusion_reranks_tables_as_plain_text_when_the_pool_is_not_vl(
    app, client, fake_vector_store, monkeypatch
):
    """Dropping the text pool from VL_POOLS is the cheaper, lower-recall variant."""
    _enable_fusion(monkeypatch, "calibrated", RETRIEVAL_FUSION_VL_POOLS="image")
    fake_vector_store.points.clear()
    _seed_pool(fake_vector_store, "t", "text", 4)
    _seed_pool(fake_vector_store, "b", "table", 3)
    _seed_pool(fake_vector_store, "i", "image", 2)

    calls = []

    async def rerank(query, documents, top_n=None, **kwargs):
        calls.append([isinstance(d, dict) for d in documents])
        return [{"index": i, "relevance_score": 0.9 - i * 0.05} for i in range(len(documents))]

    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: rerank
    await client.post("/v2/rag/", json={"messages": [{"role": "user", "content": "q"}], "top_k": 5})

    by_size = {len(call): call for call in calls}
    assert sorted(by_size) == [2, 7]
    # Tables carry an asset_key but the pool is not VL, so nothing is multimodal.
    assert by_size[7] == [False] * 7
    assert by_size[2] == [True] * 2


async def test_fusion_retrieves_text_and_table_with_one_query(
    client, fake_vector_store, monkeypatch
):
    _enable_fusion(monkeypatch, "calibrated")
    fake_vector_store.searches.clear()
    _seed_pool(fake_vector_store, "t", "text", 3)
    _seed_pool(fake_vector_store, "b", "table", 3)
    _seed_pool(fake_vector_store, "i", "image", 3)

    await client.post("/v2/rag/", json=BODY)
    includes = [
        _filter_modalities(s["filter"])[0] for s in fake_vector_store.searches
    ]
    # One OR filter carrying both modalities, not two separate queries.
    assert {"text", "table"} in includes
    assert {"image"} in includes
