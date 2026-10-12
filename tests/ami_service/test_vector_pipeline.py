"""Test VectorPipeline (fake embedder/vector_store/asset_store/DocsStatusStore)."""

import asyncio
import base64
from pathlib import Path

import pytest

from ami_rag.core.vector_pipeline import (
    ArtifactPaths,
    VectorPipeline,
    chunk_text,
)
from ami_rag.settings import Settings, resolve_image_describe_mode
from ami_rag.storage.doc_status import STATUS_INDEXED
from raganything.backend_data_extract import BackendDataParseError
from tests.ami_service.conftest import (
    FakeEmbedder,
    FakeImageWorker,
    FakeParser,
    FakeVectorStore,
)

TEXT_ID = "64b000000000000000000005"
PDF_ID = "64b000000000000000000001"
CRAWL_ID = "64b000000000000000000007"

CRAWL_URLS = [
    "https://ptit.edu.vn/wp-content/uploads/2026/05/NL_04608-scaled.jpg",
    "https://ptit.edu.vn/wp-content/uploads/2026/05/NL_04739-scaled.jpg",
]
CRAWL_CONTENT = (
    "Bài crawl PTIT.\n\n"
    "Nội dung chính.\n\n"
    "**Hình ảnh:**\n\n"
    f"- {CRAWL_URLS[0]}\n"
    f"- {CRAWL_URLS[1]}\n"
)


def _add_crawl_doc(fake_docs_repo, fake_status_store):
    fake_docs_repo.docs[CRAWL_ID] = {
        "_id": CRAWL_ID,
        "title": "Crawl PTIT",
        "content": CRAWL_CONTENT,
        "document_type": "news",
        "original_file_name": None,
        "file_path": None,
        "organization_unit_id": "64b000000000000000000002",
        "metadata": {"source_url": "https://ptit.edu.vn/tin/le-tot-nghiep"},
    }
    fake_status_store.ensure_pending(CRAWL_ID)


@pytest.fixture
def pipeline(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store, fake_docs_repo, fake_modal_processors
):
    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        EMBED_DIM=8,
        CHUNKER_VERSION="v1",
        CHUNK_SIZE=64,
        CHUNK_OVERLAP=8,
        PARSER="mineru",
        PARSE_METHOD="auto",
        MINERU_BACKEND="pipeline",
        PARSE_TEXT_SOURCE="mineru",
    )
    return VectorPipeline(
        settings,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=fake_docs_repo,
        parser=FakeParser(),
        modal_processors=fake_modal_processors,
    )


@pytest.fixture(autouse=True)
def _seed_pending(fake_status_store):
    fake_status_store.ensure_pending(TEXT_ID)
    fake_status_store.ensure_pending(PDF_ID)


async def test_full_run_mongo_text(pipeline, fake_status_store, fake_vector_store, fake_asset_store):
    outcome = await pipeline.run(TEXT_ID)

    assert outcome.ok and outcome.stage == "indexed"
    row = fake_status_store.get(TEXT_ID)
    assert row["status"] == STATUS_INDEXED
    assert row["chunk_count"] == outcome.chunk_count
    assert row["embed_model"] == "Qwen/Qwen3-VL-Embedding-2B"
    assert row["embed_dim"] == 8
    assert row["chunker_version"] == "v1"
    assert fake_vector_store.points
    assert all(p["doc_id"] == TEXT_ID for p in fake_vector_store.points.values())
    # artifacts ở MinIO
    assert fake_asset_store.load_content_list(TEXT_ID) is not None
    chunks_key = ArtifactPaths.chunks_key(fake_asset_store, TEXT_ID)
    assert fake_asset_store.json_objects.get(chunks_key) is not None
    # parse: text doc không fetch file
    assert fake_asset_store.fetched == []


async def test_full_run_describes_multimodal(
    pipeline, fake_status_store, fake_modal_processors, fake_asset_store
):
    await pipeline.run(PDF_ID)

    all_calls = []
    for proc in fake_modal_processors.values():
        all_calls.extend(proc.calls)
    assert all_calls, "describe stage phải gọi modal processor cho item multimodal"
    row = fake_status_store.get(PDF_ID)
    assert row["status"] == STATUS_INDEXED
    # descriptions.json đã lưu
    desc_key = ArtifactPaths.descriptions_key(fake_asset_store, PDF_ID)
    assert isinstance(fake_asset_store.json_objects.get(desc_key), list)


async def test_run_is_idempotent_no_duplicate_vectors(
    pipeline, fake_vector_store, fake_status_store
):
    await pipeline.run(TEXT_ID)
    count_after_first = len(fake_vector_store.points)
    await pipeline.run(TEXT_ID)  # stage == indexed -> early return, không upsert lại
    assert len(fake_vector_store.points) == count_after_first


async def test_resume_from_embed_reuses_chunks_json(
    pipeline, fake_status_store, fake_vector_store
):
    await pipeline.run(TEXT_ID)
    count_first = len(fake_vector_store.points)
    chunk_count = fake_status_store.get(TEXT_ID)["chunk_count"]

    # xoá vector nhưng giữ artifacts -> resume từ embed không gọi parse lại
    await fake_vector_store.delete_by_doc(pipeline.collection, TEXT_ID)
    outcome = await pipeline.run(TEXT_ID, from_stage="embed")

    assert outcome.ok
    assert outcome.chunk_count == chunk_count
    assert len(fake_vector_store.points) == count_first
    assert fake_status_store.get(TEXT_ID)["status"] == STATUS_INDEXED


async def test_resume_from_embed_builds_chunks_when_missing(
    pipeline, fake_asset_store, fake_status_store, fake_vector_store
):
    await pipeline.run(TEXT_ID)
    chunk_count = fake_status_store.get(TEXT_ID)["chunk_count"]

    # xoá chunks.json -> resume từ embed phải build lại từ content_list+descriptions
    fake_asset_store.json_objects.pop(ArtifactPaths.chunks_key(fake_asset_store, TEXT_ID), None)
    await fake_vector_store.delete_by_doc(pipeline.collection, TEXT_ID)

    outcome = await pipeline.run(TEXT_ID, from_stage="embed")
    assert outcome.ok
    assert outcome.chunk_count == chunk_count
    assert len(fake_vector_store.points) == chunk_count


async def test_stage_embed_replaces_old_vectors(
    pipeline, fake_vector_store, fake_status_store
):
    await pipeline.run(TEXT_ID)  # full: parse -> describe -> chunk -> embed
    n = len(fake_vector_store.points)
    # chạy lại chunk stage: delete_by_doc trước upsert -> không tăng số vector
    outcome = await pipeline.run(TEXT_ID, from_stage="chunk")
    assert outcome.ok
    assert len(fake_vector_store.points) == n


async def test_dry_run_no_embed_no_writes(
    pipeline, fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    await pipeline.run(TEXT_ID)  # full: tạo artifacts + vector
    points_before = dict(fake_vector_store.points)
    texts_before = list(fake_embedder.embedded_texts)
    status_before = fake_status_store.get(TEXT_ID)

    outcome = await pipeline.run(TEXT_ID, from_stage="embed", dry_run=True)

    assert outcome.ok
    assert outcome.chunk_count > 0
    assert outcome.cache_hits == 0  # FakeEmbedder không cache
    assert fake_vector_store.points == points_before
    assert fake_embedder.embedded_texts == texts_before
    assert fake_status_store.get(TEXT_ID) == status_before


async def test_run_without_pending_row_raises(pipeline):
    from ami_rag.storage.doc_status import DocStatusStore
    from tests.ami_service.fakes import FakeMongoCollection

    empty_store = DocStatusStore.__new__(DocStatusStore)
    empty_store._col = FakeMongoCollection()
    pipeline.store = empty_store
    with pytest.raises(ValueError):
        await pipeline.run("64b000000000000000000099")


async def test_parse_without_doc_raises_clear_error(
    fake_asset_store, fake_status_store, fake_docs_repo
):
    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        CHUNKER_VERSION="v1",
        PARSER="mineru",
    )
    pipeline = VectorPipeline(
        settings,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=None,  # không có docs_repo -> không parse được
        parser=FakeParser(),
    )
    fake_status_store.ensure_pending(TEXT_ID)
    outcome = await pipeline.run(TEXT_ID, from_stage="parse")
    assert not outcome.ok
    assert "documents" in outcome.error
    row = fake_status_store.get(TEXT_ID)
    assert row["status"] == "failed"
    assert row["error_stage"] == "parse"


async def test_vector_store_error_marks_failed(
    pipeline, fake_status_store, fake_vector_store
):
    async def boom(collection, doc_id):
        raise RuntimeError("qdrant down")

    fake_vector_store.delete_by_doc = boom
    outcome = await pipeline.run(TEXT_ID, from_stage="embed")
    assert not outcome.ok
    row = fake_status_store.get(TEXT_ID)
    assert row["status"] == "failed"


async def test_delete_doc_purges_all(
    pipeline, fake_status_store, fake_vector_store, fake_asset_store
):
    await pipeline.run(TEXT_ID)
    assert fake_vector_store.points

    await pipeline.delete_doc(TEXT_ID)

    assert fake_vector_store.points == {}
    assert TEXT_ID in fake_asset_store.deleted_assets
    assert fake_status_store.get(TEXT_ID) is None


async def test_preflight_embed(pipeline, fake_vector_store):
    await pipeline.preflight_embed()
    assert pipeline.collection in fake_vector_store.collections


def test_chunk_text_budget_and_overlap():
    text = "\n\n".join(f"đoạn {i} " + "nội dung " * 10 for i in range(20))
    chunks = chunk_text(text, max_tokens=64, overlap_tokens=8)
    assert len(chunks) > 1
    assert all(len(c) <= 512 for c in chunks)  # max_chars = max(512, 64*4)


async def test_mark_stage_meta_preserved_through_mark_indexed(
    pipeline, fake_status_store
):
    await pipeline.run(PDF_ID)
    row = fake_status_store.get(PDF_ID)
    assert row["source"] == "minio_parse"
    assert row["parser"] == "mineru"
    assert row["file_path"] == "documents/HV/a.pdf"
    assert row["page_count"] == 3
    assert row["counts"]["image"] == 1


# --- Crawl: section 'Hình ảnh' -> tải ảnh về MinIO -> item image ---


async def test_stage_parse_crawl_images_injected(
    pipeline, fake_docs_repo, fake_status_store, fake_asset_store
):
    _add_crawl_doc(fake_docs_repo, fake_status_store)
    worker = FakeImageWorker()
    pipeline.image_worker = worker
    pipeline.settings.CRAWL_IMAGES_ENABLED = True

    doc = fake_docs_repo.find_by_id(CRAWL_ID)
    intermediate = await pipeline._stage_parse(CRAWL_ID, doc)
    content_list = intermediate.content_list

    assert worker.calls == [(CRAWL_ID, CRAWL_URLS[0]), (CRAWL_ID, CRAWL_URLS[1])]
    # text đã strip section ảnh, không còn URL
    assert content_list[0] == {
        "type": "text",
        "text": "Bài crawl PTIT.\n\nNội dung chính.",
        "page_idx": 0,
    }
    images = [i for i in content_list if i["type"] == "image"]
    assert len(images) == 2
    assert [i["asset_key"] for i in images] == [
        f"rag-assets/{CRAWL_ID}/img1.jpg",
        f"rag-assets/{CRAWL_ID}/img2.jpg",
    ]
    assert all(i["image_caption"] == [] and i["page_idx"] == 0 for i in images)
    # doc crawl không fetch file gốc từ MinIO
    assert fake_asset_store.fetched == []


async def test_stage_parse_crawl_all_downloads_fail_keeps_text(
    pipeline, fake_docs_repo, fake_status_store
):
    _add_crawl_doc(fake_docs_repo, fake_status_store)
    pipeline.image_worker = FakeImageWorker(fail_urls=set(CRAWL_URLS))
    pipeline.settings.CRAWL_IMAGES_ENABLED = True

    doc = fake_docs_repo.find_by_id(CRAWL_ID)
    intermediate = await pipeline._stage_parse(CRAWL_ID, doc)

    # không tải được ảnh nào -> giữ nguyên text gốc (URL không mất khỏi index)
    assert intermediate.content_list == [
        {"type": "text", "text": CRAWL_CONTENT, "page_idx": 0}
    ]


async def test_stage_parse_crawl_partial_failure_skips_failed(
    pipeline, fake_docs_repo, fake_status_store
):
    _add_crawl_doc(fake_docs_repo, fake_status_store)
    pipeline.image_worker = FakeImageWorker(fail_urls={CRAWL_URLS[0]})
    pipeline.settings.CRAWL_IMAGES_ENABLED = True

    doc = fake_docs_repo.find_by_id(CRAWL_ID)
    intermediate = await pipeline._stage_parse(CRAWL_ID, doc)
    content_list = intermediate.content_list

    assert [i["type"] for i in content_list] == ["text", "image"]
    assert content_list[1]["asset_key"] == f"rag-assets/{CRAWL_ID}/img2.jpg"


async def test_stage_parse_crawl_disabled_keeps_old_behavior(
    pipeline, fake_docs_repo, fake_status_store
):
    _add_crawl_doc(fake_docs_repo, fake_status_store)
    worker = FakeImageWorker()
    pipeline.image_worker = worker
    # CRAWL_IMAGES_ENABLED mặc định False -> worker không được dùng
    assert pipeline.settings.CRAWL_IMAGES_ENABLED is False

    doc = fake_docs_repo.find_by_id(CRAWL_ID)
    intermediate = await pipeline._stage_parse(CRAWL_ID, doc)

    assert intermediate.content_list == [
        {"type": "text", "text": CRAWL_CONTENT, "page_idx": 0}
    ]
    assert worker.calls == []


def test_get_image_worker_lazy_builds_from_settings(pipeline, monkeypatch):
    import ami_rag.workers.crawl_images as crawl_images_mod

    captured = {}

    class _W:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(crawl_images_mod, "CrawlImageWorker", _W)

    # tắt -> None, không build
    assert pipeline._get_image_worker() is None
    assert captured == {}

    pipeline.settings.CRAWL_IMAGES_ENABLED = True
    worker = pipeline._get_image_worker()

    # lazy-build với prefix = ASSET_PREFIX (asset nằm dưới doc prefix để
    # delete_doc_assets dọn theo doc)
    assert isinstance(worker, _W)
    assert captured["prefix"] == "rag-assets"
    assert captured["bucket"] == "ami-data-documents"
    assert captured["endpoint"] == "localhost:9000"
    # cached: lần sau trả lại chính worker đó
    assert pipeline._get_image_worker() is worker


async def test_full_run_crawl_doc_indexes_images(
    pipeline,
    fake_docs_repo,
    fake_status_store,
    fake_vector_store,
    fake_asset_store,
    fake_embedder,
    fake_modal_processors,
):
    _add_crawl_doc(fake_docs_repo, fake_status_store)
    pipeline.image_worker = FakeImageWorker()
    pipeline.settings.CRAWL_IMAGES_ENABLED = True

    outcome = await pipeline.run(CRAWL_ID)

    assert outcome.ok and outcome.stage == "indexed"
    row = fake_status_store.get(CRAWL_ID)
    assert row["source"] == "mongo_text"
    assert row["counts"] == {"text": 1, "image": 2}
    payloads = [p for p in fake_vector_store.points.values() if p["doc_id"] == CRAWL_ID]
    assert len(payloads) == outcome.chunk_count
    image_payloads = [p for p in payloads if p["modality"] == "image"]
    assert len(image_payloads) == 2
    assert all(p["asset_key"] for p in image_payloads)
    # URL ảnh không lọt vào text chunk
    assert all("NL_04608" not in p["content"] for p in payloads)
    # embed: chunk image có ảnh base64 (asset lấy từ MinIO)
    last_batch = fake_embedder.embedded_texts[-1]
    image_items = [it for it in last_batch if isinstance(it, dict) and "image_b64" in it]
    assert len(image_items) == 2
    assert image_items[0]["image_b64"] == base64.b64encode(b"\x89PNG-fake-bytes").decode()
    # describe đã chạy cho ảnh crawl (descriptions.json có item image)
    desc_key = ArtifactPaths.descriptions_key(fake_asset_store, CRAWL_ID)
    descriptions = fake_asset_store.json_objects.get(desc_key)
    assert {d["type"] for d in descriptions} == {"image"}


async def test_qdrant_vector_store_upsert_batching():
    from unittest.mock import MagicMock

    from ami_rag.core.vector_store import ChunkRecord, QdrantVectorStore

    store = QdrantVectorStore("http://fake:6333")
    store._client = MagicMock()
    chunks = [
        ChunkRecord(id=f"c_{i}", vector=[0.1] * 8, payload={"idx": i})
        for i in range(150)
    ]
    await store.upsert("test_col", chunks, batch_size=64)
    assert store._client.upsert.call_count == 3
    call1_points = store._client.upsert.call_args_list[0].kwargs["points"]
    call2_points = store._client.upsert.call_args_list[1].kwargs["points"]
    call3_points = store._client.upsert.call_args_list[2].kwargs["points"]
    assert len(call1_points) == 64
    assert len(call2_points) == 64
    assert len(call3_points) == 22


async def test_qdrant_upsert_packs_by_byte_limit():
    from unittest.mock import MagicMock

    from ami_rag.core.vector_store import (
        ChunkRecord,
        QdrantVectorStore,
        _point_estimate,
    )

    store = QdrantVectorStore("http://fake:6333", max_request_bytes=64 * 1024)
    store._client = MagicMock()
    chunks = [
        ChunkRecord(id=f"b_{i}", vector=[0.1] * 8, payload={"text": "x" * 4096})
        for i in range(32)
    ]
    await store.upsert("test_col", chunks, batch_size=64)
    calls = store._client.upsert.call_args_list
    # payload ~4 KB/point -> batch bị chia theo byte trước khi chạm 64 điểm
    assert len(calls) > 1
    total = 0
    for call in calls:
        points = call.kwargs["points"]
        total += len(points)
        assert sum(_point_estimate(p) for p in points) <= 64 * 1024
        assert len(points) <= 64
    assert total == 32


async def test_qdrant_upsert_single_oversized_point_sent_alone():
    from unittest.mock import MagicMock

    from ami_rag.core.vector_store import ChunkRecord, QdrantVectorStore

    store = QdrantVectorStore("http://fake:6333", max_request_bytes=1024)
    store._client = MagicMock()
    chunks = [
        ChunkRecord(id="small", vector=[0.1] * 8, payload={"text": "ok"}),
        ChunkRecord(id="big", vector=[0.1] * 8, payload={"text": "x" * 8192}),
        ChunkRecord(id="small2", vector=[0.1] * 8, payload={"text": "ok"}),
    ]
    await store.upsert("test_col", chunks, batch_size=10)
    calls = store._client.upsert.call_args_list
    sizes = [len(c.kwargs["points"]) for c in calls]
    # point "big" vượt limit -> gửi riêng, không chặn hai điểm nhỏ
    assert sizes == [1, 1, 1]
    assert calls[1].kwargs["points"][0].payload["chunk_id"] == "big"


def test_build_chunks_with_duplicate_content_assigns_unique_ids(pipeline):
    content_list = [
        {"type": "text", "text": "Đại học Bách Khoa", "page_idx": 1},
        {"type": "text", "text": "Đại học Bách Khoa", "page_idx": 2},
        {"type": "image", "img_path": "img1.png", "page_idx": 3},
        {"type": "image", "img_path": "img2.png", "page_idx": 4},
    ]
    chunks = pipeline._build_chunks("doc_test", content_list, descriptions=[])
    assert len(chunks) == 4
    ids = [c.id for c in chunks]
    assert len(set(ids)) == 4


async def test_stage_parse_backend_data_rebuild(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store, fake_docs_repo, monkeypatch
):
    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        EMBED_DIM=8,
        CHUNKER_VERSION="v1",
        PARSER="mineru",
        PARSE_TEXT_SOURCE="backend_data",
    )
    pipeline = VectorPipeline(
        settings,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=fake_docs_repo,
        parser=FakeParser(),
    )

    rebuilt_list = [
        {"type": "text", "text": "text backend_data", "page_idx": 0},
        {"type": "table", "table_body": "<table/>", "page_idx": 0},
        {"type": "image", "img_path": "kept.png", "page_idx": 1},
    ]
    calls = {"rebuild": 0}

    async def fake_rebuild(content_list, file_path, extractor):
        calls["rebuild"] += 1
        return [dict(i) for i in rebuilt_list]

    import raganything.backend_data_extract as bde

    monkeypatch.setattr(bde, "rebuild_content_list", fake_rebuild)

    fake_status_store.ensure_pending(PDF_ID)
    outcome = await pipeline.run(PDF_ID)

    assert outcome.ok and outcome.stage == "indexed"
    assert calls["rebuild"] == 1
    saved = fake_asset_store.load_content_list(PDF_ID)
    assert [{k: v for k, v in i.items() if k != "asset_key"} for i in saved] == rebuilt_list
    # asset chỉ upload cho ảnh (text/table MinerU đã bị thay)
    assert fake_asset_store.calls.count(("upload_assets", PDF_ID)) == 1
    assert saved[2]["asset_key"] == f"rag-assets/{PDF_ID}/img2.png"


async def test_stage_parse_backend_data_failure_falls_back_to_mineru(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store, fake_docs_repo, monkeypatch
):
    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        EMBED_DIM=8,
        CHUNKER_VERSION="v1",
        PARSER="mineru",
        PARSE_TEXT_SOURCE="backend_data",
    )
    pipeline = VectorPipeline(
        settings,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=fake_docs_repo,
        parser=FakeParser(),
    )

    async def boom(content_list, file_path, extractor):
        raise BackendDataParseError("ocr down")

    import raganything.backend_data_extract as bde

    monkeypatch.setattr(bde, "rebuild_content_list", boom)

    fake_status_store.ensure_pending(PDF_ID)
    outcome = await pipeline.run(PDF_ID)

    assert outcome.ok and outcome.stage == "indexed"
    saved = fake_asset_store.load_content_list(PDF_ID)
    # fallback: giữ nguyên MinerU content_list
    assert any(i["type"] == "text" and i["text"] == "Giới thiệu" for i in saved)
    assert any(i["type"] == "table" for i in saved)


async def test_prepare_embed_items_attaches_image_b64(pipeline, fake_embedder):
    outcome = await pipeline.run(PDF_ID)
    assert outcome.ok
    # Last call to embedder should contain image item with image_b64
    last_batch = fake_embedder.embedded_texts[-1]
    image_items = [it for it in last_batch if isinstance(it, dict) and "image_b64" in it]
    assert len(image_items) >= 1
    assert image_items[0]["image_b64"]


async def test_prepare_embed_items_attaches_assets_for_table_and_equation(pipeline):
    from ami_rag.core.vector_pipeline import Chunk

    chunks = [
        Chunk(id="c1", content="text", modality="text", page_idx=0, is_multimodal=False),
        Chunk(
            id="c2",
            content="[Table Content]",
            modality="table",
            page_idx=1,
            is_multimodal=True,
            asset_key="t1.png",
        ),
        Chunk(
            id="c3",
            content="[Equation]",
            modality="equation",
            page_idx=2,
            is_multimodal=True,
            asset_key="e1.png",
        ),
        Chunk(
            id="c4",
            content="table no asset",
            modality="table",
            page_idx=3,
            is_multimodal=True,
        ),
    ]
    items = await pipeline._prepare_embed_items(chunks)
    assert items[0] == "text"
    # table có asset -> image + text (modality Image+Text)
    assert isinstance(items[1], dict)
    assert items[1]["text"] == "[Table Content]"
    assert items[1]["image_b64"]
    # equation có asset -> image + text
    assert isinstance(items[2], dict)
    assert items[2]["image_b64"]
    # table không có asset_key -> text-only
    assert items[3] == "table no asset"


# ----------------------------------------------------------------------
# Describe ảnh: materialize MinIO -> img_path + caption func (Gemini/vision)
# ----------------------------------------------------------------------

def _image_content_list(asset_key="rag-assets/docX/img0.png", caption=None, footnotes=None, page=0):
    item = {"type": "image", "img_path": "gone.png", "asset_key": asset_key, "page_idx": page}
    if caption:
        item["image_caption"] = [caption]
    if footnotes:
        item["image_footnote"] = [footnotes]
    return [
        {"type": "text", "text": "Cơ cấu tổ chức của Học viện gồm các phòng ban.", "page_idx": page},
        item,
    ]


def _describe_settings(**overrides):
    base = {
        "_env_file": None,
        "WORKSPACE": "test",
        "EMBED_MODEL": "Qwen/Qwen3-VL-Embedding-2B",
        "EMBED_DIM": 8,
        "CHUNKER_VERSION": "v1",
    }
    base.update(overrides)
    return Settings(**base)


class _FakeCaptionFunc:
    """Stand-in cho caption func (Gemini/vision): ghi call + đo concurrency."""

    def __init__(
        self,
        response=(
            '{"detailed_description": "Mô tả ảnh từ caption func", '
            '"entity_info": {"entity_name": "Sơ đồ tổ chức", "entity_type": "image", '
            '"summary": "Sơ đồ tổ chức của PTIT"}}'
        ),
        fail=False,
        delay=0.0,
    ):
        self.response = response
        self.fail = fail
        self.delay = delay
        self.calls = []  # (prompt, image_data)
        self._active = 0
        self.max_active = 0

    async def __call__(self, prompt, system_prompt=None, history_messages=None, image_data=None, messages=None, **kwargs):
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("caption func down")
            self.calls.append((prompt, image_data))
            return self.response
        finally:
            self._active -= 1


def _make_pipeline(fake_embedder, fake_vector_store, fake_asset_store, fake_status_store, **settings_kw):
    return VectorPipeline(
        _describe_settings(**settings_kw),
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        parser=FakeParser(),
        modal_processors={},
    )


def _real_image_processor(caption_func, chunk_size=64):
    """ImageModalProcessor THẬT chạy caption func giả (chứng minh img_path materialize)."""
    from raganything.modalprocessors import ImageModalProcessor

    return ImageModalProcessor(
        lightrag=None,
        modal_caption_func=caption_func,
        global_config={"chunk_token_size": chunk_size},
    )


def test_resolve_image_describe_mode():
    # auto: có CAPTION_LLM_API_KEY -> gemini
    assert resolve_image_describe_mode(_describe_settings(CAPTION_LLM_API_KEY="k")) == "gemini"
    # auto: không key nhưng có QWEN_VLM_MODEL -> vision
    assert resolve_image_describe_mode(_describe_settings(QWEN_VLM_MODEL="Qwen/Qwen3-VL")) == "vision"
    # auto: thiếu cả hai -> none (text thô, không LLM)
    assert resolve_image_describe_mode(_describe_settings()) == "none"
    # giá trị rõ được dùng luôn
    for mode in ("gemini", "vision", "none"):
        assert resolve_image_describe_mode(_describe_settings(IMAGE_DESCRIBE_MODE=mode)) == mode
    # legacy "text" (path OCR-ảnh đã xoá) -> "none"
    assert resolve_image_describe_mode(_describe_settings(IMAGE_DESCRIBE_MODE="text")) == "none"
    with pytest.raises(ValueError):
        resolve_image_describe_mode(_describe_settings(IMAGE_DESCRIBE_MODE="bogus"))


async def test_describe_image_via_real_image_processor(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """Materialize MinIO -> img_path thật -> ImageModalProcessor THẬT gửi đúng base64."""
    caption = _FakeCaptionFunc()
    pipeline = _make_pipeline(
        fake_embedder, fake_vector_store, fake_asset_store, fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    content_list = _image_content_list(caption="Sơ đồ tổ chức PTIT")
    descriptions = await pipeline._stage_describe("docX", content_list)

    assert descriptions[0]["type"] == "image"
    # JSON response được _parse_response của ImageModalProcessor parse ra description
    assert descriptions[0]["description"] == "Mô tả ảnh từ caption func"
    assert len(caption.calls) == 1
    prompt, image_data = caption.calls[0]
    # caption func nhận đúng bytes ảnh lấy từ MinIO theo asset_key
    assert base64.b64decode(image_data) == b"\x89PNG-fake-bytes"
    # prompt chứa caption + ngữ cảnh (content source = content_list gốc)
    assert "Sơ đồ tổ chức PTIT" in prompt
    assert "Cơ cấu tổ chức của Học viện" in prompt
    # content_list gốc KHÔNG bị gán img_path mới (describe_items là bản copy)
    assert content_list[1]["img_path"] == "gone.png"
    assert "asset_key" in content_list[1]


async def test_describe_image_temp_dir_cleaned_after_stage(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """img_path tồn tại LÚC describe chạy; temp dir bị xoá SAU _stage_describe."""
    seen: list[tuple[str, bool]] = []

    class _ProbeProc:
        def set_content_source(self, content_source, content_format="auto"):
            pass

        async def generate_chunk_sections(self, modal_content, content_type, item_info=None, entity_name=None):
            path = modal_content.get("img_path")
            seen.append((path, Path(path).exists()))
            return [{"description": "mô tả image", "entity_info": {}, "window_meta": None}]

    pipeline = _make_pipeline(
        fake_embedder, fake_vector_store, fake_asset_store, fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
    )
    pipeline.modal_processors = {"image": _ProbeProc()}

    content_list = _image_content_list()
    await pipeline._stage_describe("docX", content_list)

    assert len(seen) == 1
    path, existed_during = seen[0]
    assert existed_during, "img_path phải là file thật khi processor chạy"
    assert not Path(path).exists(), "temp dir phải bị xoá sau _stage_describe"
    # bản gốc giữ img_path cũ, không bị mutate
    assert content_list[1]["img_path"] == "gone.png"


async def test_describe_image_mode_none_raw_text(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """mode none: text thô caption + footnotes + ngữ cảnh; KHÔNG gọi caption func / fetch MinIO."""
    caption = _FakeCaptionFunc()
    pipeline = _make_pipeline(
        fake_embedder, fake_vector_store, fake_asset_store, fake_status_store,
        IMAGE_DESCRIBE_MODE="none",
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    fetched: list[str] = []
    original_get_bytes = fake_asset_store.get_bytes

    def _spy(key):
        fetched.append(key)
        return original_get_bytes(key)

    fake_asset_store.get_bytes = _spy

    content_list = _image_content_list(caption="Sơ đồ tổ chức PTIT", footnotes="Nguồn: PTIT")
    descriptions = await pipeline._stage_describe("docX", content_list)

    text = descriptions[0]["description"]
    assert text == "Sơ đồ tổ chức PTIT\nNguồn: PTIT\nCơ cấu tổ chức của Học viện gồm các phòng ban."
    assert caption.calls == []
    assert fetched == []


async def test_describe_image_caption_func_fails_falls_back_to_raw(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """Caption func raise -> processor nuốt lỗi (trả dict repr) -> phát hiện -> text thô, không crash."""
    caption = _FakeCaptionFunc(fail=True)
    pipeline = _make_pipeline(
        fake_embedder, fake_vector_store, fake_asset_store, fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    content_list = _image_content_list(caption="Sơ đồ tổ chức PTIT")
    descriptions = await pipeline._stage_describe("docX", content_list)

    # descriptions.json vẫn ghi, mô tả là text thô (không phải dict repr của item)
    text = descriptions[0]["description"]
    assert "Sơ đồ tổ chức PTIT" in text
    assert "Cơ cấu tổ chức của Học viện" in text
    assert not text.startswith("{")


async def test_describe_image_missing_asset_key_keeps_original(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """Ảnh thiếu asset_key -> không materialize -> img_path cũ hỏng -> text thô, không crash."""
    caption = _FakeCaptionFunc()
    pipeline = _make_pipeline(
        fake_embedder, fake_vector_store, fake_asset_store, fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    content_list = _image_content_list()
    del content_list[1]["asset_key"]
    descriptions = await pipeline._stage_describe("docX", content_list)

    # img_path cũ ("gone.png") không tồn tại -> processor lỗi -> text thô
    text = descriptions[0]["description"]
    assert "Cơ cấu tổ chức của Học viện" in text
    assert not text.startswith("{")


async def test_describe_caption_concurrency_limited(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    caption = _FakeCaptionFunc(delay=0.01)
    pipeline = _make_pipeline(
        fake_embedder, fake_vector_store, fake_asset_store, fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
        CAPTION_LLM_MAX_CONCURRENCY=2,
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    content_list = [
        {"type": "image", "img_path": "gone.png", "asset_key": f"rag-assets/docX/img{i}.png", "page_idx": 0}
        for i in range(5)
    ]
    descriptions = await pipeline._stage_describe("docX", content_list)

    assert len(descriptions) == 5
    assert len(caption.calls) == 5
    assert caption.max_active <= 2


async def test_describe_sets_content_source_on_processors(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    class _RecProc:
        def __init__(self):
            self.sources = []

        def set_content_source(self, content_source, content_format="auto"):
            self.sources.append((content_source, content_format))

        async def generate_chunk_sections(self, modal_content, content_type, item_info=None, entity_name=None):
            return [{"description": "desc", "entity_info": {}, "window_meta": None}]

    pipeline = _make_pipeline(fake_embedder, fake_vector_store, fake_asset_store, fake_status_store)
    proc = _RecProc()
    pipeline.modal_processors = {"image": proc, "table": proc}

    content_list = _image_content_list()
    await pipeline._stage_describe("docX", content_list)

    assert proc.sources and proc.sources[0][1] == "minerU"
    assert proc.sources[0][0] is content_list


async def test_describe_vision_mode_uses_processor(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store, fake_modal_processors
):
    pipeline = _make_pipeline(
        fake_embedder, fake_vector_store, fake_asset_store, fake_status_store,
        QWEN_VLM_MODEL="Qwen/Qwen3-VL",  # auto -> vision
    )
    pipeline.modal_processors = fake_modal_processors

    descriptions = await pipeline._stage_describe("docX", _image_content_list())

    # vision mode: đi qua processor, được gán img_path file tạm (đã xoá sau stage)
    assert descriptions[0]["description"] == "mô tả image"
    calls = []
    for proc in fake_modal_processors.values():
        calls.extend(proc.calls)
    img_calls = [c for c in calls if c[0] == "image"]
    assert img_calls, "processor image phải được gọi ở mode vision"
    img_item = img_calls[0][1]
    assert img_item["img_path"] != "gone.png", "materialize phải gán img_path temp"
    assert not Path(img_item["img_path"]).exists(), "temp dir xoá sau stage"


