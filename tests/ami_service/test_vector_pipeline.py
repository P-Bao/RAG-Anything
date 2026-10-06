"""Test VectorPipeline (fake embedder/vector_store/asset_store/DocsStatusStore)."""

import pytest

from ami_rag.core.vector_pipeline import (
    ArtifactPaths,
    VectorPipeline,
    chunk_text,
)
from ami_rag.settings import Settings
from ami_rag.storage.doc_status import STATUS_INDEXED
from tests.ami_service.conftest import FakeEmbedder, FakeParser, FakeVectorStore

TEXT_ID = "64b000000000000000000005"
PDF_ID = "64b000000000000000000001"


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


