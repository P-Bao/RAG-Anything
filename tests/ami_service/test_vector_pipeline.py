"""Test VectorPipeline (fake embedder/vector_store/asset_store/DocsStatusStore)."""

import pytest

from ami_rag.core.vector_pipeline import (
    ArtifactPaths,
    VectorPipeline,
    chunk_text,
)
from ami_rag.settings import Settings, resolve_image_describe_mode
from ami_rag.storage.doc_status import STATUS_INDEXED
from raganything.backend_data_extract import BackendDataParseError
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
# Describe ảnh không VLM: OCR bridge + ngữ cảnh + Qwen text-only
# ----------------------------------------------------------------------

class _FakeOCR:
    """Stand-in cho BackendDataExtractor (chỉ extract_image_ocr_text)."""

    def __init__(self, text="SƠ ĐỒ CƠ CẤU TỔ CHỨC\nGiám đốc: Đặng Hoài Bắc", fail=False, delay=0.0):
        self.text = text
        self.fail = fail
        self.delay = delay
        self.calls = []
        self._active = 0
        self.max_active = 0

    async def extract_image_ocr_text(self, image_bytes, filename="image.png"):
        self.calls.append((len(image_bytes), filename))
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                import asyncio

                await asyncio.sleep(self.delay)
            if self.fail:
                raise BackendDataParseError("OCR boom")
            return self.text
        finally:
            self._active -= 1


class _FakeLLM:
    def __init__(self, response='{"detailed_description": "Mô tả ảnh từ LLM"}'):
        self.response = response
        self.prompts = []

    async def __call__(self, prompt, system_prompt=None, **kwargs):
        self.prompts.append((prompt, system_prompt))
        return self.response


def _image_content_list(asset_key="rag-assets/docX/img0.png", caption=None, page=0):
    item = {"type": "image", "img_path": "gone.png", "asset_key": asset_key, "page_idx": page}
    if caption:
        item["image_caption"] = [caption]
    return [
        {"type": "text", "text": "Cơ cấu tổ chức của Học viện gồm các phòng ban.", "page_idx": page},
        item,
    ]


def _settings_text_mode(**overrides):
    base = {
        "_env_file": None,
        "WORKSPACE": "test",
        "EMBED_MODEL": "Qwen/Qwen3-VL-Embedding-2B",
        "EMBED_DIM": 8,
        "CHUNKER_VERSION": "v1",
        "QWEN_VLM_MODEL": "",
    }
    base.update(overrides)
    return Settings(**base)


def test_resolve_image_describe_mode():
    assert resolve_image_describe_mode(_settings_text_mode()) == "text"
    assert resolve_image_describe_mode(_settings_text_mode(QWEN_VLM_MODEL="Qwen/Qwen3-VL")) == "vision"
    assert resolve_image_describe_mode(_settings_text_mode(IMAGE_DESCRIBE_MODE="vision")) == "vision"
    assert (
        resolve_image_describe_mode(
            _settings_text_mode(IMAGE_DESCRIBE_MODE="text", QWEN_VLM_MODEL="Qwen/Qwen3-VL")
        )
        == "text"
    )
    with pytest.raises(ValueError):
        resolve_image_describe_mode(_settings_text_mode(IMAGE_DESCRIBE_MODE="bogus"))


def _make_pipeline(fake_embedder, fake_vector_store, fake_asset_store, fake_status_store, **settings_kw):
    return VectorPipeline(
        _settings_text_mode(**settings_kw),
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        parser=FakeParser(),
        modal_processors={},
    )


async def test_describe_image_text_only_with_ocr(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    pipeline = _make_pipeline(fake_embedder, fake_vector_store, fake_asset_store, fake_status_store)
    ocr = _FakeOCR()
    llm = _FakeLLM()
    pipeline._bd_extractor = ocr
    pipeline.llm_func = llm

    descriptions = await pipeline._stage_describe("docX", _image_content_list())

    assert len(descriptions) == 1
    assert descriptions[0]["description"] == "Mô tả ảnh từ LLM"
    assert descriptions[0]["type"] == "image"
    # OCR được gọi với bytes từ MinIO (asset_key), không phải img_path local
    assert ocr.calls == [(len(b"\x89PNG-fake-bytes"), "rag-assets/docX/img0.png")]
    # Prompt LLM chứa OCR text + ngữ cảnh xung quanh
    prompt, system = llm.prompts[0]
    assert "SƠ ĐỒ CƠ CẤU TỔ CHỨC" in prompt
    assert "Cơ cấu tổ chức của Học viện gồm các phòng ban." in prompt
    assert "IMAGE_TEXT_ONLY_SYSTEM" in system or "expert document analyst" in system


async def test_describe_image_skips_ocr_when_caption_present(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    pipeline = _make_pipeline(fake_embedder, fake_vector_store, fake_asset_store, fake_status_store)
    ocr = _FakeOCR()
    llm = _FakeLLM()
    pipeline._bd_extractor = ocr
    pipeline.llm_func = llm

    content_list = _image_content_list(caption="Sơ đồ tổ chức PTIT")
    await pipeline._stage_describe("docX", content_list)

    assert ocr.calls == []  # SKIP_IF_CAPTION mặc định True
    assert "Sơ đồ tổ chức PTIT" in llm.prompts[0][0]


async def test_describe_image_ocr_fail_falls_back_to_context(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    pipeline = _make_pipeline(fake_embedder, fake_vector_store, fake_asset_store, fake_status_store)
    pipeline._bd_extractor = _FakeOCR(fail=True)
    llm = _FakeLLM()
    pipeline.llm_func = llm

    descriptions = await pipeline._stage_describe("docX", _image_content_list())

    # Không crash; LLM vẫn được gọi với marker không OCR được
    assert descriptions[0]["description"] == "Mô tả ảnh từ LLM"
    assert "(không OCR được chữ trong ảnh)" in llm.prompts[0][0]
    assert "Cơ cấu tổ chức của Học viện" in llm.prompts[0][0]


async def test_describe_image_llm_fail_returns_raw_text(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    pipeline = _make_pipeline(fake_embedder, fake_vector_store, fake_asset_store, fake_status_store)
    pipeline._bd_extractor = _FakeOCR()

    async def broken_llm(prompt, system_prompt=None, **kwargs):
        raise RuntimeError("LLM down")

    pipeline.llm_func = broken_llm
    descriptions = await pipeline._stage_describe("docX", _image_content_list())

    text = descriptions[0]["description"]
    assert "SƠ ĐỒ CƠ CẤU TỔ CHỨC" in text
    assert "Cơ cấu tổ chức của Học viện" in text


async def test_describe_image_no_llm_returns_raw_text(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    pipeline = _make_pipeline(fake_embedder, fake_vector_store, fake_asset_store, fake_status_store)
    ocr = _FakeOCR()
    pipeline._bd_extractor = ocr
    pipeline.llm_func = None

    descriptions = await pipeline._stage_describe("docX", _image_content_list())
    text = descriptions[0]["description"]
    assert "SƠ ĐỒ CƠ CẤU TỔ CHỨC" in text
    assert "Cơ cấu tổ chức của Học viện" in text


async def test_describe_image_no_asset_key_skips_ocr(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    pipeline = _make_pipeline(fake_embedder, fake_vector_store, fake_asset_store, fake_status_store)
    ocr = _FakeOCR()
    pipeline._bd_extractor = ocr
    pipeline.llm_func = _FakeLLM()

    content_list = _image_content_list()
    del content_list[1]["asset_key"]
    await pipeline._stage_describe("docX", content_list)
    assert ocr.calls == []


async def test_describe_ocr_concurrency_limited(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    pipeline = _make_pipeline(
        fake_embedder, fake_vector_store, fake_asset_store, fake_status_store,
        DESCRIBE_IMAGE_OCR_MAX_CONCURRENCY=2,
    )
    ocr = _FakeOCR(delay=0.01)
    pipeline._bd_extractor = ocr
    pipeline.llm_func = None

    content_list = [
        {"type": "image", "asset_key": f"rag-assets/docX/img{i}.png", "page_idx": 0}
        for i in range(5)
    ]
    await pipeline._stage_describe("docX", content_list)
    assert len(ocr.calls) == 5
    assert ocr.max_active <= 2


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
        QWEN_VLM_MODEL="Qwen/Qwen3-VL-Embedding-2B",
    )
    pipeline.modal_processors = fake_modal_processors
    ocr = _FakeOCR()
    pipeline._bd_extractor = ocr

    descriptions = await pipeline._stage_describe("docX", _image_content_list())

    # vision mode: đi qua processor như cũ, KHÔNG gọi OCR bridge
    assert descriptions[0]["description"] == "mô tả image"
    assert ocr.calls == []
    calls = []
    for proc in fake_modal_processors.values():
        calls.extend(proc.calls)
    assert any(c[0] == "image" for c in calls)


