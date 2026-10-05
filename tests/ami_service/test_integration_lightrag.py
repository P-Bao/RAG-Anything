import pytest

pytest.importorskip("lightrag")
# LightRAG keeps asyncio locks in module-level shared storage; share one loop across tests
pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture(autouse=True)
def _real_lightrag():
    # Some legacy tests replace `lightrag` with stubs in sys.modules at import time;
    # those leak into a combined run, so check at test time.
    try:
        from lightrag.kg.shared_storage import initialize_share_data  # noqa: F401
    except ImportError:
        pytest.skip("real lightrag stubbed by other tests; run tests/ami_service separately")


@pytest.fixture
def tmp_working_dir(tmp_path):
    return str(tmp_path / "rag_storage")


async def _build_rag(working_dir: str):
    import numpy as np
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    async def fake_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        return "entity<|#|>Hoc Vien Cong Nghe<|#|>Organization<|#|>Truong dai hoc ky thuat.\n<|COMPLETE|>"

    async def fake_embed(texts):
        return np.array([[0.1, 0.2, 0.3, 0.4] for _ in texts], dtype=np.float32)

    rag = LightRAG(
        working_dir=working_dir,
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
        llm_model_func=fake_llm,
        embedding_func=EmbeddingFunc(embedding_dim=4, func=fake_embed),
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=1,
        addon_params={"language": "Tiếng Việt"},
    )
    await rag.initialize_storages()
    return rag


async def test_insert_query_delete_lifecycle(tmp_working_dir):
    from lightrag import QueryParam

    rag = await _build_rag(tmp_working_dir)
    try:
        await rag.ainsert(
            "Học viện Công nghệ bưu chính viễn thông tuyển sinh năm 2026 với học phí ổn định.",
            ids=["64b000000000000000000001"],
            file_paths=["64b000000000000000000001_a.pdf"],
        )
        status = await rag.doc_status.get_by_id("64b000000000000000000001")
        assert status is not None
        assert status.get("status") in ("processed", "pending")

        result = await rag.aquery_data("tuyển sinh", param=QueryParam(mode="naive", chunk_top_k=5))
        assert result.get("status") == "success"
        data = result.get("data") or {}
        assert len(data.get("chunks") or []) >= 1
        chunk = data["chunks"][0]
        assert chunk["file_path"].startswith("64b000000000000000000001_")
        assert chunk["file_path"].endswith("a.pdf")
        assert "reference_id" in chunk
    finally:
        await rag.finalize_storages()


async def test_delete_by_doc_id_removes_document(tmp_working_dir):
    from lightrag import QueryParam

    rag = await _build_rag(tmp_working_dir)
    try:
        await rag.ainsert(
            "Nội dung tài liệu cần xóa.",
            ids=["64b000000000000000000002"],
            file_paths=["64b000000000000000000002_b.pdf"],
        )
        await rag.adelete_by_doc_id("64b000000000000000000002")
        status = await rag.doc_status.get_by_id("64b000000000000000000002")
        assert status is None
        result = await rag.aquery_data("tài liệu", param=QueryParam(mode="naive", chunk_top_k=5))
        data = result.get("data") or {}
        assert all(
            not str(c.get("file_path") or "").startswith("64b000000000000000000002_")
            for c in (data.get("chunks") or [])
        )
    finally:
        await rag.finalize_storages()


async def test_factory_builds_with_mongo_qdrant_config(tmp_working_dir):
    """Factory wiring: constructor params valid; storages not initialized here
    (Mongo/Qdrant unreachable in unit env) — only construction is verified."""
    from ami_rag.core.factory import build_rag
    from ami_rag.settings import Settings

    settings = Settings(WORKING_DIR=tmp_working_dir)
    rag = build_rag(
        settings,
        llm_model_func=lambda prompt, **kwargs: "ok",
        embedding_func=None,
    )
    assert rag.kv_storage == "MongoKVStorage"
    assert rag.vector_storage == "QdrantVectorDBStorage"
    assert rag.graph_storage == "MongoGraphStorage"
    assert rag.doc_status_storage == "MongoDocStatusStorage"
    assert rag.workspace == "multimodal"
    assert rag.chunk_token_size == 1200
    assert rag.chunk_overlap_token_size == 100
    assert rag.entity_extract_max_gleaning == 1
