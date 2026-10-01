"""RAGAnything + real LightRAG (local json/nano storages, fake LLM/VLM/embedding).

Exercises the library contract the AMI service relies on: content_list items keep
`asset_key` down to the chunk, `aquery_data` enriches chunks with modality metadata,
and `adelete_by_doc_id` also removes multimodal chunks.
"""

import json
import uuid

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


pytest.importorskip("PIL")

DOC_ID = "64b0000000000000000000f1"


@pytest.fixture(autouse=True)
def _dummy_parser():
    """MinerU is not installed in the unit env; insert_content_list never parses, but
    RAGAnything checks that the configured parser is installed."""
    from raganything.parser import _CUSTOM_PARSERS, Parser, register_parser

    class DummyParser(Parser):
        def check_installation(self) -> bool:
            return True

    register_parser("dummy", DummyParser)
    yield
    _CUSTOM_PARSERS.pop("dummy", None)


def _png(path):
    from PIL import Image

    Image.new("RGB", (8, 8), (200, 30, 30)).save(path)
    return str(path)


async def _fake_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    text = f"{system_prompt or ''}\n{prompt}"
    if "detailed_description" in text:
        return json.dumps(
            {
                "detailed_description": "Mô tả chi tiết nội dung phương tiện.",
                "entity_info": {
                    "entity_name": "Phuong tien",
                    "entity_type": "media",
                    "summary": "Tóm tắt phương tiện.",
                },
            },
            ensure_ascii=False,
        )
    return "entity<|#|>Hoc Vien<|#|>Organization<|#|>Truong dai hoc.\n<|COMPLETE|>"


async def _fake_vision(prompt, system_prompt=None, history_messages=None, image_data=None, **kw):
    return await _fake_llm(prompt, system_prompt, history_messages)


async def _build(tmp_path, parser="dummy"):
    import numpy as np
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    from raganything import RAGAnything, RAGAnythingConfig

    async def fake_embed(texts):
        return np.array([[0.1, 0.2, 0.3, 0.4] for _ in texts], dtype=np.float32)

    rag = LightRAG(
        working_dir=str(tmp_path / "rag_storage"),
        # LightRAG keeps shared storage per workspace for the whole process
        workspace=f"t{uuid.uuid4().hex[:8]}",
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
        llm_model_func=_fake_llm,
        embedding_func=EmbeddingFunc(embedding_dim=4, func=fake_embed),
    )
    await rag.initialize_storages()
    anything = RAGAnything(
        lightrag=rag,
        vision_model_func=_fake_vision,
        config=RAGAnythingConfig(working_dir=str(tmp_path / "rag_storage"), parser=parser),
    )
    return rag, anything


def _content_list(png_path):
    return [
        {
            "type": "text",
            "text": "Học viện tuyển sinh năm 2026 với nhiều ngành đào tạo.",
            "page_idx": 0,
        },
        {
            "type": "table",
            "table_body": "| Ngành | Chỉ tiêu |\n|---|---|\n| CNTT | 500 |",
            "table_caption": ["Bảng chỉ tiêu"],
            "page_idx": 1,
        },
        {
            "type": "image",
            "img_path": png_path,
            "asset_key": f"rag-assets/{DOC_ID}/abc.png",
            "image_caption": ["Sơ đồ tuyển sinh"],
            "page_idx": 2,
        },
    ]


async def test_asset_key_and_modality_survive_ingest_and_query(tmp_path):
    rag, anything = await _build(tmp_path)
    try:
        await anything.insert_content_list(
            _content_list(_png(tmp_path / "a.png")),
            file_path=f"{DOC_ID}_a.pdf",
            doc_id=DOC_ID,
        )

        result = await anything.aquery_data("tuyển sinh", mode="naive", chunk_top_k=20)
        assert result["status"] == "success"
        chunks = result["data"]["chunks"]
        by_modality = {}
        for chunk in chunks:
            by_modality.setdefault(chunk["modality"], []).append(chunk)

        assert {"text", "image", "table"} <= set(by_modality)
        image = by_modality["image"][0]
        assert image["asset_key"] == f"rag-assets/{DOC_ID}/abc.png"
        assert image["page_idx"] == 2
        assert image["caption"] == "Sơ đồ tuyển sinh"
        assert "/tmp" not in image["content"] and f"rag-assets/{DOC_ID}/abc.png" in image["content"]
        table = by_modality["table"][0]
        assert "| CNTT | 500 |" in table["table_body"]
        assert table["page_idx"] == 1
        assert by_modality["text"][0]["asset_key"] is None
        assert all(c["file_path"].startswith(DOC_ID) for c in chunks)
    finally:
        await rag.finalize_storages()


async def test_delete_removes_multimodal_chunks(tmp_path):
    rag, anything = await _build(tmp_path)
    try:
        await anything.insert_content_list(
            _content_list(_png(tmp_path / "a.png")),
            file_path=f"{DOC_ID}_a.pdf",
            doc_id=DOC_ID,
        )
        before = await anything.aquery_data("tuyển sinh", mode="naive", chunk_top_k=20)
        assert any(c["modality"] != "text" for c in before["data"]["chunks"])

        await rag.adelete_by_doc_id(DOC_ID)

        after = await anything.aquery_data("tuyển sinh", mode="naive", chunk_top_k=20)
        remaining = (after.get("data") or {}).get("chunks") or []
        assert not [c for c in remaining if str(c.get("file_path", "")).startswith(DOC_ID)]
        assert await rag.doc_status.get_by_id(DOC_ID) is None
    finally:
        await rag.finalize_storages()


async def test_worker_ingests_pdf_end_to_end_and_drops_parse_cache(
    tmp_path, fake_docs_repo, fake_state_repo, fake_queue, fake_asset_store
):
    """IngestWorker + real RAGAnything/LightRAG; only the parser is replaced."""
    from ami_rag.queue.events import RagEvent
    from ami_rag.workers.ingest_worker import IngestWorker
    from raganything.parser import Parser, register_parser

    pdf_id = "64b000000000000000000001"

    class PdfParser(Parser):
        def check_installation(self) -> bool:
            return True

        def parse_pdf(self, pdf_path, output_dir="./output", method="auto", **kw):
            img = _png(tmp_path / "parsed.png")
            return [
                {"type": "text", "text": "Quy định học phí năm 2026.", "page_idx": 0},
                {
                    "type": "table",
                    "table_body": "| Ngành | Học phí |\n|---|---|\n| CNTT | 20tr |",
                    "page_idx": 1,
                },
                {
                    "type": "image",
                    "img_path": img,
                    "image_caption": ["Biểu đồ học phí"],
                    "page_idx": 2,
                },
            ]

        parse_document = parse_pdf

    register_parser("pdfdummy", PdfParser)
    rag, anything = await _build(tmp_path, parser="pdfdummy")
    try:

        class _S:
            WORKER_MAX_DELIVERY = 3
            WORKER_BATCH = 10
            WORKER_POLL_BLOCK_MS = 10
            RAG_STREAM = "rag:ingest"
            RAG_CONSUMER_GROUP = "ami-rag"
            PARSER = "pdfdummy"
            PARSE_METHOD = "auto"
            CRAWL_IMAGES_ENABLED = False

        worker = IngestWorker(
            rag_anything=anything,
            docs_repo=fake_docs_repo,
            state_repo=fake_state_repo,
            queue=fake_queue,
            asset_store=fake_asset_store,
            settings=_S(),
        )
        await anything._ensure_lightrag_initialized()  # creates parse_cache
        written = []
        original_upsert = anything.parse_cache.upsert

        async def spy_upsert(data):
            written.extend(data)
            await original_upsert(data)

        anything.parse_cache.upsert = spy_upsert

        result = await worker.handle_event(RagEvent(event="created", document_id=pdf_id))

        assert result["source"] == "minio_parse"
        assert result["counts"]["table"] == 1 and result["counts"]["image"] == 1
        assert result["assets"] == [f"rag-assets/{pdf_id}/img2.png"]

        found = await anything.aquery_data("học phí", mode="naive", chunk_top_k=20)
        modalities = {c["modality"] for c in found["data"]["chunks"]}
        assert {"text", "table", "image"} <= modalities
        image = next(c for c in found["data"]["chunks"] if c["modality"] == "image")
        assert image["asset_key"] == f"rag-assets/{pdf_id}/img2.png"

        # the parse cache entry (keyed by the temp path) must not be left behind
        assert written, "parse result should have been cached by RAGAnything"
        assert await anything.parse_cache.get_by_id(written[0]) is None

        await worker.handle_event(RagEvent(event="deleted", document_id=pdf_id))
        gone = await anything.aquery_data("học phí", mode="naive", chunk_top_k=20)
        left = (gone.get("data") or {}).get("chunks") or []
        assert not [c for c in left if str(c.get("file_path", "")).startswith(pdf_id)]
    finally:
        from raganything.parser import _CUSTOM_PARSERS

        _CUSTOM_PARSERS.pop("pdfdummy", None)
        await rag.finalize_storages()
