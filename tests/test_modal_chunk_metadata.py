import logging
from types import SimpleNamespace

import pytest

from raganything.processor import ProcessorMixin
from raganything.query import QueryMixin
from raganything.utils import build_modal_chunk_metadata, format_asset_line


def test_format_asset_line_prefers_asset_key():
    item = {"img_path": "/tmp/mineru/a.png", "asset_key": "rag-assets/d1/abc.png"}
    assert format_asset_line(item, "Image Path: ", item["img_path"]) == (
        "Asset: rag-assets/d1/abc.png"
    )


def test_format_asset_line_keeps_legacy_path_without_asset_key():
    item = {"img_path": "/tmp/mineru/a.png"}
    assert format_asset_line(item, "Image Path: ", item["img_path"]) == (
        "Image Path: /tmp/mineru/a.png"
    )


def test_metadata_image_has_caption_asset_and_no_table_body():
    item = {
        "img_path": "/tmp/a.png",
        "asset_key": "rag-assets/d1/abc.png",
        "image_caption": ["Hình 1", "Sơ đồ"],
        "page_idx": 3,
    }
    meta = build_modal_chunk_metadata("image", item)
    assert meta["is_multimodal"] is True
    assert meta["original_type"] == "image"
    assert meta["page_idx"] == 3
    assert meta["asset_key"] == "rag-assets/d1/abc.png"
    assert meta["caption"] == "Hình 1, Sơ đồ"
    assert meta["table_body"] is None


def test_metadata_table_keeps_raw_body_string():
    body = "| a | b |\n|---|---|\n| 1 | 2 |"
    meta = build_modal_chunk_metadata(
        "table", {"table_body": body, "table_caption": ["Bảng 1"]}, page_idx=5
    )
    assert meta["table_body"] == body
    assert meta["caption"] == "Bảng 1"
    assert meta["page_idx"] == 5
    assert meta["asset_key"] is None


def test_metadata_table_list_body_rendered_to_string():
    meta = build_modal_chunk_metadata("table", {"table_body": [["a", "b"], ["1", "2"]]})
    assert isinstance(meta["table_body"], str)
    assert "a" in meta["table_body"] and "2" in meta["table_body"]


def test_metadata_missing_fields_are_none():
    meta = build_modal_chunk_metadata("equation", {})
    assert meta["asset_key"] is None
    assert meta["caption"] is None
    assert meta["table_body"] is None


class _Host(ProcessorMixin):
    """Minimal host: only what the pure chunk helpers touch."""

    def __init__(self):
        self.logger = logging.getLogger("test")
        self.config = SimpleNamespace(use_full_path=False)
        self.lightrag = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda text: text.split())
        )


def _data(content_type, item):
    return {
        "description": "mô tả",
        "entity_info": {"entity_name": "Ent"},
        "chunk_order_index": 0,
        "content_type": content_type,
        "original_item": item,
        "item_info": {"page_idx": 2},
    }


def test_chunk_template_image_uses_asset_key_instead_of_local_path():
    host = _Host()
    item = {"img_path": "/tmp/mineru/a.png", "asset_key": "rag-assets/d1/abc.png"}
    content = host._apply_chunk_template("image", item, "mô tả")
    assert "Asset: rag-assets/d1/abc.png" in content
    assert "/tmp/mineru/a.png" not in content


def test_chunk_template_image_without_asset_key_keeps_path():
    host = _Host()
    content = host._apply_chunk_template("image", {"img_path": "/tmp/a.png"}, "mô tả")
    assert "Image Path: /tmp/a.png" in content


def test_converted_chunks_carry_structured_fields():
    host = _Host()
    table = {
        "table_body": "| a |\n|---|\n| 1 |",
        "table_caption": ["Bảng 1"],
        "asset_key": "rag-assets/d1/t.png",
        "img_path": "/tmp/t.png",
    }
    chunks = host._convert_to_lightrag_chunks_type_aware(
        [_data("table", table)], "doc.pdf", "doc-1"
    )
    assert len(chunks) == 1
    chunk = next(iter(chunks.values()))
    assert chunk["is_multimodal"] is True
    assert chunk["original_type"] == "table"
    assert chunk["asset_key"] == "rag-assets/d1/t.png"
    assert chunk["table_body"] == table["table_body"]
    assert chunk["caption"] == "Bảng 1"
    assert chunk["page_idx"] == 2
    assert chunk["full_doc_id"] == "doc-1"


class _FakeTextChunks:
    def __init__(self, records, fail=False):
        self.records = records
        self.fail = fail

    async def get_by_ids(self, ids):
        if self.fail:
            raise RuntimeError("kv down")
        return [self.records.get(i) for i in ids]


class _QueryHost(QueryMixin):
    def __init__(self, result, records, fail=False):
        self.logger = logging.getLogger("test")
        self._result = result
        self.calls = []
        self.lightrag = SimpleNamespace(
            text_chunks=_FakeTextChunks(records, fail),
            aquery_data=self._aquery_data,
        )

    async def _ensure_lightrag_initialized(self):
        return {"success": True}

    async def _aquery_data_unused(self):  # pragma: no cover
        raise AssertionError

    async def _aquery_data(self, query, param):
        self.calls.append((query, param.mode))
        return self._result


def _result():
    return {
        "status": "success",
        "data": {
            "entities": [],
            "relationships": [],
            "chunks": [
                {"content": "văn bản", "chunk_id": "c-text", "file_path": "x"},
                {"content": "ảnh", "chunk_id": "c-img", "file_path": "x"},
                {"content": "bảng", "chunk_id": "c-tab", "file_path": "x"},
                {"content": "lạ", "chunk_id": "c-missing", "file_path": "x"},
            ],
            "references": [],
        },
    }


RECORDS = {
    "c-text": {"content": "văn bản"},
    "c-img": {
        "is_multimodal": True,
        "original_type": "image",
        "asset_key": "rag-assets/d1/i.png",
        "page_idx": 4,
        "caption": "Hình",
    },
    "c-tab": {
        "is_multimodal": True,
        "original_type": "table",
        "table_body": "| a |",
        "page_idx": 7,
    },
}


@pytest.mark.asyncio
async def test_aquery_data_enriches_chunks_with_modality():
    host = _QueryHost(_result(), RECORDS)
    result = await host.aquery_data("câu hỏi", mode="mix", top_k=5)
    chunks = {c["chunk_id"]: c for c in result["data"]["chunks"]}
    assert host.calls == [("câu hỏi", "mix")]
    assert chunks["c-text"]["modality"] == "text"
    assert chunks["c-text"]["asset_key"] is None
    assert chunks["c-img"]["modality"] == "image"
    assert chunks["c-img"]["asset_key"] == "rag-assets/d1/i.png"
    assert chunks["c-img"]["page_idx"] == 4
    assert chunks["c-img"]["caption"] == "Hình"
    assert chunks["c-tab"]["modality"] == "table"
    assert chunks["c-tab"]["table_body"] == "| a |"
    assert chunks["c-missing"]["modality"] == "text"


@pytest.mark.asyncio
async def test_aquery_data_does_not_raise_when_text_chunks_fail():
    host = _QueryHost(_result(), RECORDS, fail=True)
    result = await host.aquery_data("câu hỏi")
    assert all(c["modality"] == "text" for c in result["data"]["chunks"])


@pytest.mark.asyncio
async def test_aquery_data_passes_through_failure_response():
    host = _QueryHost({"status": "failure", "message": "empty", "data": {}}, {})
    result = await host.aquery_data("")
    assert result["status"] == "failure"
