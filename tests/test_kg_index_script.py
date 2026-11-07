"""Tests for scripts/kg_index_pdfs.py item formatting (template port)."""

import asyncio
import importlib.util
import os
import sys

import pytest

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "kg_index_pdfs.py"
)
_spec = importlib.util.spec_from_file_location("kg_index_pdfs", _SCRIPT)
_kg = importlib.util.module_from_spec(_spec)
sys.modules["kg_index_pdfs"] = _kg
_spec.loader.exec_module(_kg)


def test_format_image_with_reused_caption():
    text = _kg.format_item(
        {"type": "image", "img_caption": ["fig 1"], "img_footnote": "note"},
        description="A chart of sales",
    )
    assert text == "[Hình ảnh]\nA chart of sales\nTóm tắt: fig 1\nnote"


def test_format_image_without_caption_has_no_noise_line():
    text = _kg.format_item({"type": "image"}, description="")
    assert text == "" and "Tóm tắt" not in text


def test_format_table_without_body_or_caption_is_dropped():
    assert _kg.format_item({"type": "table", "table_body": ""}, description="") == ""


def test_format_table_with_body_and_description():
    text = _kg.format_item(
        {"type": "table", "table_caption": "Revenue", "table_body": "2024 | 12"},
        description="Annual revenue table",
    )
    assert text == "[Bảng]\nChú thích: Revenue\nMô tả: Annual revenue table\n2024 | 12"


def test_format_equation():
    text = _kg.format_item({"type": "equation", "latex": "E=mc^2"}, description="")
    assert text == "[Công thức]\nCông thức: E=mc^2"


def test_assemble_preserves_order_and_counts():
    content_list = [
        {"type": "text", "text": "para one"},
        {"type": "image", "img_caption": "cap"},
        {"type": "table", "table_body": "body"},
    ]
    text, counts = _kg.assemble_document_text(content_list, {1: "img desc"})
    assert text.startswith("para one\n\n[Hình ảnh]\nimg desc\nTóm tắt: cap")
    assert "[Bảng]" in text
    assert counts == {"text": 1, "image": 1, "table": 1}


def test_assemble_drops_empty_items():
    text, counts = _kg.assemble_document_text([{"type": "text", "text": ""}], {})
    assert text == ""
    assert counts == {"text": 1}


class _Obj:
    def __init__(self, name, last_modified):
        self.object_name = name
        self.last_modified = last_modified


class _Store:
    """Duck-typed CaptionReuseStore (only what list_candidate_docs touches)."""

    bucket = "bucket"
    prefix = "rag-assets"
    _client = None

    def list_objects(self, bucket, prefix="", recursive=True):
        return iter(
            [
                _Obj("rag-assets/d1/content_list.json", 1),
                _Obj("rag-assets/d2/content_list.json", 3),
                _Obj("rag-assets/d3/content_list.json", 2),
                _Obj("rag-assets/d2/sha16.png", 9),  # noise: not a content_list
            ]
        )


def test_list_candidate_docs_limit_zero_means_all():
    store = _Store()
    store._client = store
    assert asyncio.run(_kg.list_candidate_docs(store, 0)) == ["d2", "d3", "d1"]
    assert asyncio.run(_kg.list_candidate_docs(store, 1)) == ["d2"]  # newest first


def test_reindex_allowlist_pins_the_four_real_documents():
    assert len(_kg.REINDEX_DOC_IDS) == 4
    assert "6a8ee761694f1f325e516aa9" in _kg.REINDEX_DOC_IDS  # Sổ tay SV 2026
    assert "6a8f144188f0aede5ebc8828" in _kg.REINDEX_DOC_IDS  # Cố vấn học tập 2022
    assert "6a8eec753b7bc85ec8357ed7" in _kg.REINDEX_DOC_IDS  # TB xét tuyển ThS
    assert "6a8eec423b7bc85ec8357ed6" in _kg.REINDEX_DOC_IDS  # QĐ xét tuyển tài năng


def test_reindex_wipes_every_data_storage_but_keeps_llm_cache():
    """Wipe covers chunks/graph/vdbs/doc_status; extraction cache survives."""
    assert "chunks_vdb" in _kg._WIPE_STORAGES
    assert "text_chunks" in _kg._WIPE_STORAGES
    assert "chunk_entity_relation_graph" in _kg._WIPE_STORAGES
    assert "doc_status" in _kg._WIPE_STORAGES
    assert "llm_response_cache" not in _kg._WIPE_STORAGES


def test_format_image_marker_carries_asset_key():
    text = _kg.format_item(
        {"type": "image", "asset_key": "rag-assets/d1/abc.png"}, description="desc"
    )
    assert text == "[Hình ảnh]\nẢnh: rag-assets/d1/abc.png\ndesc"


def test_format_image_without_asset_key_has_no_marker():
    text = _kg.format_item({"type": "image"}, description="desc")
    assert text == "[Hình ảnh]\ndesc"


def test_format_table_marker_only_with_asset_key():
    with_asset = _kg.format_item(
        {"type": "table", "table_body": "b", "asset_key": "k1"}, description=""
    )
    assert with_asset.startswith("[Bảng]\nẢnh: k1\n")
    without = _kg.format_item({"type": "table", "table_body": "b"}, description="")
    assert without.startswith("[Bảng]\nChú thích:")
