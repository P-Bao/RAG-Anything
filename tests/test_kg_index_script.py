"""Tests for scripts/kg_index_pdfs.py item formatting (template port)."""

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
    assert text == "[Hình ảnh]" and "Tóm tắt" not in text


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
