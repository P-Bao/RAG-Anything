"""Tests for raganything.backend_data_extract (env config + content merge)."""

import asyncio

import httpx
import pytest

from raganything.backend_data_extract import (
    DEFAULT_BASE_URL,
    DEFAULT_TABLE_ENDPOINT,
    DEFAULT_TEXT_ENDPOINT,
    BackendDataExtractor,
    BackendDataParseError,
    _group_qwen_tables,
    build_pdf_content_blocks,
    create_extractor,
    rebuild_content_list,
    resolve_ocr_fallback_modes,
)


def _clear_env(monkeypatch) -> None:
    for key in (
        "OCR_SERVICE_URL",
        "OCR_SERVICE_TIMEOUT",
        "OCR_TABLE_ENDPOINT",
        "OCR_TEXT_ENDPOINT",
        "OCR_FALLBACK_ON_EMPTY",
        "OCR_FALLBACK_MODES",
        "OCR_FALLBACK_ENABLED",
        "OCR_FALLBACK_URL",
        "OCR_FALLBACK_TOKEN",
        "OCR_FALLBACK_MODEL",
        "OCR_FALLBACK_TIMEOUT",
        "OCR_FALLBACK_RETRIES",
    ):
        monkeypatch.delenv(key, raising=False)


# ----------------------------------------------------------------------
# create_extractor env mapping
# ----------------------------------------------------------------------
def test_create_extractor_maps_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OCR_SERVICE_URL", "http://svc:9000/")
    monkeypatch.setenv("OCR_SERVICE_TIMEOUT", "45")
    monkeypatch.setenv("OCR_TABLE_ENDPOINT", "/custom/table")
    monkeypatch.setenv("OCR_TEXT_ENDPOINT", "custom/text")
    monkeypatch.setenv("OCR_FALLBACK_ON_EMPTY", "false")
    monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")

    ext = create_extractor()

    assert ext._base_url == "http://svc:9000"
    assert float(ext._timeout.read) == 45.0
    assert float(ext._timeout.connect) == 10.0
    assert ext._table_endpoint == "custom/table"
    assert ext._text_endpoint == "custom/text"
    assert ext._fallback_on_empty is False
    assert ext._fallback is None


def test_create_extractor_defaults(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")

    ext = create_extractor()

    assert ext._base_url == DEFAULT_BASE_URL
    assert float(ext._timeout.read) == 300.0
    assert ext._table_endpoint == DEFAULT_TABLE_ENDPOINT.lstrip("/")
    assert ext._text_endpoint == DEFAULT_TEXT_ENDPOINT.lstrip("/")
    assert ext._fallback_on_empty is True
    assert ext._fallback_modes == frozenset({"text", "table"})


# ----------------------------------------------------------------------
# resolve_ocr_fallback_modes
# ----------------------------------------------------------------------
def test_resolve_modes_parses_csv(monkeypatch):
    monkeypatch.setenv("OCR_FALLBACK_MODES", "text,table")
    assert resolve_ocr_fallback_modes() == frozenset({"text", "table"})


def test_resolve_modes_semicolon_and_case(monkeypatch):
    monkeypatch.setenv("OCR_FALLBACK_MODES", " TEXT ; table ")
    assert resolve_ocr_fallback_modes() == frozenset({"text", "table"})


def test_resolve_modes_invalid_raises(monkeypatch):
    monkeypatch.setenv("OCR_FALLBACK_MODES", "text,bogus")
    with pytest.raises(ValueError):
        resolve_ocr_fallback_modes()


def test_resolve_modes_unset_defaults(monkeypatch):
    monkeypatch.delenv("OCR_FALLBACK_MODES", raising=False)
    assert resolve_ocr_fallback_modes() == frozenset({"text", "table"})


def test_resolve_modes_empty_string_disables_fallback(monkeypatch):
    # Sibling parity: explicit empty value means "no fallback", unlike unset.
    monkeypatch.setenv("OCR_FALLBACK_MODES", "")
    assert resolve_ocr_fallback_modes() == frozenset()


def test_resolve_modes_whitespace_only_is_empty(monkeypatch):
    monkeypatch.setenv("OCR_FALLBACK_MODES", "   ")
    assert resolve_ocr_fallback_modes() == frozenset()


# ----------------------------------------------------------------------
# _group_qwen_tables
# ----------------------------------------------------------------------
def test_group_picks_offset_maximizing_hits():
    t1 = {"page": "1", "html": "a"}
    t2 = {"page": "2", "html": "b"}
    groups = _group_qwen_tables([t1, t2], valid_pages={2, 3})
    assert set(groups) == {2, 3}
    assert groups[2] == [t1]
    assert groups[3] == [t2]


def test_group_none_and_bool_pages():
    tn = {"page": None, "html": "n"}
    tb = {"page": True, "html": "b"}
    t2 = {"page": 2, "html": "c"}
    groups = _group_qwen_tables([tn, tb, t2], valid_pages={2})
    assert None in groups
    assert len(groups[None]) == 2
    assert groups[2] == [t2]


def test_group_no_valid_pages_keeps_zero_offset():
    t0 = {"page": 0, "html": "a"}
    t1 = {"page": 1, "html": "b"}
    groups = _group_qwen_tables([t0, t1], valid_pages=None)
    assert set(groups) == {0, 1}


# ----------------------------------------------------------------------
# build_pdf_content_blocks
# ----------------------------------------------------------------------
def _page_one_blocks():
    return [
        {"type": "table", "content": "| p1 |", "html": "<table>pdf1</table>", "page": 1, "bbox": {"top": 10}},
        {"type": "paragraph", "content": "mid", "page": 1, "bbox": {"top": 20}},
        {"type": "table", "content": "| p2 |", "html": "<table>pdf2</table>", "page": 1, "bbox": {"top": 30}},
    ]


def test_build_blocks_interleaves_and_replaces_html():
    out = build_pdf_content_blocks(_page_one_blocks(), [{"page": 1, "html": "<table>qw1</table>"}])
    assert out == [
        {"type": "table", "table_body": "<table>qw1</table>", "page_idx": 0},
        {"type": "text", "text": "mid", "page_idx": 0},
        {"type": "table", "table_body": "<table>pdf2</table>", "page_idx": 0},
    ]


def test_build_blocks_appends_no_page_qwen():
    blocks = [{"type": "paragraph", "content": "t", "page": 1, "bbox": {"top": 1}}]
    out = build_pdf_content_blocks(blocks, [{"page": None, "html": "<table>np</table>"}])
    assert out == [
        {"type": "text", "text": "t", "page_idx": 0},
        {"type": "table", "table_body": "<table>np</table>", "page_idx": None},
    ]


def test_build_blocks_appends_orphan_qwen_page():
    blocks = [{"type": "paragraph", "content": "t", "page": 1, "bbox": {"top": 1}}]
    out = build_pdf_content_blocks(blocks, [{"page": 3, "html": "<table>o</table>"}])
    assert out[-1] == {"type": "table", "table_body": "<table>o</table>", "page_idx": 2}


# ----------------------------------------------------------------------
# rebuild_content_list
# ----------------------------------------------------------------------
def test_rebuild_content_list_none_for_unsupported_ext(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    ext = BackendDataExtractor()
    assert asyncio.run(rebuild_content_list([], p, ext)) is None


# ----------------------------------------------------------------------
# BackendDataExtractor HTTP error path
# ----------------------------------------------------------------------
def test_post_file_500_raises_with_status_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ext = BackendDataExtractor(base_url="http://svc", timeout=5.0, http_client=client)
    with pytest.raises(BackendDataParseError, match="500"):
        asyncio.run(ext._post_file(ext._text_endpoint, b"x", "f.pdf", "application/pdf"))


def test_extract_tables_500_raises_without_fallback():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ext = BackendDataExtractor(base_url="http://svc", timeout=5.0, http_client=client)
    with pytest.raises(BackendDataParseError, match="500"):
        asyncio.run(ext.extract_tables(b"x", "f.pdf"))
