"""Tests for raganything.parse_pipeline (parse + describe stages)."""

import asyncio
import hashlib
import json

import raganything.backend_data_extract as bde
from raganything.parse_pipeline import (
    ParseArtifactStore,
    _describe_items,
    _parser_kwargs,
    parse_and_store,
)

BUCKET = "test-bucket"
PREFIX = "rag-assets"


class _FakeMinio:
    """Duck-typed Minio client recording put_object / get_object calls."""

    def __init__(self):
        self.puts = []
        self.objects = {}

    def put_object(self, bucket, key, data, length=None, content_type=None):
        raw = data.read() if hasattr(data, "read") else data
        self.puts.append(
            {
                "bucket": bucket,
                "key": key,
                "data": raw,
                "length": length,
                "content_type": content_type,
            }
        )
        self.objects[(bucket, key)] = raw

    def get_object(self, bucket, key):
        raise KeyError(key)


def _store():
    client = _FakeMinio()
    return ParseArtifactStore(client=client, bucket=BUCKET, prefix=PREFIX), client


def _put_keys(client):
    return [p["key"] for p in client.puts]


# --------------------------------------------------------------------------
# upload_content_list_assets
# --------------------------------------------------------------------------
def test_upload_assets_sha16_key_and_stamp(tmp_path):
    f = tmp_path / "pic.png"
    data = b"\x89PNG tiny payload"
    f.write_bytes(data)
    sha16 = hashlib.sha256(data).hexdigest()[:16]

    store, client = _store()
    content_list = [{"type": "image", "img_path": str(f)}]
    keys = store.upload_content_list_assets("doc1", content_list)

    expected = f"{PREFIX}/doc1/{sha16}.png"
    assert keys == [expected]
    assert content_list[0]["asset_key"] == expected
    assert _put_keys(client) == [expected]
    assert client.puts[0]["data"] == data


def test_upload_assets_duplicate_bytes_uploaded_once(tmp_path):
    data = b"same bytes"
    f1 = tmp_path / "a.png"
    f2 = tmp_path / "b.png"
    f1.write_bytes(data)
    f2.write_bytes(data)

    store, client = _store()
    content_list = [
        {"type": "image", "img_path": str(f1)},
        {"type": "image", "img_path": str(f2)},
    ]
    keys = store.upload_content_list_assets("doc1", content_list)

    assert len(keys) == 1
    assert len(client.puts) == 1
    assert content_list[0]["asset_key"] == content_list[1]["asset_key"] == keys[0]


def test_upload_assets_non_asset_types_skipped(tmp_path):
    f = tmp_path / "pic.png"
    f.write_bytes(b"x")
    store, client = _store()
    content_list = [{"type": "text", "img_path": str(f), "text": "hi"}]
    keys = store.upload_content_list_assets("doc1", content_list)
    assert keys == []
    assert client.puts == []
    assert "asset_key" not in content_list[0]


def test_upload_assets_missing_file_skipped(tmp_path):
    store, client = _store()
    content_list = [{"type": "image", "img_path": str(tmp_path / "nope.png")}]
    keys = store.upload_content_list_assets("doc1", content_list)
    assert keys == []
    assert client.puts == []
    assert "asset_key" not in content_list[0]


# --------------------------------------------------------------------------
# save_content_list / save_descriptions
# --------------------------------------------------------------------------
def test_save_content_list_key_and_roundtrip():
    store, client = _store()
    payload = [{"type": "text", "text": "hello"}]
    key = store.save_content_list("doc1", payload)
    assert key == f"{PREFIX}/doc1/content_list.json"
    assert _put_keys(client) == [key]
    assert json.loads(client.puts[0]["data"].decode("utf-8")) == payload


def test_save_descriptions_key_and_roundtrip():
    store, client = _store()
    payload = [{"index": 0, "type": "image", "description": "cap"}]
    key = store.save_descriptions("doc1", payload)
    assert key == f"{PREFIX}/doc1/descriptions.json"
    assert _put_keys(client) == [key]
    assert json.loads(client.puts[0]["data"].decode("utf-8")) == payload


# --------------------------------------------------------------------------
# parse_and_store
# --------------------------------------------------------------------------
class _FakeParser:
    def __init__(self, content_list):
        self.content_list = content_list
        self.calls = []

    def parse_document(self, path, **kwargs):
        self.calls.append({"path": path, **kwargs})
        return [dict(c) for c in self.content_list]


_SAMPLE = [
    {"type": "text", "text": "hello"},
    {"type": "image", "img_path": "x.png", "img_caption": "cap"},
]


def test_parse_and_store_mineru_text_source(monkeypatch):
    monkeypatch.setenv("PARSE_TEXT_SOURCE", "mineru")
    fake_parser = _FakeParser(_SAMPLE)

    store, client = _store()
    content_list, descriptions = asyncio.run(
        parse_and_store(
            store, "doc1", "some.pdf", caption_func=None, parser=fake_parser
        )
    )

    assert content_list == _SAMPLE
    assert fake_parser.calls[0]["path"] == "some.pdf"
    assert fake_parser.calls[0]["method"]
    assert descriptions == [
        {"index": 1, "type": "image", "description": "cap"}
    ]
    assert set(_put_keys(client)) == {
        f"{PREFIX}/doc1/content_list.json",
        f"{PREFIX}/doc1/descriptions.json",
    }


def test_parse_and_store_backend_data_rebuild(monkeypatch):
    monkeypatch.setenv("PARSE_TEXT_SOURCE", "backend_data")
    fake_parser = _FakeParser(_SAMPLE)

    rebuilt_list = [{"type": "text", "text": "rebuilt"}]

    async def fake_rebuild(mineru_list, file_path):
        return rebuilt_list

    store, _client = _store()
    content_list, _descriptions = asyncio.run(
        parse_and_store(
            store,
            "doc1",
            "some.pdf",
            caption_func=None,
            parser=fake_parser,
            rebuild=fake_rebuild,
        )
    )

    assert content_list == rebuilt_list


# --------------------------------------------------------------------------
# _parser_kwargs
# --------------------------------------------------------------------------
def test_parser_kwargs_from_env(monkeypatch):
    monkeypatch.setenv("MINERU_BACKEND", "pipeline")
    monkeypatch.setenv("MINERU_LANG", "en")
    monkeypatch.setenv("MINERU_SOURCE", "local")
    monkeypatch.setenv("MINERU_TIMEOUT", "90")
    monkeypatch.setenv("MINERU_DEVICE", "cuda")
    monkeypatch.setenv("MINERU_VIRTUAL_VRAM_SIZE", "3")

    assert _parser_kwargs() == {
        "backend": "pipeline",
        "lang": "en",
        "source": "local",
        "timeout": 90,
        "env": {"MINERU_DEVICE_MODE": "cuda", "MINERU_VIRTUAL_VRAM_SIZE": "3"},
    }


def test_parser_kwargs_empty_defaults(monkeypatch):
    for name in (
        "MINERU_BACKEND",
        "MINERU_LANG",
        "MINERU_SOURCE",
        "MINERU_TIMEOUT",
        "MINERU_DEVICE",
        "MINERU_VIRTUAL_VRAM_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)
    assert _parser_kwargs() == {}


# --------------------------------------------------------------------------
# describe: table item
# --------------------------------------------------------------------------
def test_describe_table_raw_mode_empty(monkeypatch):
    monkeypatch.delenv("TABLE_DESCRIBE_MODE", raising=False)
    store, _client = _store()
    content_list = [{"type": "table", "table_body": "h1 | h2"}]
    out = asyncio.run(_describe_items(store, content_list, None))
    assert out == [{"index": 0, "type": "table", "description": ""}]


def test_describe_table_llm_mode_uses_caption_func(monkeypatch):
    monkeypatch.setenv("TABLE_DESCRIBE_MODE", "llm")
    prompts = []

    async def caption_func(prompt, **kw):
        prompts.append(prompt)
        return "TBL DESC"

    store, _client = _store()
    content_list = [{"type": "table", "table_body": "h1 | h2"}]
    out = asyncio.run(_describe_items(store, content_list, caption_func))
    assert out == [{"index": 0, "type": "table", "description": "TBL DESC"}]
    assert "h1 | h2" in prompts[0]


def test_describe_table_caption_error_is_swallowed(monkeypatch):
    monkeypatch.setenv("TABLE_DESCRIBE_MODE", "llm")

    async def boom(prompt, **kw):
        raise RuntimeError("nope")

    store, _client = _store()
    content_list = [{"type": "table", "table_body": "h1 | h2"}]
    out = asyncio.run(_describe_items(store, content_list, boom))
    assert out == [{"index": 0, "type": "table", "description": ""}]
