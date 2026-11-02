"""Tests for caption binding and MinIO caption reuse."""

import base64
from types import SimpleNamespace

import pytest

from raganything.caption_binding import (
    build_caption_func,
    image_mime,
    openai_style_messages,
)
from raganything.caption_reuse import CaptionReuseStore, join_caption


class FakeCompletions:
    def __init__(self, content="caption text"):
        self.content = content
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def _png_b64() -> str:
    return base64.b64encode(b"\x89PNG\r\n\x1a\nrest").decode()


def test_image_mime_sniffing():
    assert image_mime(b"\x89PNG") == "image/png"
    assert image_mime(b"RIFF####WEBP") == "image/webp"
    assert image_mime(b"GIF89a") == "image/gif"
    assert image_mime(b"\xff\xd8\xff") == "image/jpeg"


def test_openai_style_messages_image_part():
    msgs = openai_style_messages("describe", system_prompt="sys", image_data=_png_b64())
    assert msgs[0] == {"role": "system", "content": "sys"}
    parts = msgs[1]["content"]
    assert parts[0]["type"] == "text" and parts[0]["text"] == "describe"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_caption_func_language_injection_text_and_image():
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    func = build_caption_func(
        base_url="http://x", api_key="k", model="gemini", language="vi", client=client
    )
    out = await func("describe this", image_data=_png_b64())
    assert out == "caption text"
    call = client.chat.completions.calls[0]
    tail = call["messages"][-1]["content"]
    assert tail[0]["text"].endswith("\n\nRespond in vi.")
    out2 = await func("plain prompt", messages=[{"role": "user", "content": "raw"}])
    assert out2 == "caption text"
    assert client.chat.completions.calls[1]["messages"] == [
        {"role": "user", "content": "raw"}
    ]


def test_caption_func_none_without_api_key(monkeypatch):
    monkeypatch.delenv("CAPTION_LLM_API_KEY", raising=False)
    assert build_caption_func() is None


def test_join_caption_variants():
    assert join_caption(None) == ""
    assert join_caption(["a", None, "b"]) == "a b"
    assert join_caption("s") == "s"


class FakeMinioObject:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def close(self):
        pass

    def release_conn(self):
        pass


class FakeMinio:
    def __init__(self, store: dict[str, bytes]):
        self.store = store
        self.requested = []

    def get_object(self, bucket, key):
        self.requested.append((bucket, key))
        if key not in self.store:
            from minio.error import S3Error

            raise S3Error(
                response=SimpleNamespace(status=404, headers={}, read=lambda: b""),
                code="NoSuchKey", message="nope", resource=key, request_id="r",
                host_id="h", bucket_name=bucket, object_name=key,
            )
        return FakeMinioObject(self.store[key])


def test_caption_reuse_store_reads_artifact():
    payload = bytes(
        __import__("json").dumps(
            [{"type": "image", "asset_key": "k1", "image_caption": "hello"}]
        ),
        "utf-8",
    )
    client = FakeMinio({"rag-assets/doc1/content_list.json": payload})
    store = CaptionReuseStore(
        endpoint="e", access_key="a", secret_key="s", bucket="ami-data-documents",
        prefix="rag-assets", client=client,
    )
    assert store.content_list_key("doc1") == "rag-assets/doc1/content_list.json"
    assert store.load_content_list("doc1") == [
        {"type": "image", "asset_key": "k1", "image_caption": "hello"}
    ]
    assert client.requested == [("ami-data-documents", "rag-assets/doc1/content_list.json")]


def test_missing_artifacts_return_none_not_raise():
    client = FakeMinio({})
    store = CaptionReuseStore(
        endpoint="e", access_key="a", secret_key="s", client=client,
    )
    assert store.load_content_list("gone") is None
    assert store.load_descriptions("gone") is None
    assert store.get_descriptions("gone") == {}
