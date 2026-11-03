"""Tests for raganything.ocr_fallback (Vinter OCR fallback client)."""

import asyncio

import httpx
import pytest

from raganything.ocr_fallback import (
    OcrFallbackError,
    VinterOcrClient,
    build_vinter_ocr_client,
)


def _async_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _clear_fallback_env(monkeypatch) -> None:
    for key in (
        "OCR_FALLBACK_ENABLED",
        "OCR_FALLBACK_URL",
        "OCR_FALLBACK_MODEL",
        "OCR_FALLBACK_TOKEN",
        "OCR_FALLBACK_TIMEOUT",
        "OCR_FALLBACK_RETRIES",
    ):
        monkeypatch.delenv(key, raising=False)


def test_ocr_text_success_captures_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["accept"] = request.headers.get("accept")
        captured["body"] = request.read()
        return httpx.Response(200, json={"page_texts": [{"page": 1, "text": "hello"}]})

    client = VinterOcrClient(
        base_url="http://vinter",
        token="secret",
        timeout=5.0,
        http_client=_async_client(handler),
    )
    out = asyncio.run(client.ocr_text(b"%PDF-1.4", "doc.pdf"))
    assert out == "hello"
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/ocr"
    assert captured["auth"] == "Bearer secret"
    assert captured["accept"] == "application/json"
    body = captured["body"]
    assert b'name="mode"' in body
    assert b"text" in body
    assert b'name="file"' in body
    assert b'filename="doc.pdf"' in body
    assert b"%PDF-1.4" in body


def test_retry_then_success_counts_two_attempts():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"page_texts": [{"page": 1, "text": "ok"}]})

    client = VinterOcrClient(
        base_url="http://vinter",
        retries=1,
        timeout=5.0,
        http_client=_async_client(handler),
    )
    out = asyncio.run(client.ocr_text(b"x", "a.pdf"))
    assert out == "ok"
    assert calls["n"] == 2


def test_all_attempts_fail_raises_ocr_fallback_error():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="down")

    client = VinterOcrClient(
        base_url="http://vinter",
        retries=0,
        timeout=5.0,
        http_client=_async_client(handler),
    )
    with pytest.raises(OcrFallbackError, match="503"):
        asyncio.run(client.ocr_text(b"x", "a.pdf"))
    assert calls["n"] == 1


def test_four_xx_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    client = VinterOcrClient(
        base_url="http://vinter",
        retries=3,
        timeout=5.0,
        http_client=_async_client(handler),
    )
    with pytest.raises(OcrFallbackError, match="400"):
        asyncio.run(client.ocr_text(b"x", "a.pdf"))
    assert calls["n"] == 1


def test_build_client_none_when_disabled(monkeypatch):
    _clear_fallback_env(monkeypatch)
    monkeypatch.setenv("OCR_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("OCR_FALLBACK_URL", "http://vinter")
    assert build_vinter_ocr_client() is None


def test_build_client_when_enabled_and_configured(monkeypatch):
    _clear_fallback_env(monkeypatch)
    monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("OCR_FALLBACK_URL", "http://vinter:8090")
    monkeypatch.setenv("OCR_FALLBACK_TOKEN", "secret")
    client = build_vinter_ocr_client()
    assert isinstance(client, VinterOcrClient)
    assert client._base_url == "http://vinter:8090"
    assert client._token == "secret"


def test_build_client_empty_token_still_returns_client(monkeypatch):
    # Spec expected None on empty token; module does NOT gate on token (report).
    _clear_fallback_env(monkeypatch)
    monkeypatch.setenv("OCR_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("OCR_FALLBACK_URL", "http://vinter")
    client = build_vinter_ocr_client()
    assert isinstance(client, VinterOcrClient)
    assert client._token == ""
