"""Gemini caption binding for modal processors (KG branch).

Ported from the ami-multimodal-retrieval describe stage: caption images via
an OpenAI-compatible endpoint (Google Gemini) configured by ``CAPTION_LLM_*``
envs. Returns ``None`` when no API key is set — callers fall back to reusing
captions already stored in MinIO content_list artifacts (see
``raganything.caption_reuse``) and must fail loudly when neither source
exists. VLM-enhanced paths are intentionally not provided: the KG branch is
text-only apart from image captioning.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Callable


def image_mime(data: bytes) -> str:
    """Sniff image mime from magic bytes (png/webp/gif, else jpeg)."""
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:3] == b"GIF":
        return "image/gif"
    return "image/jpeg"


def openai_style_messages(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    image_data: str | None = None,
) -> list[dict]:
    """Build chat messages; ``image_data`` (base64) becomes a data-URI part."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    if image_data:
        mime = image_mime(base64.b64decode(image_data))
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
        ]
    else:
        content = prompt
    messages.append({"role": "user", "content": content})
    return messages


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def build_caption_func(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    language: str | None = None,
    client: Any = None,
) -> Callable[..., Any] | None:
    """Build the ``modal_caption_func`` for image processors.

    Signature matches RAG-Anything vision funcs:
    ``(prompt, system_prompt=None, history_messages=None, image_data=<base64>,
    messages=None, **kwargs)``. ``language`` (default env
    ``CAPTION_LLM_LANGUAGE``) is appended as a response-language instruction.
    ``None`` when no API key is configured.
    """
    api_key = api_key if api_key is not None else _env("CAPTION_LLM_API_KEY")
    if not api_key:
        return None
    from openai import AsyncOpenAI

    client = client or AsyncOpenAI(
        base_url=base_url if base_url is not None else _env("CAPTION_LLM_BASE_URL"),
        api_key=api_key,
        timeout=float(timeout if timeout is not None else _env("CAPTION_LLM_TIMEOUT", "120") or 120),
        max_retries=int(max_retries if max_retries is not None else _env("CAPTION_LLM_RETRIES", "2") or 0),
    )
    model_name = model if model is not None else _env("CAPTION_LLM_MODEL", "gemini-3.5-flash-lite")
    lang = (language if language is not None else _env("CAPTION_LLM_LANGUAGE", "vi")).strip()

    async def gemini_caption(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages is None:
            messages = openai_style_messages(prompt, system_prompt, history_messages, image_data)
            if lang:
                tail = messages[-1]
                suffix = f"\n\nRespond in {lang}."
                if isinstance(tail["content"], str):
                    tail["content"] += suffix
                else:
                    for part in tail["content"]:
                        if part.get("type") == "text":
                            part["text"] += suffix
                            break
        resp = await client.chat.completions.create(model=model_name, messages=messages, **kwargs)
        return resp.choices[0].message.content

    return gemini_caption
