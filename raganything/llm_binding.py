"""Qwen3.5 LLM binding for LightRAG (KG branch).

Single OpenAI-compatible endpoint (vLLM). ``thinking`` is per-role:
LightRAG 1.5.7 role wrappers inject role kwargs (see ``role_llm_configs``),
so extract/keyword run with thinking OFF (structured output, cheap) and
query runs with thinking ON (recommended by the model card).

Response contract: reasoning arrives in a separate ``reasoning_content``
field and is ignored on purpose — only ``message.content`` is returned.
``content=None`` raises loudly (vLLM Qwen3.5 bug #38894) instead of
silently yielding an empty knowledge graph.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from openai import AsyncOpenAI

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
# LightRAG guidance for the Qwen3 family: 0.8-1.0 avoids greedy decode loops
# during entity extraction.
DEFAULT_TEMPERATURE = 0.9
# Enough for full chunk extraction output (measured 782 tokens on a small
# chunk); prevents truncated ("length") responses that drop entities.
DEFAULT_MAX_TOKENS = 4000


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def build_llm_model_func(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    client: AsyncOpenAI | None = None,
) -> Callable[..., Any]:
    """Build an async ``llm_model_func`` for LightRAG.

    ``thinking`` is read from kwargs; LightRAG role wrappers inject it from
    ``role_llm_configs`` (see :func:`role_llm_configs`). ``hashing_kv`` is
    accepted and unused: response caching stays off for this binding.
    """
    client = client or AsyncOpenAI(
        base_url=base_url or _env("QWEN_LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key or _env("QWEN_LLM_API_KEY", "EMPTY"),
    )
    model_name = model or _env("QWEN_LLM_MODEL", DEFAULT_MODEL)
    temp = DEFAULT_TEMPERATURE if temperature is None else temperature
    max_tok = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        hashing_kv: Any = None,
        thinking: bool = False,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        if kwargs.get("image_data"):
            # Text-only contract: a silently ignored image would produce a
            # hallucinated description (modalprocessors.py passes image_data).
            raise RuntimeError(
                "text-only Qwen binding received image_data; set "
                "CAPTION_LLM_API_KEY or rely on MinIO caption reuse"
            )
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature if temperature is not None else temp,
            max_tokens=max_tok,
            extra_body={"thinking": thinking},
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(
                f"Qwen LLM returned content=None (thinking={thinking}); "
                "see vLLM issue #38894 - check server reasoning parser"
            )
        return content

    return llm_model_func


def role_llm_configs(query_thinking: bool = True) -> dict[str, dict[str, Any]]:
    """Per-role kwargs for ``LightRAG(role_llm_configs=...)``.

    Only the query role overrides defaults: thinking ON (recommended by the
    model card) and a lower temperature for grounded answers (extraction
    keeps 0.9, the LightRAG guidance against greedy decode loops).
    """
    return {"query": {"kwargs": {"thinking": query_thinking, "temperature": 0.6}}}
