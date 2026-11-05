"""Tests for raganything.llm_binding (Qwen3.5 binding)."""

import asyncio
from types import SimpleNamespace

import pytest

from raganything.llm_binding import build_llm_model_func, role_llm_configs


class FakeCompletions:
    """SSE-shaped stream: one chunk per delta char + a usage chunk."""

    def __init__(self, content):
        self.calls = []
        self._content = content

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        async def gen():
            for ch in self._content or "":
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=ch), finish_reason=None
                        )
                    ]
                )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None), finish_reason="length"
                    )
                ]
            )
            yield SimpleNamespace(choices=[])  # usage-only frame

        return gen()


class FakeClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


def run(func, *args, **kwargs):
    return asyncio.run(func(*args, **kwargs))


def test_returns_content_and_thinking_off_by_default():
    client = FakeClient("  result")
    func = build_llm_model_func(client=client, model="m", temperature=0.9, max_tokens=4000)
    out = run(func, "prompt text")
    assert out == "  result"
    call = client.chat.completions.calls[0]
    assert call["extra_body"] == {"thinking": False}
    assert call["model"] == "m"
    assert call["max_tokens"] == 4000
    assert call["messages"][-1] == {"role": "user", "content": "prompt text"}


def test_thinking_on_via_kwarg_and_system_prompt():
    client = FakeClient("answer")
    func = build_llm_model_func(client=client, model="m")
    out = run(func, "q", system_prompt="sys", thinking=True)
    assert out == "answer"
    call = client.chat.completions.calls[0]
    assert call["extra_body"] == {"thinking": True}
    assert call["messages"][0] == {"role": "system", "content": "sys"}


def test_content_none_raises():
    client = FakeClient(None)
    func = build_llm_model_func(client=client, model="m")
    with pytest.raises(RuntimeError, match="content=None"):
        run(func, "q")


def test_image_data_raises_text_only_contract():
    client = FakeClient("hallucination")
    func = build_llm_model_func(client=client, model="m")
    with pytest.raises(RuntimeError, match="image_data"):
        run(func, "describe", image_data="aGVsbG8=")
    assert client.chat.completions.calls == []  # no LLM call made


def test_temperature_override_via_role_kwarg():
    client = FakeClient("ok")
    func = build_llm_model_func(client=client, model="m")
    run(func, "q", temperature=0.6)
    assert client.chat.completions.calls[0]["temperature"] == 0.6


def test_streams_and_ignores_usage_frame():
    client = FakeClient("Hi there")
    func = build_llm_model_func(client=client, model="m")
    out = run(func, "q")
    assert out == "Hi there"  # nối delta, bỏ chunk choices=[]
    assert client.chat.completions.calls[0]["stream"] is True


def test_role_llm_configs_thinking_query_only():
    configs = role_llm_configs()
    assert configs == {"query": {"kwargs": {"thinking": True, "temperature": 0.6}}}
    assert "extract" not in configs and "keyword" not in configs
