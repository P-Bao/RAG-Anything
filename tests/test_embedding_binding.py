"""Tests for raganything.embedding_binding (nemotron gateway binding)."""

import asyncio
from types import SimpleNamespace

import pytest

from raganything.embedding_binding import build_embedding_func


class FakeEmbeddings:
    def __init__(self, dim=4, fail=False):
        self.dim = dim
        self.fail = fail
        self.calls = []

    async def create(self, model, input, extra_body=None):
        self.calls.append({"model": model, "input": list(input), "extra_body": extra_body})
        if self.fail:
            raise RuntimeError("gateway down")
        data = [
            SimpleNamespace(index=i, embedding=[0.1] * self.dim) for i in range(len(input))
        ]
        return SimpleNamespace(data=data)


class FakeClient:
    def __init__(self, dim=4):
        self.embeddings = FakeEmbeddings(dim)


def run(coro):
    return asyncio.run(coro)


def test_document_default_and_query_context():
    client = FakeClient()
    func = build_embedding_func(client=client, model="m", embedding_dim=4)
    out = run(func(["a", "b"]))
    assert client.embeddings.calls[0]["extra_body"] == {"input_type": "document"}
    out_q = run(func(["q"], context="query"))
    assert client.embeddings.calls[1]["extra_body"] == {"input_type": "query"}
    assert len(out) == 2 and len(out_q) == 1


def test_returns_numpy_array():
    """EmbeddingFunc validates result.size -> must be an ndarray."""
    import numpy as np

    func = build_embedding_func(client=FakeClient(), model="m")
    out = run(func(["a", "b"]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 4)
    assert out.dtype == float


def test_wrapped_in_lightrag_embeddingfunc():
    """Regression: wrapping in the real EmbeddingFunc used to raise
    AttributeError: 'list' object has no attribute 'size'."""
    import numpy as np

    from lightrag.utils import EmbeddingFunc

    func = build_embedding_func(client=FakeClient(), model="m")
    wrapped = EmbeddingFunc(embedding_dim=4, func=func)
    out = run(wrapped(["q"], context="query"))
    assert isinstance(out, np.ndarray) and out.shape == (1, 4)


def test_batching_preserves_order():
    client = FakeClient()
    func = build_embedding_func(client=client, model="m", embedding_dim=4, batch_size=2)
    out = run(func(["t1", "t2", "t3", "t4", "t5"]))
    assert [len(c["input"]) for c in client.embeddings.calls] == [2, 2, 1]
    assert len(out) == 5


def test_missing_embedding_raises():
    class BadClient:
        def __init__(self):
            self.embeddings = SimpleNamespace(
                create=self._create
            )

        async def _create(self, model, input, extra_body=None):
            return SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[0.1] * 4)]  # input had 2 items
            )

    func = build_embedding_func(client=BadClient(), model="m", embedding_dim=4)
    with pytest.raises(RuntimeError, match="missing"):
        run(func(["a", "b"]))


def test_out_of_range_index_raises():
    class BadClient:
        def __init__(self):
            self.embeddings = SimpleNamespace(create=self._create)

        async def _create(self, model, input, extra_body=None):
            return SimpleNamespace(
                data=[SimpleNamespace(index=7, embedding=[0.1] * 4)]
            )

    func = build_embedding_func(client=BadClient(), model="m", embedding_dim=4)
    with pytest.raises(RuntimeError, match="out-of-range"):
        run(func(["a"]))
