"""OpenAIEmbedder - client embedding server OpenAI-compatible (vLLM).

Dùng cho ``nvidia/llama-nemotron-embed-vl-1b-v2`` serve bằng vLLM theo model card:
- POST ``/v1/embeddings`` với ``messages`` (role ``query`` | ``document``,
  content parts ``text`` / ``image_url`` data URI). Template phía server tự
  prepend prefix ``query:`` / ``passage:`` theo role.
- Handshake: ``GET /v1/models`` so model id, rồi probe embed 1 lần để lấy dim
  thực tế; lệch với config thì dừng, không bao giờ ghi vector khi chưa xác minh.
- Retry exponential backoff + jitter CHỈ cho lỗi tạm thời; KHÔNG retry 4xx.
- 1 item / request (vLLM chat-embeddings); throughput điều khiển bởi
  ``max_concurrency``.
- Cache embedding (tuỳ chọn, bật mặc định) theo khoá (model, dim, hash nội dung)
  - chia sẻ EmbeddingCache với RemoteEmbedder.
- Circuit breaker: N request liên tiếp mất kết nối -> dừng cả lô sớm.
"""

import asyncio
import base64
import hashlib
import logging
import random
from itertools import islice
from typing import Protocol

import httpx

from ami_rag.core.embedder import (
    EmbedAuthError,
    EmbedCircuitOpen,
    EmbedderError,
    EmbedInputTooLong,
    EmbedModelMismatch,
    EmbedServerOOM,
    EmbedServerUnreachable,
)

logger = logging.getLogger(__name__)

# N request liên tiếp mất kết nối -> dừng cả lô sớm (thay vì đánh failed hàng loạt)
CIRCUIT_THRESHOLD = 5


class _Cache(Protocol):
    def get_many(self, keys: list[str]) -> dict[str, list[float]]: ...
    def put_many(self, mapping: dict[str, list[float]]) -> None: ...


def _image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:3] == b"GIF":
        return "image/gif"
    return "image/jpeg"


def _data_url(b64: str) -> str:
    """base64 -> data URI; đo mime từ magic bytes (decode tối đa 12 byte đầu)."""
    n = min(len(b64), 16)
    n -= n % 4
    mime = "image/jpeg"
    if n:
        try:
            mime = _image_mime(base64.b64decode(b64[:n]))
        except ValueError:
            mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"


class OpenAIEmbedder:
    """Client embedding server OpenAI-compatible (vLLM, Nemotron VL embed)."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "nvidia/llama-nemotron-embed-vl-1b-v2",
        expected_dim: int | None = None,
        token: str = "",
        timeout: float = 60,
        max_concurrency: int = 4,
        retries: int = 3,
        backoff_base: float = 0.5,
        cache: _Cache | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_name = model
        self.expected_dim = expected_dim
        self.dim = expected_dim if expected_dim is not None else 0
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.retries = retries
        self.backoff_base = backoff_base
        self.cache = cache
        self._client = http_client or httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
        self._semaphore: asyncio.Semaphore | None = None
        self._verified = False
        self._instruction_ns = ""
        self._consecutive_unreachable = 0

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------
    async def verify(self) -> dict:
        """GET /v1/models so model id + probe embed lấy dim. Lệch -> dừng."""
        try:
            resp = await self._client.get(f"{self.base_url}/v1/models")
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise EmbedServerUnreachable(
                f"embed server không với tới được ({self.base_url}): {exc}"
            ) from exc
        if resp.status_code in (401, 403):
            raise EmbedAuthError(f"embed server từ chối token: {resp.status_code}")
        if resp.status_code != 200:
            raise EmbedServerUnreachable(
                f"embed server /v1/models trả {resp.status_code}"
            )
        info = resp.json()

        ids = [m.get("id") for m in info.get("data") or []]
        if self.model not in ids:
            raise EmbedModelMismatch(
                f"embed server chạy model {ids}, config mong đợi '{self.model}'. "
                "Không ghi vector khi model chưa xác minh."
            )

        # Probe 1 lần để lấy dim thực tế (API embeddings không khai báo dim)
        result = await self._post_embed_one({"text": "probe"}, "document")
        data = result.get("data") or []
        if not data or not data[0].get("embedding"):
            raise EmbedderError("embed server trả response /v1/embeddings rỗng")
        server_dim = len(data[0]["embedding"])
        if self.expected_dim is not None and server_dim != self.expected_dim:
            raise EmbedModelMismatch(
                f"embed server dim={server_dim}, config EMBED_DIM={self.expected_dim}. "
                "Đổi model/dim phía server thì cập nhật EMBED_DIM và dùng collection mới."
            )

        self.dim = int(server_dim)
        self._instruction_ns = hashlib.sha256(self.model.encode()).hexdigest()[:16]
        self._verified = True
        logger.info(
            "embed server handshake ok: model=%s dim=%s (vLLM OpenAI-compatible)",
            self.model,
            server_dim,
        )
        return info

    async def _ensure_verified(self) -> None:
        if not self._verified:
            await self.verify()

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    async def embed_documents(self, items: list[str | dict]) -> list[list[float]]:
        """Embed danh sách văn bản / multimodal items (type=document, có cache). Giữ thứ tự đầu vào."""
        if not items:
            return []
        await self._ensure_verified()

        vectors: list[list[float] | None] = [None] * len(items)
        keys: list[str] = [self._cache_key(it) for it in items]
        pending: list[int] = list(range(len(items)))
        if self.cache is not None:
            cached = await asyncio.to_thread(self.cache.get_many, keys)
            pending = [i for i in range(len(items)) if keys[i] not in cached]
            for i, key in enumerate(keys):
                if key in cached:
                    vectors[i] = cached[key]

        if pending:
            unreachable: EmbedServerUnreachable | None = None
            for group in self._groups(pending, self.max_concurrency):
                results = await asyncio.gather(
                    *(self._post_embed_one(items[i], "document") for i in group),
                    return_exceptions=True,
                )
                for i, result in zip(group, results):
                    if isinstance(result, EmbedServerUnreachable):
                        # Giữ lỗi đầu tiên; circuit breaker quyết định dừng hay không
                        unreachable = unreachable or result
                        continue
                    if isinstance(result, BaseException):
                        raise result
                    self._check_response(result)
                    data = result["data"]
                    if not data or not data[0].get("embedding"):
                        raise EmbedderError(
                            "embed server trả response /v1/embeddings rỗng"
                        )
                    vectors[i] = data[0]["embedding"]
                if self._consecutive_unreachable >= CIRCUIT_THRESHOLD:
                    raise EmbedCircuitOpen(
                        f"embed server không với tới {self._consecutive_unreachable} "
                        "request liên tiếp; dừng lô sớm - các doc còn lại giữ nguyên trạng thái"
                    )
            if unreachable is not None:
                raise unreachable
            if self.cache is not None:
                new_entries = {
                    keys[i]: vectors[i] for i in pending if vectors[i] is not None
                }
                if new_entries:
                    await asyncio.to_thread(self.cache.put_many, new_entries)
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            raise EmbedderError(
                f"embed_documents thiếu vector cho {len(missing)}/{len(items)} item"
            )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed một câu hỏi (role=query, không cache)."""
        await self._ensure_verified()
        result = await self._post_embed_one({"text": text}, "query")
        self._check_response(result)
        data = result.get("data") or []
        if not data or not data[0].get("embedding"):
            raise EmbedderError("embed server trả response /v1/embeddings rỗng")
        return data[0]["embedding"]

    # ------------------------------------------------------------------
    # Cache (không HTTP)
    # ------------------------------------------------------------------
    def count_cache_hits(self, items: list[str | dict]) -> int:
        """Đếm số item đã có trong cache (lookup SQLite, KHÔNG gọi HTTP).

        Yêu cầu đã verify (dim + instruction_ns từ handshake); cache tắt -> 0.
        """
        if not items or self.cache is None or not self._verified:
            return 0
        keys = [self._cache_key(it) for it in items]
        cached = self.cache.get_many(keys)
        return sum(1 for k in keys if k in cached)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    async def _post_embed_one(self, item: str | dict, type_: str) -> dict:
        """Một POST /v1/embeddings (1 item) với retry cho lỗi tạm thời; 4xx raise ngay."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        attempt = 0
        while True:
            try:
                async with self._semaphore:
                    resp = await self._client.post(
                        f"{self.base_url}/v1/embeddings",
                        json={
                            "model": self.model,
                            "messages": self._to_wire(item, type_),
                        },
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self.retries:
                    self._consecutive_unreachable += 1
                    raise EmbedServerUnreachable(
                        f"embed server không với tới được sau {attempt + 1} lần thử: {exc}"
                    ) from exc
                attempt += 1
                await asyncio.sleep(self._backoff(attempt))
                continue

            self._consecutive_unreachable = 0

            if resp.status_code == 200:
                return resp.json()

            detail = self._detail(resp)
            if resp.status_code in (401, 403):
                raise EmbedAuthError(f"embed server từ chối token: {detail}")
            if resp.status_code == 413:
                raise EmbedInputTooLong(f"input quá dài / payload vượt giới hạn: {detail}")
            if resp.status_code == 507:
                raise EmbedServerOOM(f"embed server hết GPU memory: {detail}")
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= self.retries:
                    raise EmbedServerUnreachable(
                        f"embed server lỗi tạm thời {resp.status_code} sau "
                        f"{attempt + 1} lần thử: {detail}"
                    )
                attempt += 1
                await asyncio.sleep(self._backoff(attempt))
                continue
            raise EmbedderError(f"embed server trả {resp.status_code}: {detail}")

    @staticmethod
    def _to_wire(item: str | dict, type_: str) -> list[dict]:
        """Item nội bộ -> messages OpenAI-style (1 conversation)."""
        if isinstance(item, dict):
            text = item.get("text") or ""
            img = item.get("image_b64") or ""
        else:
            text, img = item, ""
        parts: list[dict] = []
        if img:
            parts.append(
                {"type": "image_url", "image_url": {"url": _data_url(img)}}
            )
        if text:
            parts.append({"type": "text", "text": text})
        if not parts:
            raise EmbedderError("embed item rỗng (thiếu text/image_b64)")
        return [{"role": type_, "content": parts}]

    @staticmethod
    def _detail(resp: httpx.Response) -> str:
        try:
            return str(resp.json().get("detail", resp.text))[:300]
        except ValueError:
            return resp.text[:300]

    def _check_response(self, result: dict) -> None:
        """Mỗi response /v1/embeddings phải khớp model handshake."""
        model = result.get("model")
        if model != self.model:
            raise EmbedModelMismatch(
                f"embed server trả model={model!r}, không khớp handshake "
                f"(model={self.model!r}). Server có thể bị đổi model giữa chừng."
            )

    # ------------------------------------------------------------------
    # Batching / cache key
    # ------------------------------------------------------------------
    @staticmethod
    def _groups(indices: list[int], size: int):
        iterator = iter(indices)
        while group := list(islice(iterator, size)):
            yield group

    def _cache_key(self, item: str | dict) -> str:
        if isinstance(item, dict):
            t = item.get("text") or ""
            img = item.get("image_b64") or ""
            content_hash = hashlib.sha256(f"{t}|{img}".encode()).hexdigest()
        else:
            content_hash = hashlib.sha256(item.encode("utf-8")).hexdigest()
        return f"{self.model}|{self.dim}|{self._instruction_ns}|{content_hash}"

    def _backoff(self, attempt: int) -> float:
        return self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.1)

    async def close(self) -> None:
        if self.cache is not None:
            await asyncio.to_thread(self.cache.close)
        await self._client.aclose()


def build_openai_embedder(settings) -> OpenAIEmbedder:
    """Build OpenAIEmbedder từ Settings (một nguồn cấu hình duy nhất với pipeline).

    EMBED_SERVER_TOKEN chỉ đọc từ env - không bao giờ ghi vào file config.
    """
    cache = None
    if getattr(settings, "EMBED_CACHE_ENABLED", True):
        from ami_rag.core.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(settings.EMBED_CACHE_PATH)
    return OpenAIEmbedder(
        base_url=settings.EMBED_SERVER_URL,
        model=settings.EMBED_MODEL,
        expected_dim=settings.EMBED_DIM,
        token=settings.EMBED_SERVER_TOKEN,
        timeout=settings.EMBED_TIMEOUT,
        max_concurrency=settings.EMBED_MAX_CONCURRENCY,
        retries=getattr(settings, "EMBED_RETRIES", 3),
        cache=cache,
    )
