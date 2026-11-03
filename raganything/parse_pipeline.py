"""Parse + describe pipeline for production document uploads (KG branch).

Port of the ami-multimodal-retrieval parse/describe stages: the caller fetches
the original upload from MinIO; this module runs MinerU, optionally rebuilds
text from backend data, uploads extracted assets + ``content_list.json`` to
MinIO, then produces ``descriptions.json`` (``{"index", "type", "description"}``
entries, empty descriptions kept) with the same semantics as their describe
stage. MinIO writes stay under ``{ASSET_PREFIX}/{doc_id}/``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from raganything.caption_reuse import (
    CONTENT_LIST_NAME,
    DEFAULT_BUCKET,
    DEFAULT_ENDPOINT,
    DEFAULT_PREFIX,
    DESCRIPTIONS_NAME,
    _env,
    join_caption,
)

logger = logging.getLogger(__name__)

ASSET_TYPES = ("image", "table", "equation")

_MULTIMODAL_TYPES = ("image", "table", "equation", "audio", "video", "generic")

_IMAGE_PROMPT = (
    "Hãy mô tả ngắn gọn nội dung ảnh trong ngữ cảnh của tài liệu. "
    "Nếu ảnh có chứa chữ (nhãn, tiêu đề, số liệu...), hãy chép lại nguyên văn."
)

_TABLE_PROMPT = (
    "Hãy mô tả ngắn gọn nội dung bảng sau đây, giữ nguyên các số liệu quan trọng:\n\n{body}"
)

CaptionFunc = Callable[..., Any]


class ParseArtifactStore:
    """MinIO access for the parse stage (slim port of MinioAssetStore)."""

    def __init__(
        self,
        client: Any = None,
        bucket: str | None = None,
        prefix: str | None = None,
    ):
        self.bucket = bucket or _env("MINIO_BUCKET", DEFAULT_BUCKET)
        self.prefix = (prefix or _env("ASSET_PREFIX", DEFAULT_PREFIX)).strip("/")
        self._client = client
        if self._client is None:
            from minio import Minio

            self._client = Minio(
                _env("MINIO_ENDPOINT", DEFAULT_ENDPOINT),
                access_key=_env("MINIO_ACCESS_KEY", "minioadmin"),
                secret_key=_env("MINIO_SECRET_KEY"),
                secure=_env("MINIO_SECURE") == "1",
            )

    def doc_prefix(self, doc_id: str) -> str:
        return f"{self.prefix}/{doc_id}/"

    def fetch(self, object_name: str, dest_dir: Path) -> Path:
        """Download an object (original upload) into dest_dir, keeping its extension."""
        dest = Path(dest_dir) / Path(object_name).name
        self._client.fget_object(self.bucket, object_name, str(dest))
        return dest

    def get_bytes(self, key: str | None) -> bytes | None:
        """Read raw bytes of an asset; None on missing/unreadable key."""
        if not key:
            return None
        try:
            response = self._client.get_object(self.bucket, key)
        except Exception:
            return None
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self._client.put_object(
            self.bucket,
            key,
            BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def _put_json(self, key: str, data) -> str:
        raw = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.put_bytes(key, raw, "application/json")
        return key

    def upload_content_list_assets(
        self, doc_id: str, content_list: list[dict]
    ) -> list[str]:
        """Upload local image files referenced by `img_path`; set `item["asset_key"]`.

        Returns the uploaded object keys. Items without a readable `img_path` are
        left untouched (text-only table/equation). Duplicate payloads dedupe onto
        one sha16 key; every item still gets its `asset_key` stamped.
        """
        keys: list[str] = []
        for item in content_list:
            if item.get("type") not in ASSET_TYPES:
                continue
            img_path = item.get("img_path")
            if not img_path:
                continue
            path = Path(img_path)
            if not path.is_file():
                continue
            data = path.read_bytes()
            ext = path.suffix.lower() or ".png"
            key = (
                f"{self.doc_prefix(doc_id)}{hashlib.sha256(data).hexdigest()[:16]}{ext}"
            )
            if key not in keys:
                self.put_bytes(
                    key,
                    data,
                    mimetypes.guess_type(path.name)[0]
                    or "application/octet-stream",
                )
                keys.append(key)
            item["asset_key"] = key
        return keys

    def save_content_list(self, doc_id: str, content_list: list[dict]) -> str:
        return self._put_json(
            f"{self.doc_prefix(doc_id)}{CONTENT_LIST_NAME}", content_list
        )

    def save_descriptions(self, doc_id: str, descriptions: list[dict]) -> str:
        return self._put_json(
            f"{self.doc_prefix(doc_id)}{DESCRIPTIONS_NAME}", descriptions
        )


def _parser_kwargs() -> dict:
    """Extra kwargs for ``MineruParser.parse_document`` from MINERU_* envs.

    Device and VRAM cap go through the subprocess env (MINERU_DEVICE_MODE /
    MINERU_VIRTUAL_VRAM_SIZE), same as their ``settings.parser_kwargs``.
    """
    kwargs: dict = {}
    if backend := _env("MINERU_BACKEND"):
        kwargs["backend"] = backend
    if lang := _env("MINERU_LANG"):
        kwargs["lang"] = lang
    if source := _env("MINERU_SOURCE"):
        kwargs["source"] = source
    if (timeout := _env("MINERU_TIMEOUT")) and int(timeout):
        kwargs["timeout"] = int(timeout)
    env: dict[str, str] = {}
    if device := _env("MINERU_DEVICE"):
        env["MINERU_DEVICE_MODE"] = device
    if (vram := _env("MINERU_VIRTUAL_VRAM_SIZE")) and int(vram):
        env["MINERU_VIRTUAL_VRAM_SIZE"] = str(int(vram))
    if env:
        kwargs["env"] = env
    return kwargs


def _normalize_type(t: Any) -> str:
    return (
        t if t in ("text", "image", "table", "equation", "audio", "video") else "generic"
    )


def _raw_image_desc(item: dict) -> str:
    """Raw image description without LLM: caption + footnotes."""
    caption = join_caption(item.get("image_caption") or item.get("img_caption"))
    footnotes = join_caption(item.get("image_footnote") or item.get("img_footnote"))
    return "\n".join(p for p in (caption, footnotes) if p).strip()


async def _caption_llm(sem: asyncio.Semaphore, caption_func, prompt: str, **kw) -> str:
    """Call caption_func under the shared semaphore; failure/empty -> ""."""
    try:
        async with sem:
            desc = await caption_func(prompt, **kw)
    except Exception as exc:
        logger.warning("caption LLM lỗi (%s): %s", type(exc).__name__, exc)
        return ""
    return (desc or "").strip()


async def _describe_image(
    store: ParseArtifactStore, sem: asyncio.Semaphore, caption_func, item: dict
) -> str:
    raw = _raw_image_desc(item)
    if caption_func is None:
        return raw
    key = item.get("asset_key")
    data = None
    if key:
        try:
            async with sem:
                data = await asyncio.to_thread(store.get_bytes, key)
        except Exception as exc:
            logger.warning("fetch ảnh %s lỗi (bỏ qua caption): %s", key, exc)
    if not data:
        return raw
    b64 = base64.b64encode(data).decode("ascii")
    desc = await _caption_llm(sem, caption_func, _IMAGE_PROMPT, image_data=b64)
    if not desc:
        logger.warning("caption ảnh %s rỗng/thất bại -> mô tả thô", key)
        return raw
    return desc


async def _describe_one(
    store: ParseArtifactStore,
    sem: asyncio.Semaphore,
    caption_func,
    content_list: list[dict],
    index: int,
) -> dict:
    item = content_list[index]
    type_ = _normalize_type(item.get("type"))
    desc = ""
    try:
        if type_ == "image":
            desc = await _describe_image(store, sem, caption_func, item)
        elif type_ == "table":
            table_mode = _env("TABLE_DESCRIBE_MODE", "raw").strip().lower()
            body = (item.get("table_body") or "").strip()
            if table_mode != "raw" and caption_func is not None and body:
                desc = await _caption_llm(
                    sem, caption_func, _TABLE_PROMPT.format(body=body)
                )
    except Exception as exc:
        logger.warning("describe item %d lỗi: %s", index, exc)
        desc = ""
    return {"index": index, "type": type_, "description": (desc or "").strip()}


async def _describe_items(
    store: ParseArtifactStore, content_list: list[dict], caption_func
) -> list[dict]:
    """Mirror their _stage_describe: one entry per multimodal item, empty
    descriptions kept, never raises out of the describe stage."""
    indices = [
        i
        for i, item in enumerate(content_list)
        if item.get("type") in _MULTIMODAL_TYPES
    ]
    if not indices:
        return []
    sem = asyncio.Semaphore(max(1, int(_env("CAPTION_LLM_MAX_CONCURRENCY", "4"))))
    return list(
        await asyncio.gather(
            *(_describe_one(store, sem, caption_func, content_list, i) for i in indices)
        )
    )


async def parse_and_store(
    store: ParseArtifactStore,
    doc_id: str,
    local_pdf,
    caption_func: CaptionFunc | None = None,
    parser: Any = None,
    rebuild: Any = None,
) -> tuple[list[dict], list[dict]]:
    """Parse ``local_pdf`` with MinerU, store artifacts, describe modalities.

    ``parser``/``rebuild`` injection params exist for tests (the lazy imports
    resolve through a package namespace other tests may replace). ``rebuild``
    is ``async (content_list, file_path) -> list[dict] | None``.

    Returns ``(content_list, descriptions)`` — the same artifacts their pipeline
    keeps in MinIO (``content_list.json`` / ``descriptions.json``).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        if parser is None:
            from raganything.parser import get_parser

            parser = get_parser("mineru")
        content_list: list[dict] = await asyncio.to_thread(
            parser.parse_document,
            str(local_pdf),
            output_dir=tmp_dir,
            method=os.environ.get("PARSE_METHOD") or "auto",
            **_parser_kwargs(),
        )
        if _env("PARSE_TEXT_SOURCE", "backend_data") == "backend_data":
            if rebuild is None:
                from raganything.backend_data_extract import (
                    create_extractor,
                    rebuild_content_list,
                )

                async def rebuild(content_list, file_path):  # type: ignore[misc]
                    return await rebuild_content_list(
                        content_list, file_path, create_extractor()
                    )

            rebuilt = await rebuild(content_list, local_pdf)
            if rebuilt is not None:
                content_list = rebuilt
        await asyncio.to_thread(store.upload_content_list_assets, doc_id, content_list)
        await asyncio.to_thread(store.save_content_list, doc_id, content_list)

    descriptions = await _describe_items(store, content_list, caption_func)
    await asyncio.to_thread(store.save_descriptions, doc_id, descriptions)
    return content_list, descriptions
