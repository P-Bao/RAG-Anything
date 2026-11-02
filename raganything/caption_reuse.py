"""Reuse image/table captions already stored by ami-multimodal-retrieval.

Their pipeline persists each document's parsed items (including captions
written by the describe stage) to MinIO at ``{ASSET_PREFIX}/{doc_id}/
content_list.json``. Reading those artifacts instead of calling the caption
LLM again saves API cost for documents already indexed there.
Read-only access: this module never writes to their bucket.
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_ENDPOINT = "localhost:9000"
DEFAULT_BUCKET = "ami-data-documents"
DEFAULT_PREFIX = "rag-assets"
CONTENT_LIST_NAME = "content_list.json"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def join_caption(value: Any) -> str:
    """Caption/list caption -> single string (None/empty -> "")."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value if v)
    return str(value)


def extract_captions(content_list: list[dict]) -> dict[str, str]:
    """Map ``asset_key`` -> caption from a content_list artifact.

    Covers image items (``image_caption``/``img_caption``) and table items
    (``table_caption``). Items without an ``asset_key`` or caption are
    skipped.
    """
    captions: dict[str, str] = {}
    for item in content_list or []:
        key = item.get("asset_key")
        if not key:
            continue
        caption = join_caption(
            item.get("image_caption") or item.get("img_caption") or item.get("table_caption")
        ).strip()
        if caption:
            captions[key] = caption
    return captions


class CaptionReuseStore:
    """Read-only MinIO access to ami-multimodal-retrieval content_list artifacts."""

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
        prefix: str | None = None,
        secure: bool | None = None,
        client: Any = None,
    ):
        self.endpoint = endpoint or _env("MINIO_ENDPOINT", DEFAULT_ENDPOINT)
        self.bucket = bucket or _env("MINIO_BUCKET", DEFAULT_BUCKET)
        self.prefix = (prefix or _env("ASSET_PREFIX", DEFAULT_PREFIX)).strip("/")
        self._client = client
        if self._client is None:
            from minio import Minio

            self._client = Minio(
                self.endpoint,
                access_key=access_key or _env("MINIO_ACCESS_KEY", "minioadmin"),
                secret_key=secret_key or _env("MINIO_SECRET_KEY"),
                secure=secure if secure is not None else _env("MINIO_SECURE") == "1",
            )

    def content_list_key(self, doc_id: str) -> str:
        return f"{self.prefix}/{doc_id}/{CONTENT_LIST_NAME}"

    def load_content_list(self, doc_id: str) -> list[dict] | None:
        """Return the stored content_list, or None when the doc is unknown."""
        response = self._client.get_object(self.bucket, self.content_list_key(doc_id))
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()

    def get_captions(self, doc_id: str) -> dict[str, str]:
        """Captions for one document: ``asset_key`` -> caption (empty if unknown)."""
        content_list = self.load_content_list(doc_id)
        if content_list is None:
            return {}
        return extract_captions(content_list)
