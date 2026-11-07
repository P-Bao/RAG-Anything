"""Reuse image/table captions already stored by ami-multimodal-retrieval.

Their pipeline persists each document's parsed items to MinIO at
``{ASSET_PREFIX}/{doc_id}/content_list.json`` and the describe-stage LLM
captions to ``descriptions.json`` beside it. Reading those artifacts instead
of calling the caption LLM again saves API cost for documents already indexed
there. Read-only access: this module never writes to their bucket.
"""

from __future__ import annotations

import json
import os
from typing import Any

from minio.error import S3Error

DEFAULT_ENDPOINT = "localhost:9000"
DEFAULT_BUCKET = "ami-data-documents"
DEFAULT_PREFIX = "rag-assets"
CONTENT_LIST_NAME = "content_list.json"
DESCRIPTIONS_NAME = "descriptions.json"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def join_caption(value: Any) -> str:
    """Caption/list caption -> single string (None/empty -> "")."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value if v)
    return str(value)


def extract_descriptions(descriptions: list[dict]) -> dict[int, str]:
    """Map item index -> LLM description from their descriptions.json artifact.

    Their describe stage persists ``{"index", "type", "description"}`` there
    (content_list.json keeps the original MinerU caption fields untouched).
    """
    out: dict[int, str] = {}
    for entry in descriptions or []:
        if not isinstance(entry, dict):
            continue
        desc = str(entry.get("description") or "").strip()
        if desc and isinstance(entry.get("index"), int):
            out[entry["index"]] = desc
    return out


def _read_json_object(client: Any, bucket: str, key: str) -> list[dict] | None:
    """Read a JSON array object; None when it does not exist."""
    try:
        response = client.get_object(bucket, key)
    except S3Error as exc:
        if exc.code == "NoSuchKey":
            return None
        raise
    try:
        return json.loads(response.read().decode("utf-8"))
    finally:
        response.close()
        response.release_conn()


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
        return _read_json_object(self._client, self.bucket, self.content_list_key(doc_id))

    def load_descriptions(self, doc_id: str) -> list[dict] | None:
        """Their describe-stage artifact (LLM captions), None when absent."""
        return _read_json_object(
            self._client, self.bucket, f"{self.prefix}/{doc_id}/{DESCRIPTIONS_NAME}"
        )

    def get_descriptions(self, doc_id: str) -> dict[int, str]:
        """``item index`` -> LLM description (empty dict when absent)."""
        descriptions = self.load_descriptions(doc_id)
        if descriptions is None:
            return {}
        return extract_descriptions(descriptions)

    def presigned_url(self, key: str, expires_s: int | None = None) -> str:
        """Presigned GET URL for an asset key (query-time image display)."""
        raw = _env("ASSET_URL_EXPIRES_S", "3600")
        try:
            expires = expires_s or max(1, int(raw))
        except ValueError:
            expires = 3600
        return self._client.presign_get_url(self.bucket, key, expires=expires)
