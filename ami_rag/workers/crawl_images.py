import asyncio
import hashlib
import re

import httpx

_IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
_IMAGE_BULLET_RE = re.compile(r"^\s*[-*]\s*(https?://\S+)\s*$", re.MULTILINE)


def _ext_from_url(url: str, content_type: str | None) -> str:
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".webm", ".mp4"):
        if ext in url.lower():
            return ext.lstrip(".")
    if content_type:
        subtype = content_type.rsplit("/", 1)[-1].split(";")[0]
        if subtype and len(subtype) <= 5:
            return subtype
    return "bin"


class CrawlImageWorker:
    """Download crawl images into MinIO so multimodal (VLM) processing and
    /rag responses can reference durable objects instead of remote URLs.

    Enabled behind CRAWL_IMAGES_ENABLED (P4). Idempotent: an object key is
    derived from (document_id, sha256(url)) and existing objects are reused.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        prefix: str,
        secure: bool = False,
        timeout: int = 60,
    ):
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._secure = secure
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            import urllib3
            from minio import Minio

            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=3, read=60),
                retries=urllib3.Retry(total=1, backoff_factor=0),
            )
            self._client = Minio(
                self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
                http_client=http_client,
            )
        return self._client

    def collect_image_urls(self, doc: dict) -> list[str]:
        urls: list[str] = []
        metadata = doc.get("metadata") or {}
        for item in metadata.get("images") or []:
            url = item.get("url") if isinstance(item, dict) else item
            if isinstance(url, str) and url.startswith("http"):
                urls.append(url)
        content = doc.get("content") or ""
        urls.extend(m.group(1) for m in _IMAGE_MARKDOWN_RE.finditer(content))
        urls.extend(m.group(1) for m in _IMAGE_BULLET_RE.finditer(content))
        seen: set[str] = set()
        unique = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique.append(url)
        return unique

    def _download(self, url: str) -> tuple[bytes, str | None]:
        with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content, resp.headers.get("content-type")

    def _upload(self, object_name: str, data: bytes, content_type: str | None) -> None:
        client = self._get_client()
        if not client.bucket_exists(self._bucket):
            client.make_bucket(self._bucket)
        client.put_object(
            self._bucket,
            object_name,
            data=iter([data]),
            length=len(data),
            content_type=content_type or "application/octet-stream",
        )

    def _exists(self, object_name: str) -> bool:
        from minio.error import S3Error

        try:
            self._get_client().stat_object(self._bucket, object_name)
            return True
        except S3Error:
            return False

    async def ensure_in_minio(self, doc_id: str, url: str) -> str | None:
        """Download one image and store it in MinIO; returns the object key."""
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        object_name = f"{self._prefix}/{doc_id}/{url_hash}.{_ext_from_url(url, None)}"
        try:
            if await asyncio.to_thread(self._exists, object_name):
                return object_name
            data, content_type = await asyncio.to_thread(self._download, url)
            await asyncio.to_thread(self._upload, object_name, data, content_type)
            return object_name
        except Exception:
            return None
