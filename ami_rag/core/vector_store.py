"""VectorStore interface và backend Qdrant mặc định.

Lớp persistence tách khỏi vector: kết quả parse + mô tả modal lưu riêng
(MinIO), chỉ chunk/vector mới vào đây. Metadata mỗi chunk (payload):
doc_id, page, type, source_path, chunk_id, chunker_version, embed_model,
embed_dim, modality/asset_key/table_body/caption.

Quy ước collection: {prefix}__{embed_model_slug}__{chunker_version}
(không trộn hai embed model trong một collection).
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Lỗi liên quan vector store (collection lệch cấu hình, mạng, v.v.)."""


@dataclass
class ChunkRecord:
    """Một chunk (id + vector + payload) chuẩn bị upsert."""

    id: str  # chunk_id
    vector: list[float]
    payload: dict = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict


class VectorStore(Protocol):
    async def ensure_collection(self, name: str, dim: int) -> None: ...
    async def upsert(self, collection: str, chunks: list[ChunkRecord]) -> None: ...
    async def search(self, collection: str, vector: list[float], top_k: int) -> list[SearchHit]: ...
    async def delete_by_doc(self, collection: str, doc_id: str) -> int: ...
    async def count(self, collection: str, doc_id: str | None = None) -> int: ...


class QdrantVectorStore:
    """Backend mặc định: Qdrant (lệnh kế thừa từ RAG-Anything/AMI hiện có)."""

    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None):
        from qdrant_client import QdrantClient

        self._url = url
        self._client = QdrantClient(url=url, api_key=api_key or None)

    async def ensure_collection(self, name: str, dim: int) -> None:
        """Tạo collection nếu thiếu; kiểm tra tương thích dim với collection có sẵn."""
        from qdrant_client.http import models

        try:
            existing = await asyncio.to_thread(
                lambda: next(
                    (c.name for c in self._client.get_collections().collections if c.name == name),
                    None,
                )
            )
            if existing is not None:
                info = await asyncio.to_thread(self._client.get_collection, name)
                vectors = info.config.params.vectors
                size = vectors.size if vectors is not None else None
                if size is not None and size != dim:
                    raise VectorStoreError(
                        f"collection '{name}' có dim={size} nhưng cấu hình là {dim}. "
                        "Đổi EMBED_DIM/EMBED_MODEL hoặc chuyển sang collection mới."
                    )
                return
            await asyncio.to_thread(
                self._client.create_collection,
                collection_name=name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            logger.info("tạo Qdrant collection %s (dim=%s)", name, dim)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Qdrant {self._url}: {exc}") from exc

    async def upsert(self, collection: str, chunks: list[ChunkRecord]) -> None:
        from qdrant_client.http.models import PointStruct

        if not chunks:
            return
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                vector=c.vector,
                payload={"chunk_id": c.id, **c.payload},
            )
            for c in chunks
        ]
        await asyncio.to_thread(
            self._client.upsert, collection_name=collection, points=points
        )

    async def search(self, collection: str, vector: list[float], top_k: int) -> list[SearchHit]:
        results = await asyncio.to_thread(
            self._client.query_points,
            collection_name=collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchHit(id=str(p.id), score=p.score, payload=p.payload or {})
            for p in results.points
        ]

    async def delete_by_doc(self, collection: str, doc_id: str) -> int:
        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
        )

        filter_ = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        await asyncio.to_thread(
            self._client.delete,
            collection_name=collection,
            points_selector=FilterSelector(filter=filter_),
        )
        return 0  # Qdrant không trả số điểm đã xoá; 0 để batch>=1 dạng int hợp lệ

    async def count(self, collection: str, doc_id: str | None = None) -> int:
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        count_filter = (
            Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            if doc_id
            else None
        )
        result = await asyncio.to_thread(
            self._client.count, collection_name=collection, count_filter=count_filter, exact=True
        )
        return int(result.count)


QdrantVectorStoreClient = QdrantVectorStore  # alias cho test/typing mới
