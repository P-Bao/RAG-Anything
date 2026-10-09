"""VectorPipeline - pipeline vector thuần thay thế LightRAG.

Các stage (ghi DocStatusStore ở MỖI stage):
    parse -> describe -> chunk -> embed -> indexed

- parse: giữ nguyên parser (MinerU/...) qua `get_parser().parse_document`;
  assets (ảnh/table/equation render) + content_list lưu MinIO.
- describe: với item multimodal dùng modal processor sinh mô tả; lưu mô tả vào
  MinIO (`{doc_id}/descriptions.json`). Đây là chỗ duy nhất có thể gọi LLM
  (modal processors, tuỳ chọn tắt).
- chunk: text chunk bằng plain splitter; multimodal chunk bằng template
  (Nội dung = mô tả đã lưu, không bao giờ gọi LLM ở stage này); lưu
  `{doc_id}/chunks.json`.
- embed: RemoteEmbedder/OpenAIEmbedder (máy B, chọn theo EMBED_BACKEND) với
  cache; luôn delete_by_doc trước upsert (idempotent). KHÔNG tạo entity,
  KHÔNG ghi vào graph.
- indexed: mark_indexed + meta. Verify: count vector trong Qdrant ==
  số chunk đã lưu.

Data giữa các stage nằm trong MinIO: `{doc_id}/content_list.json`,
`{doc_id}/descriptions.json`, `{doc_id}/chunks.json`.
- `dry_run`: handshake (GET /info) + đếm chunk/cache hits; KHÔNG gọi /embed,
  KHÔNG ghi gì.
"""

import asyncio
import base64
import hashlib
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ami_rag.core.embedder import collection_name
from ami_rag.core.pipeline import PipelineRunner, StageOutcome
from ami_rag.core.vector_store import ChunkRecord, VectorStore, VectorStoreError
from ami_rag.settings import parser_kwargs
from ami_rag.sources import (
    SOURCE_MINIO_PARSE,
    resolve_file_path,
    select_source,
    source_hash,
)
from ami_rag.storage.doc_status import (
    STAGE_CHUNK,
    STAGE_DESCRIBE,
    STAGE_EMBED,
    STAGE_INDEXED,
    STAGE_PARSE,
    DocStatusStore,
)

logger = logging.getLogger(__name__)

# Alias stage hiển thị dev-facing (có indexed là kết thúc)
PROCESSING_STAGES = ["parse", "describe", "chunk", "embed"]


class ArtifactPaths:
    """Path helper cho artifacts trung gian trong MinIO (theo key convention của asset store)."""

    @staticmethod
    def descriptions_key(asset_store, doc_id: str) -> str:
        return f"{asset_store.doc_prefix(doc_id)}descriptions.json"

    @staticmethod
    def chunks_key(asset_store, doc_id: str) -> str:
        return f"{asset_store.doc_prefix(doc_id)}chunks.json"


def _count_by_type(content_list: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in content_list or []:
        t = item.get("type") or "other"
        counts[t] = counts.get(t, 0) + 1
    return counts


def _page_count(content_list: list[dict]) -> int:
    pages = [item.get("page_idx", 0) for item in content_list if item.get("page_idx") is not None]
    return max(pages) + 1 if pages else 0


def _normalize_item_type(t: str | None) -> str:
    return t if t in ("text", "image", "table", "equation", "audio", "video") else "generic"


@dataclass
class Chunk:
    """Chunk đã build (trước embed)."""

    id: str  # chunk_id
    content: str
    modality: str  # text|image|table|equation|audio|video|generic
    page_idx: int | None
    is_multimodal: bool
    source_path: str = ""
    asset_key: str | None = None
    table_body: str | None = None
    caption: str | None = None
    extra: dict = field(default_factory=dict)

    def payload(self, doc_id: str, *, source_path: str = "") -> dict:
        p = {
            "doc_id": doc_id,
            "modality": self.modality,
            "page": self.page_idx,
            "source_path": source_path or self.source_path,
            "chunk_id": self.id,
            "is_multimodal": self.is_multimodal,
            "content": self.content,
        }
        if self.asset_key:
            p["asset_key"] = self.asset_key
        if self.table_body:
            p["table_body"] = self.table_body
        if self.caption:
            p["caption"] = self.caption
        p.update(self.extra)
        return p

    def to_json(self, doc_id: str, source_path: str = "") -> dict:
        return {
            "chunk_id": self.id,
            "content": self.content,
            "modality": self.modality,
            "page_idx": self.page_idx,
            "is_multimodal": self.is_multimodal,
            "source_path": source_path or self.source_path,
            "asset_key": self.asset_key,
            "table_body": self.table_body,
            "caption": self.caption,
            "doc_id": doc_id,
            "extra": self.extra,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Chunk":
        return cls(
            id=data["chunk_id"],
            content=data["content"],
            modality=data.get("modality", "text"),
            page_idx=data.get("page_idx"),
            is_multimodal=data.get("is_multimodal", False),
            source_path=data.get("source_path", ""),
            asset_key=data.get("asset_key"),
            table_body=data.get("table_body"),
            caption=data.get("caption"),
            extra=data.get("extra", {}),
        )


def _chunk_id(doc_id: str, content: str, index: int | None = None) -> str:
    seed = f"{doc_id}#{index}#{content}" if index is not None else f"{doc_id}{content}"
    return f"chunk-{hashlib.md5(seed.encode()).hexdigest()[:24]}"


def _format_modal_chunk(type_: str, item: dict, description: str) -> str:
    """Template chunk cho item multimodal (không ghi entity, không gọi LLM)."""
    if type_ == "image":
        captions = _join_caption(item.get("image_caption") or item.get("img_caption"))
        footnotes = _join_caption(item.get("image_footnote") or item.get("img_footnote"))
        return (
            f"[Image Content]\n{description}\n"
            f"Tóm tắt: {captions or 'Không có'}\n{footnotes or ''}".strip()
        )
    if type_ == "table":
        caption = _join_caption(item.get("table_caption"))
        body = item.get("table_body") or item.get("text") or str(item)
        return (
            f"[Table Content]\nCaption: {caption or 'Không có'}\n"
            f"Mô tả: {description}\n{body}"
        ).strip()
    if type_ == "equation":
        text = item.get("text") or str(item.get("latex", ""))
        return f"[Equation]\nMô tả: {description}\nCông thức: {text}"
    if type_ == "audio":
        return f"[Audio Content]\n{description}"
    if type_ == "video":
        return f"[Video Content]\n{description}"
    return f"[{_normalize_item_type(type_).title()}]\n{description}"


def _join_caption(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value if v)
    return str(value)


def chunk_text(text: str, max_tokens: int = 1200, overlap_tokens: int = 100) -> list[str]:
    """Plain paragraph splitter xấp xỉ token budget (fallback ~4 chars/token)."""
    max_chars = max(512, max_tokens * 4)
    overlap_chars = overlap_tokens * 4
    paras = [p.strip() for p in re.split(r"\n{2,}|\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paras:
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), max_chars - overlap_chars):
                chunks.append(para[i : i + max_chars])
            continue
        if len(current) + len(para) + 1 > max_chars and current:
            chunks.append(current)
            # overlap: giữ đuôi của chunk trước
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = (tail + "\n" + para).strip() if tail else para
        else:
            current = (current + "\n" + para).strip() if current else para
    if current:
        chunks.append(current)
    return chunks or [text]


@dataclass
class IntermediateDoc:
    """Artifacts trung gian của một doc ở từng stage."""

    doc_id: str
    content_list: list[dict] | None = None
    file_path: str = ""


class VectorPipeline(PipelineRunner):
    """Pipeline vector thuần (không torch/transformers phía máy A)."""

    def __init__(
        self,
        settings,
        *,
        embedder,
        vector_store: VectorStore,
        asset_store,
        store: DocStatusStore,
        docs_repo=None,
        parser=None,
        modal_processors=None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.asset_store = asset_store
        self.store = store
        self.docs_repo = docs_repo
        self.parser = parser
        self.modal_processors = modal_processors or {}
        self._bd_extractor = None
        self.collection = collection_name(
            settings.WORKSPACE, settings.EMBED_MODEL, settings.CHUNKER_VERSION
        )

    # ------------------------------------------------------------------
    # Doc / JSON I/O (qua asset store)
    # ------------------------------------------------------------------
    def _load_doc(self, doc_id: str) -> dict | None:
        if self.docs_repo is None:
            return None
        try:
            return self.docs_repo.find_by_id(doc_id)
        except Exception as exc:
            logger.warning("load doc %s lỗi: %s", doc_id, exc)
            return None

    def _load_json(self, key: str) -> Any:
        return self.asset_store.load_json(key)

    def _save_json(self, key: str, data: Any) -> None:
        self.asset_store.save_json(key, data)

    def _load_descriptions(self, doc_id: str) -> list[dict] | None:
        stored = self._load_json(ArtifactPaths.descriptions_key(self.asset_store, doc_id))
        return stored if isinstance(stored, list) else None

    # ------------------------------------------------------------------
    # Stage: parse (giữ nguyên parser đã dùng)
    # ------------------------------------------------------------------
    async def _stage_parse(self, doc_id: str, doc: dict) -> IntermediateDoc:
        if doc is None:
            raise ValueError(
                f"doc {doc_id} không có bản ghi trong documents "
                "(docs_repo thiếu hoặc document đã bị xoá); không thể parse"
            )
        if self.parser is None:
            from raganything.parser import get_parser

            self.parser = get_parser(self.settings.PARSER)

        source = select_source(doc)
        content_list: list[dict]
        file_path = ""
        assets: list[str] = []

        if source == SOURCE_MINIO_PARSE:
            file_path = doc.get("file_path") or ""
            with tempfile.TemporaryDirectory() as tmp_dir:
                local = await asyncio.to_thread(
                    self.asset_store.fetch, file_path, Path(tmp_dir)
                )
                content_list = await asyncio.to_thread(
                    self.parser.parse_document,
                    str(local),
                    self.settings.PARSE_METHOD,
                    tmp_dir,
                    **parser_kwargs(self.settings),
                )
                if (
                    self.settings.PARSER == "mineru"
                    and getattr(self.settings, "PARSE_TEXT_SOURCE", "mineru") == "backend_data"
                ):
                    content_list = await self._rebuild_backend_data(
                        doc_id, content_list, local
                    )
                assets = await asyncio.to_thread(
                    self.asset_store.upload_content_list_assets, doc_id, content_list
                )
        else:
            text = doc.get("content") or ""
            content_list = [{"type": "text", "text": text, "page_idx": 0}]
            file_path = resolve_file_path(doc) or ""

        counts = _count_by_type(content_list)
        page_count = _page_count(content_list)
        await asyncio.to_thread(self.asset_store.save_content_list, doc_id, content_list)
        self.store.mark_stage(
            doc_id,
            STAGE_PARSE,
            source_path=file_path or doc_id,
            content_hash=source_hash(doc, source),
            source=source,
            parser=self.settings.PARSER,
            file_path=file_path,
            counts=counts,
            page_count=page_count,
            assets=assets,
            document_type=doc.get("document_type"),
            title=doc.get("title"),
            organization_unit_id=doc.get("organization_unit_id"),
            owner_id=doc.get("owner_id"),
        )
        return IntermediateDoc(doc_id=doc_id, content_list=content_list, file_path=file_path)

    def _get_bd_extractor(self):
        from raganything.backend_data_extract import create_extractor

        if self._bd_extractor is None:
            self._bd_extractor = create_extractor(self.settings)
        return self._bd_extractor

    async def _rebuild_backend_data(
        self, doc_id: str, content_list: list[dict], local: Path
    ) -> list[dict]:
        """Thay text/table của MinerU bằng pdfplumber/fitz + OCR endpoint + bảng
        Qwen endpoint (phong cách backend_data); ảnh MinerU giữ nguyên.
        Thất bại -> fallback MinerU content_list."""
        from raganything.backend_data_extract import rebuild_content_list

        try:
            rebuilt = await rebuild_content_list(
                content_list, local, self._get_bd_extractor()
            )
        except Exception as exc:
            logger.warning(
                "backend_data text/table parse doc=%s lỗi, giữ MinerU content_list: %s",
                doc_id,
                exc,
            )
            return content_list
        if rebuilt is None:
            return content_list
        logger.info(
            "backend_data parse doc=%s: %d items (MinerU) -> %d items (text/table backend_data + ảnh MinerU)",
            doc_id,
            len(content_list),
            len(rebuilt),
        )
        return rebuilt

    # ------------------------------------------------------------------
    # Stage: describe (LLM – duy nhất chỗ mô tả modal)
    # ------------------------------------------------------------------
    async def _stage_describe(
        self, doc_id: str, content_list: list[dict]
    ) -> list[dict]:
        multimodal_indices = [
            i
            for i, item in enumerate(content_list)
            if item.get("type") in ("image", "table", "equation", "audio", "video", "generic")
        ]
        if not multimodal_indices:
            self._save_json(
                ArtifactPaths.descriptions_key(self.asset_store, doc_id), []
            )
            self.store.mark_stage(doc_id, STAGE_DESCRIBE)
            return []

        descriptions: list[dict] = []
        for i in multimodal_indices:
            item = content_list[i]
            type_ = _normalize_item_type(item.get("type"))
            try:
                desc_text = await self._describe_item(type_, item, content_list)
            except Exception as exc:
                logger.warning("describe doc=%s item=%d lỗi: %s", doc_id, i, exc)
                desc_text = ""
            descriptions.append(
                {"index": i, "type": type_, "description": desc_text}
            )
        self._save_json(
            ArtifactPaths.descriptions_key(self.asset_store, doc_id), descriptions
        )
        self.store.mark_stage(doc_id, STAGE_DESCRIBE)
        return descriptions

    async def _describe_item(self, type_: str, item: dict, content_list: list[dict]) -> str:
        processor = self.modal_processors.get(type_) or self.modal_processors.get("generic")
        if processor is None:
            return item.get("text") or item.get("table_body") or str(item)
        try:
            sections = await processor.generate_chunk_sections(
                modal_content=item,
                content_type=type_,
                item_info={
                    "page_idx": item.get("page_idx", 0),
                    "index": 0,
                    "type": type_,
                },
                entity_name=None,
            )
            return sections[0]["description"] if sections else ""
        except Exception:
            return item.get("text") or item.get("table_body") or str(item)

    # ------------------------------------------------------------------
    # Stage: chunk (chỉ dùng mô tả đã lưu – KHÔNG gọi LLM)
    # ------------------------------------------------------------------
    def _build_chunks(
        self,
        doc_id: str,
        content_list: list[dict],
        descriptions: list[dict] | None,
        *,
        source_path: str = "",
    ) -> list[Chunk]:
        desc_by_index = {}
        for d in descriptions or []:
            if d.get("description"):
                desc_by_index.setdefault(d["index"], []).append(d["description"])

        chunks: list[Chunk] = []
        chunk_idx = 0
        for i, item in enumerate(content_list):
            t = item.get("type") or "text"
            if t == "text":
                text = item.get("text", "")
                for piece in chunk_text(
                    text,
                    max_tokens=self.settings.CHUNK_SIZE,
                    overlap_tokens=self.settings.CHUNK_OVERLAP,
                ):
                    chunks.append(
                        Chunk(
                            id=_chunk_id(doc_id, piece, chunk_idx),
                            content=piece,
                            modality="text",
                            page_idx=item.get("page_idx", 0),
                            is_multimodal=False,
                            source_path=source_path,
                        )
                    )
                    chunk_idx += 1
            else:
                # multimodal: description from describe stage if present, else raw caption
                descs = desc_by_index.get(i) or []
                description = descs[0] if descs else _join_caption(
                    item.get("image_caption")
                    or item.get("table_caption")
                    or item.get("latex")
                    or item.get("text")
                )
                content = _format_modal_chunk(t, item, description)
                chunks.append(
                    Chunk(
                        id=_chunk_id(doc_id, content, chunk_idx),
                        content=content,
                        modality=_normalize_item_type(t),
                        page_idx=item.get("page_idx", 0),
                        is_multimodal=True,
                        source_path=source_path,
                        asset_key=item.get("asset_key"),
                        table_body=item.get("table_body"),
                        caption=_join_caption(item.get("image_caption") or item.get("table_caption")) or None,
                    )
                )
                chunk_idx += 1
        return chunks

    async def _stage_chunk(
        self,
        doc_id: str,
        content_list: list[dict],
        descriptions: list[dict] | None,
        *,
        source_path: str = "",
    ) -> list[Chunk]:
        chunks = self._build_chunks(doc_id, content_list, descriptions, source_path=source_path)
        self._save_json(
            ArtifactPaths.chunks_key(self.asset_store, doc_id),
            [c.to_json(doc_id) for c in chunks],
        )
        self.store.mark_stage(
            doc_id, STAGE_CHUNK, chunk_count=len(chunks), source_path=source_path
        )
        return chunks

    async def _load_or_build_chunks(
        self,
        doc_id: str,
        content_list: list[dict] | None,
        descriptions: list[dict] | None,
        source_path: str,
    ) -> list[Chunk]:
        """Resume từ stage embed: chunks.json nếu có, else build lại từ MinIO."""
        stored = await asyncio.to_thread(
            self._load_json, ArtifactPaths.chunks_key(self.asset_store, doc_id)
        )
        if isinstance(stored, list) and stored:
            return [Chunk.from_json(c) for c in stored]
        if content_list is None:
            content_list = await asyncio.to_thread(self.asset_store.load_content_list, doc_id)
        if content_list is None:
            raise FileNotFoundError(
                f"doc {doc_id}: thiếu chunks.json + content_list trong MinIO - cần parse lại"
            )
        if descriptions is None:
            descriptions = await asyncio.to_thread(self._load_descriptions, doc_id)
        return self._build_chunks(doc_id, content_list, descriptions, source_path=source_path)

    # ------------------------------------------------------------------
    async def _prepare_embed_items(
        self, chunks: list[Chunk]
    ) -> list[str | dict]:
        """Chuẩn bị payload cho embedder: nạp ảnh từ asset_store với chunk
        image/table/equation (image+text modality - cả Qwen3-VL lẫn Nemotron
        VL đều nhận text + ảnh qua contract item {text, image_b64})."""
        asset_chunks = [
            c
            for c in chunks
            if c.modality in ("image", "table", "equation") and c.asset_key
        ]
        asset_images: dict[str, str] = {}
        if asset_chunks and hasattr(self.asset_store, "get_bytes"):
            unique_keys = {c.asset_key for c in asset_chunks if c.asset_key}

            async def _fetch(key: str) -> tuple[str, str | None]:
                try:
                    data = await asyncio.to_thread(self.asset_store.get_bytes, key)
                    if data:
                        return key, base64.b64encode(data).decode("ascii")
                except Exception as exc:
                    logger.warning("fetch asset %s failed: %s", key, exc)
                return key, None

            results = await asyncio.gather(*(_fetch(k) for k in unique_keys))
            asset_images = {k: b64 for k, b64 in results if b64}

        items: list[str | dict] = []
        for c in chunks:
            if (
                c.modality in ("image", "table", "equation")
                and c.asset_key
                and c.asset_key in asset_images
            ):
                items.append({
                    "text": c.content,
                    "image_b64": asset_images[c.asset_key],
                })
            else:
                items.append(c.content)
        return items

    # ------------------------------------------------------------------
    # Stage: embed (máy B qua RemoteEmbedder) + verify (idempotent)
    # ------------------------------------------------------------------
    async def _stage_embed(
        self, doc_id: str, chunks: list[Chunk], *, source_path: str = ""
    ) -> None:
        if not chunks:
            raise RuntimeError(f"doc {doc_id} không có chunk nào để embed")
        seen_ids: set[str] = set()
        for idx, c in enumerate(chunks):
            if c.id in seen_ids:
                c.id = f"{c.id}-{idx}"
            seen_ids.add(c.id)
        # idempotency: xoá vector cũ của doc trước khi upsert
        await self.vector_store.delete_by_doc(self.collection, doc_id)
        items = await self._prepare_embed_items(chunks)
        vectors = await self.embedder.embed_documents(items)  # [N x dim]
        if len(vectors) != len(chunks):
            raise RuntimeError("embed server trả số vector không khớp số chunk")
        records = [
            ChunkRecord(
                id=c.id,
                vector=v,
                payload=c.payload(doc_id, source_path=source_path),
            )
            for c, v in zip(chunks, vectors)
        ]
        await self.vector_store.upsert(self.collection, records)
        actual = await self.vector_store.count(self.collection, doc_id)
        if actual < len(chunks):
            raise RuntimeError(
                f"doc {doc_id}: verify count={actual} < chunk đã lưu={len(chunks)}"
            )
        self.store.mark_stage(doc_id, STAGE_EMBED, chunk_count=len(chunks))

    # ------------------------------------------------------------------
    # Run / dry_run / preflight
    # ------------------------------------------------------------------
    async def preflight_embed(self) -> None:
        await self.embedder.verify()
        await self.vector_store.ensure_collection(self.collection, self.embedder.dim)

    async def run(
        self, doc_id: str, *, from_stage: str | None = None, dry_run: bool = False
    ) -> StageOutcome:
        row = self.store.get(doc_id)
        if row is None:
            raise ValueError(f"doc {doc_id} không có bản ghi trạng thái (pending)")
        effective_start = from_stage or row.get("stage") or STAGE_PARSE
        if effective_start not in PROCESSING_STAGES:
            effective_start = STAGE_PARSE
        start_idx = PROCESSING_STAGES.index(effective_start)

        if dry_run:
            return await self._dry_run(doc_id, row, from_stage)

        try:
            # Indexed rồi: không làm lại (retry/reindex không chọn doc indexed;
            # branch này chỉ là bảo hiểm khi from_stage=None nhưng stage=indexed)
            if from_stage is None and row.get("stage") == STAGE_INDEXED:
                return StageOutcome(
                    doc_id=doc_id,
                    stage=STAGE_INDEXED,
                    ok=True,
                    chunk_count=int(row.get("chunk_count") or 0),
                )

            doc = await asyncio.to_thread(self._load_doc, doc_id)
            source_path = row.get("source_path") or ""
            content_list: list[dict] | None = None
            descriptions: list[dict] | None = None
            chunks: list[Chunk] | None = None

            if start_idx <= 0:
                intermediate = await self._stage_parse(doc_id, doc)
                content_list = intermediate.content_list
                source_path = intermediate.file_path or doc_id

            if start_idx <= 1:
                if content_list is None:
                    content_list = await asyncio.to_thread(
                        self.asset_store.load_content_list, doc_id
                    )
                if content_list is None:
                    raise FileNotFoundError(
                        f"doc {doc_id}: thiếu content_list trong MinIO - cần parse lại"
                    )
                descriptions = await self._stage_describe(doc_id, content_list)

            if start_idx <= 2:
                if content_list is None:
                    content_list = await asyncio.to_thread(
                        self.asset_store.load_content_list, doc_id
                    )
                if content_list is None:
                    raise FileNotFoundError(
                        f"doc {doc_id}: thiếu content_list trong MinIO - cần parse lại"
                    )
                if descriptions is None:
                    descriptions = await asyncio.to_thread(self._load_descriptions, doc_id)
                chunks = await self._stage_chunk(
                    doc_id, content_list, descriptions, source_path=source_path
                )

            if start_idx <= 3:
                if chunks is None:
                    chunks = await self._load_or_build_chunks(
                        doc_id, content_list, descriptions, source_path
                    )
                await self._stage_embed(doc_id, chunks, source_path=source_path)
                # Refresh row: parse/describe/chunk đã ghi meta (counts, assets, ...)
                row = self.store.get(doc_id) or row
                self.store.mark_indexed(
                    doc_id,
                    chunk_count=len(chunks),
                    embed_model=self.settings.EMBED_MODEL,
                    embed_dim=self.embedder.dim,
                    chunker_version=self.settings.CHUNKER_VERSION,
                    source_hash=row.get("content_hash"),
                    source=row.get("source"),
                    parser=row.get("parser") or self.settings.PARSER,
                    file_path=row.get("file_path"),
                    counts=row.get("counts"),
                    page_count=row.get("page_count"),
                    assets=row.get("assets"),
                    document_type=(doc or {}).get("document_type"),
                    title=(doc or {}).get("title"),
                    organization_unit_id=(doc or {}).get("organization_unit_id"),
                    owner_id=(doc or {}).get("owner_id"),
                    meta={"index_stats": {"chunks": len(chunks), "vectors": len(chunks)}},
                )

            return StageOutcome(
                doc_id=doc_id,
                stage=STAGE_INDEXED,
                ok=True,
                chunk_count=len(chunks) if chunks is not None else 0,
            )
        except VectorStoreError as exc:
            self.store.mark_failed(doc_id, f"vector store: {exc}", effective_start)
            return StageOutcome(doc_id=doc_id, stage=effective_start, ok=False, error=str(exc))
        except Exception as exc:
            self.store.mark_failed(doc_id, f"{type(exc).__name__}: {exc}", effective_start)
            return StageOutcome(doc_id=doc_id, stage=effective_start, ok=False, error=str(exc))

    async def _dry_run(self, doc_id: str, row: dict, from_stage: str | None) -> StageOutcome:
        """Handshake (GET /info) + đếm chunk / cache hits. Không POST /embed, không ghi."""
        await self.embedder.verify()
        await self.vector_store.ensure_collection(self.collection, self.embedder.dim)

        content_list = await asyncio.to_thread(self.asset_store.load_content_list, doc_id)
        descriptions = await asyncio.to_thread(self._load_descriptions, doc_id)
        source_path = row.get("source_path") or ""
        stored = await asyncio.to_thread(
            self._load_json, ArtifactPaths.chunks_key(self.asset_store, doc_id)
        )
        if isinstance(stored, list) and stored:
            chunks = [Chunk.from_json(c) for c in stored]
        elif content_list is not None:
            chunks = self._build_chunks(doc_id, content_list, descriptions, source_path=source_path)
        else:
            logger.info("doc %s: không có trung gian (content_list/chunks), dry_run trả 0", doc_id)
            chunks = []
        items = await self._prepare_embed_items(chunks)
        return StageOutcome(
            doc_id=doc_id,
            stage=from_stage or "chunk",
            ok=True,
            chunk_count=len(chunks),
            cache_hits=self.embedder.count_cache_hits(items),
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    async def delete_doc(self, doc_id: str) -> None:
        await self.vector_store.delete_by_doc(self.collection, doc_id)
        try:
            await asyncio.to_thread(self.asset_store.delete_doc_assets, doc_id)
        except Exception:
            pass
        self.store.delete(doc_id)
