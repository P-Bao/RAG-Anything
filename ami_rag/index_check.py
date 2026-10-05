"""Index completeness check and in-place repair for ingested documents.

LightRAG swallows embedding/LLM failures (quota, 5xx): a document can end up `processed`
with chunk text in Mongo but no chunk vector in Qdrant and no entities in the graph.
`inspect_document` counts what really exists for one document; `repair_document` re-runs
only the missing parts from the chunk text already stored (no re-parse, no MinIO, no
re-chunking): chunk vectors are re-embedded, and entity/relation extraction is re-run
when the document has no entities.
"""

import logging
import re
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


class IngestIncompleteError(RuntimeError):
    """The document was inserted but its index (vectors / entities) is incomplete."""


ERR_QUOTA = "QUOTA_EXHAUSTED"
ERR_RATE_LIMIT = "RATE_LIMITED"
ERR_UPSTREAM = "UPSTREAM_UNAVAILABLE"
ERR_AUTH = "AUTH_FAILED"
ERR_INCOMPLETE = "INDEX_INCOMPLETE"
ERR_OTHER = "OTHER"

# Errors that will fail identically for every remaining document: stop the whole command.
FATAL_ERROR_CODES = frozenset({ERR_QUOTA, ERR_AUTH})

ERROR_HINTS = {
    ERR_QUOTA: "API credit/quota exhausted: top up billing, then re-run `ami-rag reindex --all --repair`",
    ERR_RATE_LIMIT: "provider rate limit (429): lower concurrency or wait, then re-run the repair",
    ERR_UPSTREAM: "provider temporarily unavailable (5xx): re-run the repair later",
    ERR_AUTH: "API key rejected (401/403): fix GEMINI_API_KEY, then re-run the repair",
    ERR_INCOMPLETE: "index still incomplete after processing: re-run the repair",
    ERR_OTHER: "unexpected error: see the error message / worker logs",
}


class FatalIngestError(RuntimeError):
    """A provider error that makes every further document fail too (quota, auth)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def classify_error(exc: BaseException | str) -> str:
    """Map a provider/pipeline exception (or its message) to a stable error code."""
    if isinstance(exc, IngestIncompleteError):
        return ERR_INCOMPLETE
    if isinstance(exc, FatalIngestError):
        return exc.code
    text = str(exc)
    low = text.lower()
    if _has_code(text, "402") or "prepayment credits" in low or "billing" in low:
        return ERR_QUOTA
    if _has_code(text, "429") or "resource exhausted" in low or "rate limit" in low:
        return ERR_RATE_LIMIT
    if _has_code(text, "500", "502", "503", "504") or "unavailable" in low:
        return ERR_UPSTREAM
    if _has_code(text, "401", "403") or "api key" in low or "permission_denied" in low:
        return ERR_AUTH
    return ERR_OTHER


def _has_code(text: str, *codes: str) -> bool:
    return any(re.search(rf"(?<![0-9A-Za-z]){c}(?![0-9A-Za-z])", text) for c in codes)


def short_error(exc: BaseException | str, limit: int = 300) -> str:
    """Single-line, bounded error message for logs / registry."""
    return " ".join(str(exc).split())[:limit]


async def preflight_providers(rag) -> None:
    """Probe embedding + LLM once; raise FatalIngestError on quota/auth problems.

    Rate limit / 5xx are transient and only logged. Run before long repair/reindex jobs so
    a depleted account is reported once, up front, instead of failing every document.
    """
    probes = (
        ("embedding", lambda: rag.embedding_func(["ping"])),
        ("llm", lambda: rag.llm_model_func("Reply with: ok")),
    )
    for name, probe in probes:
        try:
            await probe()
        except Exception as exc:
            code = classify_error(exc)
            if code in FATAL_ERROR_CODES:
                raise FatalIngestError(
                    code, f"{name} preflight failed: {short_error(exc)}"
                ) from exc
            logger.warning("preflight %s: transient %s: %s", name, code, short_error(exc))


@dataclass
class IndexStats:
    doc_id: str
    chunks: int = 0
    vectors: int = 0
    entities: int = 0
    relations: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def missing_vectors(self) -> int:
        return max(self.chunks - self.vectors, 0)

    def problems(self, require_entities: bool = True) -> list[str]:
        problems = []
        if self.chunks == 0:
            problems.append("no chunks")
        if self.missing_vectors:
            problems.append(f"{self.missing_vectors}/{self.chunks} chunk vectors missing")
        if require_entities and self.chunks and self.entities == 0:
            problems.append("no entities")
        return problems

    def ok(self, require_entities: bool = True) -> bool:
        return not self.problems(require_entities)

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("doc_id")
        data.pop("reasons")
        return data

    def line(self, require_entities: bool = True) -> str:
        problems = self.problems(require_entities)
        return (
            f"doc={self.doc_id} chunks={self.chunks} vectors={self.vectors} "
            f"entities={self.entities} relations={self.relations} "
            f"{'OK' if not problems else 'INCOMPLETE: ' + '; '.join(problems)}"
        )


async def _chunk_ids(rag, doc_id: str) -> list[str]:
    status = await rag.doc_status.get_by_id(doc_id)
    return [str(c) for c in ((status or {}).get("chunks_list") or []) if c]


async def inspect_document(rag, doc_id: str) -> IndexStats:
    """Count chunks, chunk vectors, entities and relations that exist for a document."""
    chunk_ids = await _chunk_ids(rag, doc_id)
    vectors = await rag.chunks_vdb.get_by_ids(chunk_ids) if chunk_ids else []
    entities = await rag.full_entities.get_by_id(doc_id)
    relations = await rag.full_relations.get_by_id(doc_id)
    return IndexStats(
        doc_id=doc_id,
        chunks=len(chunk_ids),
        vectors=sum(1 for v in vectors if v),
        entities=len((entities or {}).get("entity_names") or []),
        relations=len((relations or {}).get("relation_pairs") or []),
    )


async def repair_document(rag, doc_id: str, stats: IndexStats | None = None) -> IndexStats:
    """Re-run the missing index parts of one document from its stored chunk text.

    Idempotent for chunk vectors (upsert by chunk id). Entity extraction runs only when the
    document has no entities, so a partially extracted document is never merged twice.
    Raises IngestIncompleteError when the chunk text itself is missing (full reindex needed).
    """
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
    from lightrag.operate import merge_nodes_and_edges

    stats = stats or await inspect_document(rag, doc_id)
    chunk_ids = await _chunk_ids(rag, doc_id)
    if not chunk_ids:
        raise IngestIncompleteError(f"document {doc_id} has no chunks in doc_status")

    rows = await rag.text_chunks.get_by_ids(chunk_ids)
    chunks = {cid: row for cid, row in zip(chunk_ids, rows) if row}
    if len(chunks) < len(chunk_ids):
        raise IngestIncompleteError(
            f"document {doc_id}: {len(chunk_ids) - len(chunks)}/{len(chunk_ids)} chunk texts "
            "missing in Mongo, full reindex needed"
        )

    if stats.missing_vectors:
        have = await rag.chunks_vdb.get_by_ids(chunk_ids)
        missing = {cid: chunks[cid] for cid, v in zip(chunk_ids, have) if not v}
        logger.info("repair doc=%s: re-embedding %d chunk(s)", doc_id, len(missing))
        await rag.chunks_vdb.upsert(
            {cid: {k: v for k, v in row.items() if k != "_id"} for cid, row in missing.items()}
        )

    if stats.entities == 0:
        logger.info(
            "repair doc=%s: re-extracting entities from %d chunk(s)",
            doc_id,
            len(chunks),
        )
        pipeline_status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        pipeline_status_lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
        chunk_results = await rag._process_extract_entities(
            chunks, pipeline_status, pipeline_status_lock
        )
        status = await rag.doc_status.get_by_id(doc_id) or {}
        await merge_nodes_and_edges(
            chunk_results=chunk_results,
            knowledge_graph_inst=rag.chunk_entity_relation_graph,
            entity_vdb=rag.entities_vdb,
            relationships_vdb=rag.relationships_vdb,
            global_config=asdict(rag),
            full_entities_storage=rag.full_entities,
            full_relations_storage=rag.full_relations,
            doc_id=doc_id,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=rag.llm_response_cache,
            entity_chunks_storage=rag.entity_chunks,
            relation_chunks_storage=rag.relation_chunks,
            file_path=status.get("file_path") or doc_id,
        )
    await rag._insert_done()
    return await inspect_document(rag, doc_id)
