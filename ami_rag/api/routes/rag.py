import asyncio
import logging
import time

from fastapi import APIRouter, Depends, Header, HTTPException

from ami_rag.api.schemas import (
    RAGRequest,
    RetrievalResponse,
    RetrievalResponseV2,
    RetrievedDoc,
    RetrievedDocV2,
)
from ami_rag.observability import (
    CHUNKS_RETRIEVED,
    DOCS_FILTERED_TOTAL,
    LIGHTRAG_FAILURES_TOTAL,
    PRESIGN_FAILURES_TOTAL,
    RERANK_FALLBACK_TOTAL,
    RETRIEVAL_DOCS_BY_MODALITY_TOTAL,
    observe_stage,
    track_retrieval,
)
from ami_rag.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


async def get_rag():
    from ami_rag.core.factory import get_raganything

    return await get_raganything()


def get_asset_store():
    from ami_rag.core.factory import get_asset_store as _get_asset_store

    return _get_asset_store()


def get_rerank_func():
    from ami_rag.core.rerank_client import build_rerank_model_func

    return build_rerank_model_func(get_settings())


_resolver_instance = None


def get_resolver():
    global _resolver_instance
    if _resolver_instance is None:
        from ami_rag.api.resolver import DocResolver
        from ami_rag.storage.mongo_docs import MongoDocumentRepo

        settings = get_settings()
        _resolver_instance = DocResolver(
            docs_repo=MongoDocumentRepo(
                mongo_uri=settings.MONGO_URI,
                db_name=settings.ORG_DB,
                collection_name=settings.DOC_COLLECTION,
            ),
            minio_endpoint=settings.MINIO_ENDPOINT,
            minio_access_key=settings.MINIO_ACCESS_KEY,
            minio_secret_key=settings.MINIO_SECRET_KEY,
            minio_bucket=settings.MINIO_BUCKET,
            minio_secure=settings.MINIO_SECURE,
            presign_expires=settings.MINIO_PRESIGN_EXPIRES,
        )
    return _resolver_instance


async def verify_api_key(authorization: str = Header(default="")):
    settings = get_settings()
    if not settings.RAG_API_KEY:
        return
    token = authorization.replace("Bearer ", "").strip()
    if token != settings.RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _last_user_message(payload: RAGRequest) -> str:
    if not payload.messages:
        return ""
    for message in reversed(payload.messages):
        if message.role == "user":
            return message.content
    return payload.messages[-1].content


async def _rerank_chunks(rerank_func, query: str, chunks: list[dict], top_n: int):
    if not chunks:
        return []
    texts = [chunk.get("content") or "" for chunk in chunks]
    reason = "empty"
    try:
        with observe_stage("rerank"):
            results = await rerank_func(query=query, documents=texts, top_n=top_n)
        scored = []
        for result in results or []:
            idx = result.get("index")
            if isinstance(idx, int) and 0 <= idx < len(chunks):
                scored.append((chunks[idx], result.get("relevance_score")))
        if scored:
            return scored
    except Exception as exc:
        reason = "error"
        logger.warning("rerank service failed, falling back to unscored chunks: %s", exc)
    RERANK_FALLBACK_TOTAL.labels(reason=reason).inc()
    return [(chunk, None) for chunk in chunks[:top_n]]


def _passes_filters(resolved: dict | None, filters) -> bool:
    if filters is None:
        return True
    resolved = resolved or {}
    return not (
        (
            filters.organization_unit_id
            and resolved.get("organization_unit_id") != filters.organization_unit_id
        )
        or (filters.document_type and resolved.get("document_type") != filters.document_type)
    )


def _chunk_modality(chunk: dict) -> str:
    return chunk.get("modality") or "text"


async def _presign_asset(asset_store, asset_key: str | None) -> str | None:
    if not asset_key:
        return None
    url = None
    if asset_store is not None:
        try:
            url = await asyncio.to_thread(asset_store.presign, asset_key)
        except Exception:
            url = None
    if url is None:
        PRESIGN_FAILURES_TOTAL.inc()
    return url


async def _build_documents(
    scored, resolver, payload: RAGRequest, entity_list: list[dict] | None = None, asset_store=None
) -> list:
    documents = []
    for chunk, score in scored:
        file_path = chunk.get("file_path")
        try:
            with observe_stage("resolve"):
                resolved = await resolver.resolve(file_path)
        except Exception:
            resolved = None
        modality = _chunk_modality(chunk)
        metadata = {
            "source": file_path,
            "page": chunk.get("page_idx"),
            "document_id": (resolved or {}).get("document_id"),
            "chunk_index": None,
            "global_id": None,
        }
        text = chunk.get("content") or ""
        if payload.version >= 2:
            if not _passes_filters(resolved, payload.filters):
                DOCS_FILTERED_TOTAL.inc()
                continue
            documents.append(
                RetrievedDocV2(
                    text=text,
                    score=score,
                    metadata=metadata,
                    reference_id=chunk.get("reference_id"),
                    doc=resolved,
                    modality=modality,
                    artifact_url=await _presign_asset(asset_store, chunk.get("asset_key")),
                    table_body=chunk.get("table_body"),
                    page=chunk.get("page_idx"),
                    caption=chunk.get("caption"),
                    entities=entity_list,
                )
            )
        else:
            documents.append(RetrievedDoc(text=text, score=score, metadata=metadata))
        RETRIEVAL_DOCS_BY_MODALITY_TOTAL.labels(modality=modality).inc()
    return documents


async def _search(payload: RAGRequest, rag, rerank_func, resolver, asset_store=None) -> dict:
    settings = get_settings()
    query = _last_user_message(payload)
    mode = payload.mode if payload.version >= 2 else "mix"
    top_k = payload.top_k or settings.RETRIEVAL_TOP_K
    with track_retrieval(query, mode, payload.version, top_k) as tracker:
        response = await _run_search(
            payload, rag, rerank_func, resolver, asset_store, query, mode, top_k
        )
        tracker.set_documents(response["documents"])
        return response


async def _run_search(
    payload: RAGRequest, rag, rerank_func, resolver, asset_store, query: str, mode: str, top_k: int
) -> dict:
    settings = get_settings()
    started = time.perf_counter()
    with observe_stage("raganything_query"):
        result = await rag.aquery_data(
            query,
            mode=mode,
            top_k=top_k,
            chunk_top_k=settings.RETRIEVAL_CHUNK_TOP_K,
            enable_rerank=False,
        )
    if result.get("status") != "success":
        LIGHTRAG_FAILURES_TOTAL.inc()
        result_metadata = {}
        data = {}
    else:
        result_metadata = result.get("metadata") or {}
        data = result.get("data") or {}
    chunks = data.get("chunks") or []
    CHUNKS_RETRIEVED.observe(len(chunks))
    modalities = payload.filters.modality if payload.filters else None
    if modalities:
        chunks = [chunk for chunk in chunks if _chunk_modality(chunk) in modalities]
    scored = await _rerank_chunks(rerank_func, query, chunks, settings.RERANK_TOP_K)
    entity_list = None
    if payload.version >= 2 and payload.include_kg:
        entity_list = [
            {
                "name": entity.get("entity_name"),
                "type": entity.get("entity_type"),
                "description": entity.get("description"),
            }
            for entity in (data.get("entities") or [])[:20]
        ]
    documents = await _build_documents(scored, resolver, payload, entity_list, asset_store)
    if payload.version >= 2:
        return RetrievalResponseV2(
            query=query,
            documents=documents,
            mode=mode,
            references=data.get("references") or [],
            meta={
                "keywords": result_metadata.get("keywords"),
                "latency_ms": int((time.perf_counter() - started) * 1000),
            },
        ).model_dump()
    return RetrievalResponse(query=query, documents=documents).model_dump()


@router.post("/", dependencies=[Depends(verify_api_key)])
async def rag_search(
    payload: RAGRequest,
    rag=Depends(get_rag),
    rerank_func=Depends(get_rerank_func),
    resolver=Depends(get_resolver),
    asset_store=Depends(get_asset_store),
):
    return await _search(payload, rag, rerank_func, resolver, asset_store)


@router.post("/stream", dependencies=[Depends(verify_api_key)])
async def rag_stream(
    payload: RAGRequest,
    rag=Depends(get_rag),
    rerank_func=Depends(get_rerank_func),
    resolver=Depends(get_resolver),
    asset_store=Depends(get_asset_store),
):
    from fastapi.responses import StreamingResponse

    async def generate():
        yield '{"status": "retrieving"}\n'
        response = await _search(payload, rag, rerank_func, resolver, asset_store)
        for doc in response["documents"]:
            metadata = {k: v for k, v in doc["metadata"].items() if k not in ("page", "source")}
            doc_payload = {"text": doc["text"], "metadata": metadata}
            if doc.get("modality") is not None:
                doc_payload["modality"] = doc["modality"]
                doc_payload["artifact_url"] = doc.get("artifact_url")
            if doc.get("score") is not None:
                doc_payload["score"] = doc["score"]
            yield f'{{"documents": [{_dumps(doc_payload)}]}}\n'
        yield '{"status": "done"}\n'

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def _dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
