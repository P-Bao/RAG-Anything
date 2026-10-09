import asyncio
import logging
import time

from fastapi import APIRouter, Depends, Header, HTTPException

from ami_rag.api.schemas import (
    RAGRequest,
    RetrievalResponseV2,
    RetrievedDocV2,
)
from ami_rag.core.fusion import (
    OTHER_POOL,
    default_score_of,
    fuse,
    gate_image_hits,
)
from ami_rag.observability import (
    CHUNKS_RETRIEVED,
    FUSION_DOCS_SELECTED_TOTAL,
    FUSION_IMAGE_GATE_DROPPED_TOTAL,
    FUSION_POOL_CANDIDATES,
    PRESIGN_FAILURES_TOTAL,
    RERANK_FALLBACK_TOTAL,
    RETRIEVAL_DOCS_BY_MODALITY_TOTAL,
    observe_stage,
    track_retrieval,
)
from ami_rag.settings import get_settings, resolve_fusion

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


async def get_pipeline():
    from ami_rag.core.factory import get_pipeline as _get_pipeline

    return await _get_pipeline()


def get_asset_store():
    from ami_rag.core.factory import get_asset_store as _get_asset_store

    return _get_asset_store()


def get_rerank_func():
    from ami_rag.core.rerank_client import build_rerank_func

    return build_rerank_func(get_settings())


_calibrator_instance = None
_calibrator_loaded = False


def get_calibrator():
    global _calibrator_instance, _calibrator_loaded
    if not _calibrator_loaded:
        settings = get_settings()
        if getattr(settings, "RERANK_CALIBRATION_ENABLED", False):
            from ami_rag.core.calibration import RerankCalibrator

            cal_path = getattr(settings, "RERANK_CALIBRATION_PATH", "./rerank_calibration.json")
            _calibrator_instance = RerankCalibrator.load(cal_path)
            if _calibrator_instance:
                logger.info("Loaded rerank calibrator from %s", cal_path)
            else:
                logger.warning(
                    "Could not load rerank calibrator from %s, continuing without calibration",
                    cal_path,
                )
        _calibrator_loaded = True
    return _calibrator_instance


def reset_calibrator():
    global _calibrator_instance, _calibrator_loaded
    _calibrator_instance = None
    _calibrator_loaded = False


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


async def _rerank_chunks(
    rerank_func,
    query: str,
    chunks: list[dict],
    top_n: int,
    asset_store=None,
    multimodal: bool | None = None,
):
    if not chunks:
        return []
    settings = get_settings()
    from ami_rag.core.rerank_client import build_rerank_documents
    from ami_rag.settings import resolve_rerank_backend

    if multimodal is None:
        # Multimodal chỉ với backend vllm (legacy BGE chỉ nhận documents text)
        multimodal = (
            getattr(settings, "RERANK_MULTIMODAL", True)
            and resolve_rerank_backend(settings) == "vllm"
        )
    documents = await build_rerank_documents(chunks, asset_store, multimodal=multimodal)
    reason = "empty"
    try:
        with observe_stage("rerank"):
            results = await rerank_func(query=query, documents=documents, top_n=top_n)
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


async def _build_documents(scored, resolver, asset_store=None) -> list:
    """Build v2 documents: resolve mỗi chunk qua doc_id (payload mang doc_id trực tiếp).

    Mỗi item là (chunk, score) hoặc (chunk, score, raw_score) hoặc
    (chunk, score, raw_score, calibrated_score): phần tử thứ 3 là điểm rerank
    thô, phần tử thứ 4 là xác suất đã calibration (chỉ khác `score` khi fusion
    dùng thang điểm riêng như RRF).
    """
    documents = []
    resolved_cache: dict[str, dict | None] = {}
    for item in scored:
        if len(item) == 4:
            chunk, score, raw_score, calibrated_score = item
            has_calibration = True
        elif len(item) == 3:
            chunk, score, raw_score = item
            calibrated_score = score
            has_calibration = True
        else:
            chunk, score = item
            raw_score = None
            calibrated_score = None
            has_calibration = False
        doc_id = chunk.get("doc_id") or ""
        if doc_id not in resolved_cache:
            try:
                resolved_cache[doc_id] = await resolver.resolve_doc_id(doc_id)
            except Exception:
                resolved_cache[doc_id] = None
        resolved = resolved_cache[doc_id]
        modality = chunk.get("modality") or "text"
        page = chunk.get("page")
        metadata = {
            "source": chunk.get("source_path") or "",
            "page": page,
            "document_id": doc_id,
            "chunk_index": None,
            "global_id": chunk.get("chunk_id"),
        }
        if has_calibration:
            if raw_score is not None:
                metadata["raw_score"] = raw_score
            if calibrated_score is not None:
                metadata["calibrated_score"] = calibrated_score
        documents.append(
            RetrievedDocV2(
                text=chunk.get("content") or "",
                score=score,
                metadata=metadata,
                reference_id=chunk.get("chunk_id"),
                doc=resolved,
                modality=modality,
                artifact_url=await _presign_asset(asset_store, chunk.get("asset_key")),
                table_body=chunk.get("table_body"),
                page=page,
                caption=chunk.get("caption"),
            )
        )
        RETRIEVAL_DOCS_BY_MODALITY_TOTAL.labels(modality=modality).inc()
    return documents


def _final_top_k(payload: RAGRequest, settings) -> int:
    """Number of documents returned to the caller (client `top_k`, else RERANK_TOP_K)."""
    return payload.top_k or settings.RERANK_TOP_K


async def _search(
    payload: RAGRequest,
    pipeline,
    rerank_func,
    resolver,
    asset_store=None,
    calibrator=None,
) -> dict:
    settings = get_settings()
    query = _last_user_message(payload)
    top_k = _final_top_k(payload, settings)
    with track_retrieval(query, "vector", 2, top_k) as tracker:
        response = await _run_search(
            payload,
            pipeline,
            rerank_func,
            resolver,
            asset_store,
            query,
            top_k,
            calibrator=calibrator,
        )
        tracker.set_documents(response["documents"])
        return response


def _chunk_dedup_key(chunk: dict) -> str:
    """Khóa xác định nội dung duy nhất của chunk để chống trùng lặp."""
    modality = chunk.get("modality") or "text"
    if modality == "image":
        asset_key = chunk.get("asset_key")
        if asset_key:
            return f"img:{asset_key}"
        return f"chunk:{chunk.get('chunk_id')}"
    if modality == "table":
        body = (chunk.get("table_body") or chunk.get("content") or "").strip()
        return f"tbl:{chunk.get('doc_id')}:{body}"
    text = (chunk.get("content") or "").strip()
    return f"txt:{text}"


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Loại bỏ các chunk có cùng nội dung chuẩn hóa, giữ lại chunk đầu tiên (điểm cao nhất)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for chunk in chunks:
        key = _chunk_dedup_key(chunk)
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


async def _retrieve_pool(pipeline, vector, limit, modalities=(), exclude=()):
    """Retrieve one pool from Qdrant.

    `modalities` selects the pool by an OR over modality values (a pool can hold
    several, e.g. text and table); `exclude` builds the catch-all pool for
    modalities outside every pool.
    """
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    should = [
        FieldCondition(key="modality", match=MatchValue(value=m)) for m in modalities
    ]
    must = []
    must_not = [
        FieldCondition(key="modality", match=MatchValue(value=m)) for m in exclude
    ]
    hits = await pipeline.vector_store.search(
        pipeline.collection,
        vector,
        limit,
        query_filter=Filter(should=should or None, must=must, must_not=must_not),
    )
    return _deduplicate_chunks([hit.payload for hit in hits])


async def _retrieve_pools(pipeline, vector, fusion) -> dict[str, list[dict]]:
    """Retrieve every modality pool concurrently, each with its own depth."""
    groups = fusion["groups"]
    sizes = fusion["sizes"]
    claimed = [m for mods in groups.values() for m in mods]
    requests = []
    for name, mods in list(groups.items()) + [(OTHER_POOL, ())]:
        limit = int(sizes.get(name, 0))
        if limit <= 0:
            continue
        if name == OTHER_POOL:
            requests.append((name, _retrieve_pool(pipeline, vector, limit, exclude=claimed)))
        else:
            requests.append((name, _retrieve_pool(pipeline, vector, limit, modalities=mods)))

    if not requests:
        return {}

    results = await asyncio.gather(*(coro for _, coro in requests))
    pools: dict[str, list[dict]] = {}
    for (name, _), chunks in zip(requests, results):
        FUSION_POOL_CANDIDATES.labels(pool=name).observe(len(chunks))
        pools[name] = chunks
    return pools


async def _rerank_pools(rerank_func, query, pools, asset_store, fusion) -> dict[str, list]:
    """Rerank every pool concurrently: wall clock is the slowest pool, not the sum.

    Each pool is reranked with `top_n=len(pool)` so the full ordering survives for
    rank-based fusion; a pool whose rerank call fails falls back to its own vector
    order (scores None) without affecting the other pools.
    """
    async def run(pool: str, chunks: list[dict]):
        with observe_stage(f"rerank.{pool}"):
            return await _rerank_chunks(
                rerank_func,
                query,
                chunks,
                len(chunks),
                asset_store,
                # Tables ride the text rerank as plain text by default; adding
                # the text pool to RETRIEVAL_FUSION_VL_POOLS sends rendered
                # table images too, which is the expensive variant.
                multimodal=pool in fusion["vl_pools"],
            )

    names = list(pools)
    if not names:
        return {}
    results = await asyncio.gather(*(run(name, pools[name]) for name in names))
    return dict(zip(names, results))


def _score_fn(calibrator):
    """The single score used by the image gate and by the merge, so the gate
    filters on exactly the number the merge would rank on."""
    def score_of(chunk, raw_score):
        if calibrator is not None:
            return calibrator.predict(raw_score, chunk.get("modality"))
        return default_score_of(chunk, raw_score)

    return score_of


def _fuse_pools(reranked, fusion, calibrator, top_k, score_of) -> list:
    """Merge the per-pool rankings and return hits ready for `_build_documents`."""
    hits = fuse(
        reranked,
        strategy=fusion["mode"],
        top_k=top_k,
        k=fusion["rrf_k"],
        quotas=fusion["quotas"],
        score_of=score_of,
    )
    for hit in hits:
        FUSION_DOCS_SELECTED_TOTAL.labels(pool=hit.pool).inc()
    if calibrator is not None:
        return [
            (hit.chunk, hit.score, hit.raw_score, calibrator.predict(hit.raw_score, hit.chunk.get("modality")))
            for hit in hits
        ]
    return [(hit.chunk, hit.score, hit.raw_score) for hit in hits]


async def _run_search_fused(
    payload: RAGRequest,
    pipeline,
    rerank_func,
    resolver,
    asset_store,
    query: str,
    top_k: int,
    calibrator=None,
    fusion: dict | None = None,
) -> dict:
    """Retrieve + rerank each modality pool on its own, then fuse the rankings."""
    settings = get_settings()
    if fusion is None:
        fusion = resolve_fusion(settings)
    started = time.perf_counter()
    with observe_stage("embed_query"):
        vector = await pipeline.embedder.embed_query(query)
    with observe_stage("vector_search"):
        pools = await _retrieve_pools(pipeline, vector, fusion)

    reranked = await _rerank_pools(rerank_func, query, pools, asset_store, fusion)

    candidates = sum(len(items) for items in reranked.values())
    CHUNKS_RETRIEVED.observe(candidates)

    score_of = _score_fn(calibrator)
    gated = 0
    if fusion.get("image_gate"):
        reranked, gated = gate_image_hits(
            reranked,
            pool_groups=fusion["groups"],
            top_k=top_k,
            score_of=score_of,
        )
        if gated:
            FUSION_IMAGE_GATE_DROPPED_TOTAL.inc(gated)

    scored = _fuse_pools(reranked, fusion, calibrator, top_k, score_of)
    documents = await _build_documents(scored, resolver, asset_store)
    references = []
    if payload.include_references:
        seen: set[str] = set()
        for doc in documents:
            doc_id = (doc.doc or {}).get("document_id")
            if doc.doc and doc_id and doc_id not in seen:
                seen.add(doc_id)
                references.append(doc.doc)
    return RetrievalResponseV2(
        query=query,
        documents=documents,
        references=references,
        meta={
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "requested_top_k": top_k,
            "returned": len(documents),
            "candidates": candidates,
            "fusion": {
                "mode": fusion["mode"],
                "pools": {name: len(items) for name, items in reranked.items()},
                "image_gate_dropped": gated,
            },
        },
    ).model_dump()


async def _run_search_single(
    payload: RAGRequest,
    pipeline,
    rerank_func,
    resolver,
    asset_store,
    query: str,
    top_k: int,
    calibrator=None,
) -> dict:
    """embed (máy B) -> vector search -> rerank -> calibrate. `top_k * RETRIEVAL_OVERFETCH`
    candidates để rerank luôn đủ `top_k` kết quả khi có."""
    settings = get_settings()
    started = time.perf_counter()
    candidates_wanted = max(
        settings.RETRIEVAL_CHUNK_TOP_K,
        top_k * max(1, settings.RETRIEVAL_OVERFETCH),
    )
    fetch_k = candidates_wanted * 2
    with observe_stage("embed_query"):
        vector = await pipeline.embedder.embed_query(query)
    with observe_stage("vector_search"):
        hits = await pipeline.vector_store.search(
            pipeline.collection,
            vector,
            fetch_k,
        )
        raw_chunks = [hit.payload for hit in hits]
        if getattr(settings, "RERANK_MULTIMODAL", True):
            try:
                from qdrant_client.http.models import FieldCondition, Filter, MatchValue

                img_filter = Filter(must=[FieldCondition(key="modality", match=MatchValue(value="image"))])
                img_hits = await pipeline.vector_store.search(
                    pipeline.collection,
                    vector,
                    top_k=5,
                    query_filter=img_filter,
                )
                raw_chunks = [hit.payload for hit in img_hits] + raw_chunks
            except Exception as exc:
                logger.debug("Multimodal candidate diversification skipped: %s", exc)

    chunks = _deduplicate_chunks(raw_chunks)[:candidates_wanted]
    CHUNKS_RETRIEVED.observe(len(chunks))
    candidates = len(chunks)

    # Rerank all candidates when calibration is active to allow calibrated reranking
    rerank_top_n = (
        len(chunks)
        if (calibrator is not None or getattr(settings, "RERANK_CALIBRATION_ENABLED", False))
        else top_k
    )
    scored_pairs = await _rerank_chunks(rerank_func, query, chunks, rerank_top_n, asset_store)

    if calibrator is not None:
        calibrated = calibrator.calibrate_and_rank(scored_pairs)
        scored = [(chunk, cal_score, raw_score) for chunk, cal_score, raw_score in calibrated[:top_k]]
    else:
        scored = [(chunk, raw_score) for chunk, raw_score in scored_pairs[:top_k]]

    documents = await _build_documents(scored, resolver, asset_store)
    references = []
    if payload.include_references:
        seen: set[str] = set()
        for doc in documents:
            doc_id = (doc.doc or {}).get("document_id")
            if doc.doc and doc_id and doc_id not in seen:
                seen.add(doc_id)
                references.append(doc.doc)
    return RetrievalResponseV2(
        query=query,
        documents=documents,
        references=references,
        meta={
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "requested_top_k": top_k,
            "returned": len(documents),
            "candidates": candidates,
        },
    ).model_dump()


async def _run_search(
    payload: RAGRequest,
    pipeline,
    rerank_func,
    resolver,
    asset_store,
    query: str,
    top_k: int,
    calibrator=None,
) -> dict:
    """Dispatch to per-modality fusion, or the original single-pool flow.

    `RETRIEVAL_FUSION_MODE=single` (default) keeps the previous behaviour
    untouched: one unfiltered retrieve plus a hardcoded image top-5, one rerank
    call, then sort. The fusion modes retrieve and rerank each modality pool
    separately and merge the rankings (see `ami_rag.core.fusion`).
    """
    fusion = resolve_fusion(get_settings())
    if fusion is None:
        return await _run_search_single(
            payload, pipeline, rerank_func, resolver, asset_store, query, top_k, calibrator
        )
    return await _run_search_fused(
        payload,
        pipeline,
        rerank_func,
        resolver,
        asset_store,
        query,
        top_k,
        calibrator=calibrator,
        fusion=fusion,
    )


@router.post("/", dependencies=[Depends(verify_api_key)])
async def rag_search(
    payload: RAGRequest,
    pipeline=Depends(get_pipeline),
    rerank_func=Depends(get_rerank_func),
    resolver=Depends(get_resolver),
    asset_store=Depends(get_asset_store),
    calibrator=Depends(get_calibrator),
):
    return await _search(
        payload, pipeline, rerank_func, resolver, asset_store, calibrator=calibrator
    )


@router.post("/stream", dependencies=[Depends(verify_api_key)])
async def rag_stream(
    payload: RAGRequest,
    pipeline=Depends(get_pipeline),
    rerank_func=Depends(get_rerank_func),
    resolver=Depends(get_resolver),
    asset_store=Depends(get_asset_store),
    calibrator=Depends(get_calibrator),
):
    from fastapi.responses import StreamingResponse

    async def generate():
        yield '{"status": "retrieving"}\n'
        response = await _search(
            payload, pipeline, rerank_func, resolver, asset_store, calibrator=calibrator
        )
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
