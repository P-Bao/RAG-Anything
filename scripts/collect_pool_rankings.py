#!/usr/bin/env python3
"""Collect per-modality pool rankings from the live stack, once.

Retrieves and reranks every modality pool for every case in
`tests/retrieval_cases.json` and dumps the raw result to JSON. Because the
output is a full ranking per pool, fusion strategies (RRF k, quota slots,
score floors, pool depths) can then be swept offline by
`scripts/eval_fusion_rules.py` without re-querying the embedder, Qdrant or the
rerank server for each variant.

Must run inside the API container (or anywhere with the same env and network):

    docker cp scripts/collect_pool_rankings.py ami-rag-api:/tmp/collect.py
    docker exec ami-rag-api python /tmp/collect.py --cases /tmp/cases.json \
        --out /tmp/pool_rankings.json
    docker cp ami-rag-api:/tmp/pool_rankings.json ./pool_rankings.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

DEFAULT_SIZES = {"text": 20, "table": 50, "image": 15, "other": 20}


def _parse_sizes(spec: str) -> dict[str, int]:
    sizes = {}
    for token in spec.split(","):
        if not token.strip():
            continue
        name, _, raw = token.partition("=")
        sizes[name.strip().lower()] = int(raw)
    return sizes


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--sizes", default=",".join(f"{k}={v}" for k, v in DEFAULT_SIZES.items()))
    parser.add_argument("--modalities", default="text,table,image")
    args = parser.parse_args()

    import httpx
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from ami_rag.core.calibration import RerankCalibrator
    from ami_rag.core.embedder import collection_name
    from ami_rag.core.factory import get_asset_store
    from ami_rag.core.rerank_client import (
        build_rerank_documents,
        build_vllm_rerank_func,
    )
    from ami_rag.settings import get_settings

    sizes = _parse_sizes(args.sizes)
    modalities = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    if args.collection:
        collection = args.collection
    elif os.environ.get("COLLECTION_NAME"):
        collection = os.environ["COLLECTION_NAME"]
    else:
        settings = get_settings()
        collection = collection_name(
            settings.WORKSPACE, settings.EMBED_MODEL, settings.CHUNKER_VERSION
        )

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        check_compatibility=False,
    )
    embed_url = os.environ["EMBED_SERVER_URL"].rstrip("/") + "/v1/embeddings"
    embed_model = os.environ["EMBED_MODEL"]

    calibrator = RerankCalibrator.load(
        os.environ.get("RERANK_CALIBRATION_PATH", "/app/rerank_calibration.json")
    )
    asset_store = get_asset_store()
    rerank_func = build_vllm_rerank_func()

    async with httpx.AsyncClient(timeout=60.0) as http:

        async def embed(query: str) -> list[float]:
            resp = await http.post(
                embed_url,
                json={"input": [query], "model": embed_model, "input_type": "query"},
            )
            return resp.json()["data"][0]["embedding"]

        async def collect(case: dict) -> dict:
            query = case["query"]
            started = time.perf_counter()
            vector = await embed(query)

            record: dict = {
                "id": case["id"],
                "modality": case["modality"],
                "expected_doc_id": case.get("expected_doc_id"),
                "expected_chunk_id": case.get("expected_chunk_id"),
                "xfail": bool(case.get("xfail")),
                "pools": {},
                "timings_ms": {},
            }

            for pool in modalities + ["other"]:
                limit = int(sizes.get(pool, 0))
                if limit <= 0:
                    continue
                must = [FieldCondition(key="modality", match=MatchValue(value=pool))] if pool != "other" else []
                must_not = (
                    [FieldCondition(key="modality", match=MatchValue(value=m)) for m in modalities]
                    if pool == "other"
                    else []
                )
                t0 = time.perf_counter()
                points = client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=limit,
                    query_filter=Filter(must=must, must_not=must_not),
                    with_payload=True,
                ).points
                chunks = [p.payload for p in points]
                retrieve_ms = (time.perf_counter() - t0) * 1000

                if not chunks:
                    record["pools"][pool] = []
                    record["timings_ms"][pool] = {"retrieve": round(retrieve_ms, 1), "rerank": 0.0}
                    continue

                t0 = time.perf_counter()
                docs = await build_rerank_documents(
                    chunks, asset_store, multimodal=(pool != "text")
                )
                build_ms = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                try:
                    results = await rerank_func(query=query, documents=docs, top_n=len(chunks))
                except Exception as exc:  # noqa: BLE001 - recorded, not raised
                    results = []
                    print(f"  [warn] {case['id']}/{pool}: rerank failed: {exc}")
                rerank_ms = (time.perf_counter() - t0) * 1000

                # Keep the server's preference order (that is what the API uses as the
                # pool ranking); only the index->chunk mapping is ours.
                ordered: list[dict] = []
                seen_indices: set[int] = set()
                for item in results:
                    index = item.get("index")
                    if not isinstance(index, int) or not 0 <= index < len(chunks):
                        continue
                    if index in seen_indices:
                        continue
                    seen_indices.add(index)
                    raw = float(item.get("relevance_score") or 0.0)
                    ordered.append(
                        {
                            "chunk_id": chunks[index].get("chunk_id"),
                            "doc_id": chunks[index].get("doc_id"),
                            "modality": chunks[index].get("modality"),
                            "raw": raw,
                            "cal": round(calibrator.predict(raw, chunks[index].get("modality")), 6)
                            if calibrator
                            else None,
                        }
                    )
                    record["pools"][pool] = ordered
                    record["timings_ms"][pool] = {
                        "retrieve": round(retrieve_ms, 1),
                        "build": round(build_ms, 1),
                        "rerank": round(rerank_ms, 1),
                    }

            record["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return record

        out = []
        for index, case in enumerate(cases, start=1):
            record = await collect(case)
            out.append(record)
            pool_max = {
                pool: round(max((c["cal"] or 0.0 for c in items), default=0.0), 3)
                for pool, items in record["pools"].items()
            }
            print(f"[{index}/{len(cases)}] {record['id']:<32} {record['total_ms']:>7.0f} ms  pool_max_cal={pool_max}")

        payload = {
            "collection": collection,
            "sizes": sizes,
            "modalities": modalities,
            "has_calibrator": calibrator is not None,
            "cases": out,
        }
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nWrote {args.out} ({len(out)} cases)")


if __name__ == "__main__":
    asyncio.run(main())