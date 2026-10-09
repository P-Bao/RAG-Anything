#!/usr/bin/env python3
"""Đo trần recall của vector search theo từng modality pool, KHÔNG rerank.

Mục đích: tách hai nguồn thất bại khác nhau mà end-to-end recall gộp lại:
1. target không nằm trong pool (retrieval kém / pool quá nông), và
2. target có trong pool nhưng bị rerank đẩy ra khỏi top_k (rerank kém).

Chỉ số 1 đo ở đây; số 2 cần `scripts/collect_pool_rankings.py`.

Must run inside the API container (hoặc nơi có cùng env + network):

    docker cp scripts/probe_pool_recall.py ami-rag-api:/tmp/probe.py
    docker cp tests/retrieval_cases.json ami-rag-api:/tmp/cases.json
    docker exec ami-rag-api python /tmp/probe.py --cases /tmp/cases.json \
        --out /tmp/pool_recall.json
    docker cp ami-rag-api:/tmp/pool_recall.json tests/pool_recall_probe.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

DEFAULT_DEPTHS = [5, 15, 30, 50]


def _parse_depths(spec: str) -> list[int]:
    return sorted({int(part.strip()) for part in spec.split(",") if part.strip()})


def _exact_rank(
    chunks: list[dict],
    case: dict,
) -> int | None:
    """Hạng (1-based) của đúng target mà case khai báo.

    Case `text` chỉ có `expected_doc_id` (không ghim chunk cụ thể) nên lấy chunk
    đầu tiên thuộc tài liệu đó. Case `table`/`image` có `expected_chunk_id` nên
    bắt buộc trúng chunk đó — dùng `doc_id` ở đây sẽ tính nhầm vì một tài liệu
    có nhiều chunk và chunk bảng/ảnh đích có thể rất sâu trong pool text.
    """
    chunk_target = case.get("expected_chunk_id")
    if chunk_target:
        for index, chunk in enumerate(chunks, start=1):
            if chunk.get("chunk_id") == chunk_target:
                return index
        return None
    doc_target = case.get("expected_doc_id")
    for index, chunk in enumerate(chunks, start=1):
        if chunk.get("doc_id") == doc_target:
            return index
    return None


def _doc_rank(chunks: list[dict], case: dict) -> int | None:
    """Hạng của chunk đầu tiên thuộc tài liệu chứa target (thông tin phụ).

    Cho biết tài liệu chứa bảng/ảnh đích có mặt trong pool hay không, tách khỏi
    việc chính chunk đích có tới được hay không.
    """
    doc_target = case.get("expected_doc_id")
    if not doc_target:
        return None
    for index, chunk in enumerate(chunks, start=1):
        if chunk.get("doc_id") == doc_target:
            return index
    return None


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--modalities", default="text,table,image")
    parser.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS))
    args = parser.parse_args()

    import httpx
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from ami_rag.core.embedder import collection_name
    from ami_rag.settings import get_settings

    depths = _parse_depths(args.depths)
    max_depth = max(depths)
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

    # Kích thước pool thật của từng modality: quyết định depth có ý nghĩa hay không.
    pool_sizes = {
        modality: client.count(
            collection_name=collection,
            count_filter=Filter(
                must=[FieldCondition(key="modality", match=MatchValue(value=modality))]
            ),
            exact=True,
        ).count
        for modality in modalities
    }

    async with httpx.AsyncClient(timeout=60.0) as http:

        async def embed(query: str) -> list[float]:
            resp = await http.post(
                embed_url,
                json={"input": [query], "model": embed_model, "input_type": "query"},
            )
            return resp.json()["data"][0]["embedding"]

        results: list[dict] = []
        for case in cases:
            started = time.perf_counter()
            vector = await embed(case["query"])
            record: dict = {
                "id": case["id"],
                "modality": case["modality"],
                "xfail": bool(case.get("xfail")),
                "exact_rank": {},
                "doc_rank": {},
                "embed_ms": 0.0,
            }
            for modality in modalities:
                points = client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=max_depth,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="modality", match=MatchValue(value=modality)
                            )
                        ]
                    ),
                    with_payload=True,
                ).points
                chunks = [p.payload for p in points]
                record["exact_rank"][modality] = _exact_rank(chunks, case)
                record["doc_rank"][modality] = _doc_rank(chunks, case)
            record["embed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            results.append(record)
            print(
                f"  {case['id']:<34}"
                + "  ".join(
                    f"{modality}={record['exact_rank'][modality] or '-'}"
                    for modality in modalities
                ),
                flush=True,
            )

    summary: dict[str, dict] = {}
    for modality in modalities:
        ranks = [r["exact_rank"][modality] for r in results if r["exact_rank"][modality]]
        entry = {
            "pool_size": pool_sizes[modality],
            "probed_depth": max_depth,
            "exact_target_found": len(ranks),
            "total_cases": len(results),
            "median_rank": statistics.median(ranks) if ranks else None,
            "recall": {},
        }
        for depth in depths:
            hit = sum(1 for rank in ranks if rank <= depth)
            entry["recall"][str(depth)] = {
                "hit": hit,
                "total": len(results),
                "rate": round(hit / len(results), 4) if results else 0.0,
            }
        summary[modality] = entry

    # Recall đúng nghĩa: chỉ tính trên các case có target thuộc modality đó.
    by_modality: dict[str, dict] = {}
    for modality in modalities:
        own = [r for r in results if r["modality"] == modality]
        entry = {
            "cases": len(own),
            "recall": {},
            "unreachable_at_probe_depth": [
                r["id"] for r in own if not r["exact_rank"][modality]
            ],
        }
        for depth in depths:
            hit = sum(1 for r in own if (r["exact_rank"][modality] or 10**9) <= depth)
            entry["recall"][str(depth)] = {
                "hit": hit,
                "total": len(own),
                "rate": round(hit / len(own), 4) if own else 0.0,
            }
        by_modality[modality] = entry

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "collection": collection,
            "embed_model": embed_model,
            "depths": depths,
            "modalities": modalities,
            "method": "vector search only, no rerank / no fusion / no gate",
            "note": (
                "exact_rank = vị trí đúng target của case trong pool: case text (không có "
                "expected_chunk_id) dùng expected_doc_id, case table/image dùng "
                "expected_chunk_id. doc_rank = vị trí chunk đầu tiên của tài liệu chứa "
                "target, cho biết tài liệu có tới được hay không."
            ),
        },
        "recall_over_all_cases": summary,
        "recall_by_case_modality": by_modality,
        "cases": results,
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== Recall theo case modality (thuần vector, không rerank) ===")
    header = "  ".join(f"@{d:<3}" for d in depths)
    print(f"{'modality':<10}{'cases':>6}{'pool':>6}  {header}")
    for modality in modalities:
        entry = by_modality[modality]
        cells = "  ".join(
            f"{entry['recall'][str(d)]['hit']:>2}/{entry['recall'][str(d)]['total']:<3}"
            for d in depths
        )
        pool = summary[modality]["pool_size"]
        print(f"{modality:<10}{entry['cases']:>6}{pool:>6}  {cells}")
    unreachable = {
        m: e["unreachable_at_probe_depth"]
        for m, e in by_modality.items()
        if e["unreachable_at_probe_depth"]
    }
    if unreachable:
        print(f"\nTarget KHÔNG tới được dù probe depth {max_depth}: {unreachable}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())