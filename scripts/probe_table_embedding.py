#!/usr/bin/env python3
"""A/B: chữ bảng mất dấu (do MinerU) có làm hại embedding hay không?

Bảng đang được embed dạng `{text: <chữ mất dấu>, image: <ảnh render>}`. Vì chữ
mất dấu chiếm một nửa nội dung và là rác, nó có thể kéo embedding của bảng lệch.
Câu hỏi: bỏ hẳn chữ, chỉ embed ảnh, thì table recall tăng hay giảm?

Đo vào **collection tạm**, không đụng collection production:

    A  text + image   (đang deploy)
    B  image  only    (bỏ chữ mất dấu)
    C  text  only     (không ảnh -- để thấy ảnh đóng góp bao nhiêu)

Chạy trong container (cần Qdrant + embed server):

    docker cp scripts/probe_table_embedding.py ami-rag-api:/tmp/t.py
    docker cp tests/retrieval_cases.json ami-rag-api:/tmp/cases.json
    docker exec ami-rag-api python /tmp/t.py --cases /tmp/cases.json \
        --out /tmp/table_embed_ab.json --keep-collection
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import statistics
import time
from pathlib import Path

SOURCE_COLLECTION = "multimodal__llama-nemotron-embed-vl-1b-v2__v1"
PREFIX = "probe_table_embed__"
ARMS = ("text+image", "image_only", "text_only")


def _build_item(text: str, image_b64: str | None, arm: str) -> str | dict:
    """Dựng item theo shape NỘI BỘ của `OpenAIEmbedder` (key `image_b64`).

    Client tự dịch `image_b64` -> `image` và bỏ `text` rỗng. Nếu tự gọi gateway mà
    truyền thẳng key `image`, hoặc truyền `image_b64` cho gateway, phép đo sẽ ra
    vector text-only (cos = 1.0 với text thuần) và kết luận sai hoàn toàn.
    """
    if arm == "text_only":
        return text
    if arm == "image_only":
        return {"image_b64": image_b64} if image_b64 else ""
    return {"text": text, "image_b64": image_b64} if image_b64 else text


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--depths", default="5,15,30,50")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--keep-collection",
        action="store_true",
        help="Giữ lại collection tạm để soi thủ công (mặc định xoá sau khi đo).",
    )
    args = parser.parse_args()

    from qdrant_client import QdrantClient, models

    from ami_rag.core.factory import get_asset_store

    depths = sorted({int(x) for x in args.depths.split(",")})
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    table_cases = [c for c in cases if c["modality"] == "table"]

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        check_compatibility=False,
    )
    asset_store = get_asset_store()
    embed_model = os.environ["EMBED_MODEL"]

    # Lấy toàn bộ chunk bảng của collection production.
    chunks: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=SOURCE_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="modality", match=models.MatchValue(value="table")
                    )
                ]
            ),
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        chunks.extend(p.payload for p in points)
        if offset is None:
            break
    total_chunks = len(chunks)
    # 24/413 chunk bảng không có ảnh render; arm `image_only` không áp dụng được cho
    # chúng (gateway từ chối item rỗng). So sánh trên tập có ảnh để công bằng cả 3 arm.
    chunks = [c for c in chunks if c.get("asset_key")]
    print(
        f"table chunks: {len(chunks)} có ảnh / {total_chunks} tổng "
        f"(bỏ {total_chunks - len(chunks)} chunk không có ảnh render)",
        flush=True,
    )

    images: dict[str, str] = {}
    keys = {c.get("asset_key") for c in chunks if c.get("asset_key")}
    for key in sorted(k for k in keys if k):
        data = await asyncio.to_thread(asset_store.get_bytes, key)
        if data:
            images[key] = base64.b64encode(data).decode("ascii")
    print(f"assets tải được: {len(images)}/{len(keys)}", flush=True)
    chunks = [c for c in chunks if c.get("asset_key") in images]
    print(f"còn {len(chunks)} chunk sau khi yêu cầu ảnh tải được", flush=True)

    # Dùng chính OpenAIEmbedder thay vì tự gọi gateway: client cắt text theo budget
    # token và dịch `image_b64` -> `image`. Tự dựng payload sẽ 400 (text quá dài) hoặc
    # âm thầm ra vector text-only (thiếu key `image`) => phép A/B sai hoàn toàn.
    from ami_rag.core.openai_embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder(os.environ["EMBED_SERVER_URL"].rstrip("/"))

    async def embed(items: list) -> list[list[float]]:
        out: list[list[float]] = []
        step = max(1, args.batch_size)
        for start in range(0, len(items), step):
            batch = items[start : start + step]
            out.extend(await embedder.embed_documents(batch))
            print(f"  embedded {len(out)}/{len(items)}", flush=True)
        return out

    async def embed_query(query: str) -> list[float]:
        return await embedder.embed_query(query)

    results: dict[str, dict] = {}
    for arm in arms:
        started = time.perf_counter()
        items = [
            _build_item(
                c.get("content") or "",
                images.get(c.get("asset_key") or ""),
                arm,
            )
            for c in chunks
        ]
        vectors = await embed(items)

        name = PREFIX + arm.replace("+", "_")
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=2048, distance=models.Distance.COSINE),
        )
        client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(
                    id=i,
                    vector=v,
                    payload={
                        "chunk_id": c.get("chunk_id"),
                        "doc_id": c.get("doc_id"),
                        "modality": "table",
                        "page": c.get("page"),
                    },
                )
                for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )

        max_depth = max(depths)
        ranks: dict[str, int | None] = {}
        for case in table_cases:
            points = client.query_points(
                collection_name=name,
                query=await embed_query(case["query"]),
                limit=max_depth,
                with_payload=["chunk_id"],
            ).points
            ids = [p.payload.get("chunk_id") for p in points]
            target = case["expected_chunk_id"]
            ranks[case["id"]] = ids.index(target) + 1 if target in ids else None

        found = [r for r in ranks.values() if r]
        results[arm] = {
            "collection": name,
            "chunks": len(chunks),
            "embed_s": round(time.perf_counter() - started, 1),
            "reachable": len(found),
            "median_rank": statistics.median(found) if found else None,
            "recall": {
                str(d): sum(1 for r in ranks.values() if (r or 10**9) <= d)
                for d in depths
            },
            "ranks": ranks,
        }
        print(f"\n[{arm}] {results[arm]['recall']}", flush=True)

        if not args.keep_collection and arm != arms[-1]:
            client.delete_collection(name)

    payload = {
    "meta": {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_collection": SOURCE_COLLECTION,
        "embed_model": embed_model,
        "table_chunks_with_image": len(chunks),
        "table_chunks_total": total_chunks,
        "depths": depths,
        "note": (
            "Đo trên collection tạm, production không bị đụng. Bỏ collection tạm "
            "bằng `python -c 'client.delete_collection(name)'` sau khi xem xong."
        ),
    },
    "results": results,
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Table recall theo arm (9 case bảng) ===")
    print(f"{'arm':<12}" + "".join(f"@{d:<5}" for d in depths) + "reachable")
    for arm, res in results.items():
        cells = "".join(f"{res['recall'][str(d)]:<6}" for d in depths)
        print(f"{arm:<12}{cells}{res['reachable']}")
    print(f"\nWrote {args.out}")
    if not args.keep_collection:
        last = PREFIX + arms[-1].replace("+", "_")
        if client.collection_exists(last):
            client.delete_collection(last)
        print(f"đã xoá collection tạm {last}")


if __name__ == "__main__":
    asyncio.run(main())