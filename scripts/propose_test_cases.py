#!/usr/bin/env python3
"""Nguyên liệu để mở rộng bộ test: chunk đích, chunk dễ nhầm, và truy vấn nháp.

Bộ case hiện tại chỉ 28 mục, mỗi mục gắn đúng một truy vấn. Không có LLM trong
stack nên script không thể tự viết câu hỏi hay đoạn phân biệt có chất lượng; thay
vào đoán nó bỏ ra thứ thật sự cần để viết:

1. `target` -- nội dung chunk đích đầy đủ (để soạn paraphrase).
2. `confusables` -- các chunk cùng pool hay bị nhầm với target (nguyên liệu cho
   minimal-pair: câu hỏi buộc phải phân biệt đúng hai chunk này).
3. `mechanical_drafts` -- câu hỏi dựng bằng mẫu từ từ khoá, đánh dấu
   `needs_authoring` vì chất lượng thấp.
4. `validate` -- cách kiểm tra ngay một truy vấn vừa viết.

Must run inside the API container:

    docker cp scripts/propose_test_cases.py ami-rag-api:/tmp/propose.py
    docker cp tests/retrieval_cases.json ami-rag-api:/tmp/cases.json
    docker exec ami-rag-api python /tmp/propose.py --cases /tmp/cases.json \
        --out /tmp/proposals.json
    docker cp ami-rag-api:/tmp/proposals.json tests/test_case_proposals.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

SNIPPET_CHARS = 260


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _distinctive_terms(text: str, limit: int = 6) -> list[str]:
    """Từ khoá đặc trưng: cụm từ viết hoa + số + từ dài, loại từ dừng tiếng Việt."""
    lowered = _clean(text).lower()
    words = re.findall(r"[\wÀ-ỹ]+", lowered, flags=re.UNICODE)
    noise = {
        "của", "các", "và", "cho", "những", "một", "trong", "với", "là", "được",
        "theo", "này", "như", "khi", "về", "tại", "để", "có", "không", "đã", "sẽ",
        "người", "năm", "công", "thông", "tin", "học", "viện", "sinh", "viên",
    }
    candidates = [w for w in words if len(w) > 3 and w not in noise]
    counter = Counter(candidates)
    ranked = [term for term, _ in counter.most_common() if len(term) > 3]
    if not ranked:
        return []
    return ranked[:limit]


def _title_of(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip().lstrip("# ").strip()
        if len(stripped) >= 12:
            return stripped[:SNIPPET_CHARS]
    return _clean(text)[:SNIPPET_CHARS]


def _is_target(payload: dict, case: dict) -> bool:
    chunk_target = case.get("expected_chunk_id")
    if chunk_target:
        return payload.get("chunk_id") == chunk_target
    return payload.get("doc_id") == case.get("expected_doc_id")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--probe-depth", type=int, default=15)
    parser.add_argument("--confusables", type=int, default=6)
    args = parser.parse_args()

    import httpx
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from ami_rag.core.embedder import collection_name
    from ami_rag.settings import get_settings

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

    async with httpx.AsyncClient(timeout=60.0) as http:

        async def embed(query: str) -> list[float]:
            resp = await http.post(
                embed_url,
                json={"input": [query], "model": embed_model, "input_type": "query"},
            )
            return resp.json()["data"][0]["embedding"]

        proposals: list[dict] = []
        for case in cases:
            vector = await embed(case["query"])
            modality_filter = Filter(
                must=[
                    FieldCondition(
                        key="modality", match=MatchValue(value=case["modality"])
                    )
                ]
            )
            points = client.query_points(
                collection_name=collection,
                query=vector,
                limit=args.probe_depth,
                query_filter=modality_filter,
                with_payload=True,
            ).points

            target_payload = None
            target_rank = None
            confusables = []
            for rank, point in enumerate(points, start=1):
                payload = point.payload
                if _is_target(payload, case):
                    if target_rank is None:
                        target_payload = payload
                        target_rank = rank
                    continue
                if len(confusables) < args.confusables:
                    confusables.append(
                        {
                            "vector_rank": rank,
                            "chunk_id": payload.get("chunk_id"),
                            "doc_id": payload.get("doc_id"),
                            "page": payload.get("page"),
                            "snippet": _clean(payload.get("content") or "")[
                                :SNIPPET_CHARS
                            ],
                            "distinctive_terms": _distinctive_terms(
                                payload.get("content") or ""
                            ),
                        }
                    )

            target_text = ""
            if target_payload:
                target_text = (
                    target_payload.get("content")
                    or target_payload.get("table_body")
                    or target_payload.get("caption")
                    or ""
                )
            else:
                # target không nằm trong pool depth này: lấy trực tiếp theo id.
                chunk_target = case.get("expected_chunk_id")
                if chunk_target:
                    found = client.scroll(
                        collection_name=collection,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="chunk_id", match=MatchValue(value=chunk_target)
                                )
                            ]
                        ),
                        limit=1,
                        with_payload=True,
                    )[0]
                    if found:
                        target_payload = found[0].payload
                        target_text = (
                            target_payload.get("content")
                            or target_payload.get("table_body")
                            or ""
                        )

            terms = _distinctive_terms(target_text)
            drafts = []
            if terms:
                drafts.append(
                    {
                        "kind": "keyword_rephrase",
                        "query": " ".join(terms),
                        "needs_authoring": True,
                        "note": "Ghép từ khoá đặc trưng; phải viết lại thành câu hỏi.",
                    }
                )
            if confusables and terms:
                sibling = confusables[0]["distinctive_terms"][:3]
                drafts.append(
                    {
                        "kind": "minimal_pair",
                        "query": "Phân biệt " + " ".join(sibling) + " với " + terms[0],
                        "needs_authoring": True,
                        "against_chunk_id": confusables[0]["chunk_id"],
                        "note": (
                            "Câu hỏi buộc phải tách target khỏi chunk dễ nhầm nhất; "
                            "đáp án phải là target."
                        ),
                    }
                )

            proposals.append(
                {
                    "source_case_id": case["id"],
                    "modality": case["modality"],
                    "current_query": case["query"],
                    "current_xfail": bool(case.get("xfail")),
                    "expected_doc_id": case.get("expected_doc_id"),
                    "expected_chunk_id": case.get("expected_chunk_id"),
                    "target": {
                        "found_in_pool_at_probe_depth": target_rank is not None,
                        "vector_rank": target_rank,
                        "page": (target_payload or {}).get("page"),
                        "text": target_text,
                        "distinctive_terms": terms,
                    },
                    "confusables": confusables,
                    "mechanical_drafts": drafts,
                    "validate": {
                        "how": (
                            "Đưa câu hỏi mới vào retrieval_cases.json với "
                            "expected_chunk_id/expected_doc_id của target rồi chạy "
                            "`python scripts/probe_pool_recall.py --cases ...` để xem "
                            "target có nằm trong pool @15 không."
                        ),
                        "keep_if": (
                            "target nằm trong pool @15 (exact_rank <= 15). Nếu > 15 thì "
                            "case đó sẽ hỏng vì pool sâu, đánh dấu xfail với "
                            "vector_rank ghi lại."
                        ),
                    },
                }
            )
            print(
                f"  {case['id']:<34} target_rank={target_rank}  "
                f"confusables={len(confusables)}  drafts={len(drafts)}",
                flush=True,
            )

    reachable = sum(1 for p in proposals if p["target"]["found_in_pool_at_probe_depth"])
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "collection": collection,
            "probe_depth": args.probe_depth,
            "source_cases": len(proposals),
            "target_reachable_at_probe_depth": reachable,
            "how_to_use": [
                (
                    "Đây là NGUYÊN LIỆU, chưa phải bộ test. Chưa case nào được thêm "
                    "vào tests/retrieval_cases.json."
                ),
                (
                    "`mechanical_drafts` chỉ là nháp sinh bằng mẫu, chất lượng thấp: "
                    "hãy viết lại bằng tiếng Việt tự nhiên trước khi dùng."
                ),
                (
                    "`confusables[0..]` là các chunk hay bị nhặt nhầm: dùng để viết "
                    "minimal-pair có tác dụng thật."
                ),
                "Chỉ giữ câu hỏi mà target nằm trong pool @15, xem `validate.keep_if`.",
                (
                    "Ưu tiên mở rộng text/table: collection chỉ có 70 chunk ảnh nên "
                    "case ảnh không mở rộng được nhiều."
                ),
            ],
        },
        "proposals": proposals,
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {args.out} ({len(proposals)} sources, target trong pool@"
          f"{args.probe_depth}: {reachable}/{len(proposals)})")


if __name__ == "__main__":
    asyncio.run(main())