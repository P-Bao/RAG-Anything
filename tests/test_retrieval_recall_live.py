"""Live integration test for top_k=5 retrieval recall over indexed documents in Qdrant.

Requires running ami-rag stack (API server on port 8009, Qdrant, embed & rerank).
Automatically skips if API server is unreachable.
"""

import json
import os
import time
from pathlib import Path

import httpx
import pytest

CASES_FILE = Path(__file__).parent / "retrieval_cases.json"
BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8009").rstrip("/")
API_KEY = os.getenv("RAG_API_KEY", "")
TIMEOUT = float(os.getenv("RAG_TIMEOUT", "60.0"))


def _load_cases() -> list[dict]:
    with open(CASES_FILE, encoding="utf-8") as f:
        return json.load(f)


def _check_service_ready() -> bool:
    try:
        res = httpx.get(f"{BASE_URL}/readyz", timeout=3.0)
        return res.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _check_service_ready(),
        reason=f"ami-rag API server is not accessible or not ready at {BASE_URL}",
    ),
]


def _build_params():
    cases = _load_cases()
    params = []
    for c in cases:
        marks = []
        if c.get("xfail"):
            marks.append(pytest.mark.xfail(reason=c.get("xfail_reason") or "Expected low recall"))
        params.append(pytest.param(c, id=f"{c['modality']}-{c['id']}", marks=marks))
    return params


def _check_hit(case: dict, documents: list[dict]) -> tuple[bool, int]:
    for rank, doc in enumerate(documents, start=1):
        if case["modality"] == "text":
            doc_id = (doc.get("doc") or {}).get("document_id")
            if doc_id == case["expected_doc_id"]:
                return True, rank
        else:
            ref_id = doc.get("reference_id")
            if ref_id == case["expected_chunk_id"]:
                return True, rank
    return False, -1


def run_case(case: dict, top_k: int = 5) -> dict:
    """Run one case against the live API and report the outcome.

    Shared by the pytest tests below and by `scripts/run_fusion_bench.py`, so a
    strategy comparison and the official assertions score cases identically.
    """
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "messages": [{"role": "user", "content": case["query"]}],
        "top_k": top_k,
    }
    started = time.perf_counter()
    with httpx.Client(timeout=TIMEOUT) as client:
        res = client.post(f"{BASE_URL}/v2/rag/", json=payload, headers=headers)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if res.status_code != 200:
        return {
            "id": case["id"],
            "modality": case["modality"],
            "status": "error",
            "hit": False,
            "rank": -1,
            "latency_ms": elapsed_ms,
            "detail": res.text[:200],
        }

    body = res.json()
    documents = body.get("documents", [])
    hit, rank = _check_hit(case, documents)
    return {
        "id": case["id"],
        "modality": case["modality"],
        "status": "hit" if hit else "miss",
        "hit": hit,
        "rank": rank,
        "latency_ms": elapsed_ms,
        "server_latency_ms": (body.get("meta") or {}).get("latency_ms"),
        "fusion": (body.get("meta") or {}).get("fusion"),
        "documents": documents,
    }


@pytest.mark.parametrize("case", _build_params())
def test_retrieval_top5_recall(case: dict):
    result = run_case(case, top_k=5)
    documents = result["documents"]
    details = [
        {
            "rank": idx + 1,
            "modality": d.get("modality"),
            "reference_id": d.get("reference_id"),
            "doc_id": (d.get("doc") or {}).get("document_id"),
            "score": round(d.get("score") or 0.0, 4),
        }
        for idx, d in enumerate(documents)
    ]

    expected_target = (
        f"doc_id={case['expected_doc_id']}"
        if case["modality"] == "text"
        else f"chunk_id={case['expected_chunk_id']}"
    )

    assert result["status"] != "error", f"API error: {result['detail']}"
    assert len(documents) <= 5, f"Expected at most 5 documents, got {len(documents)}"

    assert result["hit"], (
        f"Case '{case['id']}' ({case['modality']}) MISS in top 5.\n"
        f"Query: {case['query']}\n"
        f"Expected: {expected_target}\n"
        f"Returned ({len(documents)} docs): {details}"
    )


def test_retrieval_summary_report():
    """Summary test collecting Hit@5 and MRR stats across all modalities."""
    cases = _load_cases()
    results = {
        "text": {"hit": 0, "total": 0, "rr_sum": 0.0},
        "table": {"hit": 0, "total": 0, "rr_sum": 0.0},
        "image": {"hit": 0, "total": 0, "rr_sum": 0.0},
    }

    for case in cases:
        modality = case["modality"]
        results[modality]["total"] += 1
        result = run_case(case, top_k=5)
        if result["hit"]:
            results[modality]["hit"] += 1
            results[modality]["rr_sum"] += 1.0 / result["rank"]

    total_hits = sum(s["hit"] for s in results.values())
    total_cases = sum(s["total"] for s in results.values())
    total_rr = sum(s["rr_sum"] for s in results.values())

    print("\n" + "=" * 65)
    print("           RETRIEVAL RECALL@5 & MRR SUMMARY REPORT")
    print("=" * 65)
    print(f"{'MODALITY':<10} | {'HIT@5':<12} | {'HIT RATE':<10} | {'MRR':<8}")
    print("-" * 65)
    for modality, stats in results.items():
        hit = stats["hit"]
        total = stats["total"]
        pct = (hit / total * 100) if total > 0 else 0.0
        mrr = (stats["rr_sum"] / total) if total > 0 else 0.0
        print(f"{modality.upper():<10} | {hit}/{total:<10} | {pct:6.1f}%    | {mrr:.4f}")
    print("-" * 65)
    overall_pct = (total_hits / total_cases * 100) if total_cases > 0 else 0.0
    overall_mrr = (total_rr / total_cases) if total_cases > 0 else 0.0
    print(f"{'OVERALL':<10} | {total_hits}/{total_cases:<10} | {overall_pct:6.1f}%    | {overall_mrr:.4f}")
    print("=" * 65 + "\n")

    assert results["text"]["hit"] >= 8, "Text retrieval should hit at least 8/10"
    assert results["table"]["hit"] >= 3, "Table retrieval should hit at least 3/9"
    assert results["image"]["hit"] >= 1, "Image retrieval should hit at least 1/9 (baseline raw)"
