#!/usr/bin/env python3
"""Xuất bộ dữ liệu A/B để đánh giá tác hại của việc chèn ảnh thừa vào context.

Câu hỏi cần trả lời: với một truy vấn mà đáp án nằm hoàn toàn ở chunk text, việc
thêm 2 ảnh không liên quan vào context có làm câu trả lời tệ đi không?

Mỗi case sinh hai arm lấy từ **cùng một lần retrieve**:
  - arm có ảnh: toàn bộ document trả về
  - arm bỏ ảnh: cùng đó, chỉ gỡ các block modality=image

Thứ tự arm được đảo ngẫu nhiên theo case id và ánh xạ nằm riêng ở `meta`, nên khi
dán từng arm vào web LLM ở hai lượt khác nhau thì LLM không suy ra lượt nào có ảnh.

Script không sinh câu trả lời (stack này không có LLM). Nó chỉ xuất context +
nguồn chứng để người đánh giá chấm, và tự trích một nháp `expected_answer` từ
chunk đích để tiết kiệm thao tác, đánh dấu `needs_review` để người dùng sửa.

    python scripts/export_text_only_ab.py \
        --cases tests/retrieval_cases.json \
        --config-label "quota text=3,image=2 gate=off" \
        --out tests/text_only_ab.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

RUBRIC = [
    "Trung thực: mọi khẳng định trong câu trả lời đều bám được CONTEXT?",
    "Đầy đủ: có trả lời hết các ý trong câu hỏi không?",
    "Bịa từ ảnh: có mô tả chi tiết hình ảnh mà CONTEXT không nêu không?",
    "Thừa: có đoạn vòng, lặp lại, không cần thiết nào không?",
    "Sai nguồn: có trích dẫn sai tài liệu/trang không?",
]

JUDGE_PROMPT_TEMPLATE = """Bạn được đưa CONTEXT và một CÂU HỎI. Hãy trả lời CÂU HỎI chỉ dựa vào CONTEXT.

CONTEXT:
{context}

CÂU HỎI: {query}

Yêu cầu: trả lời ngắn gọn, không bịa ngoài CONTEXT. Nếu CONTEXT không đủ để trả lời,
hãy nói rõ thiếu gì."""

STOPWORDS = {
    "và", "của", "các", "cho", "những", "một", "trong", "với", "là", "được", "gì",
    "nào", "có", "không", "để", "như", "này", "khi", "về", "tại", "theo", "ra",
    "từ", "đã", "sẽ", "cũng", "nhưng", "mà", "ở", "thì", "bị",
}


def _find_rank(
    pool: list[dict],
    target_chunk: str | None,
    target_doc: str | None,
) -> int | None:
    """Hạng của target trong pool; case text khớp theo doc, case bảng/ảnh theo chunk."""
    for index, block in enumerate(pool, start=1):
        if target_chunk:
            if block["reference_id"] == target_chunk:
                return index
        elif block.get("document_id") == target_doc:
            return index
    return None


def _context_block(doc: dict) -> dict:
    """Rút gọn document của API thành một block context cho LLM đọc."""
    modality = doc.get("modality") or "text"
    meta = doc.get("metadata") or {}
    block: dict = {
        "modality": modality,
        "page": doc.get("page"),
        "reference_id": doc.get("reference_id"),
        "document_id": (doc.get("doc") or {}).get("document_id"),
        "score": round(float(doc.get("score") or 0.0), 4),
        "raw_score": (
            round(float(meta["raw_score"]), 4) if meta.get("raw_score") is not None else None
        ),
    }
    if modality == "image":
        block["caption"] = doc.get("caption")
        block["asset_url"] = doc.get("artifact_url")
        block["text_fallback"] = doc.get("text") or None
    elif modality == "table":
        block["table_body"] = doc.get("table_body")
        block["caption"] = doc.get("caption")
        block["text"] = doc.get("text")
    else:
        block["text"] = doc.get("text")
    return block


def _render_context(blocks: list[dict]) -> str:
    lines = []
    for index, block in enumerate(blocks, start=1):
        head = f"[{index}] modality={block['modality']}"
        if block.get("page") is not None:
            head += f" page={block['page']}"
        lines.append(head)
        if block["modality"] == "image":
            if block.get("caption"):
                lines.append(f"    mô tả ảnh: {block['caption']}")
            if block.get("text_fallback"):
                lines.append(f"    chữ trong ảnh: {block['text_fallback']}")
            if block.get("asset_url"):
                lines.append(f"    url ảnh: {block['asset_url']}")
        else:
            body = block.get("table_body") or block.get("text") or ""
            for line in str(body).splitlines():
                if line.strip():
                    lines.append(f"    {line.strip()}")
    return "\n".join(lines)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text or "")
    return [p.strip() for p in parts if len(p.strip()) >= 25]


def _draft_answer(query: str, chunk_text: str) -> str:
    """Nháp câu trả lời: các câu trong chunk đích trùng từ khoá nhất với truy vấn.

    Cố tình không tự diễn giải: chỉ ghép nguyên văn câu đã có trong context để
    không bịa nội dung. Người dùng sửa lại trước khi dùng làm chuẩn.
    """
    query_tokens = {
        token for token in re.split(r"\W+", query.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in STOPWORDS
    }
    scored = []
    for sentence in _sentences(chunk_text):
        tokens = set(re.split(r"\W+", sentence.lower(), flags=re.UNICODE))
        overlap = len(query_tokens & tokens)
        if overlap:
            scored.append((overlap, sentence))
    if not scored:
        first = _sentences(chunk_text)
        return " ".join(first[:2]) if first else ""
    scored.sort(key=lambda item: -item[0])
    return " ".join(sentence for _, sentence in scored[:3])


def _arm_order(case_id: str) -> bool:
    """True nghĩa là arm có ảnh đứng ở vị trí arm_1 (quyết định ổn định theo id)."""
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="tests/retrieval_cases.json")
    parser.add_argument("--out", default="tests/text_only_ab.json")
    parser.add_argument("--base-url", default="http://localhost:8009")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config-label", default="")
    parser.add_argument("--only-modality", default="text")
    args = parser.parse_args()

    import httpx

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = [c for c in cases if c["modality"] == args.only_modality]

    out_cases: list[dict] = []
    arm_map: dict[str, str] = {}
    totals = {
        "queries": 0,
        "with_image": 0,
        "image_slots": 0,
        "slots": 0,
        "lost_to_images": 0,
        "missing_regardless": 0,
    }

    with httpx.Client(timeout=180.0) as client:
        for case in cases:
            resp = client.post(
                f"{args.base_url.rstrip('/')}/v2/rag/",
                json={
                    "messages": [{"role": "user", "content": case["query"]}],
                    "top_k": args.top_k,
                },
            )
            resp.raise_for_status()
            documents = resp.json().get("documents", [])
            blocks = [_context_block(d) for d in documents]

            image_blocks = [b for b in blocks if b["modality"] == "image"]
            kept_blocks = [b for b in blocks if b["modality"] != "image"]

            target_doc = case.get("expected_doc_id")
            target_chunk = case.get("expected_chunk_id")

            rank_with = _find_rank(blocks, target_chunk, target_doc)
            rank_without = _find_rank(kept_blocks, target_chunk, target_doc)

            # Chunk đích đầy đủ để người đánh giá đối chiếu nháp câu trả lời.
            ground_text = ""
            for block in blocks:
                if target_chunk and block["reference_id"] == target_chunk:
                    ground_text = block.get("text") or block.get("table_body") or ""
                    break
                if not target_chunk and block.get("document_id") == target_doc:
                    ground_text = ground_text or block.get("text") or ""
            if not ground_text and kept_blocks:
                ground_text = kept_blocks[0].get("text") or ""

            with_images_first = _arm_order(case["id"])
            arms = {
                "arm_1": {
                    "blocks": blocks if with_images_first else kept_blocks,
                    "has_images": with_images_first,
                    "rendered_context": _render_context(
                        blocks if with_images_first else kept_blocks
                    ),
                },
                "arm_2": {
                    "blocks": kept_blocks if with_images_first else blocks,
                    "has_images": not with_images_first,
                    "rendered_context": _render_context(
                        kept_blocks if with_images_first else blocks
                    ),
                },
            }
            arm_map[case["id"]] = (
                "arm_1=with_images" if with_images_first else "arm_2=with_images"
            )

            totals["queries"] += 1
            totals["with_image"] += 1 if image_blocks else 0
            totals["image_slots"] += len(image_blocks)
            totals["slots"] += len(blocks)
            # Chỉ quy cho việc chèn ảnh khi bản bỏ ảnh vẫn tìm thấy target.
            lost_to_images = rank_with is None and rank_without is not None
            missing_anyway = rank_without is None
            totals["lost_to_images"] += 1 if lost_to_images else 0
            totals["missing_regardless"] += 1 if missing_anyway else 0

            out_cases.append(
                {
                    "case_id": case["id"],
                    "query": case["query"],
                    "modality_expectation": "text_only",
                    "ground_truth": {
                        "expected_doc_id": target_doc,
                        "expected_chunk_id": target_chunk,
                        "chunk_text": ground_text,
                        "expected_answer_draft": _draft_answer(case["query"], ground_text),
                        "needs_review": True,
                    },
                    "retrieval": {
                        "config": args.config_label,
                        "returned": len(blocks),
                        "image_slots": len(image_blocks),
                        "image_urls": [
                            b["asset_url"] for b in image_blocks if b.get("asset_url")
                        ],
                        "text_target_rank_with_images": rank_with,
                        "text_target_rank_images_removed": rank_without,
                        "evidence_retained": rank_with is not None,
                        "evidence_lost_to_images": lost_to_images,
                        "evidence_missing_regardless_of_images": missing_anyway,
                    },
                    "judge_prompt_template": JUDGE_PROMPT_TEMPLATE,
                    "arms": arms,
                }
            )
            print(
                f"  {case['id']:<32} img={len(image_blocks)}  "
                f"rank(with)={rank_with}  rank(without)={rank_without}",
                flush=True,
            )

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": (
                "Chấm mù chất lượng câu trả lời khi context có và không có ảnh thừa, "
                "trên các truy vấn mà đáp án nằm hoàn toàn ở chunk text."
            ),
            "config_under_test": args.config_label,
            "top_k": args.top_k,
            "how_to_use": [
                (
                    "Với mỗi case, thay {{context}} trong judge_prompt_template bằng "
                    "`arms.<arm>.rendered_context`, rồi chạy arm_1 và arm_2 ở hai "
                    "lượt riêng biệt."
                ),
                "Chạy arm_1 và arm_2 ở hai lượt riêng biệt, không nói tên arm.",
                "Chấm theo `rubric`, rồi so sánh.",
                "`meta.arm_map` cho biết arm nào thực sự có ảnh -- đọc SAU khi chấm.",
                (
                    "`ground_truth.expected_answer_draft` là nháp trích tự động từ "
                    "chunk đích, chưa kiểm chứng: sửa trước khi dùng làm chuẩn."
                ),
            ],
            "rubric": RUBRIC,
            "arm_map": arm_map,
            "totals": {
                **totals,
                "image_slot_rate": round(totals["image_slots"] / totals["slots"], 4)
                if totals["slots"]
                else 0.0,
                "query_with_image_rate": round(
                    totals["with_image"] / totals["queries"], 4
                )
                if totals["queries"]
                else 0.0,
            },
        },
        "cases": out_cases,
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {args.out}")
    print(
        f"  ảnh chèn: {totals['image_slots']}/{totals['slots']} slot "
        f"({payload['meta']['totals']['image_slot_rate']:.0%}), "
        f"{totals['with_image']}/{totals['queries']} query có ảnh, "
        f"mất chunk đúng do ảnh: {totals['lost_to_images']}, "
        f"mất dù không có ảnh: {totals['missing_regardless']}"
    )


if __name__ == "__main__":
    main()