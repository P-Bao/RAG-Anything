"""Đo table_repair v2.1: ablation từng bước sửa bbox chảy + mẫu mới + độ phủ thật.

Chạy TRONG CONTAINER:
    docker cp ami_rag/core/table_repair.py ami-rag-worker:/opt/venv/lib/python3.12/site-packages/ami_rag/core/
    docker cp scripts/run_table_repair_eval.py ami-rag-worker:/tmp/
    docker cp tests/table_repair_sample.json ami-rag-worker:/tmp/table_repair_sample_old.json
    docker exec ami-rag-worker python /tmp/run_table_repair_eval.py            # mode=run
    docker exec ami-rag-worker python /tmp/run_table_repair_eval.py report     # sau khi gán nhãn

mode=run: chạy 5 stage ablation (S0=baseline v2.0, S1=+cửa sổ, S2=+gán từ,
S3=+snap lưới, S4=+ô ngắn strict), dump mẫu để gán nhãn + chỉ số mục 3 + ảnh.
mode=report: đọc nhãn (labels_file), tính ablation + tiêu chí mẫu mới + CI.
mode=S4: chỉ chạy S4 (kiểm tra tất định: chạy 2 lần, diff /tmp/table_repair_s4_check.json).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

import cv2
import numpy as np
import pypdfium2 as pdfium
from minio import Minio

sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages")

from ami_rag.core.table_repair import (
    CAT_ALREADY_CORRECT,
    CAT_EMPTY_BOTH,
    CAT_KEPT_DIGITS,
    CAT_KEPT_LOW_SIM,
    CAT_KEPT_NO_TEXT,
    CAT_KEPT_NUMERIC,
    CAT_KEPT_OCR_EMPTY,
    CAT_REPLACED,
    RENDER_SCALE,
    TableRepairer,
    extract_ocr_cells,
    html_td_per_tr,
    map_cells_to_page,
    normalize_text,
    snap_cells_to_grid,
    structure_cells,
    structure_td_per_tr,
)

BUCKET = "ami-data-documents"
PDF_KEY = "documents/VAN_PHONG/32b8a415227d4405a7535120dc2e951b_So sinh vien ban in.pdf"
CL_KEY = "rag-assets/6a8ee761694f1f325e516aa9/content_list.json"
DOC_ID = "6a8ee761694f1f325e516aa9"
EXPECTED_SHA16 = "5998716b9436a770"
OUT_JSON = "/tmp/table_repair_ablation.json"
OUT_SAMPLE = "/tmp/table_repair_samples.json"
LABELS_FILE = "/tmp/table_repair_labels.json"
OLD_SAMPLE_FILE = "/tmp/table_repair_sample_old.json"  # mẫu cũ (đã dùng chỉnh digit gate)

N_SAMPLE_STAGE = 40       # S1/S2/S3: mỗi stage 40 ô thay để gán nhãn
N_SAMPLE_FINAL_REPLACED = 150
N_SAMPLE_FINAL_KEPT = 60
SEED_STAGE = 101
SEED_FINAL = 202
SEED_FINAL2 = 303
SEED_VISUAL = 303
N_VISUAL = 5

STAGES = [
    ("S0", {"use_window": False, "word_assign": False, "grid_snap": False, "strict_short": False, "restore_diacritics": False, "restrict_bounded": False, "flat_threshold": False, "no_split_extra": False, "reject_fragment": False}),
    ("S1", {"use_window": True, "word_assign": False, "grid_snap": False, "strict_short": False, "restore_diacritics": True, "restrict_bounded": False, "flat_threshold": False, "no_split_extra": False, "reject_fragment": False}),
    ("S2", {"use_window": True, "word_assign": True, "grid_snap": False, "strict_short": False, "restore_diacritics": True, "restrict_bounded": False, "flat_threshold": False, "no_split_extra": False, "reject_fragment": False}),
    ("S3", {"use_window": True, "word_assign": True, "grid_snap": True, "strict_short": False, "restore_diacritics": True, "restrict_bounded": False, "flat_threshold": False, "no_split_extra": False, "reject_fragment": False}),
    ("S4", {"use_window": True, "word_assign": True, "grid_snap": True, "strict_short": True, "restore_diacritics": True, "restrict_bounded": False, "flat_threshold": False, "no_split_extra": False, "reject_fragment": False}),
    ("S5", {"use_window": True, "word_assign": True, "grid_snap": True, "strict_short": True, "restore_diacritics": True, "restrict_bounded": True, "flat_threshold": False, "no_split_extra": False, "reject_fragment": False}),
    ("S6", {"use_window": True, "word_assign": True, "grid_snap": True, "strict_short": True, "restore_diacritics": True, "restrict_bounded": True, "flat_threshold": True, "no_split_extra": False, "reject_fragment": False}),
    ("S7", {"use_window": True, "word_assign": True, "grid_snap": True, "strict_short": True, "restore_diacritics": True, "restrict_bounded": True, "flat_threshold": True, "no_split_extra": True, "reject_fragment": True}),
]
FINAL_STAGE = "S7"
SEED_FINAL3 = 404

ALL_CATEGORIES = [
    CAT_ALREADY_CORRECT,
    CAT_REPLACED,
    CAT_KEPT_NUMERIC,
    CAT_KEPT_DIGITS,
    CAT_KEPT_LOW_SIM,
    CAT_KEPT_NO_TEXT,
    CAT_KEPT_OCR_EMPTY,
    CAT_EMPTY_BOTH,
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Khoảng tin cậy 95% (Wilson) cho tỉ lệ k/n -> (low, p, high)."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), p, min(1.0, (c + h) / d)


def cell_items(results):
    """Gom (replaced_items, kept_items) từ kết quả chạy — kèm ngữ cảnh.

    key = page|bảng|ô — bắt buộc có chỉ số bảng: cùng trang có thể có nhiều
    bảng trùng số ô (đã vấp: page 38 có 2 bảng cùng cell 19).
    """
    replaced, kept = [], []
    for r in results:
        if r.status != "repaired":
            continue
        tbl_key = getattr(r, "entry_key", 0)
        for cidx, cell in enumerate(r.cells):
            ctx = [
                r.cells[j].final_text
                for j in range(max(0, cidx - 2), min(len(r.cells), cidx + 3))
                if j != cidx
            ]
            item = {
                "key": f"{r.page_idx + 1}|{tbl_key}|{cidx}",
                "page": r.page_idx + 1,
                "cell_index": cidx,
                "ocr": cell.ocr_text,
                "match": cell.match_text,
                "src": cell.match_source,
                "sim": cell.similarity,
                "category": cell.category,
                "context": ctx,
                "layer": cell.layer_text,
                "words": cell.words_text,
            }
            if cell.category == CAT_REPLACED:
                replaced.append(item)
            elif cell.category in (
                CAT_KEPT_LOW_SIM,
                CAT_KEPT_NO_TEXT,
                CAT_KEPT_OCR_EMPTY,
                CAT_KEPT_NUMERIC,
                CAT_KEPT_DIGITS,
            ):
                kept.append(item)
    return replaced, kept


def stratified_page_sim(items, n, rng):
    """Phân tầng theo tercile sim, trong mỗi tercile round-robin theo trang."""
    if not items:
        return []
    items = sorted(items, key=lambda x: (x["sim"] if x["sim"] is not None else 0.0))
    t1 = items[: len(items) // 3]
    t2 = items[len(items) // 3 : 2 * len(items) // 3]
    t3 = items[2 * len(items) // 3 :]
    out = []
    quota = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    for grp, q in zip((t1, t2, t3), quota):
        by_page = defaultdict(list)
        for it in grp:
            by_page[it["page"]].append(it)
        pages = list(by_page)
        rng.shuffle(pages)
        picked, i = 0, 0
        while picked < q and any(by_page.values()):
            cands = by_page[pages[i % len(pages)]]
            if cands:
                out.append(cands.pop(0))
                picked += 1
            i += 1
    return out[:n]


def run_stage(pdf, tables, minio, flags):
    repairer = TableRepairer(minio_client=minio, bucket=BUCKET, **flags)
    results = []
    for k, entry in tables:
        res = repairer.repair_table(page=pdf[entry["page_idx"]], table_entry=entry)
        res.entry_key = k
        results.append(res)
    return repairer, results


def counts_of(results):
    counts = {cat: 0 for cat in ALL_CATEGORIES}
    for r in results:
        if r.status != "repaired":
            continue
        for cat, n_cat in r.count_by_category().items():
            counts[cat] = counts.get(cat, 0) + n_cat
    return counts


def token_cov(results):
    pairs = [
        (r.token_coverage_before, r.token_coverage_after)
        for r in results
        if r.status == "repaired"
        and r.token_coverage_before is not None
        and r.token_coverage_after is not None
    ]
    if not pairs:
        return None, None
    return (
        statistics.mean(p[0] for p in pairs),
        statistics.mean(p[1] for p in pairs),
    )


def draw_bbox_images(pdf, tables, repairer, results, out_prefix, rng, n=N_VISUAL):
    """5 ảnh bbox (ô đã snap) ở trang ngẫu nhiên trong các bảng đã vá."""
    repaired = [r for r in results if r.status == "repaired" and r.cells]
    picks = rng.sample(repaired, min(n, len(repaired)))
    entries = {k: it for k, it in tables}
    for vi, r in enumerate(picks, start=1):
        entry = entries[r.entry_key]
        page = pdf[entry["page_idx"]]
        bmp = page.render(scale=2.0)
        img = cv2.cvtColor(np.asarray(bmp.to_pil()), cv2.COLOR_RGB2BGR)
        crop = repairer._render_crop(page, entry["bbox"], key=entry["page_idx"])
        cells_px, structures = structure_cells(repairer.structurer, crop)
        if structure_td_per_tr(structures) != html_td_per_tr(entry["table_body"]):
            continue
        cells, _ = map_cells_to_page(
            cells_px, entry["bbox"], page.get_size(), crop, render_scale=RENDER_SCALE
        )
        cells = snap_cells_to_grid(cells)
        for cell in cells:
            x0 = float(cell[:, 0].min()) * 2.0
            y0 = float(cell[:, 1].min()) * 2.0
            x1 = float(cell[:, 0].max()) * 2.0
            y1 = float(cell[:, 1].max()) * 2.0
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 0), 2)
        out_img = f"{out_prefix}_{vi}_page{entry['page_idx'] + 1}.jpg"
        cv2.imwrite(out_img, img)
        print(f"[anh-bbox] {out_img} ({len(cells)} o, page {entry['page_idx'] + 1})")


def draw_cell_examples(pdf, tables, minio, items, out_prefix, rng, n=5):
    """5 ảnh ví dụ ô sai / đúng+rác từ S0 (baseline) để người dùng tự xem."""
    wrong = [it for it in items if (it["sim"] or 0) < 0.68]
    garbage = [
        it
        for it in items
        if it["sim"] is not None
        and len(str(it["match"]).split()) > len(str(it["ocr"]).split()) + 1
    ]
    picks = []
    for pool in (garbage, wrong):
        rng.shuffle(pool)
        picks.extend(pool[: n // 2 + 1])
    picks = picks[:n]
    repairer = TableRepairer(
        minio_client=minio,
        bucket=BUCKET,
        use_window=False,
        word_assign=False,
        grid_snap=False,
        strict_short=False,
        restore_diacritics=False,
    )
    tbl_by_page = defaultdict(list)
    for k, it in tables:
        tbl_by_page[it["page_idx"] + 1].append((k, it))
    for idx, item in enumerate(picks, start=1):
        page_no = item["page"]
        target = None
        for k, entry in tbl_by_page.get(page_no, []):
            ocr_cells = extract_ocr_cells(entry.get("table_body", ""))
            if item["cell_index"] < len(ocr_cells) and ocr_cells[item["cell_index"]] == item["ocr"]:
                target = (k, entry)
                break
        if target is None:
            continue
        _k, entry = target
        page = pdf[entry["page_idx"]]
        crop = repairer._render_crop(page, entry["bbox"], key=entry["page_idx"])
        cells_px, _structures = structure_cells(repairer.structurer, crop)
        cells, _ = map_cells_to_page(
            cells_px, entry["bbox"], page.get_size(), crop, render_scale=RENDER_SCALE
        )
        cell = cells[item["cell_index"]]
        x0 = float(cell[:, 0].min())
        y0 = float(cell[:, 1].min())
        x1 = float(cell[:, 0].max())
        y1 = float(cell[:, 1].max())
        bmp = page.render(scale=4.0)
        img = cv2.cvtColor(np.asarray(bmp.to_pil()), cv2.COLOR_RGB2BGR)
        for c2 in cells:
            a0, b0 = float(c2[:, 0].min()) * 4.0, float(c2[:, 1].min()) * 4.0
            a1, b1 = float(c2[:, 0].max()) * 4.0, float(c2[:, 1].max()) * 4.0
            cv2.rectangle(img, (int(a0), int(b0)), (int(a1), int(b1)), (160, 160, 160), 1)
        cv2.rectangle(img, (int(x0 * 4), int(y0 * 4)), (int(x1 * 4), int(y1 * 4)), (0, 0, 255), 4)
        m = 120
        cy0 = max(int(y0 * 4) - m, 0)
        cy1 = min(int(y1 * 4) + m, img.shape[0])
        cx0 = max(int(x0 * 4) - m, 0)
        cx1 = min(int(x1 * 4) + m, img.shape[1])
        out_img = f"{out_prefix}_{idx}_p{page_no}_c{item['cell_index']}.jpg"
        cv2.imwrite(out_img, img[cy0:cy1, cx0:cx1])
        print(
            f"[anh-vi-du] {out_img} | page {page_no} cell {item['cell_index']} | "
            f"OCR={item['ocr']!r} -> extract={item['match']!r} | sim={item['sim']}"
        )


def item3_metrics(pdf, tables, results_s4, minio):
    """Mục 3: bảng bỏ (ô, chunk Qdrant, kích thước) + nguyên liệu chỉ số tổng."""
    entries = {k: it for k, it in tables}
    dropped = [
        (r, entries[r.entry_key])
        for r in results_s4
        if r.status == "skipped" and r.reason == "row_structure_mismatch"
    ]
    repaired = [r for r in results_s4 if r.status == "repaired"]

    drop_tds = [r.n_html_td for r, _e in dropped]
    rep_tds = [r.n_html_td for r in repaired]

    # chunk Qdrant: match theo table_body
    matched_dropped = matched_all = n_chunks = None
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        from ami_rag.core.embedder import collection_name
        from ami_rag.settings import get_settings

        s = get_settings()
        col = collection_name(s.WORKSPACE, s.EMBED_MODEL, s.CHUNKER_VERSION)
        qc = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ.get("QDRANT_API_KEY") or None,
            check_compatibility=False,
        )
        chunk_bodies = []
        _offset = None
        while True:
            pts, _offset = qc.scroll(
                col,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=DOC_ID)),
                        FieldCondition(key="modality", match=MatchValue(value="table")),
                    ]
                ),
                limit=256,
                offset=_offset,
                with_payload=["table_body"],
                with_vectors=False,
            )
            chunk_bodies.extend(p.payload.get("table_body") or "" for p in pts)
            if not _offset:
                break
        n_chunks = len(chunk_bodies)
        matched_dropped = sum(
            1 for r, e in dropped if (e.get("table_body") or "") in chunk_bodies
        )
        matched_all = sum(
            1 for r in results_s4 if (entries[r.entry_key].get("table_body") or "") in chunk_bodies
        )
    except Exception as ex:  # noqa: BLE001
        print(f"[qdrant] lỗi: {ex}")

    # ước lượng ô hỏng của bảng bị bỏ: substring sau bỏ dấu trong vùng bảng
    est_broken = 0
    est_total = 0
    for r, entry in dropped:
        page = pdf[entry["page_idx"]]
        page_w, page_h = page.get_size()
        bb = entry["bbox"]
        region = (
            bb[0] * page_w / 1000.0,
            bb[1] * page_h / 1000.0,
            bb[2] * page_w / 1000.0,
            bb[3] * page_h / 1000.0,
        )
        tp = page.get_textpage()
        layer = tp.get_text_bounded(
            left=region[0], bottom=page_h - region[3], right=region[2], top=page_h - region[1]
        )
        layer_norm = normalize_text(layer or "")
        for cell in extract_ocr_cells(entry.get("table_body", "")):
            if not cell.strip():
                continue
            est_total += 1
            if normalize_text(cell) not in layer_norm:
                est_broken += 1

    n_restored = sum(
        1
        for r in repaired
        for c in r.cells
        if c.category == CAT_ALREADY_CORRECT and c.final_text != c.ocr_text
    )
    n_replaced = sum(1 for r in repaired for c in r.cells if c.category == CAT_REPLACED)
    n_kept_low = sum(1 for r in repaired for c in r.cells if c.category == CAT_KEPT_LOW_SIM)
    n_kept_empty = sum(1 for r in repaired for c in r.cells if c.category == CAT_KEPT_OCR_EMPTY)
    n_unknown = sum(
        1
        for r in repaired
        for c in r.cells
        if c.category in (CAT_KEPT_NO_TEXT, CAT_KEPT_NUMERIC, CAT_KEPT_DIGITS)
    )
    return {
        "dropped": {
            "n_tables": len(dropped),
            "cells_total": int(sum(drop_tds)),
            "td_median": statistics.median(drop_tds) if drop_tds else None,
            "td_max": max(drop_tds) if drop_tds else None,
            "qdrant_chunks_matched": matched_dropped,
            "qdrant_chunks_all_tables": matched_all,
            "qdrant_table_chunks_total": n_chunks,
            "est_broken_cells": est_broken,
            "est_broken_of_nonempty": round(est_broken / est_total, 4) if est_total else None,
            "est_nonempty_cells": est_total,
        },
        "repaired": {
            "n_tables": len(repaired),
            "td_median": statistics.median(rep_tds) if rep_tds else None,
            "td_max": max(rep_tds) if rep_tds else None,
        },
        "aggregate_input": {
            "n_restored_diacritics": n_restored,
            "n_replaced": n_replaced,
            "n_kept_low_sim": n_kept_low,
            "n_kept_ocr_empty": n_kept_empty,
            "n_unknown": n_unknown,
            "est_broken_dropped": est_broken,
        },
    }


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    rng_final = random.Random(SEED_FINAL)

    c = Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=False,
    )
    stat = c.stat_object(BUCKET, PDF_KEY)
    data = c.get_object(BUCKET, PDF_KEY).read()
    sha = hashlib.sha256(data).hexdigest()
    assert stat.size == 18906422 and sha[:16] == EXPECTED_SHA16
    cl = json.loads(c.get_object(BUCKET, CL_KEY).read())
    tables = [
        (k, it)
        for k, it in enumerate(cl)
        if it.get("type") == "table" and it.get("table_body")
    ]
    tables.sort(key=lambda kv: kv[1].get("page_idx", -1))
    pdf = pdfium.PdfDocument(data)
    print(f"[data] {len(tables)} bảng, sha OK")

    stage_sel = [FINAL_STAGE] if mode == "final" else [s for s, _f in STAGES]
    stage_results = {}
    out: dict = {"stages": {}}

    for name in stage_sel:
        flags = dict(STAGES[[s for s, _ in STAGES].index(name)][1])
        repairer, results = run_stage(pdf, tables, c, flags)
        stage_results[name] = (repairer, results)
        counts = counts_of(results)
        cb, ca = token_cov(results)
        skipped: dict[str, int] = {}
        for r in results:
            if r.status == "skipped" and r.reason:
                skipped[r.reason] = skipped.get(r.reason, 0) + 1
        out["stages"][name] = {
            "flags": flags,
            "cell_counts": counts,
            "n_repaired_tables": sum(1 for r in results if r.status == "repaired"),
            "skipped": skipped,
            "token_before": round(cb, 4) if cb is not None else None,
            "token_after": round(ca, 4) if ca is not None else None,
            "n_restored": sum(
                1
                for r in results
                if r.status == "repaired"
                for cc in r.cells
                if cc.category == CAT_ALREADY_CORRECT and cc.final_text != cc.ocr_text
            ),
        }
        st = out["stages"][name]
        print(
            f"[{name}] bảng vá {st['n_repaired_tables']} | "
            f"thay {counts[CAT_REPLACED]} | đúng-sau-bỏ-dấu {counts[CAT_ALREADY_CORRECT]} "
            f"(restored {st['n_restored']}) | giữ-low {counts[CAT_KEPT_LOW_SIM]} "
            f"| ocr-empty {counts[CAT_KEPT_OCR_EMPTY]} | token {st['token_before']}->{st['token_after']}"
        )

    if mode == "final":
        with open("/tmp/table_repair_final_check.json", "w") as f:
            json.dump(
                [
                    {
                        "page_idx": r.page_idx,
                        "status": r.status,
                        "reason": r.reason,
                        "cats": r.count_by_category(),
                    }
                    for r in stage_results[FINAL_STAGE][1]
                ],
                f,
                ensure_ascii=False,
                sort_keys=True,
            )
        print(f"[{FINAL_STAGE}] xong — diff /tmp/table_repair_final_check.json giữa 2 lần chạy")
        return

    # ===== mẫu để gán nhãn =====
    excluded: set[str] = set()
    try:
        with open(OLD_SAMPLE_FILE, encoding="utf-8") as f:
            old = json.load(f)
        for it in old.get("sample_replaced", []) + old.get("sample_kept", []):
            excluded.add(f"{it['page']}|{it['cell_index']}")
        print(f"[loai-tru] {len(excluded)} ô của mẫu cũ (đã dùng chỉnh digit gate)")
    except FileNotFoundError:
        print("[loai-tru] không tìm thấy mẫu cũ — không loại trừ")
    # loại trừ thêm mọi ô đã gán nhãn ở các vòng đo trước (đã dùng để chỉnh (e)/(f))
    prev_labeled: set[str] = set()
    try:
        with open(LABELS_FILE, encoding="utf-8") as f:
            prev = json.load(f)
        for grp in prev.values():
            prev_labeled.update(grp.keys())
        print(f"[loai-tru] {len(prev_labeled)} ô đã gán nhãn ở vòng trước")
    except FileNotFoundError:
        print("[loai-tru] chưa có nhãn cũ")
    # key của mẫu cũ là page|cell (không có bảng) — loại trừ gần đúng theo page|cell
    prev_labeled_approx = {k.split("|", 1)[0] + "|" + k.rsplit("|", 1)[1] for k in prev_labeled}

    samples: dict[str, list] = {}
    used_keys: set[str] = set()
    for name in ("S1", "S2", "S3", "S5"):
        replaced, _kept = cell_items(stage_results[name][1])
        rng = random.Random(SEED_STAGE + int(name[1]))
        pool = [it for it in replaced if it["key"] not in excluded]
        picks = stratified_page_sim(pool, N_SAMPLE_STAGE, rng)
        samples[name] = picks
        used_keys.update(it["key"] for it in picks)

    rep_final, kept_final = cell_items(stage_results[FINAL_STAGE][1])
    # tercile sim của pool final để reweight tỉ lệ quần thể
    sims_sorted = sorted((it["sim"] or 0.0) for it in rep_final)
    n_all = len(sims_sorted)
    out["final_tercile_sizes"] = [n_all // 3, n_all // 3, n_all - 2 * (n_all // 3)]
    out["final_tercile_bounds"] = {
        "t1_max": sims_sorted[n_all // 3 - 1] if n_all else None,
        "t2_max": sims_sorted[2 * n_all // 3 - 1] if n_all else None,
    }
    pool_rep = [
        it
        for it in rep_final
        if it["key"] not in excluded
        and it["key"] not in used_keys
        and it["key"] not in prev_labeled
        and f"{it['page']}|{it['cell_index']}" not in prev_labeled_approx
    ]
    rng_final = random.Random(SEED_FINAL3)
    sample_rep = stratified_page_sim(pool_rep, N_SAMPLE_FINAL_REPLACED, rng_final)
    by_cat = defaultdict(list)
    for it in kept_final:
        if it["key"] not in excluded and it["key"] not in prev_labeled:
            by_cat[it["category"]].append(it)
    sample_kept = []
    cats = sorted(by_cat, key=lambda k: -len(by_cat[k]))
    quota = max(5, N_SAMPLE_FINAL_KEPT // max(len(cats), 1))
    for ci, cat in enumerate(cats):
        rng = random.Random(SEED_FINAL3 + 1000 + ci)
        sample_kept.extend(stratified_page_sim(by_cat[cat], quota, rng))
    sample_kept = sample_kept[:N_SAMPLE_FINAL_KEPT]
    samples[FINAL_STAGE] = sample_rep + sample_kept

    with open(OUT_SAMPLE, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)

    # ===== ảnh =====
    rng_v = random.Random(SEED_VISUAL)
    draw_bbox_images(
        pdf, tables, stage_results[FINAL_STAGE][0], stage_results[FINAL_STAGE][1],
        "/tmp/table_repair_bbox_v4", rng_v,
    )
    rep0, _ = cell_items(stage_results["S0"][1])
    draw_cell_examples(pdf, tables, c, rep0, "/tmp/table_repair_cell_ex", rng_v)

    # ===== mục 3 =====
    out["item3"] = item3_metrics(pdf, tables, stage_results[FINAL_STAGE][1], c)
    print(json.dumps(out["item3"]["dropped"], ensure_ascii=False))

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[luu] {OUT_JSON}")

    # in dòng gán nhãn
    for name in ("S1", "S2", "S3", "S5"):
        print(f"\n########## NHAN {name} (thay) ##########")
        for it in samples[name]:
            print(
                f"[{name}] {it['key']} sim={it['sim']} src={it['src']} "
                f"ocr={it['ocr']!r} match={it['match']!r} | ctx={it['context'][:3]}"
            )
    print(f"\n########## NHAN {FINAL_STAGE}-THAY (mẫu cuối) ##########")
    for it in sample_rep:
        print(
            f"[FINAL_R] {it['key']} sim={it['sim']} src={it['src']} "
            f"ocr={it['ocr']!r} match={it['match']!r} | ctx={it['context'][:3]}"
        )
    print(f"\n########## NHAN {FINAL_STAGE}-GIU ##########")
    for it in sample_kept:
        print(
            f"[FINAL_K] {it['key']} cat={it['category']} sim={it['sim']} "
            f"ocr={it['ocr']!r} layer={it['layer']!r} words={it['words']!r}"
        )


def _label_counts(items, lab):
    n = k = g = w = d = 0
    for it in items:
        tag = lab.get(it["key"], "?")
        if tag == "?":
            continue
        n += 1
        k += tag == "clean"
        g += tag == "garbage"
        w += tag == "wrong"
        d += tag == "digits"
    return n, k, g, w, d


def report() -> None:
    """Đọc nhãn, tính ablation + tiêu chí + CI + chỉ số tổng."""
    with open(OUT_SAMPLE, encoding="utf-8") as f:
        samples = json.load(f)
    with open(LABELS_FILE, encoding="utf-8") as f:
        labels = json.load(f)
    with open(OUT_JSON, encoding="utf-8") as f:
        out = json.load(f)

    print("\n===== ABLATION (độ chính xác ô thay, gán nhãn bằng mắt) =====")
    print("S0 (mẫu cũ n=99, gán nhãn vòng trước): sạch ~43% | đúng+rác ~44% | sai ~13%")

    def counts_from_labels(lab: dict) -> tuple[int, int, int, int, int]:
        vals = list(lab.values())
        return (
            len(vals),
            sum(1 for v in vals if v == "clean"),
            sum(1 for v in vals if v == "garbage"),
            sum(1 for v in vals if v == "wrong"),
            sum(1 for v in vals if v == "digits"),
        )

    rows: list[tuple[str, int, int, int, int, int]] = [("S0", 99, 43, 44, 13, 0)]
    by_name: dict[str, tuple[int, int, int, int, int]] = {}
    # S4, S6: toàn bộ mẫu đã gán nhãn, đọc thẳng từ labels
    for name, grp in (("S4", "S4R"), ("S6", "S6R")):
        by_name[name] = counts_from_labels(labels.get(grp, {}))
    for name in ("S1", "S2", "S3", "S5", FINAL_STAGE):
        lab_key = "FINAL_R" if name == FINAL_STAGE else name
        if name == FINAL_STAGE:
            pool = [it for it in samples[FINAL_STAGE] if it["category"] == CAT_REPLACED]
        else:
            pool = samples[name]
        by_name[name] = _label_counts(pool, labels.get(lab_key, {}))
    for name in ("S1", "S2", "S3", "S4", "S5", "S6", FINAL_STAGE):
        if name in by_name:
            rows.append((name, *by_name[name]))

    prev_p = 0.43
    for name, n, k, g, w, d in rows:
        if n == 0:
            continue
        lo, p, hi = wilson(k, n)
        print(
            f"{name}: sạch {k}/{n} = {p:.0%} (CI95 {lo:.0%}-{hi:.0%}) | "
            f"đúng+rác {g} | sai {w} | đổi số {d}"
            + (f" | delta {p - prev_p:+.0%}" if name != "S0" else "")
        )
        if name not in ("S0",):
            prev_p = p

    # tiêu chí trên mẫu cuối (FINAL): thô + reweight quần thể
    fn, fk, _fg, fw, fd = _label_counts(
        [it for it in samples[FINAL_STAGE] if it["category"] == CAT_REPLACED],
        labels.get("FINAL_R", {}),
    )
    crit1 = fn > 0 and fk / fn >= 0.85
    crit2 = fn > 0 and (fw + fd) / fn <= 0.03
    print(
        f"TIÊU CHÍ {FINAL_STAGE} (mẫu thô): sạch≥85% -> {'DAT' if crit1 else 'KHONG DAT'} "
        f"({fk / fn:.0%}) | sai+đổi số≤3% -> {'DAT' if crit2 else 'KHONG DAT'} "
        f"({(fw + fd) / fn:.1%})"
    )
    sizes = out.get("final_tercile_sizes") or [1, 1, 1]
    bounds = out.get("final_tercile_bounds") or {}
    rep_items = [it for it in samples[FINAL_STAGE] if it["category"] == CAT_REPLACED]
    lab_final = labels.get("FINAL_R", {})

    def tercile_of(sim):
        if sim is None or sim <= (bounds.get("t1_max") or 1):
            return 0
        if sim <= (bounds.get("t2_max") or 1):
            return 1
        return 2

    tot_w = sum(sizes)
    w_clean = w_wrong = 0.0
    for t in range(3):
        tk = tw = tn = 0
        for it in rep_items:
            if tercile_of(it["sim"]) != t:
                continue
            tag = lab_final.get(it["key"], "?")
            if tag == "?":
                continue
            tn += 1
            tk += tag == "clean"
            tw += tag in ("wrong", "digits")
        if tn:
            w_clean += sizes[t] * tk / tn
            w_wrong += sizes[t] * tw / tn
    print(
        f"TIÊU CHÍ {FINAL_STAGE} (reweight quần thể, tercile {sizes}): "
        f"sạch {w_clean / tot_w:.1%} | sai+đổi số {w_wrong / tot_w:.1%} -> "
        f"{'DAT' if w_wrong / tot_w <= 0.03 else 'KHONG DAT'}"
    )

    # ô giữ: an toàn vs bỏ nhầm (S4K/S6K từ nhãn + FINAL_K từ mẫu)
    for grp, items_src in (("S4K", None), ("S6K", None), ("FINAL_K", samples[FINAL_STAGE])):
        safe = missed = other = 0
        if items_src is None:
            labk = labels.get(grp, {})
            safe = sum(1 for v in labk.values() if v == "safe")
            missed = sum(1 for v in labk.values() if v == "missed")
            other = sum(1 for v in labk.values() if v not in ("safe", "missed"))
        else:
            for it in items_src:
                if it["category"] == CAT_REPLACED:
                    continue
                tag = labels.get("FINAL_K", {}).get(it["key"], "?")
                if tag == "safe":
                    safe += 1
                elif tag == "missed":
                    missed += 1
                else:
                    other += 1
        print(f"{grp}: giữ đúng (an toàn) {safe} | bỏ nhầm {missed} | chưa nhãn/khác {other}")

    agg = out["item3"]["aggregate_input"]
    dropped = out["item3"]["dropped"]
    clean_rate = (fk / fn) if fn else 1.0
    wrong_rate = (fw + fd) / fn if fn else 0.0
    n_replaced = agg["n_replaced"]
    fixed_clean = agg["n_restored_diacritics"] + round(n_replaced * clean_rate)
    denom_all = (
        fixed_clean
        + round(n_replaced * wrong_rate)
        + agg["n_kept_low_sim"]
        + agg["n_kept_ocr_empty"]
        + agg["n_unknown"]
        + dropped["est_broken_cells"]
    )
    print("\n===== CHỈ SỐ TỔNG (mục 3, trên cấu hình cuối) =====")
    print(f"ô hỏng được sửa sạch   : {fixed_clean}")
    print(f"tổng ô hỏng (ước tính) : {denom_all} (gồm ~{dropped['est_broken_cells']} của bảng bỏ)")
    print(f"=> tỉ lệ = {fixed_clean / denom_all:.1%}")
    print("chi tiết:", json.dumps(agg, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        main()
