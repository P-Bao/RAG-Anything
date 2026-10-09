"""Dry-run table_repair trên toàn bộ bảng của Sổ tay sinh viên (SBV).

Chạy TRONG CONTAINER (cần mineru models + MinIO + pypdfium2):
    docker cp scripts/run_table_repair_dryrun.py ami-rag-worker:/tmp/
    docker exec ami-rag-worker python /tmp/run_table_repair_dryrun.py

Chỉ đo, không ghi index/Qdrant. Xuất /app/output/table_repair_dryrun.json
(+ 5 ảnh bbox để kiểm tra bằng mắt).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import cv2
import numpy as np
import pypdfium2 as pdfium
from minio import Minio

sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages")

from ami_rag.core.table_repair import TableRepairer

BUCKET = "ami-data-documents"
PDF_KEY = "documents/VAN_PHONG/32b8a415227d4405a7535120dc2e951b_So sinh vien ban in.pdf"
CL_KEY = "rag-assets/6a8ee761694f1f325e516aa9/content_list.json"
# sha256 (16 hex đầu) + size của bản PDF đã parse trước đó (AGENTS.md handoff)
EXPECTED_SHA16 = "5998716b9436a770"
EXPECTED_SIZE = 18906422
OUT_JSON = "/tmp/table_repair_dryrun.json"
N_VISUAL = 5


def main() -> None:
    c = Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=False,
    )

    # Constraint (c): PDF trong kho trùng sha với bản đã parse
    stat = c.stat_object(BUCKET, PDF_KEY)
    data = c.get_object(BUCKET, PDF_KEY).read()
    sha = hashlib.sha256(data).hexdigest()
    assert stat.size == EXPECTED_SIZE, f"size lệch: {stat.size} != {EXPECTED_SIZE}"
    assert sha[:16] == EXPECTED_SHA16, f"sha lệch: {sha[:16]} != {EXPECTED_SHA16}"
    print(f"[sha] OK: {stat.size} bytes, sha256[:16]={sha[:16]}")

    cl = json.loads(c.get_object(BUCKET, CL_KEY).read())
    tables = [it for it in cl if it.get("type") == "table" and it.get("table_body")]
    with_asset = [it for it in tables if it.get("asset_key")]
    pages = sorted({it["page_idx"] for it in tables})
    print(
        f"[data] {len(tables)} bảng (có table_body), {len(with_asset)} có ảnh crop, "
        f"{len(pages)} trang"
    )

    pdf = pdfium.PdfDocument(data)
    repairer = TableRepairer(minio_client=c, bucket=BUCKET)

    results = []
    for k, entry in enumerate(with_asset):
        page_idx = entry["page_idx"]
        res = repairer.repair_table(page=pdf[page_idx], table_entry=entry)
        results.append(res)
        if (k + 1) % 50 == 0:
            print(f"  ... {k + 1}/{len(with_asset)}")

    # Tổng hợp
    n_repaired = sum(1 for r in results if r.status == "repaired")
    skipped: dict[str, int] = {}
    for r in results:
        if r.status == "skipped" and r.reason:
            skipped[r.reason] = skipped.get(r.reason, 0) + 1

    # Tỉ lệ ô khớp + tương đồng (trọng số theo số ô của từng bảng)
    tot_cells = sum(r.n_cells for r in results if r.status == "repaired")
    tot_filled = sum(r.n_filled for r in results if r.status == "repaired")
    tot_replaced = sum(r.n_replaced for r in results if r.status == "repaired")
    tot_kept = sum(r.n_kept_ocr for r in results if r.status == "repaired")
    tot_short = sum(r.n_short_cells for r in results if r.status == "repaired")
    tot_long = sum(r.n_long_cells for r in results if r.status == "repaired")
    w_short = sum(
        (r.short_sim or 0) * r.n_short_cells for r in results if r.status == "repaired"
    )
    w_long = sum(
        (r.long_sim or 0) * r.n_long_cells for r in results if r.status == "repaired"
    )
    avg_short = w_short / tot_short if tot_short else None
    avg_long = w_long / tot_long if tot_long else None

    print("\n===== BẢNG SỐ LIỆU (dry-run, không ghi) =====")
    print(f"Tổng bảng có ảnh crop: {len(with_asset)} | vá được: {n_repaired} "
          f"({n_repaired / max(len(with_asset), 1):.0%})")
    print(f"Bị bỏ: {len(with_asset) - n_repaired} theo lý do: {json.dumps(skipped, ensure_ascii=False)}")
    print(f"Ô (bảng vá được): {tot_cells} | có chữ text layer: {tot_filled} | "
          f"đã thay bằng text layer: {tot_replaced} ({tot_replaced / max(tot_filled, 1):.0%} "
          f"của ô có chữ, {tot_replaced / max(tot_cells, 1):.0%} của mọi ô)")
    print(f"Giữ chữ OCR (bbox lệch / không có chữ): {tot_kept}")
    print(f"Ô ngắn (<12 ký tự): {tot_short} | tương đồng TB: "
          f"{avg_short:.3f}" if avg_short is not None else "Ô ngắn: 0")
    print(f"Ô dài: {tot_long} | tương đồng TB: "
          f"{avg_long:.3f}" if avg_long is not None else "Ô dài: 0")

    # Phân bố tương đồng ô dài để chỉnh ngưỡng 0.6
    long_below = [
        (r.page_idx, r.long_sim)
        for r in results
        if r.status == "repaired" and r.long_sim is not None and r.long_sim < 0.6
    ]
    if long_below:
        print(f"Bảng có tương đồng ô dài < 0.6: {len(long_below)} "
              f"(page_idx: {[p for p, _ in long_below][:10]})")

    # Lưu kết quả (bỏ repaired_html để file nhỏ)
    out = {
        "doc": "SBV (Sổ tay sinh viên 2026)",
        "sha256_16": sha[:16],
        "total_tables_with_asset": len(with_asset),
        "total_tables": len(tables),
        "pages_with_tables": len(pages),
        "repaired": n_repaired,
        "skipped_by_reason": skipped,
        "cells_total": tot_cells,
        "cells_filled": tot_filled,
        "cells_replaced": tot_replaced,
        "cells_kept_ocr": tot_kept,
        "short_cells": tot_short,
        "long_cells": tot_long,
        "avg_sim_short": round(avg_short, 4) if avg_short is not None else None,
        "avg_sim_long": round(avg_long, 4) if avg_long is not None else None,
        "per_table": [
            {
                "page_idx": r.page_idx,
                "status": r.status,
                "reason": r.reason,
                "n_cells": r.n_cells,
                "n_html_td": r.n_html_td,
                "n_filled": r.n_filled,
                "n_replaced": r.n_replaced,
                "n_kept_ocr": r.n_kept_ocr,
                "scale": list(r.scale) if r.scale else None,
                "short_sim": r.short_sim,
                "long_sim": r.long_sim,
            }
            for r in results
        ],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[luu] {OUT_JSON}")

    # Ảnh bbox: vẽ ô lên trang render (5 bảng vá được đầu tiên, mỗi bảng đúng ô của nó)
    from ami_rag.core.table_repair import map_cells_to_page, structure_cells

    visual = []
    seen_pages: set[int] = set()
    for k, r in enumerate(results):
        if r.status != "repaired" or r.n_cells == 0 or r.page_idx in seen_pages:
            continue
        seen_pages.add(r.page_idx)
        visual.append((k, r))
        if len(visual) >= N_VISUAL:
            break
    render_scale = 2.0
    for vi, (k, r) in enumerate(visual, start=1):
        entry = with_asset[k]
        page = pdf[r.page_idx]
        bitmap = page.render(scale=render_scale)
        pil = bitmap.to_pil()
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        cells_px, _ = structure_cells(repairer.structurer, repairer._load_crop(entry["asset_key"]))
        cells, _scale = map_cells_to_page(
            cells_px, entry["bbox"], page.get_size(), repairer._load_crop(entry["asset_key"])
        )
        for cell in cells:
            x0 = float(cell[:, 0].min()) * render_scale
            y0 = float(cell[:, 1].min()) * render_scale
            x1 = float(cell[:, 0].max()) * render_scale
            y1 = float(cell[:, 1].max()) * render_scale
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 0), 2)
        out_img = f"/tmp/table_repair_bbox_{vi}_page{r.page_idx + 1}.jpg"
        cv2.imwrite(out_img, img)
        print(f"[anh] {out_img} ({len(cells)} ô)")


if __name__ == "__main__":
    main()
