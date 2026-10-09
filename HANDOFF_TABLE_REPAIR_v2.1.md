# Handoff — Sửa chữ mất dấu bảng PDF (table_repair v2.1)

## Mục tiêu
Sửa chữ mất dấu tiếng Việt trong ô bảng do MinerU OCR (model `ch_PP-OCRv6` chỉ biết Latin) bằng cách splice **text layer PDF** (đúng dấu) vào HTML bảng OCR. Lưu kết quả ở trường riêng `repaired_table_body`, giữ nguyên `table_body` gốc.

Vòng này (theo chỉ thị user): đạt **độ chính xác ô thay ≥95%** (chấp nhận thay ít hơn), **ablation từng bước** (1a–1d + khắc phục), đo bằng **mẫu mới có nhãn** (CI 95%), tính **độ phủ thật** (bảng bỏ + chỉ số tổng), các việc nhỏ (env, ảnh, tất định). KHÔNG ghi index/re-embed.

## Kết quả cốt lõi

### Ablation (độ chính xác ô thay, gán nhãn bằng mắt, đo từng bước)

| stage | thay đổi | sạch | đúng+rác | sai | delta |
|---|---|---|---|---|---|
| S0 | baseline v2.0 (mẫu cũ n=99) | 43% | 44% | 13% | — |
| S1 | (a) cửa sổ khớp tốt nhất n±1 trên token | 85% (40) | 2 | 4 | +42 |
| S2 | (b) gán theo TỪ (max-overlap, không cắt token) | 88% (40) | 2 | 3 | +3 |
| S3 | (c) snap cạnh ô về trung vị lưới | 88% (40) | 3 | 2 | +0 |
| S4 | (d) ô ngắn không hạ ngưỡng (0.85) | 85% (150) | 10 | **13 (8.7%)** | −3 |
| S5 | (e) bounded chỉ fallback (không từ/chữ dọc) | 72% (40) | 4 | 7 | −12 |
| S6 | (f) ngưỡng phẳng 0.85, bỏ relax mọi ô | 90% (150) | 11 | 4 (2.7%) | +18 |
| S7 | (g) không tách trước `_-` + (h) chặn fragment | **99% (150)** | 2 | **0** | +9 |

**Đọc nhanh:**
- (a) cửa sổ: bước lớn nhất (+42đ).
- (c) snap lưới: +0đ đo được — giữ cho bbox sạch, không claim độ chính xác.
- (d) một mình không đủ: 13 sai là ô dài lách qua relaxed 0.45 + bounded garble.
- (e) một mình TỆ hơn: ép về nguồn words nhưng relaxed vẫn cho qua bản dở.
- (f) ngưỡng phẳng 0.85: **lá chắn chính** (sai 8.7%→2.7%).
- (g)+(h): nhắm 2 lỗi cuối: email bị xé `_` (sim 0.95!) + thay bằng fragment (`EMAIL`→`MAIL`).

### Mẫu mới (mục 2) — **TIÊU CHÍ ĐẠT**
- Mẫu S7: 150 ô thay + 60 ô giữ, seed 404, loại 529 ô đã gán nhãn mọi vòng + 146 ô mẫu digit-gate, phân tầng tercile sim × trang.
- **Sạch 148/150 = 99% (CI95 95–100%)** ✓ ≥85%
- **Sai + đổi số = 0/150 = 0%** ✓ ≤3% (reweight quần thể: 98.8%/0%)
- Ô giữ: 56 an toàn / 4 bỏ nhầm (kept_ocr_empty layer sạch: '2','4','HK2','Giáo dục thể chất 2') — đúng lớp đã biết.
- Đổi số: 0 trên 210 ô — digit gate + luật fragment giữ chân.

### Độ phủ thật (mục 3)
- **145 bảng bỏ = 10 042 ô** (median 50, max 381) vs bảng vá (median 20, max 138) → đúng, bảng bỏ là lớn nhất. 145/145 có chunk Qdrant (375/375 bảng có chunk, doc 397 chunk bảng).
- Ước lượng ô hỏng bảng bỏ (substring bỏ dấu): **3568/7621 không rỗng (46.8%)**.
- **Chỉ số tổng: 1444 / 5583 = 25.9%** ô hỏng sửa sạch (num = 151 restored + 1310 thay × 98.8%). Trong 227 bảng vá: **72%** ô hỏng được sửa.
- **Đề xuất (chưa làm)**: cứu bảng bỏ bằng khớp mờ *tập từ* trong vùng — bỏ per-tr gate, gán từ theo overlap + word-set overlap với OCR từng ô.

### Việc nhỏ (mục 4)
- **Env thread**: `MINERU_INTRA/INTER_OP_NUM_THREADS=1` chỉ quanh tạo session, khôi phục sau (test chạy cả đường lỗi). Parse MinerU = subprocess CLI → không ảnh hưởng. S7: ~35s/375 bảng.
- **Ảnh cho bạn xem** (tôi không xem được): 5 bbox `/tmp/opencode/bbox/table_repair_bbox_v4_{1..5}_page{19,41,90,38,29}.jpg`; 5 ví dụ ô sai/đúng+rác baseline: `/tmp/opencode/bbox/table_repair_cell_ex_{1..5}_p{115,50,38,99,113}_c{59,13,2,8,0}.jpg`.
- **Tất định**: S7 chạy 2 lần → JSON giống hệt.

## Phát hiện/sửa thêm (nói đủ)

1. **Đính chính**: "52% already_correct bị bỏ sót mất dấu" **sai** — đa số đúng thật (email/số/tiếng Anh). Chỉ **~151 ô** mất dấu thuần, đã vá qua `restore_diacritics`.
2. Bug cache `id(page)` tái sử dụng → key `page_idx`.
3. Bug key mẫu `page|cell` va chạm giữa bảng cùng trang → `page|bảng|ô`.
4. Email/hyphen bị tách từ (`_-` thiếu) → hỏng dữ liệu sim 0.95.
5. Thay bằng fragment (`EMAIL`→`MAIL`, `Plus)`→`Plu`) → luật chặn substring.
6. Script eval bị ghi đè mất (untracked) — đã viết lại, backup `/tmp/opencode/backup_run_table_repair_eval.py`. **Đề nghị `git add scripts/`**.

## Files & Artifacts

### Code chính
- `ami_rag/core/table_repair.py` — module v2.1 (`MODULE_VERSION="table_repair/2.1"`), flags 9 bước, defaults = S7 config.
- `tests/test_table_repair.py` — 32 test (host 27 pass/5 skip, container 32 pass).

### Eval & Labels
- `scripts/run_table_repair_eval.py` — 8-stage eval, modes `run`/`report`/`final`.
- `tests/table_repair_ablation.json` — ablation table + item3 metrics (S7).
- `tests/table_repair_samples_final.json` — mẫu S1,S2,S3,S5 (40 mỗi) + S7 (150+60).
- `tests/table_repair_labels.json` — 580 nhãn (S1–S3,S4R/K,S5,S6R/K,FINAL_R/K).

### Images (host)
- `/tmp/opencode/bbox/table_repair_bbox_v4_*.jpg` (5 bbox S7)
- `/tmp/opencode/bbox/table_repair_cell_ex_*.jpg` (5 ví dụ S0)

### Container
- `/opt/venv/lib/python3.12/site-packages/ami_rag/core/table_repair.py` (deployed)
- `/tmp/table_repair_labels.json`, `/tmp/table_repair_samples.json`, `/tmp/table_repair_ablation.json`
- `/tmp/eval_v6.log` (full log 361 lines)

## Ràng buộc còn hiệu lực
- Chưa ghi MinIO/Qdrant, chưa re-embed.
- Canary fusion trước khi re-embed.
- 4 tài liệu bảng còn lại cần PDF từ backend crawl.
- `rerank_calibration.json` chờ refit.
- `tests/proposed_test_cases.json`, `tests/text_only_ab.json`, `tests/pool_recall_probe.json` chờ review.
- 375 bảng vs 397 chunk (22 không `table_body`).

## Test Suite
- 606 passed, 9 skipped, 9 xfailed, ruff clean (ami_rag + tests/ami_service + scripts).

---

**Trạng thái**: S7 ĐẠT tiêu chí, tất định, đầy đủ artifact. Sẵn sàng cho canary fusion → re-embed → reindex khi bạn duyệt.