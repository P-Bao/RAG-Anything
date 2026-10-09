<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **RAG-Anything** (4655 symbols, 8755 relationships, 266 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/RAG-Anything/context` | Codebase overview, check index freshness |
| `gitnexus://repo/RAG-Anything/clusters` | All functional areas |
| `gitnexus://repo/RAG-Anything/processes` | All execution flows |
| `gitnexus://repo/RAG-Anything/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

# Handoff — Hai nhánh retrieve + rerank + fusion (cập nhật: 08/10/2026)

### Kiến trúc (đúng sơ đồ, 2 nhánh)

```
Query
 ├─► Vector search (text|table) depth 50 ─► Rerank (kèm ảnh bảng) ─► list T
 └─► Vector search (image)       depth 15 ─► VL Rerank               ─► list I
                                                             │
                                        cổng lọc ảnh ─► fuse ─► TopK
```

Ranh giới tách ở **text-vs-image**, không phải table-vs-image: điểm rerank của text và bảng
cùng thang đo, chênh lệch thang điểm chỉ tồn tại ở ảnh — nên chỉ cần một lần sửa thang cho ảnh.
Bảng + text dùng **chung 1 lệnh rerank**; chỉ nhánh ảnh mới gọi VL reranker.

Cấu hình đo tốt nhất: `RETRIEVAL_FUSION_MODE=quota` + `QUOTA=text=3,image=2` (không ngưỡng),
`POOLS=text+table,image`, `POOL_SIZES=text=50,image=15`, `VL_POOLS=text,image`, `IMAGE_GATE=true`.
`RETRIEVAL_FUSION_MODE` vẫn để mặc định `single` — bật fusion là quyết định deploy, không tự đổi.

### Đã đo (28 case `tests/retrieval_cases.json`, top_k=5, `scripts/run_fusion_bench.py`)

Cấu hình chung: `text=50`, `image=15`, bảng rerank kèm ảnh, gate bật. Cột cuối là tham số đổi.

| cấu hình | Hit@5 | text | table | image | MRR | p50 |
|---|---|---|---|---|---|---|
| `single` (mặc định, cũ) | 19/28 | 10/10 | 5/9 | 4/9 | 0.6161 | **876 ms** |
| **`quota` không ngưỡng `text=3,image=2`** | **22/28** | 10/10 | **7/9** | 5/9 | 0.644 | 1345 ms |
| `rrf` k=60 | 22/28 | 10/10 | 7/9 | 5/9 | 0.6131 | 1353 ms |
| `calibrated` | 21/28 | 10/10 | 6/9 | 5/9 | 0.637 | 1353 ms |
| `quota` có ngưỡng `text=3@0.45,image=2@0.40` | 21/28 | 10/10 | 6/9 | 5/9 | 0.6369 | 1353 ms |
| … `calibrated`, tắt gate | 21/28 | 10/10 | 6/9 | 5/9 | 0.6339 | 1354 ms |
| … `calibrated`, `image=30` | 21/28 | 10/10 | 6/9 | 5/9 | 0.6339 | 2048 ms |
| … `calibrated`, `text=60` | 21/28 | 10/10 | 6/9 | 5/9 | 0.6458 | 1470 ms |
| … `calibrated`, `text=40` | 20/28 | 10/10 | 5/9 | 5/9 | 0.6280 | 1187 ms |
| … `calibrated`, bảng rerank **text thuần** | 18/28 | 10/10 | **3/9** | 5/9 | 0.5804 | 1249 ms |
| … `calibrated`, tắt calibration | 18/28 | 10/10 | 7/9 | 1/9 | 0.5458 | 1341 ms |

- **Tính tất định**: chạy lại 4 strategy hai lần → **Hit@5 và hit theo modality giống hệt**, MRR
  dao động ±0.01 (reranker hơi không tất định nên thứ tự *trong* top 5 đổi). Tin Hit@5; đừng tin
  chênh lệch MRR dưới ~0.01.
- **Bảng CẦN ảnh render trong rerank** — điểm đắt nhất và dễ sai nhất. Rerank bảng thuần text làm
  table tụt **6/9 → 3/9** (mất `table_01`, `table_03`). Reranker không có lỗi: trong pool 60,
  raw rerank xếp `table_01` rank 2, `table_03` rank 1. **Calibrator mới là thủ phạm** — nó được
  fit cho modality `table` trên bảng *đã rerank kèm ảnh*, nên áp cho điểm bảng rerank text thuần là
  lệch phân phối. Bù lại chỉ tốn ~72 ms. ⇒ đừng tin rằng "bỏ ảnh khỏi rerank bảng là tiết kiệm miễn phí".
- **Ép đa dạng modality thắng sort thuần**: `quota` không ngưỡng và `rrf` đều 22/28 > `calibrated`
  21/28, và cả hai cùng nhặp lại `table_09` mà `calibrated` xếp ngoài top 5. Cả hai đều **ép**
  slot cho modality yếu (quota reserve cứng; rrf với pool rời nhau + weight bằng nhau là
  round-robin). `calibrated` không sai, nhưng nó xếp hạng thuần theo điểm nên modality yếu không
  bao giờ có mặt ở cuối trang.
- **Ngưỡng quota phá chính quota**: có ngưỡng → rơi về đúng bằng `calibrated` (21/28), vì slot
  dự phòng bị chặn khi ứng viên dưới ngưỡng — mà ứng viên đó lại là ứng viên đúng.
- **Calibration là tiên quyết**: tắt → 18/28, image 5/9 → 1/9 (dù table lên 7/9).
- **Depth 50 là đủ**: `table_08` ở vector rank 44 trong pool gộp; depth 60 không thêm hit nào,
  thêm ~120 ms. `image=30` cũng không thêm hit, thêm ~700 ms.
- **Cổng lọc ảnh không đổi kết quả** với `calibrated` (21/28 bật lẫn tắt). Lý do nằm ngay trong
  định nghĩa: ảnh dưới đường cắt không thể thắng sort toàn cục, nên cổng chỉ **chặn ảnh thừa mang
  vào context**, không đổi trang kết quả. Nó *có* đổi với `quota` (ảnh bị lọc thì trả slot dự phòng
  cho pool khác). Đừng kỳ vọng nó nâng recall.
- `rrf` với pool rời nhau + weight bằng nhau là round-robin theo rank ⇒ `k` không đổi thứ tự.
- 7/9 case `xfail` vẫn hỏng vì **chunk đích không có trong pool** (`table_06` vượt depth 280;
  `image_overview_01`/`image_units_01` vector rank 22/30). Không sửa cờ `xfail` — chỉ báo delta.
- **So với thiết kế 3 pool** (text/table/image): cùng 21–22/28 nhưng **nhanh hơn 38%**
  (1345 ms vs 2178 ms) và chỉ 2 lệnh rerank thay vì 3.

### Rerank 2 lớp (prescreen) — ĐÃ GỠ, đừng khôi phục

Đã viết rồi xoá: `RETRIEVAL_FUSION_PRESCREEN_TOP_N`, `_PRESCREEN_MIN_DOCS`, `_rerank_pool()`,
`FUSION_PRESCREEN_TOTAL`. Lý do: prescreen sinh ra để xử lý pool `table` 50 doc **có ảnh render**.
Trong kiến trúc 2 nhánh, pool text+table đã gộp thành 1 lệnh rerank và pool ảnh chỉ 15 doc ⇒
không còn pool sâu nào để lọc 2 lớp. Giữ lại chỉ là code chết + 2 knob bị người khác tưởng còn tác dụng.

### Code

- `ami_rag/core/fusion.py`: `fuse`/`fuse_calibrated`/`fuse_rrf`/`fuse_quota`, `FusedHit`,
  `QuotaRule`, `parse_pool_groups` (nhóm modality `+`), `parse_pool_sizes`, `parse_quota`,
  `resolve_pool`, `pool_is_multimodal`, `rrf_score`, **`gate_image_hits`**. Thuần Python.
- `ami_rag/api/routes/rag.py`: `_run_search` dispatcher → `_run_search_single` (code cũ **giữ nguyên**)
  hoặc `_run_search_fused` (`_retrieve_pools` → `_rerank_pools` gather → `gate_image_hits` → `_fuse_pools`).
  `_retrieve_pool` lọc **OR nhiều modality** (`Filter.should`). `_score_fn` dùng chung cho gate và
  fuse để cổng lọc đúng bằng con số mà phép merge sẽ xếp hạng.
- `settings.py`: `RETRIEVAL_FUSION_MODE` (`single|calibrated|rrf|quota`), `_POOLS`, `_POOL_SIZES`,
  `_RRF_K`, `_QUOTA`, `_VL_POOLS`, `_IMAGE_GATE`; helper `resolve_fusion(settings)` trả `None` khi tắt.
- `observability.py`: `FUSION_POOL_CANDIDATES{pool}`, `FUSION_DOCS_SELECTED_TOTAL{pool}`,
  `FUSION_IMAGE_GATE_DROPPED_TOTAL`; span `rag.rerank.<pool>`.
- `docker-compose.ami.yml`: 7 biến fusion + `RERANK_CALIBRATION_ENABLED` nằm trong `environment:`
  (ghi đè `env_file`) để sweep bằng shell env mà không sửa `.env`.
- Tests: `tests/ami_service/test_fusion.py` (50), `test_api.py` +13 case;
  `FakeVectorStore` đọc `should`/`must`/`must_not` và ghi `searches`.

### Đo sâu hơn (08/10/2026) — trần recall, chi phí ảnh thừa, lỗi dữ liệu

**Trần recall (vector thuần, không rerank — `scripts/probe_pool_recall.py`):**

| modality | pool | @5 | @15 | @30 | @50 |
|---|---|---|---|---|---|
| text | 4862 | 9/10 | 10/10 | 10/10 | 10/10 |
| table | 413 | 6/9 | 6/9 | 6/9 | 7/9 |
| image | **70** | 4/9 | **7/9** | 9/9 | 9/9 |

- Collection chỉ có **70 chunk ảnh** ⇒ `image=15` đã lấy 21% cả pool. Không thể mở rộng case ảnh.
- `table_06`/`table_07` **không tới được dù probe depth 50/413**. Đã kiểm tra trực tiếp: 2 chunk đó
  tồn tại trong collection, đúng `modality=table`, đúng doc/page ⇒ **không phải lỗi nhãn, mà vector
  rank > 50/413**. `xfail_reason` chung chung "low semantic alignment" là sai nguyên nhân.
- **Recall@15 ảnh = 7/9 nhưng end-to-end hit ảnh = 5/9** ⇒ rerank *nâng* được target từ rank 6–15
  lên top-5. 2 case hỏng (`image_overview_01` rank 22, `image_units_01` rank 30) là do **rerank
  hạ chúng xuống**, không phải do thiếu pool — nên `image=30` không thêm hit là hệ quả, không phải ngẫu nhiên.

**Chi phí ảnh thừa trên 10 truy vấn thuần text (`quota text=3,image=2`, top_k=5):**

| cổng lọc ảnh | query có ảnh | slot ảnh | mất chunk đúng **do** ảnh |
|---|---|---|---|
| **bật** (đang deploy) | 1/10 | 2/50 = **4%** | 0 |
| tắt | 10/10 | 20/50 = **40%** | 0 |

- Cổng lọc **vô hiệu hoá phần lớn lực ép của quota**: quota reserve 2 slot ảnh cứng, nhưng ảnh dưới
  đường cắt bị gate loại nên slot dự phòng trả lại cho pool text ⇒ câu hỏi thuần text vẫn ra text.
- **Đính chính một kết luận sai**: `text_07_thiet_ke_game` không có target trong top-5 *kể cả khi bỏ
  hết ảnh* ⇒ không thể quy cho việc chèn ảnh. Nó là lỗi retrieval, tách riêng.
- Gate **không** nâng recall (đã biết) — nhưng nó cắt 36/50 slot ảnh thừa. Đó là lý do giữ nó.

**Lỗi dữ liệu: text trong bảng mất dấu tiếng Việt — do MinerU, KHÔNG phải do PDF hay code ta.**

Toàn bộ 413 chunk `modality=table` bị hỏng ký tự dấu ở **cả** `table_body` và `content`:
`Tiếng Anh`→`Ting Anh`, `Cấu trúc dữ liệu`→`Cu trúc d liu`, `Kinh tế cơ sở`→`Tin hc cơ s`,
`Giải tích`→`Gii tích`, `Thương mại điện tử`→`Thưong mi đin t`, `Quản trị giá`→`Quån tri giá`.

Đã truy nguyên từng tầng trên PDF `Sổ tay sinh viên 2026` (266 trang, tải từ MinIO):

| tầng | kết quả |
|---|---|
| PDF text layer (pypdfium2, độc lập) | **đúng dấu** — "KẾ HOẠCH VÀ TIẾN TRÌNH HỌC TẬP", "Tiếng Anh (Course 2)" |
| `pdftext` 0.7.1 (tầng khai thác char của MinerU) | **đúng dấu** |
| code chuyển đổi của ta (`mineru_content.py`) | không có `unicodedata`/strip dấu nào ⇒ **vô tội** |
| MinerU `type=text` | **đúng dấu** |
| MinerU `type=table` → `table_body` | **mất dấu** |

⇒ Chỉ nhánh **nhận dạng bảng** của MinerU hỏng. Cùng một trang, MinerU cho "Tiếng Anh" ở
text block và "Ting Anh cho sn xut bándn" ở table block.

- **BẪY: `-t false` KHÔNG phải cách sửa.** Đo A/B cùng trang: `-t false` cho 0 chữ "Ting Anh" —
  nhưng `table_body` dài **0**, tức MinerU **xoá sạch nội dung bảng**. Chữ "Tiếng Anh" còn lại là
  của text block, không phải bảng. Bật/tắt table parsing đều không dùng được: bật thì mất dấu,
  tắt thì mất bảng. Đừng đọc "0 chữ hỏng" thành "đã sửa".
- **ĐÃ THỬ MinerU 4.0.10 (bản mới nhất) ⇒ KHÔNG sửa được, đừng nâng cấp vì lý do này.**
  Đã dựng sandbox tại `/mnt/data/baopv/mineru4-sandbox` (base package, tier `basic`, ONNX CPU,
  model `MinerU-4_models_onnx` 819 MB) và parse đúng trang mốc:
  - `ocr-mode auto` → mất dấu y hệt 3.4.5 (`Gii tích`, `Ting Anh`, `Cu trúc d liu`).
  - `ocr-mode txt` → **y hệt**, chứng minh chữ trong ô bảng KHÔNG lấy từ text layer mà đi qua
    OCR model bất kể chế độ nào.
  - 3.4.5 + `-l latin` → cũng không đổi. **Gợi ý `MINERU_LANG=latin` trước đó là sai, đã bị bác bỏ.**
- **NGUYÊN NHÂN GỐC: model nhận dạng chữ là bản Trung Quốc.** File
  `ch_PP-OCRv6_small_rec_infer.onnx` — tiền tố `ch_` = Chinese. Bảng chữ cái của nó không có
  `ế ộ ữ ạ ầ ậ` nên ép ra chữ Latin gần nhất. Cấu trúc bảng (`slanet-plus.onnx`) thì vẫn tốt.
  Caption/text block đúng dấu vì đọc thẳng text layer. MinerU 4.0 **không còn cờ `-l/--lang`**
  và không cho đổi model Rec từ CLI ⇒ không có sẵn tuỳ chọn nào trong MinerU.
- **MinerU 4.0 là bản viết lại, không phải bump thường** (nếu sau này cân nhắc nâng cấp):
  entrypoint `mineru -p x.pdf -o out` → `mineru-kit parse x.pdf --output out`; backends
  `pipeline|vlm|hybrid` → tiers `flash|basic|standard|advanced`; extras `mineru[core]` → base +
  `mineru[torch]`/`mineru[full]`; model `MinerU-4_models_onnx|_torch`; cờ `-t` (bảng) **bị bỏ
  hẳn** ⇒ `PARSE_TABLE` sẽ không còn ý nghĩa; schema output đổi sang `docvortex.middle 2.0`
  (`pages[].blocks[].content[]`, `table_body.content` là HTML) khác `content_list` V1 mà
  `mineru_content.py` đang đọc. `/models` mount **read-only** nên cần volume mới cho 4.0.
- **ĐÃ ĐO: chữ bảng mất dấu KHÔNG giới hạn retrieval ⇒ không cần reindex.** Phép A/B
  `scripts/probe_table_embedding.py` embed lại 389 chunk bảng (có ảnh) theo 3 arm vào
  collection tạm, đo 9 case bảng:

  | arm | @5 | @15 | @30 | @50 | reachable | embed |
  |---|---|---|---|---|---|---|
  | text+image (đang deploy) | 6 | 6 | 6 | 7 | 7 | 37.4s |
  | **image only** (bỏ chữ mất dấu) | 6 | 6 | 6 | 7 | 7 | 35.0s |
  | text only (không ảnh) | 6 | 7 | 8 | 8 | 8 | 6.0s |

  ⇒ Bỏ hẳn chữ mất dấu **không đổi một điểm nào**; ảnh render cũng đóng góp **0** cho
  embedding bảng, trong khi làm embedding chậm **6×** (37.4s vs 6.0s). Thiệt hại thật sự
  của chữ mất dấu nằm ở **context LLM đọc**, không phải ở retrieval.
- **24/413 chunk bảng không có ảnh render** (`asset_key` rỗng) ⇒ arm `image_only` không áp
  dụng được cho chúng (gateway từ chối item rỗng); script chỉ so trên 389 chunk có ảnh.
- **BẪY khi tự gọi embed gateway:** contract là key `image`; `OpenAIEmbedder._to_embed_payload`
  mới dịch `image_b64` → `image`. Gửi thẳng `image_b64` cho gateway **không báo lỗi** mà ra
  vector y hệt text-only (đo được `cos(text, text+image_b64) = 1.0`, còn với `image` là 0.369).
  Probe tự dựng payload mà không đi qua client sẽ ra kết luận sai hoàn toàn. Ngoài ra client còn
  tự cắt text theo `EMBED_MAX_INPUT_TOKENS=6000`; bỏ qua thì payload >8192 token bị 400.
- **Đã thêm `PARSE_TABLE`** (`settings.py`, truyền `table` tường minh vào lệnh MinerU) để hành vi
  này không còn ẩn trong mặc định của MinerU. **Mặc định vẫn là `true`** — không được đổi sang
  `false` khi chưa có backend thay thế.
- Backend VLM (`MinerU2.5-Pro-2605-1.2B`) nhiều khả năng đọc đúng dấu vì nó đọc pixel, nhưng model
  **chưa được tải** trong `/models` (chỉ có `PDF-Extract-Kit-1.0`) và worker **không có GPU**
  (`torch.cuda.is_available() == False`) nên chưa thể thử.
- **Reindex toàn bộ hiện không khả thi**: 973 tài liệu, worker chạy CPU. Cần GPU trước.
- **RapidOCR cũng không cứu được**: đọc ảnh bảng ra "Tieng Anh", "Giaitich", "Kien truc" — mất dấu
  y hệt, vì model mặc định không phải tiếng Việt. Không có OCR engine sẵn trong image.
- Hệ quả cho retrieval: đây là lý do thật sự của "bảng rerank kèm ảnh 6/9 vs text thuần 3/9" —
  ảnh render là kênh sạch duy nhất còn lại cho bảng, không phải lựa chọn tối ưu hoá.
- **Đừng dùng tỉ lệ ký tự có dấu để kết luận phạm vi.** Lần đo đầu (table 9,3% vs text 17,3%)
  bị nhiễu: 397/413 bảng đến từ một tài liệu mix tiếng Anh nên tỉ lệ thấp là bình thường. Quét
  toàn tài liệu cho thấy chỉ ~9/973 tài liệu dưới 11%. Số liệu đáng tin là so sánh **cùng trang,
  cùng tài liệu**, như bảng trên.

**Bug calibrator đã sửa:** `scripts/fit_rerank_calibration.py` đọc `doc["score"]` — nhưng đó là điểm
**đã calibration** (`score == metadata.calibrated_score`, đã kiểm chứng bằng API thật: image raw
0.1436 → calibrated 1.0000). Nghĩa là calibrator được fit trên output của chính nó. Đã đổi sang
`metadata.raw_score` và in cảnh báo khi mẫu thiếu raw score. **Chưa refit lại model** — số liệu CV cũ
và `rerank_calibration.json` hiện tại vẫn là kết quả của cách lấy mẫu cũ.

### Bẫy đã vấp (đừng lặp lại)

- **`RETRIEVAL_FUSION_POOL_SIZES`/`_QUOTA` key theo TÊN pool, không phải nhóm modality.**
  Ghi `text+table=40` sẽ parse được rồi khớp **không pool nào** → quota im lặng không dùng slot.
  `parse_pool_sizes`/`parse_quota` giờ reject `+` để lỗi này lộ ra sớm.
- **Pool gộp làm bảng chìm.** Một `Filter.should` + một limit ⇒ pool lấy theo thứ tự điểm vector
  toàn cục, chunk text chiếm đa số chỗ. Không giải quyết bằng depth lớn hơn (depth 60 vẫn 3/9).
- **Bench để lại container ở strategy cuối.** Sau `run_fusion_bench.py` phải deploy lại config cần
  đo, nếu không sẽ đọc nhầm số liệu (case này đã từng làm tôi tưởng kết quả không tất định).
- **Đừng tin dòng bench cho tới khi env trong container đã đúng.** `docker-compose.ami.yml` giữ
  `environment:` với default riêng cho từng biến fusion; lỡ đổi default ở `settings.py` mà quên
  compose thì container vẫn chạy giá trị cũ và dòng bench trông y hệt nhau (tôi đã mất 2 vòng
  đo vì thế). `run_fusion_bench.py` giờ có `_assert_deployed()` đọc `printenv` trong container và
  abort nếu lệch; `tests/ami_service/test_config_consistency.py` chặn lệch compose↔settings.
- **`.gitignore` có `test_*` nuốt dữ liệu trong `tests/`.** Đã thêm `!tests/*.json` + `!tests/*.md`.
  Trước đó `tests/test_case_proposals.json` không hiện trong `git status` — dễ mất file không hay biết.
  File đã đổi tên thành `tests/proposed_test_cases.json` cho an toàn.
- **File có tiếng Việt: dùng tool `read`/`edit`/`write`, tuyệt đối không tự viết escape `\uXXXX`.**
  Tôi đã hỏng hàng loạt comment trong `settings.py` bằng cách viết escape unicode bằng tay
  (`thự` thay vì `thử`, `phách` thay vì `phân`), và một lần trong docstring `propose_test_cases.py`.
  Rà lại bằng mắt — mojibake kiểu này là chữ hợp lệ nên không detector nào bắt được.
  `grep -n "�" <file>` bắt được thì dùng.
- **Đừng suy ra nguyên nhân từ một phép đo thiếu đối chứng.** Tôi đã kết luận "ảnh đẩy mất chunk
  text" chỉ vì rank=None ở bản có ảnh; đo thêm bản bỏ ảnh mới thấy target vốn không có. Luôn so
  `with_images` vs `images_removed` trước khi quy tội cho cơ chế.
- `gitnexus` CLI không có trên máy này (AGENTS.md trỏ path Windows) → dùng grep thay cho
  `gitnexus_impact` / `detect_changes`.

### Scripts

- `scripts/run_fusion_bench.py` — recreate container theo từng strategy + chốp live test, xuất bảng delta + latency.
- `scripts/probe_pool_recall.py` — chạy **trong container**: trần recall thuần vector, tách lỗi retrieval khỏi lỗi rerank. Xuất `tests/pool_recall_probe.json`.
- `scripts/export_text_only_ab.py` — chạy từ host: xuất `tests/text_only_ab.json` (2 arm có/không ảnh, thứ tự đảo ngẫu nhiên) để đưa lên web LLM để chấm.
- `scripts/propose_test_cases.py` — chạy **trong container**: nguyên liệu mở rộng bộ test (chunk đích + chunk dễ nhầm + nháp câu hỏi). Xuất `tests/proposed_test_cases.json`. **Chưa case nào được thêm vào `retrieval_cases.json`.**
- `scripts/probe_table_embedding.py` — chạy **trong container**: A/B 3 arm embedding bảng (text+image / image only / text only) vào collection tạm. Kết luận: bỏ chữ mất dấu không đổi recall, ảnh đóng góp 0 cho embedding bảng. Xuất `tests/table_embed_ab.json`.
- `scripts/collect_pool_rankings.py` — chạy **trong container** (`docker cp` + `docker exec`): thu thập ranking đầy đủ của pool 1 lần.
- `scripts/eval_fusion_rules.py` — sweep rule offline trên ranking đã thu thập (giây thay phút). Lưu ý: cắt pool theo thứ tự **rerank** không mô phỏng được độ sâu retrieve thật (`table_08` rerank rank 3 nhưng vector rank 44) → phải đo thật bằng `run_fusion_bench.py`.
# Handoff — Nemotron VL embed + rerank multimodal (cập nhật: 06/10/2026)

## Đã xong (commits: `663bc12`, `88ae7f7`, `9ab5ce5`, `bd5ad49`, gateway-sync commit tiếp theo)
- **Embed default mới**: `nvidia/llama-nemotron-embed-vl-1b-v2` serve qua **gateway `nemotron-vl-vllm`** (máy B, repo riêng `D:\Code\Python\Ami\RAG\nemotron-vl-vllm\`: gateway FastAPI :8080 trước 2 vLLM embed + rerank; contract: `/health`, `/v1/embeddings` với `{"input": [items], "input_type": "query"|"document"}` — item `str` | `{"text", "image"=<base64>}` | `{"content": [parts]}`; `/v1/rerank` alias `/rerank` với `{query, documents, top_n}` → `{results: [{index, relevance_score}]}`). Client `OpenAIEmbedder` (`ami_rag/core/openai_embedder.py`): handshake `/health` + probe embed xác minh model + dim, batch `EMBED_BATCH_SIZE` item/request + `EMBED_MAX_CONCURRENCY`, retry/circuit breaker/cache như RemoteEmbedder. Đã đối chiếu contract với gateway `gateway/app.py` — khớp.
- **Rerank default mới**: `nvidia/llama-nemotron-rerank-vl-1b-v2` (vLLM `--runner pooling` + template override `nemotron-vl-rerank.jinja`): endpoint `/rerank`, documents `str` hoặc `{"content": [text/image_url parts]}`. `build_vllm_rerank_func` + `build_rerank_documents` (`ami_rag/core/rerank_client.py`): chunk image/table/equation có `asset_key` kèm ảnh render MinIO (data URI, bounded concurrency, thiếu → fallback text). Multimodal chỉ với backend `vllm`; legacy BGE (`build_rerank_model_func`) giữ nguyên text-only.
- **Backend dispatch**: `EMBED_BACKEND` (`auto|custom|openai`, auto: prefix `Qwen/` → custom) + `RERANK_BACKEND` (`auto|legacy|vllm`, auto: rỗng/chứa `bge` → legacy) + `RERANK_MULTIMODAL` (default True). Resolver: `settings.resolve_embed_backend`/`resolve_rerank_backend`; factory `build_embedder`; cli `_build_embedder` qua factory.
- **Multimodal embed**: `_prepare_embed_items` gắn ảnh asset cho modality image/**table**/equation (trước chỉ image) — image+text cho cả 2 backend.
- Settings: `EMBED_MODEL` default → nemotron (`EMBED_DIM` 2048 giữ nguyên, trùng với Qwen); `RERANK_MODEL` default → nemotron.
- Tests: `tests/test_openai_embedder.py` (14), `tests/test_rerank_client.py` (8), `tests/test_gateway_parity.py` (4 — chạy mã gateway thật qua ASGI, skip nếu repo không có); suite 476 pass / 1 skip (reportlab pre-existing); ruff sạch `ami_rag` + `tests/ami_service`.

## Lưu ý
- Đổi `EMBED_MODEL` → collection mới `multimodal__llama-nemotron-embed-vl-1b-v2__v1` → **phải `ami-rag reindex --scan`**.
- Query API chỉ text (không đổi schema `RAGRequest`).
- `_check_embed_server` (cli) backend-agnostic: `/health` + `embedder.verify()`; print dùng `embedder.model`/`dim` (getattr fallback cho FakeEmbedder).
- URL defaults: `EMBED_SERVER_URL`/`RERANK_BASE_URL` = `http://localhost:8080` (gateway); backend `custom` (Qwen) phải đổi `EMBED_SERVER_URL` về 8007 qua env.
- Cleanup model cũ: `ami-rag cleanup --stale-models` (xoá Qdrant `{WORKSPACE}__*` không phải collection hiện tại — ví dụ bản Qwen sau khi đổi Nemotron) + `--purge-cache-model "Qwen/Qwen3-VL-Embedding-2B"` (xoá cache SQLite theo prefix `{MODEL}|`). Helpers: `legacy_cleanup.find_stale_model_collections` + `EmbeddingCache.delete_by_prefix`.

# Handoff — Migration sang vector pipeline thuần (cập nhật: 06/10/2026)

Chi tiết đầy đủ xem `PLAN.md`. Tóm tắt cho phiên tiếp theo:

## Objective
Migrate RAG-Anything fork (service `ami_rag`) từ LightRAG KG + Gemini embedding sang **vector pipeline thuần** với `Qwen/Qwen3-VL-Embedding-2B` chạy remote trên máy B (HTTP, port 8007). API chỉ giữ `POST /rag` + `/rag/stream` (v2-only, drop v1/filters/include_kg/mode). CLI (`ami-rag`) chỉ còn status/retry/reindex.

## Trạng thái phase
| Phase | Nội dung | Trạng thái |
|-------|----------|-----------|
| 0 | Inventory (`docs/migration_inventory.md`) | ✅ commit `7221759` |
| 1 | Thiết kế ( duyệt) | ✅ |
| 2 | Qwen embed server (repo riêng) + client RemoteEmbedder + cache SQLite | ✅ commit `394c8df` |
| 3 | CLI + DocStatusStore + lockfile + PipelineRunner protocol | ✅ commit `32bf44c` |
| 4 | VectorPipeline + wiring factory/worker/API + strip LightRAG | ✅ commit `b2f8828` |
| 5 | Reindex docs cũ (scan legacy → pending) | ✅ commit `787ff0d` |
| 6 | Dọn docs + xóa LightRAG còn sót | ✅ |

## Phase 4 — ✅ (commit `b2f8828`)
- `vector_pipeline.py`: run() tuần tự (load row → doc → parse → describe → chunk → embed → mark_indexed); resume từ MinIO artifacts (chunks.json reuse, build lại nếu thiếu); `_stage_parse` raise khi doc None; dry_run không embed/không ghi; junk đã xóa (`ArtifactPaths`, `_load_doc`, `_load_or_build_chunks`).
- Wiring: `create_runner` → `build_pipeline`; factory.py không còn LightRAG (build_pipeline/get_pipeline/close_pipeline + LLM/vision funcs thuần OpenAI-compatible qua `openai` lib, `_build_modal_processors` lightrag=None).
- Worker: runner-based (`IngestWorker(runner=...)`), DocStatusStore làm state repo; xóa `index_check.py` + `storage/rag_documents.py`.
- API v2-only: `RAGRequest` = messages/top_k/include_references; `/rag` = embed_query → vector_search → rerank → `resolve_doc_id`; admin dùng DocStatusStore; `DocResolver.resolve_doc_id`.
- raganything thin: query.py (aquery → llm, aquery_data → embed+search), processor.py (parse-only), raganything.py (parser + modal processors + query), utils.py (bỏ insert_text_content*), config.py (`get_env_value` local — KHÔNG import lightrag nữa, toàn service import được không cần lightrag).
- modalprocessors: BaseModalProcessor lightrag-tolerant (llm_model_func/tokenizer/global_config qua kwargs), xóa `_create_entity_and_chunk`/`_process_chunk_for_extraction`/`process_multimodal_content` (mọi file), `compute_mdhash_id` local.
- Tests: 480 pass khi commit Phase 4; xóa 15 test file old-pipeline; **thêm `tests/__init__.py`** (bắt buộc: site-packages có package `tests` shadow local dir → `tests.ami_service` import lỗi nếu thiếu).

## Phase 5 — ✅ (commit `787ff0d`)
- `ami_rag/core/scan.py`: `scan_pending` (scan backend documents → ensure_pending cho docs thiếu registry row) + `reindex_candidates` (pending + stale, gồm legacy `processed`; indexed/failed loại trừ).
- CLI: `ami-rag reindex --scan`; from_stage per-doc (`_from_stage_for`: chunk khi content_list đã có trong MinIO, None=full parse khi thiếu).
- Tests: 483 pass, 2 skip (pre-existing: reportlab + lightrag); ruff clean trên `ami_rag`.

## Phase 6 — ✅
- Xóa `reproduce/`, `examples/`, `notebooks/`, `env.example`, `scripts/create_tiktoken_cache.py`; dead code `raganything/batch.py` + `raganything/resilience.py` (+ export trong `__init__.py`). Giữ `callbacks.py` (dùng bởi processor/query) + `batch_parser.py` (parse-only).
- Bỏ dep `lightrag-hku` (pyproject + requirements.txt + MANIFEST.in); bỏ settings dead (`WORKING_DIR`, `RETRIEVAL_TOP_K`, `INGEST_VERIFY`, `INGEST_REQUIRE_ENTITIES`, `MAX_GLEANING`, `SUMMARY_LANGUAGE`).
- Bỏ metric dead `LIGHTRAG_FAILURES_TOTAL` + panel dashboard Grafana tương ứng; viết lại `docs/ami_service.md`; xóa docs thuần LightRAG (`offline_setup.md`, `architecture.md`, `api_reference.md`).
- Không còn `import lightrag` nào trong repo (chỉ còn tên param compat `lightrag=None` trong modalprocessors/factory).
- Tests: 435 pass, 1 skip (reportlab pre-existing); ruff clean `ami_rag` + `tests/ami_service`. Migration HOÀN TẤT.

## Quy tắc (bắt buộc)
- **MỖI phase commit riêng, STOP xin duyệt sau mỗi phase.** Trước commit: `gitnexus_detect_changes()`.
- Trước khi sửa symbol: `gitnexus_impact` trước (AGENTS.md gitnexus block ở trên).
- Test: `pytest` cần `$env:PYTHONIOENCODING="utf-8"`; lint ruff 0.16.0 global (`ami_rag/**` đã per-file-ignores BLE001/S110/B008).
- GitNexus CLI phải dùng bản global 1.6.5 (`C:\Users\phanb\AppData\Local\pnpm\bin\gitnexus.CMD`), KHÔNG `npx` latest.
- File có tiếng Việt: chỉ dùng tool write/edit, KHÔNG Add-Content/Set-Content (cp1252 mojibake).
- Không gọi LLM/API ngoài ý muốn; describe là stage duy nhất gọi LLM (tuỳ chọn).
- Repo Qwen server nằm RIÊNG ngoài RAG-Anything: `D:\Code\Python\Ami\RAG\qwen-embedding-server\` (git `31a5393`, 20 tests pass/1 skip).
