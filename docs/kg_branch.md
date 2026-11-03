# KG branch — LightRAG 1.5.7 + Qwen3.5 + nemotron embedding

Nhánh `feature/kg-qwen35-nemotron` (từ `main`): knowledge graph LightRAG, khác
pipeline vector thuần của `feature/ami-service`. Tài liệu này là runbook duy
nhất của nhánh; docs còn lại trong `docs/` là của upstream RAG-Anything.

## Kiến trúc

```
MinIO (ami-multimodal-retrieval)
  └─ rag-assets/{doc_id}/content_list.json + descriptions.json
       └─ kg_index_pdfs.py: ghép text theo template (caption tái sử dụng, 0 call API)
            └─ LightRAG.ainsert

Doc mới (registry Mongo, `make index-mongo`) — chỉ khi artifact chưa có:
  file gốc (MinIO) → MinerU parse (layout + ảnh)
    → PARSE_TEXT_SOURCE=backend_data: pdfplumber/PyMuPDF + OCR :12006
      (fallback Vintern :8090) thay text/bảng
    → upload ảnh + content_list.json → describe (caption Gemini cho ảnh,
      bảng raw, công thức LaTeX) → descriptions.json
    → ghép text → LightRAG.ainsert (như trên)
                 ├─ chunking (LightRAG, ~1200 tokens)
                 ├─ entity/relation extraction: Qwen3.5-35B-A3B GPTQ-Int4, thinking OFF
                 ├─ gleaning: cùng cấu hình
                 └─ embedding: nemotron-embed-vl-1b-v2 qua gateway :8080
Storage: file-based (NanoVectorDB + NetworkX graphml) trong KG_WORKING_DIR
API server: kg-api :8009 (thường trú) — /v2/rag (retrieval + generate), /v2/rag/index
  (nhận docs từ MinIO hoặc Mongo; 1 worker, query song song), /v2/rag/status, /readyz
Query: LightRAG mode "mix" (KG + vector) + BGE rerank :8010 — binding gửi
hết documents (top_k = số documents), trả TẤT CẢ kết quả rerank, LightRAG
cắt theo RERANK_TOP_K (chunk_top_k); service chết → LightRAG warning + giữ
thứ tự gốc,
thinking ON
VLM-enhanced query: TẮT HẲN (vision_model_func không bao giờ được set —
xem raganything/kg_factory.py)
```

## Setup

```bash
make setup        # uv sync --extra text (lightrag-hku>=1.5.7,<2, minio, mineru)
make test         # pytest (tests/ami_service là của nhánh vector, bỏ qua)
```

### Từ lúc clone xong

Nhánh chỉ tồn tại local (chưa push) — clone xong phải có nhánh
`feature/kg-qwen35-nemotron` (push trước hoặc làm việc trực tiếp trên máy này).

1. **Yêu cầu hạ tầng sống** (kiểm tra trước khi chạy):
   - Qwen3.5 vLLM OpenAI-compatible (`QWEN_LLM_BASE_URL`, mặc định `:8000/v1`)
   - Gateway nemotron (`NEMOTRON_EMBED_URL`, mặc định `:8080`)
   - MinIO của ami-multimodal-retrieval (`MINIO_ENDPOINT`, mặc định `:9000`,
     bucket `ami-data-documents`, prefix `rag-assets`) — chỉ đọc
2. `git switch feature/kg-qwen35-nemotron && make setup`
3. Export env (xem bảng dưới): tối thiểu `QWEN_LLM_BASE_URL`,
   `NEMOTRON_EMBED_URL`, `MINIO_ENDPOINT`, `MINIO_SECRET_KEY`;
   `CAPTION_LLM_API_KEY` chỉ cần khi muốn fallback caption Gemini.
4. `make test` — sanity 450 tests, không cần hạ tầng.
5. **Dry-run** (không gọi LLM, chỉ xem text assemble + caption reuse):
   `make index-pdf ARGS="--limit 2 --dry-run"`
6. **Index thử**: `make index-pdf ARGS="--limit 2"` → dòng
   `KG totals: entities=N ... in Ns` phải có N > 0 (chặn silent-zero).
   KHÔNG chạy 2 tiến trình index song song (xem cảnh báo ở dưới).
7. **Kiểm tra KG**: `make index-status`
8. **Query smoke**: `uv run python scripts/kg_index_pdfs.py query "câu hỏi"`
   (mode mặc định `mix`, thinking ON) — xác nhận answer non-null, không rỗng.
9. Index thêm: `make index-pdf BG=1 ARGS="--limit 10"` rồi
   `tail -f logs/index-pdf-*.log`.

### Server API (cổng 8009)

Service thường trú thay thế API của `ami-multimodal-retrieval` (cùng cổng 8009,
cùng kiểu auth bearer). 1 image dùng cho cả `kg-api` (server) và `kg-rag` (CLI).

```bash
make api                        # local: uvicorn raganything.api (API_PORT=8009)
make docker-up                  # build + chạy kg-api detached (restart: unless-stopped)
make docker-logs                # theo dõi log
make docker-restart             # down + up lại
make docker-down                # dừng
```

| Route | Việc |
|---|---|
| `GET /` | `{service, version}` |
| `GET /healthz` | liveness `{status:"ok"}` |
| `GET /readyz` | readiness: LightRAG đã init + counts KG (503 khi chưa xong) |
| `POST /v2/rag/` | retrieval: `{query \| messages, top_k, include_references, generate}` → `{query, documents[], references[], meta{latency_ms, requested_top_k, returned, candidates, fusion}}`; `generate=true` thêm `answer` (LLM) |
| `POST /v2/rag/index` | 202 `{job_id}` — `{source:"minio"\|"mongo", document_ids?, limit?, dry_run?}` **tiếp nhận docs từ MinIO hoặc Mongo** |
| `GET /v2/rag/index/{job_id}` | trạng thái job + tiến độ từng doc |
| `GET /v2/rag/status` | counts KG + queue + `doc_status_counts` |

- Auth: `Authorization: Bearer $RAG_API_KEY` cho mọi `/v2/rag/*`; `RAG_API_KEY` rỗng = không auth (nội bộ). `GET /`, `/healthz`, `/readyz` mở.
- Ingest chạy **1 worker** (queue trong process) — storage file-based không ghi song song; **query chạy song song** (mọi storage LightRAG ghi atomic).
- Trạng thái doc ghi vào collection riêng `KG_STATE_COLLECTION` (mặc định `kg_rag_documents`) — **không** đụng `multimodal_rag_documents`; không có `MONGO_URI` → fallback file JSON.
- Phiên này **chưa có**: `/admin/*`, `/metrics`, Redis queue, Qdrant, OTEL, `/v2/rag/stream`.

### Docker CLI (one-shot runner)

| Var | Mặc định | Vai trò |
|---|---|---|
| `QWEN_LLM_BASE_URL` | `http://localhost:8000/v1` | vLLM Qwen3.5 (OpenAI-compatible) |
| `QWEN_LLM_MODEL` | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | |
| `QWEN_LLM_API_KEY` | `EMPTY` | |
| `NEMOTRON_EMBED_URL` | `http://localhost:8080` | gateway nemotron-vl-vllm |
| `NEMOTRON_EMBED_MODEL` | `nvidia/llama-nemotron-embed-vl-1b-v2` | dim 2048 |
| `NEMOTRON_EMBED_BATCH_SIZE` | `16` | |
| `NEMOTRON_EMBED_MAX_CONCURRENCY` | `4` | |
| `RERANK_BASE_URL` | `http://localhost:8010` | BGE rerank service; **rỗng** = tắt rerank |
| `RERANK_TOP_K` | `5` | số chunk cuối cho query (`QueryParam.chunk_top_k`) — binding gửi **hết** documents cho BGE, không chọn khi gọi API |
| `RERANK_TIMEOUT` | `60` | giây |
| `MIN_RERANK_SCORE` | `0` | LightRAG lọc ngầm điểm rerank (default 0.5) — `0` = giữ tất cả kết quả rerank, chỉ cắt theo `RERANK_TOP_K` |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO của ami-multimodal-retrieval; parse stage **ghi** artifact mới (ảnh + content_list + descriptions) |
| `MINIO_BUCKET` | `ami-data-documents` | |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` / rỗng | |
| `ASSET_PREFIX` | `rag-assets` | prefix artifact bên kia |
| `KG_WORKING_DIR` | `./rag_storage` | storage file-based của KG |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8009` | server kg-api (8009 = cổng API cũ) |
| `RAG_API_KEY` | rỗng | bearer cho `/v2/rag/*`; rỗng = không auth |
| `KG_STATE_COLLECTION` | `kg_rag_documents` | collection trạng thái ingest (riêng, không dùng chung) |
| `MONGO_URI` | `mongodb://localhost:27017` | registry production (`make index-mongo`) — **read-only** |
| `ORG_DB` / `DOC_COLLECTION` | `organization_db` / `documents` | query `{status: "active"}` |
| `PARSE_METHOD` | `auto` | method MinerU (`auto`/`ocr`/`txt`) |
| `PARSE_TEXT_SOURCE` | `backend_data` | `backend_data` = thay text/table bằng pdfplumber/PyMuPDF + OCR; `mineru` = giữ nguyên output MinerU |
| `MINERU_BACKEND` / `MINERU_LANG` / `MINERU_SOURCE` / `MINERU_TIMEOUT` | rỗng / rỗng / rỗng / `0` | kwargs truyền `parse_document` (parser_kwargs như pipeline bên kia) |
| `MINERU_DEVICE` / `MINERU_VIRTUAL_VRAM_SIZE` | rỗng / `0` | env subprocess MinerU (`MINERU_DEVICE_MODE`/`MINERU_VIRTUAL_VRAM_SIZE`) |
| `OCR_SERVICE_URL` | `http://171.226.10.153:12006` | OCR service (text hierarchy-parse + bảng Qwen) |
| `OCR_SERVICE_TIMEOUT` | `300` | giây |
| `OCR_TABLE_ENDPOINT` / `OCR_TEXT_ENDPOINT` | `/api/v1/raw_miner_ocr/qwen` / `/api/v1/hierarchy-parse` | |
| `OCR_FALLBACK_ENABLED` | `true` | fallback Vintern khi OCR service lỗi/timeout/trả rỗng (tắt = `false` hoặc URL rỗng) |
| `OCR_FALLBACK_URL` / `OCR_FALLBACK_TOKEN` | `http://localhost:8090` / rỗng | vLLM Vintern; token là bearer tuỳ chọn, **không** dùng để tắt |
| `OCR_FALLBACK_MODEL` / `OCR_FALLBACK_TIMEOUT` / `OCR_FALLBACK_RETRIES` | `5CD-AI/Vintern-1B-v3_5` / `300` / `1` | |
| `OCR_FALLBACK_ON_EMPTY` / `OCR_FALLBACK_MODES` | `true` / `text,table` | bật fallback khi kết quả rỗng; kind được fallback — **giá trị rỗng = tắt fallback** |
| `CAPTION_LLM_API_KEY` | rỗng | caption Gemini cho ảnh doc mới parse (không có → fallback `img_caption` trong content_list) |
| `CAPTION_LLM_BASE_URL` | Google OpenAI-compatible | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `CAPTION_LLM_MODEL` | `gemini-3.5-flash-lite` | |
| `CAPTION_LLM_LANGUAGE` | `vi` | |

## Runbook

```bash
# 1. Xem trước text sẽ index (không gọi LLM)
make index-pdf ARGS="--limit 2 --dry-run"

# 2. Index thật (Qwen3.5 + gateway phải sống); kiểm tra dòng "KG totals: entities=..."
make index-pdf ARGS="--limit 2"

# 3. Doc cụ thể / chạy nền
make index-pdf DOC=<doc-id>
make index-pdf BG=1 ARGS="--limit 10"

# 4. Trạng thái KG
make index-status

# 5. Doc mới theo danh sách Mongo (production upload) — parse nếu thiếu artifact
make index-mongo ARGS="--limit 5 --dry-run"
make index-mongo ARGS="--limit 5"

# 6. API server (thay API cũ, cổng 8009)
make api                                   # local; hoặc: make docker-up (thường trú)
curl localhost:8009/readyz
curl -X POST localhost:8009/v2/rag/ -H "Content-Type: application/json" \
     -d '{"query":"chương trình AI-RAG do ai phụ trách?","generate":true}'
curl -X POST localhost:8009/v2/rag/index -H "Content-Type: application/json" \
     -d '{"source":"mongo","limit":5}'
curl localhost:8009/v2/rag/index/<job_id>
```

- Doc đã được ami-multimodal-retrieval index → caption tái sử dụng từ
  `descriptions.json`, **không tốn API caption**.
- `make index-pdf` (scan MinIO): thiếu artifact → skip + warn. Dùng
  `make index-mongo` cho doc mới chưa parse (xem dưới).
- Ingest chậm: ~3-7 phút/chunk (extract + gleaning ở ~17 tok/s). Tăng
  song song bằng `EXTRACT_MAX_ASYNC_LLM`/`KEYWORD_MAX_ASYNC_LLM` (LightRAG
  role env, mặc định 4) khi GPU cho phép.
- Cache LLM: LightRAG tự cache extraction/keywords/query qua
  `llm_response_cache` ở tầng operate (không phụ thuộc binding) — chạy lại
  sau lỗi không phải trả token cho chunk đã extract thành công.
- **Chỉ chạy 1 tiến trình index** trên cùng `KG_WORKING_DIR`: storage
  file-based (NanoVectorDB/graphml) là last-write-wins, 2 process song song
  gây mất dữ liệu im lặng.

### Index theo danh sách Mongo (`make index-mongo`)

Danh sách doc lấy từ registry production (`{ORG_DB}.{DOC_COLLECTION}`,
query `{status: "active"}` — **read-only, không ghi Mongo**). Mỗi doc:

```
select_source(doc):
├─ mongo_text (txt/crawl, không có file) → content trong Mongo → index trực tiếp
└─ minio_parse (pdf/docx) → rag-assets/<doc_id>/content_list.json tồn tại?
     ├─ có → reuse artifact + caption (như make index-pdf)
     └─ không → PARSE: fetch file gốc (doc.file_path) → MinerU (layout + ảnh)
                → PARSE_TEXT_SOURCE=backend_data: thay text/table bằng
                  pdfplumber/PyMuPDF + OCR service (fallback Vintern)
                → upload ảnh + content_list.json (layout như pipeline bên kia)
                → DESCRIBE: image → CAPTION_LLM (fallback img_caption),
                  table → TABLE_DESCRIBE_MODE (raw mặc định, không gọi LLM),
                  equation → LaTeX có sẵn
                → descriptions.json → assemble → index
```

- Idempotent: artifact đã tồn tại → skip parse; insert lại do LightRAG dedup.
- Hạ tầng parse cần sống: MinIO, OCR service (`OCR_SERVICE_URL`),
  fallback Vintern (`OCR_FALLBACK_URL`, token). OCR service chết/timeout →
  fallback Vintern; cả hai chết → parse fail cho doc đó, các doc khác chạy tiếp.
- `PARSE_TEXT_SOURCE=mineru` = escape hatch giữ nguyên output MinerU khi
  không muốn OCR (chậm hơn về chất lượng text/bảng).
- Flags: `--limit N`, `--doc <ObjectId>` (lặp được), `--dry-run`
  (dry-run KHÔNG parse — chỉ liệt kê nguồn/reuse).

## Audit LLM — mọi call site trên nhánh này

| Call site | Model | thinking | temp | max_tokens | Verified |
|---|---|---|---|---|---|
| Entity/relation extraction + gleaning (LightRAG `extract`) | Qwen3.5 | **OFF** | 0.9 | 4000 | ✅ test thật (tuple format sạch, reasoning_content=null) |
| Keyword extraction query time (`keyword`) | Qwen3.5 | **OFF** | 0.9 | 4000 | ⚠️ cùng wrapper — verify khi chạy smoke query |
| Query answer (`query`) | Qwen3.5 | **ON** | 0.6 | 4000 | ⚠️ verify khi smoke (content phải non-null cả 2 mode) |
| Rerank (`rerank_model_func`, query-time chunk ordering) | BGE `bge-reranker-v2-m3` :8010 | n/a | n/a | n/a | ⚠️ verify khi smoke (LightRAG fallback chunks gốc nếu service chết — warning, không fatal) |
| Embedding (chunk/entity/relation/query) | nemotron vl 1b v2 | n/a | n/a | n/a | ⚠️ verify gateway khi smoke (`/health` + dim 2048); query dùng `input_type=query` — `EmbeddingFunc(supports_asymmetric=True)` (thiếu cờ này LightRAG bỏ `context`) |
| Caption ảnh (doc mới parse hoặc doc chưa có artifact, có key) | Gemini 3.5 flash lite | n/a | server default | server default | ⚠️ port từ ami-multimodal-retrieval - verify 1 call |
| Parse text/bảng (`make index-mongo`, doc mới, `PARSE_TEXT_SOURCE=backend_data`) | OCR service :12006 (hierarchy-parse + raw_miner_ocr/qwen) | n/a | n/a | n/a | ⚠️ verify khi parse doc mới; lỗi/timeout → fallback Vintern :8090 |
| VLM-enhanced query, audio/video processors | — | — | — | — | ⛔ TẮT: không cấu hình, không đường gọi |

Không còn call site LLM nào khác: `kg_factory.py` là điểm wire duy nhất;
`grep -rn "llm_model_func\|modal_caption_func\|chat.completions" raganything/`
chỉ trỏ về 3 binding đã liệt kê.

## Monitoring (thu gọn)

Nhánh này không có service/process dài hạn (script chạy rồi thoát) → không
dựng Grafana dashboard/ServiceMonitor như `ami-multimodal-retrieval/monitoring/`.
Số đo khả dụng:

- Timing per-doc + tổng trong output `make index-pdf` (`indexed <id>: ... Ns`,
  `KG totals: ... in Ns`) — theo dõi tok/s thực tế của Qwen3.5.
- `make index-status` — đếm entity/relation/chunk trước/sau mỗi batch.
- Metrics phía hạ tầng (đọc trực tiếp): vLLM Qwen (`/metrics` cổng server,
  `vllm:request_*`), gateway nemotron (`:8080/metrics` nếu bật).

Khi nhánh này có worker/API chạy dài hạn, port pattern
`monitoring/` của ami-multimodal-retrieval (ServiceMonitor + PrometheusRule +
dashboard JSON) — đừng viết lại từ đầu.

## Khác biệt với nhánh vector (`feature/ami-service`)

| | Nhánh này (KG) | feature/ami-service |
|---|---|---|
| Retrieval | LightRAG mix (KG + vector), graphml/NanoVectorDB file-based | Qdrant thuần + rerank |
| LLM | Qwen3.5-35B-A3B GPTQ-Int4 (localhost) | Qwen self-host cho describe/answer |
| Embed | nemotron gateway :8080 | nemotron gateway :8080 (giống) |
| Rerank | BGE `bge-reranker-v2-m3` :8010 | BGE :8010 / nemotron :8080 |
| Caption | Tái sử dụng MinIO artifacts, fallback Gemini | Qwen VLM describe |
| Ingest | script + Makefile; danh sách doc từ Mongo (read-only) + parse MinerU/OCR tại chỗ cho doc mới | API + worker + Mongo registry |
| Parse text/bảng | pdfplumber/PyMuPDF + OCR :12006 + fallback Vintern :8090 (port) | như trên (nguồn gốc) |

Quay lại nhánh vector: `git switch feature/ami-service && uv sync --extra text`
(venv được sync theo branch).
