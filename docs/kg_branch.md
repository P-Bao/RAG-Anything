# KG branch — LightRAG 1.5.7 + Qwen3.5 + nemotron embedding

Nhánh `feature/kg-qwen35-nemotron` (từ `main`): knowledge graph LightRAG, khác
pipeline vector thuần của `feature/ami-service`. Tài liệu này là runbook duy
nhất của nhánh; docs còn lại trong `docs/` là của upstream RAG-Anything.

## Kiến trúc

```
MinIO (ami-multimodal-retrieval, read-only)
  └─ rag-assets/{doc_id}/content_list.json + descriptions.json
       └─ kg_index_pdfs.py: ghép text theo template (caption tái sử dụng, 0 call API)
            └─ LightRAG.ainsert
                 ├─ chunking (LightRAG, ~1200 tokens)
                 ├─ entity/relation extraction: Qwen3.5-35B-A3B GPTQ-Int4, thinking OFF
                 ├─ gleaning: cùng cấu hình
                 └─ embedding: nemotron-embed-vl-1b-v2 qua gateway :8080
Storage: file-based (NanoVectorDB + NetworkX graphml) trong KG_WORKING_DIR
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

### Docker (one-shot runner)

Image cài đủ deps pyproject (kể cả `mineru[core]` — nặng nhưng bắt buộc vì
`RAGAnything` init kiểm tra `MineruParser.check_installation()`, dù script
không parse). Storage nằm trong volume `rag-storage` — **không chạy 2 index
song song**.

```bash
cp .env.example .env            # host = tên container trên ami-network khi chạy kèm hạ tầng
make docker-build
make docker-index-pdf ARGS="--limit 2 --dry-run"
make docker-index-pdf ARGS="--limit 2"
make docker-index-status
make docker-query Q="chương trình AI-RAG do ai phụ trách?"
make docker-clean               # xóa container + volume + image local
```

Yêu cầu: network `ami-network` tồn tại (`docker network create ami-network`)
và `.env` có trước khi `run` (env_file). Local `make index-pdf` và docker
dùng chung quy ước `ARGS`/`Q`/`MODE`/`BG`-không-có — BG chỉ cho local.

## Environment variables

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
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO của ami-multimodal-retrieval (read-only) |
| `MINIO_BUCKET` | `ami-data-documents` | |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` / rỗng | |
| `ASSET_PREFIX` | `rag-assets` | prefix artifact bên kia |
| `KG_WORKING_DIR` | `./rag_storage` | storage file-based của KG |
| `CAPTION_LLM_API_KEY` | rỗng | đặt thì fallback caption Gemini cho doc chưa có artifact |
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
```

- Doc đã được ami-multimodal-retrieval index → caption tái sử dụng từ
  `descriptions.json`, **không tốn API caption**.
- Thiếu artifact → script skip + warn; muốn index doc mới chưa parse thì phải
  qua pipeline bên kia (parse MinerU) rồi chạy lại script.
- Ingest chậm: ~3-7 phút/chunk (extract + gleaning ở ~17 tok/s). Tăng
  song song bằng `EXTRACT_MAX_ASYNC_LLM`/`KEYWORD_MAX_ASYNC_LLM` (LightRAG
  role env, mặc định 4) khi GPU cho phép.
- Cache LLM: LightRAG tự cache extraction/keywords/query qua
  `llm_response_cache` ở tầng operate (không phụ thuộc binding) — chạy lại
  sau lỗi không phải trả token cho chunk đã extract thành công.
- **Chỉ chạy 1 tiến trình index** trên cùng `KG_WORKING_DIR`: storage
  file-based (NanoVectorDB/graphml) là last-write-wins, 2 process song song
  gây mất dữ liệu im lặng.

## Audit LLM — mọi call site trên nhánh này

| Call site | Model | thinking | temp | max_tokens | Verified |
|---|---|---|---|---|---|
| Entity/relation extraction + gleaning (LightRAG `extract`) | Qwen3.5 | **OFF** | 0.9 | 4000 | ✅ test thật (tuple format sạch, reasoning_content=null) |
| Keyword extraction query time (`keyword`) | Qwen3.5 | **OFF** | 0.9 | 4000 | ⚠️ cùng wrapper — verify khi chạy smoke query |
| Query answer (`query`) | Qwen3.5 | **ON** | 0.6 | 4000 | ⚠️ verify khi smoke (content phải non-null cả 2 mode) |
| Rerank (`rerank_model_func`, query-time chunk ordering) | BGE `bge-reranker-v2-m3` :8010 | n/a | n/a | n/a | ⚠️ verify khi smoke (LightRAG fallback chunks gốc nếu service chết — warning, không fatal) |
| Embedding (chunk/entity/relation/query) | nemotron vl 1b v2 | n/a | n/a | n/a | ⚠️ verify gateway khi smoke (`/health` + dim 2048) |
| Caption ảnh (chỉ khi doc chưa có artifact + có key) | Gemini 3.5 flash lite | n/a | server default | server default | ⚠️ port từ ami-multimodal-retrieval — verify 1 call |
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
| Ingest | script + Makefile | API + worker + Mongo registry |

Quay lại nhánh vector: `git switch feature/ami-service && uv sync --extra text`
(venv được sync theo branch).
