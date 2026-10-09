# AMI multimodal RAG service (`ami_rag`)

Package `ami_rag/` trong fork RAG-Anything này là service multimodal RAG của hệ thống AMI: một ingest worker (parse tài liệu bằng MinerU, chunk + embed bằng vector pipeline thuần) và một API truy xuất `POST /v2/rag`. Nó thay thế hoàn toàn repo cũ `ami-multimodal-retrieval`. Pipeline đã chuyển sang **vector thuần**: không còn LightRAG (không entity, không knowledge graph, không neo4j).

## 1. Kiến trúc

```
ami_data (backend docs: MinIO + MongoDB organization_db.documents)
   │  XADD event mỏng (id + content_hash)
   ▼
Redis Streams  rag:ingest  (consumer group ami-rag)
   ▼
ami_rag worker  (ami-rag-worker, hoặc nhúng trong API khi WORKER_ENABLED=true)
   ├─ select_source(doc)
   │    ├─ mongo_text : text Mongo `content`  →  content_list = [text]
   │    └─ minio_parse: tải file gốc từ MinIO → MinerU parse → content_list (text/image/table/equation)
   ├─ upload ảnh/bảng/công thức lên MinIO  rag-assets/{doc_id}/…  (gắn item["asset_key"])
   ├─ lưu content_list.json vào MinIO
   ├─ describe: mô tả modal (LLM, tuỳ chọn) → descriptions.json
   ├─ chunk: cắt chunk theo token (CHUNK_SIZE/CHUNK_OVERLAP) → chunks.json
   ├─ embed: RemoteEmbedder (HTTP máy B) → upsert Qdrant (delete_by_doc trước, idempotent)
   └─ mark_indexed: ghi trạng thái vào DocStatusStore
        ▼
   Qdrant (vector, collection `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}`)
   MongoDB `organization_db.multimodal_rag_documents` (DocStatusStore, một dòng/doc theo stage)

Hệ thống AI ──POST /v2/rag──▶ ami_rag API: embed_query (máy B) → vector search Qdrant → rerank service
                               → resolve doc Mongo → presign MinIO (artifact_url, file_url) → response
```

- Event mỏng: queue chỉ mang `event` (`created`/`updated`/`deleted`), `document_id`, `content_hash`, `document_type`, `org_id`. Worker đọc nội dung từ Mongo/MinIO (`ami_rag/queue/events.py`).
- Doc id = MongoDB ObjectId. Mỗi chunk payload mang `doc_id` trực tiếp nên API tra ngược về Mongo (`ami_rag/api/resolver.py::resolve_doc_id`).
- `created`/`updated` → `ensure_pending` + chạy pipeline từ stage `parse`; `deleted` → xoá vector Qdrant + asset MinIO + dòng registry. Nguồn không đổi (`content_hash` trùng, status `indexed`, đúng `embed_model`/`chunker_version`) thì bị bỏ qua, cả ở worker lẫn ở CLI `reindex`.
- Pipeline (`ami_rag/core/vector_pipeline.py`): 5 stage `parse → describe → chunk → embed → indexed`, mỗi stage ghi trạng thái vào DocStatusStore. Kết quả trung gian lưu MinIO (`content_list.json`, `descriptions.json`, `chunks.json`) nên retry chạy tiếp từ stage lỗi, không cần parse lại.
- Embed: stage duy nhất gọi mạng thật; luôn `delete_by_doc` trước khi upsert (idempotent) rồi verify count. Describe là stage duy nhất có thể gọi LLM (tuỳ chọn).
- Retry: `WORKER_MAX_DELIVERY` lần; quá ngưỡng thì dòng registry được đánh dấu `failed` và message được ack. Event lỗi chưa đủ ngưỡng được để pending và tự giao lại (XAUTOCLAIM) sau `WORKER_RETRY_IDLE_MS` (mặc định 10 phút; cần Redis ≥ 6.2) — đặt đủ lớn hơn thời gian parse MinerU dài nhất để worker khác không "cướp" message đang xử lý. Khi `failed`, `attempts` được reset nên `reprocess_failed`/`retry --all-failed` có đủ ngân sách thử lại.
- LLM (describe): profile `qwen-selfhost` (vLLM, OpenAI-compatible) qua `QWEN_LLM_*`. Gemini không còn được hỗ trợ (đã gỡ). Rerank là service ngoài `POST {RERANK_BASE_URL}/rerank` (mặc định vLLM `nvidia/llama-nemotron-rerank-vl-1b-v2`, multimodal: chunk image/table/equation kèm ảnh + text; lỗi rerank thì trả chunk không điểm (fallback)).
- Embedding: mặc định `nvidia/llama-nemotron-embed-vl-1b-v2` serve qua **gateway** `nemotron-vl-vllm` (máy B, repo riêng: gateway FastAPI trước 2 vLLM embed + rerank, port 8080; endpoint `/v1/embeddings` với `{"input", "input_type": "query"|"document"}`, item text/ảnh base64/bảng); client `OpenAIEmbedder` (`ami_rag/core/openai_embedder.py`). Còn hỗ trợ `Qwen/Qwen3-VL-Embedding-2B` trên `qwen-embedding-server` (port 8007) qua backend `custom` (`RemoteEmbedder`, `ami_rag/core/remote_embedder.py`). Chọn qua `EMBED_BACKEND` (`auto`: prefix `Qwen/` → custom, còn lại → openai). Máy A chỉ gọi HTTP — không cài torch/transformers, không cần GPU cho embed. Chunk image/table/equation có asset_key được embed dạng image+text (ảnh asset từ MinIO).

## 2. Quyết định nguồn nội dung

`ami_rag/sources.py::select_source`:

| Điều kiện trên `organization_db.documents` | Nguồn | Hành vi |
|---|---|---|
| `original_file_name` rỗng hoặc đuôi `.txt` (text, crawl, seed) | `mongo_text` | dùng text trong Mongo `content`; rỗng thì bỏ qua |
| còn lại (pdf, doc, docx, ...) | `minio_parse` | tải file `file_path` từ MinIO, parse lại bằng MinerU; text/bảng Mongo **không dùng** |

`source_hash` (dấu vân tay quyết định có nạp lại hay không):
- `mongo_text`: sha256 của `content.strip()`.
- `minio_parse`: sha256 của `"minio|{file_path}"` (key MinIO duy nhất cho mỗi lần upload; `content_hash` Mongo bị bỏ qua có chủ đích).

Doc `minio_parse` không có `file_path` thì lỗi (`ValueError`). Ảnh crawl (tuỳ chọn): `CRAWL_IMAGES_ENABLED=true` thì worker tải ảnh của doc crawl về MinIO prefix `CRAWL_IMAGES_PREFIX` và thêm item `image`.

## 3. Dữ liệu

| Thành phần | Giá trị | Ghi chú |
|---|---|---|
| Mongo DB | `RAG_DB` = `organization_db` | dùng chung DB với backend ami_data |
| Registry / DocStatusStore | `organization_db.multimodal_rag_documents` (`RAG_DOCUMENTS_COLLECTION`) | một dòng/doc, `_id` = ObjectId dạng string, ghi bởi worker/CLI/admin |
| Qdrant | `QDRANT_URL` | vector chunk, collection `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}` (ví dụ `multimodal__llama-nemotron-embed-vl-1b-v2__v1`) |
| MinIO asset | `{ASSET_PREFIX}/{doc_id}/` (`rag-assets/…`) trong `MINIO_BUCKET` | ảnh/bảng/công thức (tên `sha256[:16]` + ext), `content_list.json`, `descriptions.json`, `chunks.json` |
| Nguồn doc | `organization_db.documents` (`ORG_DB`/`DOC_COLLECTION`) | **chỉ đọc**, lọc `status: "active"` khi duyệt toàn bộ |

Quy ước collection vector (`ami_rag/core/embedder.py::collection_name`): `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}` — đổi embed model hoặc `CHUNKER_VERSION` là sang collection mới, không bao giờ trộn vector hai embed model.

Ranh giới:

| Collection | Quyền của RAG |
|---|---|
| `documents` | chỉ ĐỌC (`ORG_DB`/`DOC_COLLECTION`) |
| `document_versions`, `organization_units`, `users` | không đọc trong code; chỉ dùng để `$lookup` khi truy vấn/báo cáo từ bên ngoài |
| `multimodal_rag_documents` | GHI (tạo collection, index, upsert/xoá) |
| Qdrant collection `multimodal__*` | GHI |

Lưu ý vận hành:
- Quyền Mongo: user của service RAG cần `createCollection` + `createIndex` + đọc/ghi trên `multimodal_rag_documents` trong `organization_db` (`_ensure_indexes` tạo index `status`, `stage`; lỗi chỉ log warning), và chỉ cần `find` trên `documents`.
- Backup/restore/drop `organization_db` ảnh hưởng cả backend lẫn RAG; khi chỉ muốn xoá dữ liệu RAG, drop collection `multimodal_rag_documents` + collection Qdrant + prefix MinIO `rag-assets/`, không drop database.

### Liên kết với organization_units / users

Mỗi dòng registry được ghi thêm các field liên kết khi `indexed` (`ami_rag/sources.py::doc_link_fields`, truyền qua `meta=` vào `mark_indexed`; dòng `pending`/`processing`/`failed` chưa có):

| Field registry | Kiểu | Liên kết tới |
|---|---|---|
| `_id` | string | `documents._id` dưới dạng chuỗi |
| `document_oid` | ObjectId | `documents._id` (dùng cho `$lookup`) |
| `organization_unit_id` | giữ kiểu gốc của `documents` (ObjectId) | `organization_units._id` |
| `owner_id` | string | định danh OIDC: khớp `users.sub` / `users.preferred_username` / `users.email`, KHÔNG phải `users._id` (xem `ami_data/.../app/repositories/user_repository.py`, `find_all_users_with_upload_count`) |
| `document_type`, `title` | string | sao chép từ `documents` |

`$lookup` chỉ chạy trong cùng một database, vì vậy RAG phải dùng chung `organization_db`. Ví dụ:

```javascript
db.multimodal_rag_documents.aggregate([
  { $match: { status: "indexed" } },
  { $lookup: { from: "documents", localField: "document_oid", foreignField: "_id", as: "doc" } },
  { $lookup: { from: "organization_units", localField: "organization_unit_id", foreignField: "_id", as: "unit" } },
  { $lookup: {
      from: "users",
      let: { owner: "$owner_id" },
      pipeline: [
        { $match: { $expr: { $and: [
            { $ne: ["$$owner", null] },
            { $or: [
              { $eq: ["$sub", "$$owner"] },
              { $eq: ["$preferred_username", "$$owner"] },
              { $eq: ["$email", "$$owner"] } ] } ] } } }
      ],
      as: "owner" } },
  { $project: { status: 1, title: 1, "unit.name": 1, "owner.preferred_username": 1, "doc.status": 1 } }
])
```

(Tên field hiển thị như `unit.name` tuỳ schema `organization_units` của backend; điều chỉnh cho khớp.)

### Field của `multimodal_rag_documents`

Các field (`ami_rag/storage/doc_status.py`):

| Field | Ý nghĩa |
|---|---|
| `_id` | document id |
| `document_oid`, `organization_unit_id`, `owner_id`, `document_type`, `title` | field liên kết (bảng trên), chỉ có sau lần `indexed` |
| `status` | `pending` / `processing` / `indexed` / `failed` / `stale`; legacy `processed` (pipeline cũ) đọc là `stale` |
| `stage` | stage hiện tại: `parse` / `describe` / `chunk` / `embed` / `indexed` |
| `source`, `content_hash` | nguồn (`mongo_text`/`minio_parse`) và vân tay nội dung |
| `source_path`, `file_path` | key MinIO của file gốc và đường dẫn citation |
| `parser` | giá trị `PARSER` lúc nạp |
| `assets` | danh sách key MinIO đã upload |
| `counts`, `page_count` | số item theo modality (`text/image/table/equation/other`), số trang |
| `chunk_count`, `embed_model`, `embed_dim`, `chunker_version` | thông tin vector (để kiểm tra tương thích khi reindex) |
| `attempts`, `error`, `error_stage` | theo dõi lần thử/lỗi (`error` cắt 2000 ký tự) |
| `created_at`, `updated_at` | thời điểm |

## 4. Biến môi trường

File mẫu: `.env.ami.example` (copy thành `.env`). Nguồn sự thật: `ami_rag/settings.py`.

| Nhóm | Biến | Mặc định | Ý nghĩa |
|---|---|---|---|
| LLM | `QWEN_LLM_BASE_URL` / `QWEN_LLM_MODEL` / `QWEN_LLM_API_KEY` | `http://vllm:8000/v1` / `Qwen/Qwen3-32B` / rỗng | qwen-selfhost (OpenAI-compatible), dùng cho `describe_func` |
| | `QWEN_VLM_MODEL` | rỗng | rỗng = dùng `QWEN_LLM_MODEL` |
| Embed server | `EMBED_SERVER_URL` | `http://localhost:8080` | trỏ đúng server tương ứng `EMBED_BACKEND` (gateway nemotron-vl-vllm: 8080; qwen-embedding-server: 8007) |
| | `EMBED_SERVER_TOKEN` | rỗng | **chỉ từ env**, không ghi vào file trong git |
| | `EMBED_MODEL` | `nvidia/llama-nemotron-embed-vl-1b-v2` | dùng để xác minh với server lúc handshake |
| | `EMBED_BACKEND` | `auto` | `auto` / `custom` (qwen-embedding-server: `/info` + `/embed`) / `openai` (gateway nemotron-vl-vllm: `/health` + `/v1/embeddings` với `{"input", "input_type"}`); auto: prefix `Qwen/` → custom, còn lại → openai |
| | `EMBED_DIM` | `2048` | để xác minh (dim thực tế do server quyết định) |
| | `EMBED_TIMEOUT` | `60` | giây |
| | `EMBED_BATCH_SIZE` | `32` | số item mỗi request |
| | `EMBED_MAX_CONCURRENCY` | `4` | số request đồng thời |
| | `EMBED_RETRIES` | `3` | retry cho lỗi tạm thời (timeout/5xx/429) |
| | `EMBED_MAX_INPUT_TOKENS` | `6000` | budget token mỗi item embed (máy A không tokenizer, xấp xỉ ~3 chars/token); item vượt bị cắt đầu trước khi gửi — content đầy đủ vẫn lưu Qdrant. Default tính cho gateway `max_model_len=8192`; server chạy 4096 thì giảm qua env |
| | `EMBED_IMAGE_TOKEN_RESERVE` | `1792` | một ảnh Nemotron VL tốn tối đa ~1792 visual token (6 tile + thumbnail) — budget text kèm ảnh bị trừ trước |
| | `EMBED_CACHE_ENABLED` | `true` | cache embedding SQLite ở máy A |
| | `EMBED_CACHE_PATH` | `./embed_cache.db` | |
| Mongo | `MONGO_URI` | `mongodb://localhost:27017/?directConnection=true` | |
| | `RAG_DB` | `organization_db` | dùng chung DB với backend |
| | `RAG_DOCUMENTS_COLLECTION` | `multimodal_rag_documents` | registry / DocStatusStore |
| | `ORG_DB` / `DOC_COLLECTION` | `organization_db` / `documents` | chỉ đọc |
| Qdrant | `QDRANT_URL` / `QDRANT_API_KEY` | `http://localhost:6333` / rỗng | |
| | `QDRANT_UPSERT_MAX_MB` | `16` | giới hạn ước lượng byte mỗi request upsert (Qdrant server mặc định giới hạn request 32 MB); batch được đóng gói theo byte VÀ số điểm |
| Index | `WORKSPACE` | `multimodal` | tiền tố collection Qdrant `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}` |
| Parser | `PARSER` | `mineru` | |
| | `PARSE_METHOD` | `auto` | |
| | `PARSER_OUTPUT_DIR` | `./output` | |
| MinerU (chỉ khi `PARSER=mineru`) | `MINERU_BACKEND` | `pipeline` | truyền `-b`; xem mục 5.1 vì sao không bỏ trống |
| | `MINERU_DEVICE` | rỗng | rỗng = tự nhận; `cuda` / `cuda:0` / `cpu`; map sang env `MINERU_DEVICE_MODE` của tiến trình mineru |
| | `MINERU_VIRTUAL_VRAM_SIZE` | `0` | GB; `0` = dùng VRAM thật; >0 map sang env cùng tên, MinerU chọn batch size như thể GPU có chừng đó VRAM |
| | `MINERU_LANG` | rỗng | gợi ý ngôn ngữ OCR (`ch`, `en`, `latin`, ...); rỗng = mặc định MinerU |
| | `MINERU_SOURCE` | rỗng | nguồn model: `huggingface` / `modelscope` / `local`; rỗng = mặc định |
| | `MINERU_TIMEOUT` | `1800` | giây/tài liệu; `0` = không giới hạn |
| Chunking | `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `100` | token |
| Queue | `REDIS_URL` | `redis://localhost:6379/0` | |
| | `RAG_STREAM` | `rag:ingest` | |
| | `RAG_CONSUMER_GROUP` | `ami-rag` | |
| | `RAG_CONSUMER_NAME` | rỗng | rỗng = tự sinh |
| Worker | `WORKER_ENABLED` | `true` | `true`: API nhúng worker; compose đặt `false` cho cả hai container |
| | `WORKER_BATCH` | `10` | |
| | `WORKER_POLL_BLOCK_MS` | `5000` | |
| | `WORKER_MAX_DELIVERY` | `3` | |
| | `WORKER_RETRY_IDLE_MS` | `600000` | Sau bao lâu (ms) message pending lỗi được giao lại |
| | `WORKER_METRICS_PORT` | `9109` | `/metrics` của `ami-rag-worker` |
| Rerank | `RERANK_BASE_URL` | `http://localhost:8080` | gateway nemotron-vl-vllm phục vụ cả embed + rerank |
| | `RERANK_MODEL` | `nvidia/llama-nemotron-rerank-vl-1b-v2` | rỗng hoặc chứa `bge` → legacy BGE (documents text-only) |
| | `RERANK_BACKEND` | `auto` | `auto` / `legacy` (BGE) / `vllm` (Nemotron VL, multimodal) |
| | `RERANK_MULTIMODAL` | `true` | chỉ với backend `vllm`: chunk image/table/equation có asset_key kèm ảnh (data URI từ MinIO) + text |
| | `RERANK_TOP_K` | `5` | số tài liệu trả về mặc định khi request không chỉ định `top_k` |
| | `RERANK_TIMEOUT` | `60` | giây |
| | `RERANK_MAX_INPUT_TOKENS` | `6000` | budget token text mỗi document rerank; vượt bị cắt đầu (kèm ảnh → trừ `RERANK_IMAGE_TOKEN_RESERVE`) — rerank server cùng vLLM giới hạn `max_model_len` |
| | `RERANK_IMAGE_TOKEN_RESERVE` | `1792` | như `EMBED_IMAGE_TOKEN_RESERVE` |
| Retrieval | `RETRIEVAL_CHUNK_TOP_K` | `40` | số chunk tối thiểu lấy từ Qdrant trước khi rerank |
| | `RETRIEVAL_OVERFETCH` | `4` | hệ số nhân lấy ứng viên: `max(RETRIEVAL_CHUNK_TOP_K, top_k * RETRIEVAL_OVERFETCH)` để sau khi rerank vẫn đủ `top_k` kết quả |
| | `RETRIEVAL_FUSION_MODE` | `single` | `single` = một pool chung (retrieve không filter + 1 query ảnh top-5, 1 lệnh rerank). `calibrated` \| `rrf` \| `quota` = retrieve + rerank **2 nhánh** rồi fuse xếp hạng (xem [6.1](#61-two-branch-retrieve--rerank--fusion-experimental)) |
| | `RETRIEVAL_FUSION_POOLS` | `text+table,image` | pool = nhóm modality, phân cách bằng `+`. Văn xuất và bảng dùng chung 1 lệnh rerank; ảnh tách riêng. Modality không thuộc pool nào vào pool `other` |
| | `RETRIEVAL_FUSION_POOL_SIZES` | `text=50,image=15` | sâu retrieve từng pool, key là **tên pool** (`text`, không phải `text+table`). Cần ≥ 50 vì `table_08` ở vector rank 44 trong pool gộp |
| | `RETRIEVAL_FUSION_VL_POOLS` | `text,image` | pool nào rerank **kèm ảnh render**. `text,image` = bảng cũng gửi ảnh; `image` = bảng rerank text thuần → **mất 2 hit bảng** (3/9) |
| | `RETRIEVAL_FUSION_IMAGE_GATE` | `true` | cổng lọc ảnh: ảnh chỉ vào nếu điểm ≥ điểm text tại đường cắt |
| | `RETRIEVAL_FUSION_RRF_K` | `60` | chỉ dùng cho mode `rrf`; với pool rời nhau + weight bằng nhau thì `k` **không** đổi thứ tự |
| | `RETRIEVAL_FUSION_QUOTA` | `text=3,image=2` | chỉ dùng cho mode `quota`: `tên_pool=slots[@ngưỡng]`. **Bỏ ngưỡng** là cấu hình đo tốt nhất (22/28); thêm ngưỡng rơi về 21/28 |
| Pipeline/CLI | `CHUNKER_VERSION` | `v1` | gắn vào collection + registry để kiểm tra tương thích lúc reindex |
| | `CLI_STUCK_PROCESSING_MINUTES` | `60` | doc `processing` quá lâu bị `status` báo là treo |
| MinIO | `MINIO_ENDPOINT` | `localhost:9000` | |
| | `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` / rỗng | |
| | `MINIO_BUCKET` | `ami-data-documents` | |
| | `MINIO_SECURE` | `false` | |
| | `ASSET_PREFIX` | `rag-assets` | |
| | `CRAWL_IMAGES_PREFIX` / `CRAWL_IMAGES_ENABLED` | `crawl-images` / `false` | |
| | `MINIO_PRESIGN_EXPIRES` | `3600` | giây, cho `artifact_url`/`file_url` |
| API | `API_HOST` / `API_PORT` | `0.0.0.0` / `8009` | |
| | `RAG_API_KEY` | rỗng | rỗng = không auth |
| Tracing | `OTEL_EXPORTER_OTLP_ENDPOINT` | rỗng | rỗng = tracing tắt; đọc từ `os.environ`. Chỉ API dùng; compose để trống cho worker. Trỏ tới OTLP/HTTP của Tempo riêng `tempo-multimodal-rag` (xem 9) |
| | `OTEL_SERVICE_NAME` | `multimodal-rag-retrieval` | |
| | `TRACE_OUTPUT_MAX_LEN` | `2000000` | giới hạn ký tự event `documents_json`; `0` = không cắt |

Lưu ý: `OTEL_*` và `TRACE_OUTPUT_MAX_LEN` do `observability.py` đọc trực tiếp từ `os.environ` (không qua `Settings`). Docker Compose nạp `.env` vào môi trường container nên hoạt động; khi chạy local bằng `make start_backend`, cần `export` các biến này trong shell.

## 5. Chạy local và Docker

Local (cần Redis, Mongo, Qdrant, MinIO, rerank service, embed server sẵn sàng):

```bash
make setup                    # uv sync --extra service
cp .env.ami.example .env      # điền EMBED_SERVER_TOKEN, MINIO_SECRET_KEY, hosts...
make start_backend            # API :8009 (uvicorn --reload)   | hoặc: ami-rag-api
make start_worker             # ingest worker                  | hoặc: ami-rag-worker
make test                     # pytest -v tests/ami_service/
make style / make lint        # ruff trên ami_rag và tests/ami_service
```

Docker (`docker-compose.ami.yml`, `Dockerfile.ami`, `python:3.12-slim`):

```bash
cp .env.ami.example .env      # host Mongo/Qdrant/Redis/MinIO/rerank/embed = tên container trên ami-network
make start_docker             # tạo network ami-network nếu thiếu, rồi up --build -d
make ps && make health        # /healthz + đếm series multimodal_rag_retrieval_* và multimodal_rag_ingest_*
make logs SERVICE=ami-rag-worker
make restart | make down | make docker-clean
```

- Build image chậm (~20 phút khi đổi code): `Dockerfile.ami` chép `ami_rag/` trước `pip install ".[service]"` nên mỗi lần sửa code `pip` tải lại toàn bộ gói.
- Hai service dùng chung image: `ami-rag-api` (port 8009) và `ami-rag-worker` (port 9109 cho `/metrics`). Mongo, Qdrant, Redis, MinIO, rerank chạy ngoài compose trên network external `ami-network`; trong container, host trong `.env` phải là tên container, không phải `localhost`.
- GPU: `ami-rag-worker` có `deploy.resources.reservations.devices` (nvidia, count 1) cho MinerU. Bỏ khối này để parse bằng CPU (xem 5.1). Embed không cần GPU ở máy A.
- Cache model MinerU: volume `mineru-models` mount tại `/models` (`HF_HOME=/models/huggingface`, `MINERU_MODEL_SOURCE=huggingface`), tránh tải lại mỗi lần rebuild. Volume `rag-output` mount tại `/app/output`.
- LibreOffice (`libreoffice-writer`) đã cài trong image để chuyển doc/docx sang PDF trước khi MinerU parse; chạy local cần tự cài LibreOffice.
- Biến Make ghi đè: `PORT`, `WORKER_METRICS_PORT`, `COMPOSE_FILE`, `SERVICE`, `NETWORK`.

### 5.1 MinerU: backend, GPU/VRAM, chạy CPU

Worker gọi `parse_document(..., **parser_kwargs(settings))` (`ami_rag/workers/ingest_worker.py`, `ami_rag/settings.py::parser_kwargs`). Hàm này đổi `MINERU_BACKEND/LANG/SOURCE/TIMEOUT` thành kwargs `backend/lang/source/timeout`, còn `MINERU_DEVICE` và `MINERU_VIRTUAL_VRAM_SIZE` thành `env` của tiến trình `mineru` (`MINERU_DEVICE_MODE`, `MINERU_VIRTUAL_VRAM_SIZE`). Biến rỗng/`0` thì không truyền. Parser khác `mineru` thì trả `{}`.

**Vì sao phải đặt `MINERU_BACKEND=pipeline`**: MinerU 3.4.x khi không có `-b` mặc định dùng `hybrid-engine`, chậm hơn ~6 lần và tốn VRAM gấp ~8 lần `pipeline`. Đừng để trống.

Số đo trên Colab T4, PDF 10 trang, MinerU 3.x:

| Cấu hình | Thời gian | VRAM đỉnh |
|---|---|---|
| `pipeline`, GPU auto (không đặt `MINERU_VIRTUAL_VRAM_SIZE`) | 50.5 s | 1821 MiB |
| `pipeline`, `MINERU_VIRTUAL_VRAM_SIZE=4` | 57.0 s | 1135 MiB |
| `pipeline`, `MINERU_VIRTUAL_VRAM_SIZE=8` | 49.8 s | 1817 MiB |
| `pipeline`, `MINERU_VIRTUAL_VRAM_SIZE=16` | 47.4 s | 3369 MiB |
| `hybrid-engine` (mặc định MinerU khi không có `-b`) | 286 s | 14539 MiB |
| Đường thư viện `parse_document` (GPU auto) | 46.4 s | 1821 MiB |
| CPU (3 trang) | 82.7 s (~27 s/trang) | n/a |
| GPU | ~5 s/trang | |

Khuyến nghị:
- `pipeline` dưới 4 GiB VRAM nên chạy GPU trực tiếp (mặc định, `MINERU_DEVICE` rỗng).
- GPU dùng chung với service khác (reranker, vLLM, ...): đặt `MINERU_VIRTUAL_VRAM_SIZE=4` (đỉnh ~1.1 GiB, chậm hơn nhẹ). Giá trị lớn hơn làm MinerU dùng batch lớn hơn nên tốn thêm VRAM, không nhanh hơn đáng kể.
- Chạy CPU: đặt `MINERU_DEVICE=cpu` và bỏ khối `deploy.resources.reservations.devices` của `ami-rag-worker` trong `docker-compose.ami.yml`. Chậm ~5 lần (~27 s/trang): tăng `MINERU_TIMEOUT` cho file dài và `WORKER_RETRY_IDLE_MS` cho đủ lớn hơn thời gian parse.

### 5.2 Embed server (máy B) và Redis (đã gặp thực tế)

- Embedding + rerank chạy trên **server riêng** ở máy B, chọn qua `EMBED_BACKEND`:
  - **`openai` (mặc định)**: repo `nemotron-vl-vllm` (máy B, port 8080) — gateway FastAPI trước 2 vLLM (`nvidia/llama-nemotron-embed-vl-1b-v2` + `nvidia/llama-nemotron-rerank-vl-1b-v2`, vLLM ≥ 0.17.0 với template override; gateway nhận text/ảnh base64/bảng, fan-out batch song song xuống vLLM). Contract gateway: `GET /health` (200 khi cả hai sẵn sàng), `POST /v1/embeddings` với `{"input": [items], "input_type": "query"|"document"}` (item `str` | `{"text":..., "image": <base64>}` | `{"content": [parts]}`), `POST /v1/rerank` (alias `/rerank`) với `{query, documents, top_n}` → `{results: [{index, relevance_score}]}`. Client `OpenAIEmbedder` (`ami_rag/core/openai_embedder.py`): handshake `/health` + probe embed 1 lần lấy dim, batch `EMBED_BATCH_SIZE` item/request.
  - **`custom`**: `qwen-embedding-server/` (repo ngoài RAG-Anything, port 8007), model `Qwen/Qwen3-VL-Embedding-2B`: dim 2048 (MRL 64–2048), last-token pooling + L2 normalize, instruction qua system message, `transformers>=4.57`/`qwen-vl-utils>=0.0.14`. Tổng quan: `docs/qwen_embedding_notes.md`.
- Rerank mặc định qua gateway `nemotron-vl-vllm` (documents text hoặc multimodal `{"content": [text/image_url parts]}`): chunk image/table/equation có `asset_key` được kèm ảnh render từ MinIO (`RERANK_MULTIMODAL`); thiếu ảnh thì rơi về text. Score là logit (có thể âm), chỉ dùng so thứ hạng — giống legacy BGE.
- `OpenAIEmbedder`/`RemoteEmbedder` handshake lúc khởi tạo: so model (và dim — probe embed 1 lần với `OpenAIEmbedder`); lệch thì dừng với lỗi rõ ràng (không ghi vector khi chưa xác minh). Mỗi response được kiểm tra model để phát hiện server bị đổi model giữa chừng.
- Retry chỉ cho lỗi tạm thời (timeout, 5xx, 429, mất kết nối), backoff + jitter; **không retry** 4xx. Circuit breaker: 5 lô liên tiếp không với tới server thì dừng cả lô sớm (doc chưa xử lý không bị đánh fail hàng loạt). Lỗi `embed server unreachable` trong một doc không làm hỏng doc khác.
- Cache embedding SQLite ở máy A (`EMBED_CACHE_ENABLED`, key = `model|dim|instruction_ns|sha256(text)`); `reindex --dry-run` dùng `count_cache_hits` để báo trước số chunk trúng cache, không gọi HTTP.
- **Kích thước input gửi lên máy B** (đã gặp thực tế: vLLM từ chối `decoder prompt ... longer than the maximum model length` với HTTP 400):
  - Gateway vLLM giới hạn `max_model_len` (khuyến nghị đặt `EMBED/RERANK_MAX_MODEL_LEN=8192` trong `.env` máy B, kèm `*_MAX_NUM_BATCHED_TOKENS >= 8192`; một ảnh tốn tối đa ~1792 visual token).
  - Client cắt đầu (head-truncate) item vượt budget trước khi gửi: embed qua `EMBED_MAX_INPUT_TOKENS` (trừ `EMBED_IMAGE_TOKEN_RESERVE` khi kèm ảnh), rerank qua `RERANK_MAX_INPUT_TOKENS`. Content đầy đủ vẫn lưu Qdrant payload nên LLM vẫn thấy toàn văn khi trả lời; chỉ vector/rerank tính trên phần đầu.
  - Qdrant giới hạn request (mặc định 32 MB): `QDRANT_UPSERT_MAX_MB=16` — upsert đóng gói batch theo byte ước lượng (payload + vector) và số điểm; doc nhiều bảng lớn (payload `content` + `table_body`) không còn gộp vượt giới hạn trong một request.
- Hết credit/quota hay model quá tải trên server (5xx sau retry) → doc ở stage `embed` thành `failed` với lý do rõ ràng; chạy lại `retry` sau khi server ổn định.
- Redis: `redis-py` ≥ 8 mặc định `socket_timeout=5s`, trùng với `XREADGROUP BLOCK` (`WORKER_POLL_BLOCK_MS`) nên worker báo `queue read failed: Timeout reading from redis`. Client của worker đặt `socket_timeout = WORKER_POLL_BLOCK_MS/1000 + 5` và `health_check_interval=30` (`ingest_worker.py::_build_default_deps`).

## 6. API

Mọi route `/v2/rag/*` và `/admin/*` yêu cầu `Authorization: Bearer <RAG_API_KEY>` khi `RAG_API_KEY` khác rỗng (rỗng = mở; nên chỉ dùng trong mạng nội bộ). `/`, `/healthz`, `/readyz`, `/metrics` không yêu cầu auth; chặn `/metrics` ở reverse proxy nếu cần.

Chỉ còn **v2**: request chỉ gồm `messages`, `top_k`, `include_references`. Đã bỏ v1 response, bỏ `version`/`mode`/`filters`/`include_kg`.

### POST /v2/rag/

Request:

```json
{
  "messages": [{"role": "user", "content": "bảng học phí các ngành"}],
  "top_k": 10,
  "include_references": true
}
```

Response (rút gọn; mỗi document có đủ các field, giá trị không áp dụng là `null`):

```json
{
  "query": "bảng học phí các ngành",
  "documents": [
    { "text": "Đoạn văn mô tả học phí ...", "score": 0.93, "modality": "text",
      "artifact_url": null, "table_body": null, "page": null, "caption": null,
      "reference_id": "<chunk_id>",
      "metadata": {"source": "665f..._hocphi.pdf", "page": null, "document_id": "665f...", "chunk_index": null, "global_id": "<chunk_id>"},
      "doc": {"document_id": "665f...", "title": "Học phí 2026", "document_type": "pdf",
              "organization_unit_id": "...", "file_url": "https://minio/presigned/...", "source_url": null} },
    { "text": "... Asset: rag-assets/665f.../ab12cd34ef567890.png ...", "score": 0.81, "modality": "image",
      "artifact_url": "https://minio/presigned/rag-assets/665f.../ab12cd34ef567890.png",
      "table_body": null, "page": 2, "caption": "Sơ đồ tổ chức", "reference_id": "<chunk_id>", "doc": {"...": "..."} },
    { "text": "...", "score": 0.77, "modality": "table",
      "artifact_url": null, "table_body": "| Ngành | Học phí |\n| --- | --- |\n| CNTT | 30tr |",
      "page": 4, "caption": "Bảng 1. Học phí", "reference_id": "<chunk_id>", "doc": {"...": "..."} }
  ],
  "references": [{"reference_id": "<chunk_id>", "file_path": "665f..._hocphi.pdf"}],
  "meta": {"latency_ms": 820}
}
```

Luồng: `embed_query` (máy B) → vector search Qdrant (`max(RETRIEVAL_CHUNK_TOP_K, top_k * RETRIEVAL_OVERFETCH)` ứng viên) → rerank service → trả `top_k` document đầu.

- `top_k` (1..200) là số document trả về cuối cùng (mặc định `RERANK_TOP_K` khi không gửi).
- `modality`: `text` | `image` | `table` | `equation` (từ `original_type` của chunk multimodal).
- `artifact_url`: presigned URL (hết hạn sau `MINIO_PRESIGN_EXPIRES`) của `asset_key`; `null` khi chunk không có asset hoặc presign lỗi (tăng `presign_failures_total`).
- `references`: dedup theo document, chỉ có khi `include_references: true`.
- Lỗi rerank → trả chunk không điểm (fallback, đo `rerank_fallback_total`).
- Khi bật fusion, `meta` có thêm `fusion: {mode, pools: {pool: số ứng viên}}`.

### 6.1 Two-branch retrieve + rerank + fusion (thử nghiệm)

`RETRIEVAL_FUSION_MODE` khác `single` sẽ **retrieve + rerank theo 2 nhánh**, rồi merge
các bảng xếp hạng (`ami_rag/core/fusion.py`, thuần Python — không numpy/sklearn, giống
`ami_rag/core/calibration.py`):

```
embed_query
  ├─ vector search (filter modality = text|table, depth 50) ─► Rerank ─► list T
  └─ vector search (filter modality = image,       depth 15) ─► VL Rerank ─► list I
                                                                  │
                                          cổng lọc ảnh (gate) ─► fuse ─► top_k
```

Ranh giới tách nằm ở **text-vs-image**, không phải table-vs-image: điểm rerank của văn xuất
và bảng cùng thang đo, còn ảnh thì không — nên chỉ cần một lần sửa thang điểm cho ảnh. Bảng và
text dùng **chung một lệnh rerank**; chỉ nhánh ảnh mới gọi VL reranker.

| mode | cách gộp |
| --- | --- |
| `single` | pool chung, sort theo điểm rerank |
| `calibrated` | sort toàn cục theo P(relevant) của `RerankCalibrator` |
| `rrf` | `weight / (k + rank)` |
| `quota` | reserve slot/pool + ngưỡng, phần dư fill theo điểm calibrated |

#### Kết quả đo (28 case `tests/retrieval_cases.json`, top_k=5)

Cấu hình chung: `text=50`, `image=15`, bảng rerank kèm ảnh, gate bật. Chỉ đổi tham số được ghi.

| cấu hình | Hit@5 | text | table | image | MRR | p50 |
| --- | --- | --- | --- | --- | --- | --- |
| `single` (baseline) | 19/28 | 10/10 | 5/9 | 4/9 | 0.6161 | 876 ms |
| `quota` **không ngưỡng** `text=3,image=2` | **22/28** | 10/10 | **7/9** | 5/9 | 0.644 | 1345 ms |
| `rrf` k=60 | 22/28 | 10/10 | 7/9 | 5/9 | 0.6131 | 1353 ms |
| `calibrated` | 21/28 | 10/10 | 6/9 | 5/9 | 0.637 | 1353 ms |
| `quota` **có ngưỡng** `text=3@0.45,image=2@0.40` | 21/28 | 10/10 | 6/9 | 5/9 | 0.6369 | 1353 ms |
| … `calibrated`, tắt gate | 21/28 | 10/10 | 6/9 | 5/9 | 0.6339 | 1354 ms |
| … `calibrated`, `image=30` | 21/28 | 10/10 | 6/9 | 5/9 | 0.6339 | 2048 ms |
| … `calibrated`, `text=60` | 21/28 | 10/10 | 6/9 | 5/9 | 0.6458 | 1470 ms |
| … `calibrated`, `text=40` | 20/28 | 10/10 | 5/9 | 5/9 | 0.6280 | 1187 ms |
| … `calibrated`, bảng rerank text thuần | 18/28 | 10/10 | **3/9** | 5/9 | 0.5804 | 1249 ms |
| … `calibrated`, tắt calibration | 18/28 | 10/10 | 7/9 | 1/9 | 0.5458 | 1341 ms |

> **Tính tất định**: chạy lại `single` + `calibrated` + `quota` + `rrf` hai lần cho **Hit@5 và
> số hit theo modality giống hệt**, nhưng MRR dao động ±0.01 (reranker hơi không tất định nên
> thứ tự *trong* top 5 đổi). Hãy tin Hit@5, đừng tin chênh lệch MRR dưới ~0.01.

Kết luận đo được:

- **Bảng CẦN ảnh render trong rerank.** Đây là điểm đắt nhất và dễ sai nhất: rerank bảng thuần
  text làm table tụt **6/9 → 3/9** (mất `table_01`, `table_03`), dù raw rerank xếp bảng rất tốt
  (`table_01` rank 2, `table_03` rank 1 trong pool 60) — tức **calibrator hạ bảng**, không phải
  reranker. Nguyên nhân hợp lý: calibrator cho modality `table` được fit trên bảng đã rerank kèm
  ảnh, nên áp cho điểm bảng rerank text thuần là lệch phân phối. Bù lại chỉ tốn ~72 ms.
  ⇒ Đừng tin rằng "bỏ ảnh khỏi rerank bảng là tiết kiệm miễn phí".
- **Ép đa dạng modality thắng sort thuần.** `quota` không ngưỡng và `rrf` đều 22/28, hơn
  `calibrated` 21/28 — cả hai đều **ép** slot cho modality yếu (`quota` reserve cứng, `rrf` với
  pool rời nhau + weight bằng nhau là round-robin theo rank). Cả hai cùng thắp thêm `table_09`,
  mà `calibrated` xếp nó dưới top 5. Nói cách khác: calibrated không sai, nhưng nó **xếp hạng
  thuần theo điểm** nên modality yếu không bao giờ có mặt ở cuối trang.
- **Ngưỡng của quota phá chính quota.** `quota` có ngưỡng rơi về đúng bằng `calibrated` (21/28):
  slot dự phòng bị chặn khi ứng viên dưới ngưỡng, và ứng viên đó lại là ứng viên đúng.
- **Calibration vẫn là tiên quyết**: tắt → 18/28, image 5/9 → 1/9 (dù table lên 7/9, tức ảnh mất
  sạch vì điểm thô của text ~0.9 ≫ image ~0.14).
- **Depth 50 là đủ.** `table_08` nằm ở vector rank 44 trong pool gộp; depth 60 không thêm hit nào
  mà thêm ~120 ms. Cần depth sâu vì trong pool gộp, bảng phải cạnh tranh với text theo điểm vector
  (`table_08` rank 43 trong pool table riêng → rank 44+ trong pool gộp). `image=30` cũng không
  thêm hit mà tốn thêm ~700 ms.
- **Cổng lọc ảnh không đổi kết quả** với `calibrated` (21/28 cả bật lẫn tắt). Lý do nằm ngay
  trong định nghĩa: ảnh dưới đường cắt không thể thắng phép sort toàn cục, nên cổng chỉ **chặn
  việc mang ảnh thừa vào context**, không đổi trang kết quả. Nó *có* đổi với `quota`: ảnh bị
  lọc thì trả lại slot dự phòng cho pool khác. Đừng kỳ vọng nó nâng recall.
- `rrf` với pool rời nhau + weight bằng nhau là round-robin theo rank ⇒ `k` không đổi thứ tự.
- 7/9 case `xfail` vẫn hỏng vì **chunk đích không nằm trong pool**: `table_06/07` (vượt depth 280
  trong pool gộp) và `image_overview_01`/`image_units_01` (vector rank 22/30, cần pool image ≥ 30
  nhưng khi đó rerank rank 12/15 — vẫn không kịp vào top 5).

#### So với thiết kế 3 pool (text / table / image)

Cùng đạt 21–22/28, nhưng 2 nhánh **nhanh hơn 38%** (1345 ms vs 2178 ms) và chỉ còn 2 lệnh rerank thay
vì 3: gộp bảng vào nhánh text nghĩa là pool sâu 50 doc chỉ rerank một lần thay vì hai lần.

#### Trần recall (vector thuần, không rerank)

Đo bằng `scripts/probe_pool_recall.py`, kết quả lưu ở `tests/pool_recall_probe.json`.

| modality | pool | @5 | @15 | @30 | @50 |
| --- | --- | --- | --- | --- | --- |
| text | 4862 | 9/10 | 10/10 | 10/10 | 10/10 |
| table | 413 | 6/9 | 6/9 | 6/9 | 7/9 |
| image | **70** | 4/9 | **7/9** | 9/9 | 9/9 |

Ba điều rút ra:

- Collection chỉ có **70 chunk ảnh**, nên `image=15` đã lấy 21% cả pool và không thể mở rộng case ảnh.
- `table_06` / `table_07` không tới được dù probe depth 50/413. Đã kiểm tra trực tiếp trong
  collection: cả hai chunk tồn tại, đúng `modality=table`, đúng doc và page ⇒ là lỗi vector rank,
  **không phải nhãn sai**. Lý do `xfail` ghi "low semantic alignment" là mô tả sai triệu chứng.
- **Recall@15 ảnh = 7/9 nhưng end-to-end hit ảnh chỉ 5/9**: rerank *nâng* được target từ rank 6–15
  lên top-5. Hai case hỏng là do rerank hạ chúng xuống, không phải thiếu pool — đó là lý do
  `image=30` không thêm hit nào.

#### Chi phí ảnh thừa trên truy vấn thuần text

10 truy vấn có đáp án ở chunk text, `quota text=3,image=2`, top_k=5:

| cổng lọc ảnh | query có ảnh | slot ảnh chiếm | mất chunk đúng **do** ảnh |
| --- | --- | --- | --- |
| **bật** (mặc định) | 1/10 | 2/50 = **4%** | 0 |
| tắt | 10/10 | 20/50 = **40%** | 0 |

Quota reserve 2 slot ảnh cứng, nhưng cổng lọc loại ảnh dưới đường cắt nên slot dự phòng được trả lại
cho pool text — đây là lý do cổng lọc đáng giữ, dù nó không nâng recall.

> Lưu ý: `text_07_thiet_ke_game` không có target trong top-5 **kể cả khi bỏ hết ảnh**. Không quy
> được lỗi này cho việc chèn ảnh; nó là lỗi retrieval riêng.

Bộ dữ liệu chấm tay nằm ở `tests/text_only_ab.json` (sinh bằng `scripts/export_text_only_ab.py`):
mỗi truy vấn có 2 arm lấy từ cùng một lần retrieve, thứ tự arm đảo ngẫu nhiên, kèm rubric và
prompt mẫu để đưa lên web LLM chấm mù.

#### ⚠️ Lỗi dữ liệu: text trong bảng mất dấu tiếng Việt (lỗi MinerU)

Toàn bộ 413 chunk `modality=table` mất ký tự dấu ở **cả** `table_body` và `content`:

| đúng | đã lưu |
| --- | --- |
| Tiếng Anh | Ting Anh |
| Cấu trúc dữ liệu | Cu trúc d liu |
| Kinh tế cơ sở | Tin hc cơ s |
| Giải tích | Gii tích |
| Quản trị giá | Quån tri giá |

Truy từng tầng trên PDF `Sổ tay sinh viên 2026` (266 trang):

| tầng | kết quả |
| --- | --- |
| PDF text layer (pypdfium2, độc lập) | **đúng dấu** |
| `pdftext` 0.7.1 (tầng khai thác char của MinerU) | **đúng dấu** |
| code chuyển đổi của ta | **vô tội** (không có `unicodedata`/strip dấu) |
| MinerU `type=text` | **đúng dấu** |
| MinerU `type=table` → `table_body` | **mất dấu** |

⇒ Chỉ nhánh **nhận dạng bảng** của MinerU hỏng. Không phải lỗi PDF, cũng không phải lỗi service.

> **Cảnh báo**: `PARSE_TABLE=false` **không** phải cách sửa — đo A/B cùng trang cho thấy
> `table_body` dài **0**, tức MinerU xoá sạch nội dung bảng. Bật thì mất dấu, tắt thì mất bảng.

Biến `PARSE_TABLE` (mặc định `true`) tồn tại để hành vi này không còn ẩn trong mặc định MinerU;
không được đổi sang `false` khi chưa có backend thay thế.

#### Đã đo: chữ mất dấu KHÔNG giới hạn retrieval ⇒ không cần reindex

Phép A/B embed lại 389 chunk bảng (có ảnh) theo 3 arm vào collection tạm
(`scripts/probe_table_embedding.py`, kết quả `tests/table_embed_ab.json`):

| arm | @5 | @15 | @30 | @50 | reachable | thời gian embed |
| --- | --- | --- | --- | --- | --- | --- |
| text+image (đang deploy) | 6 | 6 | 6 | 7 | 7 | 37.4s |
| **image only** (bỏ chữ mất dấu) | 6 | 6 | 6 | 7 | 7 | 35.0s |
| text only (không ảnh) | 6 | 7 | 8 | 8 | 8 | 6.0s |

Bỏ hẳn chữ mất dấu **không đổi một điểm nào**, và ảnh render đóng góp **0** cho embedding bảng
trong khi làm chậm **6×**. Nghĩa là thiệt hại thật sự của chữ mất dấu nằm ở **context LLM đọc**,
không phải ở retrieval — nên không đáng reindex toàn bộ corpus chỉ để sửa nó.

Hệ quả cho retrieval: đây là lý do thật sự khiến bảng chỉ dùng được qua ảnh render. Ảnh render là
kênh sạch duy nhất còn lại cho bảng, không phải một lựa chọn tối ưu hoá.