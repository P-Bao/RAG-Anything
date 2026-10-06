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
- LLM (describe): profile `qwen-selfhost` (vLLM, OpenAI-compatible) qua `QWEN_LLM_*`. Gemini không còn được hỗ trợ (đã gỡ). Rerank là service ngoài `POST {RERANK_BASE_URL}/rerank`; lỗi rerank thì trả chunk không điểm (fallback).
- Embedding: model `Qwen/Qwen3-VL-Embedding-2B` chạy trên server riêng (máy B, thư mục `qwen-embedding-server`, repo ngoài RAG-Anything, port 8007). Máy A chỉ gọi HTTP qua `RemoteEmbedder` (`ami_rag/core/remote_embedder.py`) — không cài torch/transformers, không cần GPU cho embed.

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
| Qdrant | `QDRANT_URL` | vector chunk, collection `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}` (ví dụ `multimodal__qwen3-vl-embedding-2b__v1`) |
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
| Embed server | `EMBED_SERVER_URL` | `http://localhost:8007` | |
| | `EMBED_SERVER_TOKEN` | rỗng | **chỉ từ env**, không ghi vào file trong git |
| | `EMBED_MODEL` | `Qwen/Qwen3-VL-Embedding-2B` | dùng để xác minh với server lúc handshake |
| | `EMBED_DIM` | `2048` | để xác minh (dim thực tế do server quyết định) |
| | `EMBED_TIMEOUT` | `60` | giây |
| | `EMBED_BATCH_SIZE` | `32` | số item mỗi request |
| | `EMBED_MAX_CONCURRENCY` | `4` | số request đồng thời |
| | `EMBED_RETRIES` | `3` | retry cho lỗi tạm thời (timeout/5xx/429) |
| | `EMBED_CACHE_ENABLED` | `true` | cache embedding SQLite ở máy A |
| | `EMBED_CACHE_PATH` | `./embed_cache.db` | |
| Mongo | `MONGO_URI` | `mongodb://localhost:27017/?directConnection=true` | |
| | `RAG_DB` | `organization_db` | dùng chung DB với backend |
| | `RAG_DOCUMENTS_COLLECTION` | `multimodal_rag_documents` | registry / DocStatusStore |
| | `ORG_DB` / `DOC_COLLECTION` | `organization_db` / `documents` | chỉ đọc |
| Qdrant | `QDRANT_URL` / `QDRANT_API_KEY` | `http://localhost:6333` / rỗng | |
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
| Rerank | `RERANK_BASE_URL` | `http://localhost:8010` | |
| | `RERANK_TOP_K` | `5` | số tài liệu trả về mặc định khi request không chỉ định `top_k` |
| | `RERANK_TIMEOUT` | `60` | giây |
| Retrieval | `RETRIEVAL_CHUNK_TOP_K` | `40` | số chunk tối thiểu lấy từ Qdrant trước khi rerank |
| | `RETRIEVAL_OVERFETCH` | `4` | hệ số nhân lấy ứng viên: `max(RETRIEVAL_CHUNK_TOP_K, top_k * RETRIEVAL_OVERFETCH)` để sau khi rerank vẫn đủ `top_k` kết quả |
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

- Embedding chạy trên **server riêng** (`qwen-embedding-server/`, repo ngoài RAG-Anything, port 8007). Model `Qwen/Qwen3-VL-Embedding-2B`: dim 2048 (MRL 64–2048), last-token pooling + L2 normalize, instruction qua system message, `transformers>=4.57`/`qwen-vl-utils>=0.0.14`. Tổng quan: `docs/qwen_embedding_notes.md`.
- `RemoteEmbedder` handshake lúc khởi tạo: gọi `/info`, so `model_name`/`dim` với config; lệch thì dừng với lỗi rõ ràng (không ghi vector khi chưa xác minh). Mỗi response `/embed` được kiểm tra `model_name`/`dim` để phát hiện server bị đổi model giữa chừng.
- Retry chỉ cho lỗi tạm thời (timeout, 5xx, 429, mất kết nối), backoff + jitter; **không retry** 4xx. Circuit breaker: 5 lô liên tiếp không với tới server thì dừng cả lô sớm (doc chưa xử lý không bị đánh fail hàng loạt). Lỗi `embed server unreachable` trong một doc không làm hỏng doc khác.
- Cache embedding SQLite ở máy A (`EMBED_CACHE_ENABLED`, key = `model|dim|instruction_ns|sha256(text)`); `reindex --dry-run` dùng `count_cache_hits` để báo trước số chunk trúng cache, không gọi HTTP.
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

### POST /v2/rag/stream

Cùng request, trả NDJSON (`application/x-ndjson`): `{"status": "retrieving"}`, rồi mỗi document một dòng `{"documents": [{"text", "metadata", "score", "modality", "artifact_url"}]}`, cuối cùng `{"status": "done"}`.

### Admin (`/admin/*`)

| Endpoint | Mô tả |
|---|---|
| `GET /admin/pipeline_status` | `{workspace, doc_status_counts (DocStatusStore), queue_pending}` |
| `POST /admin/reprocess_failed` | publish lại event `updated` cho mọi doc `failed` trong registry → `{requeued}` |
| `POST /admin/reindex` | body `{"all": true}` hoặc `{"document_ids": [...]}`; publish event `updated` → `{published}` |
| `GET /admin/documents/{id}` | trạng thái DocStatusStore: `status, stage, source, counts, page_count, parser, document_type, title, organization_unit_id, owner_id, error, updated_at` (ObjectId trả về dạng string, `null` nếu dòng chưa `indexed`); 404 nếu chưa có |
| `GET /admin/documents/{id}/content` | từ `content_list.json` trên MinIO: `{document_id, markdown, blocks, tables}` (block có `type`, `page_idx`, `text`/`table_body`/`caption`/`latex`, `asset_url`); 404 nếu chưa có |

Khác: `GET /healthz`, `GET /readyz`, `GET /metrics`.

## 7. CLI `ami-rag`

CLI phục vụ 4 việc: xem trạng thái doc, chạy lại doc lỗi/hỏng, reindex thủ công, dọn collection legacy.

```bash
ami-rag status [--failed] [--stale] [--doc <id|path>] [--check-embed-server]
ami-rag retry (--doc <id|path> [...] | --all-failed) [--from-stage S] [--max-attempts N] [--yes]
ami-rag reindex [--stale | --all | --doc <id|path> [...]] [--scan] [--from-stage S] [--dry-run] [--yes]
ami-rag cleanup [--dry-run] [--mongo-only | --qdrant-only] [--yes]
```

### `status` — xem trạng thái (mặc định chỉ đọc local: 0 LLM, 0 embed, 0 model call)

In workspace, config đang dùng (`embed_model`/`embed_dim`/`chunker_version`), collection đang dùng theo quy ước `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}`, số doc theo trạng thái, doc thiếu bản ghi (báo `pending`), doc `processing` quá lâu (`CLI_STUCK_PROCESSING_MINUTES`, mặc định 60 phút) được báo `TREO`.

| Cờ | Ý nghĩa |
|---|---|
| `--failed` | chỉ hiện doc `failed` (kèm stage lỗi, error, attempts, updated_at) |
| `--stale` | chỉ hiện doc `stale` (lệch embed_model/chunker_version, file đổi hash, pipeline cũ `processed`) |
| `--doc <id\|path>` | xem chi tiết một doc |
| `--check-embed-server` | gọi `/health` + `/info` của embed server: sẵn sàng không, đúng model/dim theo config không |

### `retry` — chạy lại doc `failed` từ stage lỗi (bằng dữ liệu trung gian đã lưu)

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--doc <id\|path> [...]` / `--all-failed` | - | chọn doc để retry (không phải `failed` thì bỏ qua) |
| `--from-stage` | stage lỗi đã lưu | `parse`/`describe`/`chunk`/`embed`; ép làm lại từ stage đó |
| `--max-attempts` | `3` | chỉ retry doc có `attempts < N` |
| `--yes` | hỏi xác nhận | bỏ qua xác nhận |
| `--embed-server-url` / `--embed-batch-size` | - | override tối thiểu |

- Trước khi chạy, kiểm tra embed server (health + handshake); không với tới thì **dừng sớm** (không đánh fail từng doc).
- Một doc lỗi không dừng cả lô; cuối lệnh in tổng kết thành công/thất bại/bỏ qua.
- Lockfile `./.ami_rag_cli.lock` chống chạy hai `retry`/`reindex` cùng lúc trên một kho dữ liệu.

### `reindex` — xử lý lại thủ công (chunk → embed từ dữ liệu parse + mô tả modal đã lưu)

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--stale` | **mặc định** | chỉ doc `stale` |
| `--all` | - | mọi doc có bản ghi |
| `--scan` | - | quét `organization_db.documents` active: tạo `pending` cho doc chưa có bản ghi, rồi xử lý pending + stale (dùng cho migration từ pipeline cũ — chỉ đọc Mongo, 0 LLM/0 embed) |
| `--doc <id\|path> [...]` | - | doc cụ thể |
| `--from-stage` | theo từng doc | `chunk` khi `content_list.json` đã có trong MinIO, full parse khi thiếu; giá trị explicit override tất cả |
| `--dry-run` | - | in số doc, số chunk cần embed, số chunk trúng cache; không gọi embed, không ghi gì |
| `--yes` | hỏi xác nhận | bỏ qua xác nhận |
| `--embed-server-url` / `--embed-batch-size` | - | override tối thiểu |

- Trước khi embed, kiểm tra tương thích `embed_model`/`embed_dim`/`chunker_version` với collection đích (qua preflight `/info`).
- Khoá bất đồng bộ: lệnh từ chối chạy nếu một `retry`/`reindex` khác đang giữ lockfile.

Mã thoát: `0` xong, `1` một số doc thất bại, `2` dừng sớm (preflight embed server, lockfile, validation).

### `cleanup` — kiểm tra + xoá collection legacy của pipeline LightRAG cũ

Sau migration, các collection của LightRAG không còn được đọc/ghi nhưng vẫn chiếm chỗ. Lệnh dò và xoá chúng (`ami_rag/core/legacy_cleanup.py`):

- **Mongo** (`RAG_DB`): mọi collection `{WORKSPACE}_*` trừ registry `RAG_DOCUMENTS_COLLECTION` — tức `multimodal_full_docs`, `multimodal_text_chunks`, `multimodal_llm_response_cache`, `multimodal_doc_status`, `multimodal_chunk_entity_relation`(+`_edges`), `multimodal_parse_cache`, ... Registry được giữ nguyên vì DocStatusStore còn đọc dòng legacy `processed` (như `stale`) để `reindex --scan`.
- **Qdrant**: mọi collection `{WORKSPACE}_*` **một gạch** (`multimodal_chunks`, `multimodal_entities`, `multimodal_relationships`, ...). Các collection `{WORKSPACE}__*` (hai gạch — quy ước vector pipeline, gồm cả embed model/chunker version cũ để rollback) luôn giữ.
- Collection ngoài tiền tố workspace (dịch vụ khác dùng chung Mongo/Qdrant) không bao giờ bị đụng.

| Cờ | Ý nghĩa |
|---|---|
| `--dry-run` | chỉ liệt kê collection legacy kèm số document/points, không xoá |
| `--yes` | bỏ qua xác nhận (mặc định hỏi trước khi xoá — không hoàn tác được) |
| `--mongo-only` / `--qdrant-only` | giới hạn dò/xoá một loại DB |

Mã thoát: `0` xong (hoặc không có gì xoá), `1` một số collection xoá lỗi, `2` không kết nối được Mongo/Qdrant.

Ví dụ:

```bash
ami-rag cleanup --dry-run       # xem cái gì sẽ bị xoá
ami-rag cleanup --yes           # xoá hết legacy (sau khi reindex --scan thành công)
ami-rag cleanup --qdrant-only   # chỉ xoá collection Qdrant legacy
```

Lưu ý: MinIO `rag-assets/` và cache embedding SQLite **không thuộc** cleanup — assets/content_list.json vẫn dùng cho resume, cache embedding key theo model nên tự vô hiệu khi đổi model. Thư mục local `rag_storage/` của LightRAG cũ (nếu còn trên đĩa) xoá tay.

### 7.1 Chạy CLI khi dùng Docker

`docker-compose.ami.yml` chỉ có hai service: `ami-rag-api` và `ami-rag-worker`. Không có service `ami-rag` riêng; `ami-rag` là lệnh bên trong image (entry point `ami_rag.cli:main`). Chạy lệnh trong container đang chạy, dùng cùng `.env` (Mongo, Redis, Qdrant, MinIO, embed server) với service:

```bash
# từ thư mục gốc repo; -T tắt TTY (cần khi pipe/tee/cron, vô hại khi chạy tay)
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag status
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag status --failed
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag status --check-embed-server
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag reindex --stale --dry-run
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag reindex --all --yes
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag retry --all-failed --yes
```

Rút gọn bằng Makefile (`make cli`, chạy `ami-rag <lệnh con>` trong `ami-rag-worker`):

```bash
make cli status
make cli status --failed
make cli -- reindex --stale --dry-run      # cờ dạng --xxx cần "--" để make không tự parse
make cli -- reindex --all --yes
make cli -- retry --all-failed --yes
make cli CLI_SERVICE=ami-rag-api status    # đổi container
```

Chạy nền cho lượng dữ liệu lớn (`reindex --all` trên cả nghìn doc): thêm `BG=1`. Lệnh chạy detached trong container (`exec -d`), sống tiếp khi đóng terminal/SSH; log ghi vào `/app/output/cli/cli-<thời gian>.log` (volume `rag-output`, symlink `latest.log`), dòng cuối `# exit=<mã>`.

```bash
make cli BG=1 -- reindex --all --yes    # in đường dẫn log, trả về ngay
make cli-logs                               # tail -f log của lệnh nền gần nhất (Ctrl-C chỉ dừng tail)
make cli-ps                                 # liệt kê mọi tiến trình `ami-rag <lệnh>` đang chạy (pid, thời gian)
make cli-stop                               # dừng lệnh nền gần nhất (chỉ pid đó)
make cli-stop PID=<pid>                     # dừng pid lấy từ cli-ps
```

`cli-stop` chỉ kill đúng pid đã ghi, không đụng worker/API hay phiên `ami-rag` mở tay khác. Dừng giữa chừng an toàn vì `retry`/`reindex` idempotent (stage embed luôn `delete_by_doc` trước khi upsert). `retry`/`reindex` bị khoá bởi lockfile `./.ami_rag_cli.lock`, nên không chạy hai lệnh này song song được.

Không có `--`, `make cli reindex --all` báo `unrecognized option '--all'` (lệnh con không cờ như `status` thì không cần). Target kiểm tra container đang chạy theo `docker compose exec`, nên cần `make start_docker` trước.

Sai thường gặp: `docker compose exec ami-rag reindex --all` lỗi vì `ami-rag` là tên lệnh chứ không phải tên service. Đúng: `exec <service> ami-rag <lệnh con>`, với `<service>` là `ami-rag-worker` hoặc `ami-rag-api`.

Chọn container:

| Lệnh | Nên chạy trong | Lý do |
|---|---|---|
| `status` | `ami-rag-worker` hoặc `ami-rag-api` | chỉ đọc Mongo; `--check-embed-server` cần `.env` trỏ đúng embed server |
| `retry`, `reindex` | `ami-rag-worker` | chạy trực tiếp trong tiến trình CLI; cần tới embed server + Mongo của worker |

Lưu ý:
- Lệnh `exec` dài nên chạy nền, có log: `docker compose -f docker-compose.ami.yml exec -T ami-rag-worker ami-rag reindex --all --yes > reindex.log 2>&1 &`, rồi `tail -f reindex.log`. Hoặc dùng `tmux`/`screen`. Ngắt phiên SSH có thể làm dừng lệnh đang chạy ở foreground; chạy lại là an toàn (idempotent).
- Container phải đang chạy. Nếu chưa: `make start_docker` (hoặc `docker compose -f docker-compose.ami.yml up -d`). Nếu cần chạy mà không dựa vào container sẵn có (không đụng worker):
  `docker compose -f docker-compose.ami.yml run --rm --no-deps ami-rag-worker ami-rag status`
  (`run` tạo container tạm cùng image/`.env`/volume, không publish cổng 9109; `--rm` xoá khi xong).
- Image đang chạy phải có code CLI mới. Sau khi sửa `ami_rag/`, `docker compose -f docker-compose.ami.yml up -d --build` (lâu, xem mục 5) rồi mới chạy `retry`/`reindex`; image cũ báo `invalid choice` hoặc thiếu cờ mới.
- Mã thoát của `exec` là mã thoát của lệnh trong container, nên `retry`/`reindex` (trả 1 khi còn doc thất bại) dùng được trong script/CI: `docker compose -f docker-compose.ami.yml exec -T ami-rag-worker ami-rag retry --all-failed --yes || echo "còn doc failed"`.
- Ngoài Docker (máy dev, đã `uv sync`): `uv run ami-rag <lệnh con>` với cùng cờ; host trong `.env` phải truy cập được từ máy đó (tên container như `redis`, `mongo` chỉ phân giải trong `ami-network`; `EMBED_SERVER_URL` phải trỏ đúng địa chỉ embed server).

## 8. Runbook backfill dữ liệu cũ

Các lệnh `ami-rag …` dưới đây viết ở dạng ngắn; khi chạy bằng Docker thêm tiền tố `docker compose -f docker-compose.ami.yml exec ami-rag-worker` (mục 7.1).

Mục tiêu: nạp toàn bộ doc có sẵn trong `organization_db.documents` vào index mới (collection Qdrant `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}`, registry `organization_db.multimodal_rag_documents`). Doc cũ của pipeline LightRAG (status `processed` trong registry cũ) được `reindex --scan` quét lại thành `pending` rồi xử lý theo pipeline mới: parse → describe → chunk → embed. Doc cũ chỉ có text/bảng đơn giản trong Mongo vẫn được parse lại từ file MinIO (nếu `original_file_name` không rỗng/`.txt`); text Mongo của pdf/docx **không** được dùng.

1. Chuẩn bị: `.env` đúng, embed server máy B chạy (`ami-rag status --check-embed-server` OK), worker chạy (`make start_docker` hoặc `make start_worker`), `make health` OK, queue pending 0. Worker cần MinerU + model cache sẵn (xem mục 5).
2. Quét doc cũ chưa có bản ghi: `ami-rag reindex --scan --dry-run` — in số pending/stale sẽ xử lý.
3. Dry-run toàn bộ, kiểm tra cột `source` (`mongo_text`/`minio_parse`) hợp lý và số chunk trúng cache:
   `ami-rag reindex --all --dry-run`
4. Thử text/crawl nhỏ (nhanh, không cần MinerU):
   `ami-rag reindex --doc <id> --yes` với vài doc text trước.
5. Thử file thật (MinerU, chậm): `ami-rag reindex --doc <id> --from-stage parse --yes`. Kiểm tra `GET /admin/documents/{id}` (`counts` có `table`/`image`), `GET /admin/documents/{id}/content`, và một truy vấn `POST /v2/rag/`.
6. Chạy toàn bộ: `ami-rag reindex --all --yes` (hoặc `--scan --yes` để quét + xử lý). Trước đó kiểm tra embed server sẵn sàng bằng `ami-rag status --check-embed-server`, vì embed server chết thì mọi doc ở stage `embed` sẽ thành `failed`.
7. Theo dõi:
    - `ami-rag status` (đếm theo status, doc failed, doc treo); queue pending xem qua metric/admin API;
   - `curl -s localhost:8009/admin/pipeline_status` (thêm header `Authorization: Bearer ...` nếu có `RAG_API_KEY`);
   - metric worker `:9109/metrics`: `multimodal_rag_ingest_events_total{result}`, `..._stream_pending`, `..._stream_lag`, `..._documents{status}` (đếm theo `status` trong `organization_db.multimodal_rag_documents`), `..._parse_failures_total`; dashboard row "Ingest pipeline".
   - `make logs SERVICE=ami-rag-worker`.
8. Xử lý lỗi:
    - doc `failed`: `ami-rag retry --all-failed` (hoặc `ami-rag retry --doc <id>`);
    - nạp lại khi đổi model/chunker: `ami-rag reindex --stale` (doc được đánh `stale` tự động khi lệch `embed_model`/`chunker_version`);
    - ép làm lại toàn bộ từ parse: `ami-rag retry --doc <id> --from-stage parse`;
    - doc không cần nữa: xoá qua event `deleted` trong queue hoặc xoá dòng registry + asset thủ công (CLI không còn lệnh purge).
9. Kiểm tra index: `ami-rag status` (đếm theo status/stage; `--check-embed-server` khi nghi ngờ máy B). Doc lỗi nằm ở `failed` với lý do trong `error` + `error_stage`.
10. Hoàn tất khi không còn `processing`/`failed`/`pending` trong `ami-rag status`, số doc `indexed` xấp xỉ số doc active.
11. Dọn collection legacy của LightRAG: `ami-rag cleanup --dry-run` rồi `ami-rag cleanup --yes` (xem mục 7 `cleanup`). Chỉ chạy sau khi đã xác nhận retrieval trên index mới hoạt động tốt.

## 9. Monitoring

Chi tiết triển khai: [`monitoring/README.md`](../monitoring/README.md).

Tracing (OTLP/HTTP): chỉ truy vấn `POST /v2/rag` được trace — span `rag.retrieval` (input: attribute `input.query`/`input.mode`/`input.top_k`; output: danh sách document đầy đủ trong event `documents_json`) cùng span con theo stage (`embed_query`/`vector_search`/`rerank`/`resolve`). Worker không trace (ingest chỉ có metric Prometheus) và FastAPI telemetry tự động bị tắt (`FastAPI(telemetry={"auto_configure": False, ...})` trong `ami_rag/api/main.py`), nếu không FastAPI ≥ 0.142 đọc `OTEL_EXPORTER_OTLP_ENDPOINT` rồi trace mọi route và đẩy metrics/logs tới `/v1/metrics`, `/v1/logs` (Tempo trả `404`: log `Failed to export metrics batch code: 404`). Tempo dùng riêng cho service này: release helm `tempo-multimodal-rag` (`monitoring/helm/tempo-multimodal-rag-values.yaml`, chart `grafana/tempo` 1.24.4, tách khỏi Tempo của conversational-agent); datasource Grafana uid `multimodal_rag_log` nạp qua sidecar bằng ConfigMap `monitoring/helm/grafana-datasource-tempo-multimodal-rag.yaml`; URL Grafana `http://tempo-multimodal-rag.monitoring.svc.cluster.local:3200`; `OTEL_EXPORTER_OTLP_ENDPOINT` trỏ tới NodePort OTLP/HTTP 4318 của service (`kubectl get svc tempo-multimodal-rag -n monitoring`).

Metric API (`GET :8009/metrics`, prefix `multimodal_rag_retrieval_`):

| Metric | Loại | Label |
|---|---|---|
| `requests_total`, `errors_total` | Counter | `mode` (`vector`) |
| `requests_by_day_total` / `_by_hour_of_day_total` / `_by_day_hour_total` | Counter | `day` / `hod` / `day,hod` (Asia/Ho_Chi_Minh) |
| `duration_seconds` | Histogram | `mode` |
| `stage_duration_seconds` | Histogram | `stage` = `embed_query` / `vector_search` / `rerank` / `resolve` |
| `docs_returned`, `chunks_retrieved` | Histogram | |
| `requests_in_flight` | Gauge | |
| `docs_filtered_total`, `presign_failures_total` | Counter | |
| `rerank_fallback_total` | Counter | `reason` = `error` / `empty` |
| `docs_by_modality_total` | Counter | `modality` |

Metric worker (`:9109/metrics` của `ami-rag-worker`; khi `WORKER_ENABLED=true` chúng nằm chung `/metrics` của API), prefix `multimodal_rag_ingest_`:

| Metric | Loại | Label |
|---|---|---|
| `events_total` | Counter | `event` (created/updated/deleted), `result` (indexed/skipped/failed/retry) |
| `in_flight` | Gauge | |
| `duration_seconds` | Histogram | `source` (minio_parse/mongo_text/none) |
| `stage_duration_seconds` | Histogram | `stage` (đường delete của worker; các stage pipeline tính trong `multimodal_rag_retrieval_*` và log) |
| `parse_failures_total` | Counter | `document_type` |
| `asset_upload_failures_total`, `pages_total` | Counter | |
| `items_total` | Counter | `modality` |
| `stream_pending`, `stream_lag` | Gauge | |
| `documents` | Gauge | `status` (đọc từ `multimodal_rag_documents`) |

Scrape và dashboard:
- Service Docker trên host: điền `__NODE_IP__` trong `monitoring/k8s/multimodal-rag-retrieval-metrics-scrape.yaml` rồi `make metrics-scrape-apply`. Service trong k8s: `monitoring/helm/servicemonitor.yaml`. Cả hai có 2 endpoint: `metrics` (8009) và `worker-metrics` (9109).
- `monitoring/helm/prometheusrule.yaml` (PrometheusRule `multimodal-rag-retrieval`): `RagIngestBacklogHigh` (pending+lag > 50 trong 15 phút), `RagIngestFailures` (> 5 event failed/30 phút), `RagIngestParseFailures` (> 3 parse lỗi/30 phút), `RagIngestWorkerDown` (target `worker-metrics` không UP 5 phút, critical), `RagRetrievalErrorRate` (> 5%), `RagRerankFallback` (> 20%), `RagPresignFailures` (> 10/15 phút), `RagMetricsDown` (target `metrics` không UP 5 phút, critical).
- Dashboard `monitoring/dashboards/multimodal-rag-retrieval-dashboard.json`: thêm row "Multimodal retrieval" (docs theo modality, presign failures) và "Ingest pipeline" (stream pending/lag, in-flight, events theo kết quả, duration, stage time, documents theo status, parse failures, items theo modality).
- Lệnh: `make dashboard-apply` (ConfigMap + ServiceMonitor + PrometheusRule, namespace `monitoring`), `make metrics-scrape-apply`, `make health`.

## 10. Tests

Hai bộ chạy riêng:

```bash
uv run pytest tests --ignore=tests/ami_service     # test gốc RAG-Anything
uv run pytest tests/ami_service                    # test của ami_rag (hoặc: make test)
```

Toàn bộ test dùng fake (không cần dịch vụ ngoài, không cần lightrag — đã gỡ). Suite `tests/ami_service` gồm: API/worker/CLI/pipeline (`test_api.py`, `test_worker.py`, `test_cli.py`, `test_vector_pipeline.py`, `test_doc_status_store.py`) với fake embedder/vector store/parser; observability và stream tests. Test gốc (`tests/*.py`) kiểm tra parser, modal processors, prompt, batch_parser của thư viện.

## 11. Phiên bản

`pyproject.toml` không còn `lightrag-hku` — LightRAG đã gỡ hoàn toàn khỏi dependency. Extra `service` thêm FastAPI, uvicorn, pydantic-settings, redis, pymongo, qdrant-client, httpx, minio, prometheus-client, OpenTelemetry, openai. Embedding service nằm ở repo ngoài `qwen-embedding-server` (máy B, port 8007), máy A không cài torch/transformers. Entry point: `ami-rag-api`, `ami-rag-worker`, `ami-rag`.

`mineru[core]>=3.4.1,<4`: MinerU 4.x đổi CLI (`mineru parse <path>`, `-p` = pages) nên không tương thích với lệnh `mineru -p <file> -o <dir> -m ...` mà `raganything/parser.py` gọi; bắt buộc ghim `<4`. MinerU 3.4.x mặc định backend `hybrid-engine` (nặng VRAM) khi không truyền `-b`, và chọn thiết bị bằng biến môi trường `MINERU_DEVICE_MODE` (không có cờ `-d`); `MINERU_VIRTUAL_VRAM_SIZE` (GB) buộc MinerU chọn batch size như thể GPU có chừng đó VRAM.

Chunk multimodal: parse-only (`raganything/processor.py`) + mô tả modal (`generate_description_only`/`generate_chunk_sections`); metadata có cấu trúc (`original_type`, `page_idx`, `asset_key`, `table_body`, `caption`) đi thẳng vào chunk payload để API trả về client.
