# AMI multimodal RAG service (`ami_rag`)

Package `ami_rag/` trong fork RAG-Anything này là service multimodal RAG của hệ thống AMI: một ingest worker (parse tài liệu bằng MinerU, nạp vào RAG-Anything/LightRAG) và một API truy xuất `POST /v2/rag`. Nó thay thế hoàn toàn repo cũ `ami-multimodal-retrieval`.

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
   └─ RAGAnything.insert_content_list(doc_id=<mongo id>)
        ▼
   LightRAG: Mongo `organization_db` (KV, doc_status, graph; collection `multimodal_*`) + Qdrant (vector), workspace `multimodal`

Hệ thống AI ──POST /v2/rag──▶ ami_rag API: RAGAnything.aquery_data (mix) → rerank service → resolve doc Mongo
                              → presign MinIO (artifact_url, file_url) → response
```

- Event mỏng: queue chỉ mang `event` (`created`/`updated`/`deleted`), `document_id`, `content_hash`, `document_type`, `org_id`. Worker đọc nội dung từ Mongo/MinIO (`ami_rag/queue/events.py`).
- Id 1:1: LightRAG doc id = MongoDB ObjectId; `file_path` của chunk = `{doc_id}_{basename}` nên API tra ngược về Mongo (`ami_rag/api/resolver.py`).
- Update = xoá index cũ rồi insert lại; `deleted` xoá index + asset MinIO + dòng registry. Nguồn không đổi (`source_hash` trùng, status `processed`) thì bị bỏ qua.
- Retry: `WORKER_MAX_DELIVERY` lần; quá ngưỡng thì dòng registry được đánh dấu `failed` và message được ack. Event lỗi chưa đủ ngưỡng được để pending và tự giao lại (XAUTOCLAIM) sau `WORKER_RETRY_IDLE_MS` (mặc định 10 phút; cần Redis ≥ 6.2) — đặt đủ lớn hơn thời gian parse MinerU dài nhất để worker khác không "cướp" message đang xử lý. Khi `failed`, `attempts` được reset nên `reprocess_failed`/`reindex --force` có đủ ngân sách thử lại.
- LLM profile: `gemini` (mặc định) hoặc `qwen-selfhost` (vLLM, OpenAI-compatible) qua `LLM_PROFILE`. Rerank là service ngoài `POST {RERANK_BASE_URL}/rerank`; lỗi rerank thì trả chunk không điểm (fallback).

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

## 3. Dữ liệu: dùng chung `organization_db`, tách bằng tiền tố `multimodal_`

RAG dùng chung database `organization_db` với backend ami_data. Mọi collection của RAG có tiền tố `multimodal_` nên không đụng collection của backend (`documents`, `users`, ...).

| Thành phần | Giá trị | Ghi chú |
|---|---|---|
| Mongo DB của RAG | `RAG_DB` = `organization_db` | LightRAG KV/doc_status/graph (`MONGO_DATABASE` được set từ `RAG_DB` bằng cách gán trực tiếp `os.environ` trong `ami_rag/core/factory.py::_inject_storage_env`, nên `Settings` luôn thắng biến môi trường `MONGO_DATABASE`/`MONGO_URI` có sẵn) |
| Registry | `organization_db.multimodal_rag_documents` (`RAG_DOCUMENTS_COLLECTION`) | một dòng/doc, `_id` = ObjectId dạng string |
| Workspace | `WORKSPACE` = `multimodal` | tiền tố collection LightRAG (`{workspace}_{namespace}`) và phân vùng Qdrant |
| Qdrant | `QDRANT_URL` | vector, partition theo workspace |
| MinIO asset | `{ASSET_PREFIX}/{doc_id}/` (`rag-assets/…`) trong `MINIO_BUCKET` | ảnh/bảng/công thức (tên `sha256[:16]` + ext), `content_list.json` |
| Nguồn doc | `organization_db.documents` (`ORG_DB`/`DOC_COLLECTION`) | **chỉ đọc**, lọc `status: "active"` khi duyệt toàn bộ |

### Các collection `multimodal_*` và ranh giới

LightRAG đặt tên collection Mongo là `{workspace}_{namespace}` (`lightrag/kg/mongo_impl.py`, `final_namespace`). Với `WORKSPACE=multimodal`:

| Collection | Nguồn | Ghi chú |
|---|---|---|
| `multimodal_rag_documents` | `ami_rag` (`RAG_DOCUMENTS_COLLECTION`) | registry, ghi bởi worker/CLI/admin |
| `multimodal_full_docs`, `multimodal_text_chunks`, `multimodal_llm_response_cache`, `multimodal_full_entities`, `multimodal_full_relations`, `multimodal_entity_chunks`, `multimodal_relation_chunks` | LightRAG KV (`lightrag/namespace.py`) | |
| `multimodal_doc_status` | LightRAG doc status | |
| `multimodal_chunk_entity_relation`, `multimodal_chunk_entity_relation_edges` | LightRAG graph (`MongoGraphStorage`: node + `_edges`) | |
| `multimodal_parse_cache`, `multimodal_multimodal_status` | RAGAnything (`raganything/raganything.py`, namespace `parse_cache` / `multimodal_status`) | tên lặp `multimodal_multimodal_status` vì workspace `multimodal` + namespace `multimodal_status` |
| collection khác | do LightRAG tạo | vector (entities/relationships/chunks) nằm ở Qdrant, không ở Mongo |

Ranh giới:

| Collection | Quyền của RAG |
|---|---|
| `documents` | chỉ ĐỌC (`ORG_DB`/`DOC_COLLECTION`) |
| `document_versions`, `organization_units`, `users` | không đọc trong code; chỉ dùng để `$lookup` khi truy vấn/báo cáo từ bên ngoài |
| `multimodal_*` | GHI (tạo collection, index, upsert/xoá) |

Lưu ý vận hành:
- Quyền Mongo: user của service RAG cần `createCollection` + `createIndex` + đọc/ghi trên `multimodal_*` trong `organization_db` (`_ensure_indexes` tạo index `status`, `organization_unit_id`, `document_oid`; lỗi chỉ log warning), và chỉ cần `find` trên `documents`.
- Biến `MONGODB_WORKSPACE` (nếu đặt trong môi trường) sẽ ghi đè workspace trong tên collection Mongo của LightRAG (`mongo_impl.py`, đọc trực tiếp từ môi trường), khiến tên collection không còn theo `WORKSPACE`; đừng đặt biến này trừ khi cố ý.
- Tên collection phụ thuộc `WORKSPACE`: đổi `WORKSPACE` = tạo bộ collection/partition Qdrant mới (dữ liệu cũ không được dùng nữa, cần backfill lại). `RAG_DOCUMENTS_COLLECTION` không theo `WORKSPACE`, nên registry cũ còn `processed` có thể khiến doc bị bỏ qua: dùng `--force`.
- Backup/restore/drop `organization_db` ảnh hưởng cả backend lẫn RAG; khi chỉ muốn xoá dữ liệu RAG, drop từng collection `multimodal_*` (kèm dữ liệu Qdrant), không drop database.

### Liên kết với organization_units / users

Mỗi dòng registry được ghi thêm các field liên kết khi `processed` (`ami_rag/sources.py::doc_link_fields`, truyền qua `meta=` vào `mark_processed`; dòng `failed`/`processing` chưa có):

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
  { $match: { status: "processed" } },
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

Các field (`ami_rag/storage/rag_documents.py`):

| Field | Ý nghĩa |
|---|---|
| `_id` | document id |
| `document_oid`, `organization_unit_id`, `owner_id`, `document_type`, `title` | field liên kết (bảng trên), chỉ có sau lần `processed` |
| `status` | `processing` / `processed` / `failed` / `stale` |
| `source`, `source_hash` | nguồn (`mongo_text`/`minio_parse`) và vân tay |
| `file_path` | `{doc_id}_{basename}` dùng cho citation |
| `parser` | giá trị `PARSER` lúc nạp |
| `assets` | danh sách key MinIO đã upload |
| `counts`, `page_count` | số item theo modality (`text/image/table/equation/other`), số trang |
| `attempts`, `last_hash`, `error` | theo dõi lần thử/lỗi (`error` cắt 2000 ký tự) |
| `created_at`, `updated_at` | thời điểm |

## 4. Biến môi trường

File mẫu: `.env.ami.example` (copy thành `.env`). Nguồn sự thật: `ami_rag/settings.py`.

| Nhóm | Biến | Mặc định | Ý nghĩa |
|---|---|---|---|
| LLM | `LLM_PROFILE` | `gemini` | `gemini` hoặc `qwen-selfhost` |
| | `GEMINI_API_KEY` | rỗng | |
| | `GEMINI_LLM_MODEL` | `gemini-2.5-flash` | |
| | `GEMINI_VISION_MODEL` | rỗng | rỗng = dùng `GEMINI_LLM_MODEL` |
| | `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | |
| | `EMBEDDING_DIM` | `1536` | chiều embedding Gemini |
| | `QWEN_LLM_BASE_URL` / `QWEN_LLM_MODEL` / `QWEN_LLM_API_KEY` | `http://vllm:8000/v1` / `Qwen/Qwen3-32B` / rỗng | |
| | `QWEN_VLM_MODEL` | rỗng | rỗng = dùng `QWEN_LLM_MODEL` |
| | `QWEN_EMBED_BASE_URL` / `QWEN_EMBED_MODEL` / `QWEN_EMBED_API_KEY` | `http://vllm:8000/v1` / `Qwen/Qwen3-Embedding-0.6B` / rỗng | |
| | `QWEN_EMBED_DIM` | `1024` | |
| Mongo | `MONGO_URI` | `mongodb://localhost:27017/?directConnection=true` | |
| | `RAG_DB` | `organization_db` | dùng chung DB với backend |
| | `RAG_DOCUMENTS_COLLECTION` | `multimodal_rag_documents` | registry |
| | `ORG_DB` / `DOC_COLLECTION` | `organization_db` / `documents` | chỉ đọc |
| Qdrant | `QDRANT_URL` / `QDRANT_API_KEY` | `http://localhost:6333` / rỗng | |
| Index | `WORKSPACE` | `multimodal` | tiền tố collection LightRAG + phân vùng Qdrant |
| | `WORKING_DIR` | `./rag_storage` | working dir LightRAG |
| Parser | `PARSER` | `mineru` | |
| | `PARSE_METHOD` | `auto` | |
| | `PARSER_OUTPUT_DIR` | `./output` | |
| Chunking | `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `100` | token |
| | `MAX_GLEANING` | `1` | |
| | `SUMMARY_LANGUAGE` | `Tiếng Việt` | |
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
| | `RERANK_TOP_K` | `5` | số chunk trả về sau rerank |
| | `RERANK_TIMEOUT` | `60` | giây |
| Retrieval | `RETRIEVAL_TOP_K` | `40` | top_k mặc định khi request không gửi |
| | `RETRIEVAL_CHUNK_TOP_K` | `40` | |
| MinIO | `MINIO_ENDPOINT` | `localhost:9000` | |
| | `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` / rỗng | |
| | `MINIO_BUCKET` | `ami-data-documents` | |
| | `MINIO_SECURE` | `false` | |
| | `ASSET_PREFIX` | `rag-assets` | |
| | `CRAWL_IMAGES_PREFIX` / `CRAWL_IMAGES_ENABLED` | `crawl-images` / `false` | |
| | `MINIO_PRESIGN_EXPIRES` | `3600` | giây, cho `artifact_url`/`file_url` |
| API | `API_HOST` / `API_PORT` | `0.0.0.0` / `8009` | |
| | `RAG_API_KEY` | rỗng | rỗng = không auth |
| Tracing | `OTEL_EXPORTER_OTLP_ENDPOINT` | rỗng | rỗng = tracing tắt; đọc từ `os.environ` |
| | `OTEL_SERVICE_NAME` | `multimodal-rag-retrieval` | |
| | `TRACE_OUTPUT_MAX_LEN` | `2000000` | giới hạn ký tự event `documents_json`; `0` = không cắt |

Lưu ý: `OTEL_*` và `TRACE_OUTPUT_MAX_LEN` do `observability.py` đọc trực tiếp từ `os.environ` (không qua `Settings`). Docker Compose nạp `.env` vào môi trường container nên hoạt động; khi chạy local bằng `make start_backend`, cần `export` các biến này trong shell.

## 5. Chạy local và Docker

Local (cần Redis, Mongo, Qdrant, MinIO, rerank service sẵn sàng):

```bash
make setup                    # uv sync --extra service
cp .env.ami.example .env      # điền GEMINI_API_KEY, MINIO_SECRET_KEY, hosts...
make start_backend            # API :8009 (uvicorn --reload)   | hoặc: ami-rag-api
make start_worker             # ingest worker                  | hoặc: ami-rag-worker
make test                     # pytest -v tests/ami_service/
make style / make lint        # ruff trên ami_rag và tests/ami_service
```

Docker (`docker-compose.ami.yml`, `Dockerfile.ami`, `python:3.12-slim`):

```bash
cp .env.ami.example .env      # host Mongo/Qdrant/Redis/MinIO/rerank = tên container trên ami-network
make start_docker             # tạo network ami-network nếu thiếu, rồi up --build -d
make ps && make health        # /healthz + đếm series multimodal_rag_retrieval_* và multimodal_rag_ingest_*
make logs SERVICE=ami-rag-worker
make restart | make down | make docker-clean
```

- Hai service dùng chung image: `ami-rag-api` (port 8009) và `ami-rag-worker` (port 9109 cho `/metrics`). Mongo, Qdrant, Redis, MinIO, rerank chạy ngoài compose trên network external `ami-network`; trong container, host trong `.env` phải là tên container, không phải `localhost`.
- GPU: `ami-rag-worker` có `deploy.resources.reservations.devices` (nvidia, count 1) cho MinerU. Bỏ khối này để parse bằng CPU (chậm hơn nhiều).
- Cache model MinerU: volume `mineru-models` mount tại `/models` (`HF_HOME=/models/huggingface`, `MINERU_MODEL_SOURCE=huggingface`), tránh tải lại mỗi lần rebuild. Volume `rag-output` mount tại `/app/output`.
- LibreOffice (`libreoffice-writer`) đã cài trong image để chuyển doc/docx sang PDF trước khi MinerU parse; chạy local cần tự cài LibreOffice.
- Biến Make ghi đè: `PORT`, `WORKER_METRICS_PORT`, `COMPOSE_FILE`, `SERVICE`, `NETWORK`.

## 6. API

Mọi route `/v2/rag/*` và `/admin/*` yêu cầu `Authorization: Bearer <RAG_API_KEY>` khi `RAG_API_KEY` khác rỗng (rỗng = mở; nên chỉ dùng trong mạng nội bộ). `/`, `/healthz`, `/readyz`, `/metrics` không yêu cầu auth; chặn `/metrics` ở reverse proxy nếu cần.

### POST /v2/rag/ — v1 (không đổi)

```json
// Request
{ "messages": [{"role": "user", "content": "học phí năm 2026"}], "top_k": 5 }
// Response
{ "query": "học phí năm 2026",
  "documents": [{ "text": "...", "score": 0.91,
    "metadata": {"source": "<doc_id>_file.pdf", "page": 3, "document_id": "665f...", "chunk_index": null, "global_id": null} }] }
```

v1 luôn dùng mode `mix`. `metadata.page` lấy từ `page_idx` của chunk (null với chunk text thường); `chunk_index`/`global_id` luôn null.

### POST /v2/rag/ — v2 (`"version": 2`)

Request:

```json
{
  "messages": [{"role": "user", "content": "bảng học phí các ngành"}],
  "top_k": 10, "version": 2, "mode": "mix",
  "include_kg": false,
  "filters": {"organization_unit_id": null, "document_type": null, "modality": ["table", "image"]}
}
```

Response (rút gọn; mỗi document có đủ các field, giá trị không áp dụng là `null`):

```json
{
  "query": "bảng học phí các ngành",
  "mode": "mix",
  "documents": [
    { "text": "Đoạn văn mô tả học phí ...", "score": 0.93, "modality": "text",
      "artifact_url": null, "table_body": null, "page": null, "caption": null,
      "reference_id": "1", "entities": null,
      "metadata": {"source": "665f..._hocphi.pdf", "page": null, "document_id": "665f...", "chunk_index": null, "global_id": null},
      "doc": {"document_id": "665f...", "title": "Học phí 2026", "document_type": "pdf",
              "organization_unit_id": "...", "file_url": "https://minio/presigned/...", "source_url": null} },
    { "text": "... Asset: rag-assets/665f.../ab12cd34ef567890.png ...", "score": 0.81, "modality": "image",
      "artifact_url": "https://minio/presigned/rag-assets/665f.../ab12cd34ef567890.png",
      "table_body": null, "page": 2, "caption": "Sơ đồ tổ chức", "reference_id": "1", "doc": {"...": "..."} },
    { "text": "...", "score": 0.77, "modality": "table",
      "artifact_url": null, "table_body": "| Ngành | Học phí |\n| --- | --- |\n| CNTT | 30tr |",
      "page": 4, "caption": "Bảng 1. Học phí", "reference_id": "1", "doc": {"...": "..."} }
  ],
  "references": [{"reference_id": "1", "file_path": "665f..._hocphi.pdf"}],
  "meta": {"keywords": {"high_level": ["..."], "low_level": ["..."]}, "latency_ms": 820}
}
```

- `modality`: `text` | `image` | `table` | `equation` (từ `original_type` của chunk multimodal).
- `artifact_url`: presigned URL (hết hạn sau `MINIO_PRESIGN_EXPIRES`) của `asset_key`; `null` khi chunk không có asset hoặc presign lỗi (tăng `presign_failures_total`).
- `filters.modality` lọc chunk trước rerank (áp dụng cả v1/v2); `filters.organization_unit_id` / `filters.document_type` chỉ áp dụng cho v2, đo bằng `docs_filtered_total`.
- `include_kg: true` thêm `entities` (tối đa 20: `name`, `type`, `description`) vào mỗi document.
- Mode: `naive`, `local`, `global`, `hybrid`, `mix` (mặc định). `top_k` 1..200, mặc định `RETRIEVAL_TOP_K`; số document trả về tối đa `RERANK_TOP_K`.

### POST /v2/rag/stream

Cùng request, trả NDJSON (`application/x-ndjson`): `{"status": "retrieving"}`, rồi mỗi document một dòng `{"documents": [{"text", "metadata", "score", ["modality", "artifact_url"]}]}` (với v2 có thêm `modality`/`artifact_url`; `metadata` bỏ `page`/`source`), cuối cùng `{"status": "done"}`.

### Admin (`/admin/*`)

| Endpoint | Mô tả |
|---|---|
| `GET /admin/pipeline_status` | `{workspace, doc_status_counts (LightRAG), queue_pending}` |
| `POST /admin/reprocess_failed` | publish lại event `updated` cho mọi doc `failed` trong registry → `{requeued}` |
| `POST /admin/reindex` | body `{"all": true}` hoặc `{"document_ids": [...]}`; publish event `updated` → `{published}` (không lọc type, không có `force`) |
| `GET /admin/documents/{id}` | trạng thái registry: `status, source, counts, page_count, parser, document_type, title, organization_unit_id, owner_id, error, updated_at` (4 field `document_type/title/organization_unit_id/owner_id` lấy từ `multimodal_rag_documents`, ObjectId trả về dạng string, `null` nếu dòng chưa `processed`); 404 nếu chưa có |
| `GET /admin/documents/{id}/content` | từ `content_list.json` trên MinIO: `{document_id, markdown, blocks, tables}` (block có `type`, `page_idx`, `text`/`table_body`/`caption`/`latex`, `asset_url`); 404 nếu chưa có |

Khác: `GET /healthz`, `GET /readyz`, `GET /metrics`.

## 7. CLI `ami-rag`

```bash
ami-rag reindex (--all | --doc-ids ID [ID ...]) [--type pdf,docx,text,crawl] [--limit N] \
                [--batch-size 100] [--force] [--dry-run] [--direct]
ami-rag status
ami-rag purge-doc --doc-id <id>
```

| Cờ | Ý nghĩa |
|---|---|
| `--all` / `--doc-ids` | bắt buộc một trong hai; `--all` duyệt doc `status: active` của `organization_db.documents` |
| `--type` | lọc `document_type` (danh sách phân tách bằng dấu phẩy) |
| `--limit`, `--batch-size` | giới hạn số doc; kích thước batch cursor Mongo |
| `--dry-run` | in bảng `doc_id / type / source / hash / processed` rồi thoát, không ghi gì |
| `--force` | đánh dấu `stale` các doc `processed` đã chọn để nạp lại dù `source_hash` không đổi |
| `--direct` | nạp ngay trong tiến trình CLI (không qua Redis/worker), in `indexed/skipped/failed` |

Mặc định (không `--direct`) lệnh chỉ XADD event `created` vào `RAG_STREAM`; worker xử lý. `status` in workspace, số doc active, đếm registry theo status, `queue pending`, id doc failed và tình trạng rerank service. `purge-doc` xoá doc khỏi LightRAG, asset MinIO và registry.

## 8. Runbook backfill dữ liệu cũ

Mục tiêu: nạp toàn bộ doc có sẵn trong `organization_db.documents` vào index mới (workspace `multimodal`, collection `organization_db.multimodal_*`). Doc đã `processed` từ trước khi có field liên kết (`document_oid`, `organization_unit_id`, `owner_id`, ...) chỉ được bổ sung các field này khi nạp lại (`--force`). Doc cũ chỉ có text/bảng đơn giản trong Mongo vẫn được parse lại từ file MinIO (nếu `original_file_name` không rỗng/`.txt`); text Mongo của pdf/docx **không** được dùng.

1. Chuẩn bị: `.env` đúng, worker chạy (`make start_docker` hoặc `make start_worker`), `make health` OK, `ami-rag status` thấy `rerank service: ok` và `queue pending: 0`. Worker cần GPU/MinerU và model cache sẵn (xem mục 5).
2. Dry-run toàn bộ, kiểm tra cột `source` (`mongo_text`/`minio_parse`) hợp lý:
   `ami-rag reindex --all --dry-run`
3. Thử text/crawl nhỏ (nhanh, không cần MinerU):
   `ami-rag reindex --all --type text,crawl --limit 20`
4. Thử file thật (MinerU, chậm): `ami-rag reindex --all --type pdf,docx --limit 3`. Kiểm tra `GET /admin/documents/{id}` (`counts` có `table`/`image`), `GET /admin/documents/{id}/content`, và một truy vấn `POST /v2/rag/` v2 với `filters.modality`.
5. Chạy toàn bộ: `ami-rag reindex --all` (doc đã `processed` và không đổi sẽ bị worker bỏ qua; thêm `--force` để nạp lại tất cả).
6. Theo dõi:
   - `ami-rag status` (đếm theo status, `queue pending`, doc failed);
   - `curl -s localhost:8009/admin/pipeline_status` (thêm header `Authorization: Bearer ...` nếu có `RAG_API_KEY`);
   - metric worker `:9109/metrics`: `multimodal_rag_ingest_events_total{result}`, `..._stream_pending`, `..._stream_lag`, `..._documents{status}` (đếm theo `status` trong `organization_db.multimodal_rag_documents`), `..._parse_failures_total`; dashboard row "Ingest pipeline".
   - `make logs SERVICE=ami-rag-worker`.
7. Xử lý lỗi:
   - doc `failed`: `POST /admin/reprocess_failed` (hoặc `ami-rag reindex --doc-ids <id>`);
   - nạp lại bắt buộc (đổi cấu hình parser/chunking): `ami-rag reindex --doc-ids <id> --force` hoặc `--all --type pdf,docx --force`;
   - cần chạy đồng bộ để xem lỗi trực tiếp: thêm `--direct` (dừng worker trước nếu không muốn hai tiến trình cùng ghi);
   - doc không cần nữa: `ami-rag purge-doc --doc-id <id>`.
8. Hoàn tất khi `queue pending` và `lag` về 0, `status` không còn `processing`/`failed`, `multimodal_rag_ingest_documents{status="processed"}` xấp xỉ số doc active.

## 9. Monitoring

Chi tiết triển khai: [`monitoring/README.md`](../monitoring/README.md).

Metric API (`GET :8009/metrics`, prefix `multimodal_rag_retrieval_`):

| Metric | Loại | Label |
|---|---|---|
| `requests_total`, `errors_total` | Counter | `mode` |
| `requests_by_day_total` / `_by_hour_of_day_total` / `_by_day_hour_total` | Counter | `day` / `hod` / `day,hod` (Asia/Ho_Chi_Minh) |
| `duration_seconds` | Histogram | `mode` |
| `stage_duration_seconds` | Histogram | `stage` = `raganything_query` / `rerank` / `resolve` |
| `docs_returned`, `chunks_retrieved` | Histogram | |
| `requests_in_flight` | Gauge | |
| `lightrag_failures_total`, `docs_filtered_total`, `presign_failures_total` | Counter | |
| `rerank_fallback_total` | Counter | `reason` = `error` / `empty` |
| `docs_by_modality_total` | Counter | `modality` |

Metric worker (`:9109/metrics` của `ami-rag-worker`; khi `WORKER_ENABLED=true` chúng nằm chung `/metrics` của API), prefix `multimodal_rag_ingest_`:

| Metric | Loại | Label |
|---|---|---|
| `events_total` | Counter | `event` (created/updated/deleted), `result` (processed/skipped/failed/retry) |
| `in_flight` | Gauge | |
| `duration_seconds` | Histogram | `source` (minio_parse/mongo_text/none) |
| `stage_duration_seconds` | Histogram | `stage` (download/parse/upload_assets/insert/delete) |
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

Một số test gốc stub module `lightrag` trong `sys.modules`, nên các test tích hợp trong `tests/ami_service/test_integration_*.py` tự skip khi chạy chung hai bộ; chạy `tests/ami_service` riêng để chúng chạy với `lightrag` thật. Phần còn lại của `tests/ami_service` dùng fake (không cần dịch vụ ngoài).

## 11. Phiên bản

`pyproject.toml` ghim `lightrag-hku>=1.4.9,<1.5` (cài từ PyPI, không dùng checkout local). Extra `service` thêm FastAPI, uvicorn, pydantic-settings, redis, pymongo, qdrant-client, httpx, minio, prometheus-client, OpenTelemetry, openai, google-genai. Entry point: `ami-rag-api`, `ami-rag-worker`, `ami-rag`.

`mineru[core]>=3.4.1,<4`: MinerU 4.x đổi CLI (`mineru parse <path>`, `-p` = pages) nên không tương thích với lệnh `mineru -p <file> -o <dir> -m ...` mà `raganything/parser.py` gọi; bắt buộc ghim `<4`. MinerU 3.4.x mặc định backend `hybrid-engine` (nặng VRAM) khi không truyền `-b`, và chọn thiết bị bằng biến môi trường `MINERU_DEVICE_MODE` (không có cờ `-d`); `MINERU_VIRTUAL_VRAM_SIZE` (GB) buộc MinerU chọn batch size như thể GPU có chừng đó VRAM. Đo VRAM thực tế bằng `notebooks/mineru_vram_check.ipynb` (Colab).

Thay đổi liên quan trong thư viện `raganything`: chunk multimodal lưu field có cấu trúc và `RAGAnything.aquery_data` làm giàu chunk; xem `docs/architecture.md` và `docs/api_reference.md`.
