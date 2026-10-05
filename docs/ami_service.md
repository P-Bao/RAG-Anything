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
   ├─ RAGAnything.insert_content_list(doc_id=<mongo id>)
   └─ index check: chunk vector (Qdrant) + entity (graph) phải có thật, nếu không → retry/`failed`
        ▼
   LightRAG: Mongo `organization_db` (KV, doc_status, graph; collection `multimodal_*`) + Qdrant (vector), workspace `multimodal`

Hệ thống AI ──POST /v2/rag──▶ ami_rag API: RAGAnything.aquery_data (mix) → rerank service → resolve doc Mongo
                              → presign MinIO (artifact_url, file_url) → response
```

- Event mỏng: queue chỉ mang `event` (`created`/`updated`/`deleted`), `document_id`, `content_hash`, `document_type`, `org_id`. Worker đọc nội dung từ Mongo/MinIO (`ami_rag/queue/events.py`).
- Id 1:1: LightRAG doc id = MongoDB ObjectId; `file_path` của chunk = `{doc_id}_{basename}` nên API tra ngược về Mongo (`ami_rag/api/resolver.py`).
- Update = xoá index cũ rồi insert lại; `deleted` xoá index + asset MinIO + dòng registry. Nguồn không đổi (`source_hash` trùng, status `processed`) thì bị bỏ qua, cả ở worker lẫn ở CLI `reindex` (CLI không publish event cho doc đó, trừ khi `--force`).
- Index check: LightRAG nuốt lỗi embedding/LLM (hết quota, 402/429/5xx), nên `insert_content_list` có thể trả về bình thường dù không có vector chunk hay entity nào. Sau mỗi lần insert, worker đếm lại thực tế (`ami_rag/index_check.py::inspect_document`: chunk trong `doc_status`, vector trong Qdrant, entity/relation trong `full_entities`/`full_relations`), ghi log `index check: doc=… chunks=N vectors=N entities=N relations=N OK`. Thiếu vector hoặc không có entity thì ghi `WARNING`, ném `IngestIncompleteError` (đi theo đường retry → `failed`) thay vì ghi `processed`. Kết quả đếm lưu vào registry (`index_stats`). Tắt bằng `INGEST_VERIFY=false`; cho phép doc không có entity bằng `INGEST_REQUIRE_ENTITIES=false`.
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
| `index_stats` | `{chunks, vectors, entities, relations}` đếm được ngay sau lần nạp thành công hoặc sau `reindex --repair` |
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
| | `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | `gemini-embedding-2` cũng dùng được: embedding được gọi từng text một vì model này gộp cả batch thành 1 vector (xem 5.2) |
| | `EMBEDDING_DIM` | `1536` | chiều embedding Gemini; được gửi cho API (`send_dimensions`), không thì Gemini trả 3072 chiều |
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
| MinerU (chỉ khi `PARSER=mineru`) | `MINERU_BACKEND` | `pipeline` | truyền `-b`; xem mục 5.1 vì sao không bỏ trống |
| | `MINERU_DEVICE` | rỗng | rỗng = tự nhận; `cuda` / `cuda:0` / `cpu`; map sang env `MINERU_DEVICE_MODE` của tiến trình mineru |
| | `MINERU_VIRTUAL_VRAM_SIZE` | `0` | GB; `0` = dùng VRAM thật; >0 map sang env cùng tên, MinerU chọn batch size như thể GPU có chừng đó VRAM |
| | `MINERU_LANG` | rỗng | gợi ý ngôn ngữ OCR (`ch`, `en`, `latin`, ...); rỗng = mặc định MinerU |
| | `MINERU_SOURCE` | rỗng | nguồn model: `huggingface` / `modelscope` / `local`; rỗng = mặc định |
| | `MINERU_TIMEOUT` | `1800` | giây/tài liệu; `0` = không giới hạn |
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
| | `INGEST_VERIFY` | `true` | kiểm tra index (vector chunk, entity) sau mỗi lần nạp; thiếu thì retry/`failed` |
| | `INGEST_REQUIRE_ENTITIES` | `true` | `false` = chỉ bắt buộc có vector chunk, cho phép doc không có entity |
| Rerank | `RERANK_BASE_URL` | `http://localhost:8010` | |
| | `RERANK_TOP_K` | `5` | số tài liệu trả về mặc định khi request không chỉ định `top_k` |
| | `RERANK_TIMEOUT` | `60` | giây |
| Retrieval | `RETRIEVAL_TOP_K` | `40` | số entity/quan hệ trích xuất tối thiểu từ LightRAG |
| | `RETRIEVAL_CHUNK_TOP_K` | `40` | số chunk tối thiểu lấy từ LightRAG trước khi lọc và rerank |
| | `RETRIEVAL_OVERFETCH` | `4` | hệ số nhân lấy ứng viên: `chunk_top_k = max(RETRIEVAL_CHUNK_TOP_K, top_k * RETRIEVAL_OVERFETCH)` để sau khi lọc org/type vẫn đủ `top_k` kết quả |
| Repair | `REPAIR_CONCURRENCY` | `2` | số tài liệu sửa song song trong `reindex --repair` (thấp để tránh rate limit 429) |
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

- Build image chậm (~20 phút khi đổi code): `Dockerfile.ami` chép `ami_rag/` trước `pip install ".[service]"` nên mỗi lần sửa code `pip` tải lại toàn bộ gói (kể cả CUDA).
- Hai service dùng chung image: `ami-rag-api` (port 8009) và `ami-rag-worker` (port 9109 cho `/metrics`). Mongo, Qdrant, Redis, MinIO, rerank chạy ngoài compose trên network external `ami-network`; trong container, host trong `.env` phải là tên container, không phải `localhost`.
- GPU: `ami-rag-worker` có `deploy.resources.reservations.devices` (nvidia, count 1) cho MinerU. Bỏ khối này để parse bằng CPU (xem 5.1).
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

Reranker `bge-reranker-v2-m3` fp16, batch 40x512: đỉnh ~1.9 GiB, 1.0 s (nếu chạy chung GPU với worker, cộng vào ngân sách VRAM).

Khuyến nghị:
- `pipeline` dưới 4 GiB VRAM nên chạy GPU trực tiếp (mặc định, `MINERU_DEVICE` rỗng).
- GPU dùng chung với service khác (reranker, vLLM, ...): đặt `MINERU_VIRTUAL_VRAM_SIZE=4` (đỉnh ~1.1 GiB, chậm hơn nhẹ). Giá trị lớn hơn làm MinerU dùng batch lớn hơn nên tốn thêm VRAM, không nhanh hơn đáng kể.
- Chạy CPU: đặt `MINERU_DEVICE=cpu` và bỏ khối `deploy.resources.reservations.devices` của `ami-rag-worker` trong `docker-compose.ami.yml`. Chậm ~5 lần (~27 s/trang): tăng `MINERU_TIMEOUT` cho file dài và `WORKER_RETRY_IDLE_MS` cho đủ lớn hơn thời gian parse.
- Tự đo lại trên GPU của bạn bằng `notebooks/mineru_vram_check.ipynb`.

### 5.2 Gemini embedding / LLM và Redis (đã gặp thực tế)

- `ami_rag/core/factory.py::_build_embedding_func`: `EmbeddingFunc(send_dimensions=True)` để gửi `EMBEDDING_DIM`; thiếu thì Gemini trả vector 3072 chiều và LightRAG báo `Vector count mismatch: expected 1 vectors but got 2 vectors`. Embedding gọi từng text một (`_gemini_embed_one_per_text`, đồng thời tối đa 8) vì `gemini-embedding-2` gộp cả batch thành 1 vector. Nhiều `429` khi nạp hàng loạt thì giảm `_GEMINI_EMBED_CONCURRENCY`.
- `_build_llm_func` bọc `gemini_complete_if_cache` / `openai_complete_if_cache` bằng hàm nhận `prompt` đầu tiên: helper của lightrag nhận `model` ở vị trí đầu, dùng `partial(..., model=...)` sẽ gây `got multiple values for argument 'model'` và mọi bước trích entity thất bại.
- Hết credit Gemini (`402 RESOURCE_EXHAUSTED: prepayment credits are depleted`) hay model quá tải (`503`) làm LightRAG bỏ qua vector/entity. Nạp credit rồi dùng `ami-rag verify` / `reindex --repair` (mục 7) để xác định và sửa doc bị ảnh hưởng.
- Redis: `redis-py` ≥ 8 mặc định `socket_timeout=5s`, trùng với `XREADGROUP BLOCK` (`WORKER_POLL_BLOCK_MS`) nên worker báo `queue read failed: Timeout reading from redis`. Client của worker đặt `socket_timeout = WORKER_POLL_BLOCK_MS/1000 + 5` và `health_check_interval=30` (`ingest_worker.py::_build_default_deps`).

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
- Mode: `naive`, `local`, `global`, `hybrid`, `mix` (mặc định). `top_k` (1..200) là số document trả về cuối cùng (mặc định `RERANK_TOP_K` khi không gửi). Hệ thống lấy `top_k * RETRIEVAL_OVERFETCH` ứng viên từ LightRAG để đảm bảo sau khi lọc modality / org / document_type và rerank vẫn đủ số tài liệu yêu cầu.

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
                [--batch-size 100] [--force] [--dry-run] [--direct] [--repair]
ami-rag verify [--doc-ids ID [ID ...]] [--type …] [--limit N] [--batch-size 100] [--all]
ami-rag status
ami-rag purge-doc --doc-id <id>
```

| Cờ | Ý nghĩa |
|---|---|
| `--all` / `--doc-ids` | bắt buộc một trong hai; `--all` duyệt doc `status: active` của `organization_db.documents` |
| `--type` | lọc `document_type` (danh sách phân tách bằng dấu phẩy) |
| `--limit`, `--batch-size` | giới hạn số doc; kích thước batch cursor Mongo |
| `--dry-run` | in bảng theo loại `total / skip(done) / to_ingest` rồi thoát, không ghi gì (với `--repair`: liệt kê doc thiếu vector/entity) |
| `--force` | đánh dấu `stale` các doc `processed` đã chọn để nạp lại dù `source_hash` không đổi |
| `--direct` | nạp ngay trong tiến trình CLI (không qua Redis/worker), in `indexed/skipped/failed` |
| `--repair` | với doc `processed`/`failed`: chạy lại phần còn thiếu của index từ chunk text đã lưu trong Mongo (chuyển thành `processed` khi xong); doc chưa từng có chunk tự động đưa vào full ingest |

Mặc định (không `--direct`, không `--force`) lệnh bỏ qua doc đã `processed` có `source_hash` không đổi (in `skipping N already processed document(s)`), chỉ XADD event `created` cho phần còn lại vào `RAG_STREAM`; worker xử lý. Doc `failed`/`stale`/chưa có trong registry luôn được nạp lại.

`--repair` (`ami_rag/index_check.py::repair_document`): chạy trên tập doc `processed` và `failed` đã có chunk text trong `multimodal_text_chunks`. Lệnh kiểm tra index thật: (1) re-embed các chunk chưa có vector trong Qdrant; (2) chạy lại trích entity/relation chỉ khi doc chưa có entity nào. Khi hoàn tất, doc được cập nhật `status: processed` và lưu `index_stats`. Nếu thiếu chunk text hoặc chưa từng ingest, doc tự động chuyển sang hàng đợi full ingest. Lệnh có cơ chế **preflight** và **circuit breaker**: gặp lỗi cạn credit (`QUOTA_EXHAUSTED`) hoặc sai khóa (`AUTH_FAILED`) sẽ **dừng ngay lập tức** toàn bộ tiến trình thay vì đánh dấu fail hàng loạt các doc còn lại. Sau khi kết thúc, CLI in bảng phân loại lỗi theo mã (`QUOTA_EXHAUSTED`, `RATE_LIMITED`, `UPSTREAM_UNAVAILABLE`) kèm hướng dẫn khắc phục cụ thể.

`verify` đếm, cho từng doc: số chunk (`doc_status.chunks_list`), vector chunk trong Qdrant, entity/relation (`full_entities`/`full_relations`); in dòng `doc=… chunks=… vectors=… entities=… relations=… OK|INCOMPLETE: lý do` kèm trạng thái registry (`processed`, `failed(MÃ_LỖI)`, `never-ingested`), in bảng tóm tắt lỗi theo nguyên nhân và `index ready: X/Y docs`; thoát mã 1 nếu còn doc thiếu. `status` in workspace, số doc active, đếm registry theo status, `queue pending`, danh sách doc failed nhóm theo mã lỗi (`failed documents by cause`) và tình trạng rerank service. `purge-doc` xoá doc khỏi LightRAG, asset MinIO và registry.

### 7.1 Chạy CLI khi dùng Docker

`docker-compose.ami.yml` chỉ có hai service: `ami-rag-api` và `ami-rag-worker`. Không có service `ami-rag` riêng; `ami-rag` là lệnh bên trong image (entry point `ami_rag.cli:main`). Chạy lệnh trong container đang chạy, dùng cùng `.env` (Mongo, Redis, Qdrant, Gemini) với service:

```bash
# từ thư mục gốc repo; -T tắt TTY (cần khi pipe/tee/cron, vô hại khi chạy tay)
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag status
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag reindex --all --dry-run
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag reindex --all
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag verify
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag reindex --all --repair --type text --limit 20
```

Rút gọn bằng Makefile (`make cli`, chạy `ami-rag <lệnh con>` trong `ami-rag-worker`):

```bash
make cli status
make cli verify
make cli -- reindex --all --dry-run        # cờ dạng --xxx cần "--" để make không tự parse
make cli -- reindex --all --repair --type text --limit 20
make cli ARGS="reindex --doc-ids <id> --force"   # hoặc truyền qua ARGS, không cần "--"
make cli CLI_SERVICE=ami-rag-api status    # đổi container
```

Chạy nền cho lượng dữ liệu lớn (`--repair`, `--direct`, `--force` trên cả nghìn doc): thêm `BG=1`. Lệnh chạy detached trong container (`exec -d`), sống tiếp khi đóng terminal/SSH; log ghi vào `/app/output/cli/cli-<thời gian>.log` (volume `rag-output`, symlink `latest.log`), dòng cuối `# exit=<mã>`.

```bash
make cli BG=1 -- reindex --all --repair     # in đường dẫn log, trả về ngay
make cli-logs                               # tail -f log của lệnh nền gần nhất (Ctrl-C chỉ dừng tail)
make cli-ps                                 # liệt kê mọi tiến trình `ami-rag <lệnh>` đang chạy (pid, thời gian)
make cli-stop                               # dừng lệnh nền gần nhất (chỉ pid đó)
make cli-stop PID=<pid>                     # dừng pid lấy từ cli-ps
```

`cli-stop` chỉ kill đúng pid đã ghi, không đụng worker/API hay phiên `ami-rag` mở tay khác. Dừng giữa chừng an toàn vì `--repair`/`reindex` idempotent: chạy lại sẽ bỏ qua doc đã đủ. Lệnh `reindex` mặc định chỉ XADD event nên chạy xong rất nhanh, việc nặng do worker làm nền; `BG=1` chủ yếu có ích cho `--repair` và `--direct`. Đừng chạy hai `--repair`/`--direct` song song (cùng ghi một index và cùng quota Gemini); `cli-ps` để kiểm tra trước khi bắt đầu.

Không có `--`, `make cli reindex --all` báo `unrecognized option '--all'` (lệnh con không cờ như `status`, `verify` thì không cần). Target kiểm tra container đang chạy theo `docker compose exec`, nên cần `make start_docker` trước.

Sai thường gặp: `docker compose exec ami-rag reindex --all` lỗi vì `ami-rag` là tên lệnh chứ không phải tên service. Đúng: `exec <service> ami-rag <lệnh con>`, với `<service>` là `ami-rag-worker` hoặc `ami-rag-api`.

Chọn container:

| Lệnh | Nên chạy trong | Lý do |
|---|---|---|
| `status`, `reindex` (mặc định, `--dry-run`, `--force`), `purge-doc` | `ami-rag-worker` hoặc `ami-rag-api` | chỉ đọc Mongo/Redis hoặc XADD event; worker đang chạy xử lý tiếp |
| `verify` | `ami-rag-worker` | đọc Mongo + Qdrant, cần `.env` của worker |
| `reindex --repair`, `reindex --direct` | `ami-rag-worker` | khởi tạo LightRAG ngay trong tiến trình CLI và gọi Gemini; worker có đủ model cache và biến môi trường |

Lưu ý:
- `--direct` và `--repair` ghi thẳng vào index từ tiến trình CLI. Nên chạy khi queue rỗng (`status` cho `queue pending: 0`) để không trùng với doc worker đang xử lý. Hai lệnh này cùng quota Gemini với worker.
- Lệnh `exec` dài nên chạy nền, có log: `docker compose -f docker-compose.ami.yml exec -T ami-rag-worker ami-rag reindex --all --repair > repair.log 2>&1 &`, rồi `tail -f repair.log`. Hoặc dùng `tmux`/`screen`. Ngắt phiên SSH có thể làm dừng lệnh đang chạy ở foreground; `reindex` mặc định chỉ XADD event nên chạy lại sau khi bị ngắt là an toàn (doc đã `processed` bị bỏ qua).
- Container phải đang chạy. Nếu chưa: `make start_docker` (hoặc `docker compose -f docker-compose.ami.yml up -d`). Nếu cần chạy mà không dựa vào container sẵn có (không đụng worker):
  `docker compose -f docker-compose.ami.yml run --rm --no-deps ami-rag-worker ami-rag verify`
  (`run` tạo container tạm cùng image/`.env`/volume, không publish cổng 9109; `--rm` xoá khi xong).
- Image đang chạy phải có code CLI mới. Sau khi sửa `ami_rag/`, `docker compose -f docker-compose.ami.yml up -d --build` (lâu, xem mục 5) rồi mới chạy `verify`/`--repair`; image cũ báo `invalid choice` hoặc thiếu cờ `--repair`.
- Mã thoát của `exec` là mã thoát của lệnh trong container, nên `verify` (trả 1 khi còn doc thiếu) dùng được trong script/CI: `docker compose -f docker-compose.ami.yml exec -T ami-rag-worker ami-rag verify || echo "còn doc thiếu"`.
- Ngoài Docker (máy dev, đã `uv sync`): `uv run ami-rag <lệnh con>` với cùng cờ; host trong `.env` phải truy cập được từ máy đó (tên container như `redis`, `mongo` chỉ phân giải trong `ami-network`).

## 8. Runbook backfill dữ liệu cũ

Các lệnh `ami-rag …` dưới đây viết ở dạng ngắn; khi chạy bằng Docker thêm tiền tố `docker compose -f docker-compose.ami.yml exec ami-rag-worker` (mục 7.1).

Mục tiêu: nạp toàn bộ doc có sẵn trong `organization_db.documents` vào index mới (workspace `multimodal`, collection `organization_db.multimodal_*`). Doc đã `processed` từ trước khi có field liên kết (`document_oid`, `organization_unit_id`, `owner_id`, ...) chỉ được bổ sung các field này khi nạp lại (`--force`). Doc cũ chỉ có text/bảng đơn giản trong Mongo vẫn được parse lại từ file MinIO (nếu `original_file_name` không rỗng/`.txt`); text Mongo của pdf/docx **không** được dùng.

1. Chuẩn bị: `.env` đúng, worker chạy (`make start_docker` hoặc `make start_worker`), `make health` OK, `ami-rag status` thấy `rerank service: ok` và `queue pending: 0`. Worker cần GPU/MinerU và model cache sẵn (xem mục 5).
2. Dry-run toàn bộ, kiểm tra cột `source` (`mongo_text`/`minio_parse`) hợp lý:
   `ami-rag reindex --all --dry-run`
3. Thử text/crawl nhỏ (nhanh, không cần MinerU):
   `ami-rag reindex --all --type text,crawl --limit 20`
4. Thử file thật (MinerU, chậm): `ami-rag reindex --all --type pdf,docx --limit 3`. Kiểm tra `GET /admin/documents/{id}` (`counts` có `table`/`image`), `GET /admin/documents/{id}/content`, và một truy vấn `POST /v2/rag/` v2 với `filters.modality`.
5. Chạy toàn bộ: `ami-rag reindex --all` (doc đã `processed` và không đổi bị bỏ qua ngay ở CLI; thêm `--force` để nạp lại tất cả). Trước đó kiểm tra Gemini còn credit (một lần embed thử), vì hết credit thì index rỗng dù doc báo `processed` ở bản worker cũ.
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
8. Kiểm tra index thật: `ami-rag verify` (chỉ đọc). Báo cáo số doc theo tầng: tổng doc active trong Mongo → có dòng registry `processed` → có chunk text → chunk có vector Qdrant → có entity; số "index ready" là doc đủ vector (và entity). Doc thiếu: `ami-rag reindex --all --repair --dry-run` rồi bỏ `--dry-run`; lỗi table/image của pdf/docx: `--doc-ids <id> --force`.
9. Hoàn tất khi `queue pending` và `lag` về 0, `status` không còn `processing`/`failed`, `multimodal_rag_ingest_documents{status="processed"}` xấp xỉ số doc active, và `ami-rag verify` báo `index ready` bằng số doc active (doc lỗi `IngestIncompleteError` nằm ở `failed` với lý do trong `error`).

## 9. Monitoring

Chi tiết triển khai: [`monitoring/README.md`](../monitoring/README.md).

Tracing (OTLP/HTTP): chỉ truy vấn `POST /v2/rag` được trace — span `rag.retrieval` (input: attribute `input.query`/`input.mode`/`input.version`/`input.top_k`; output: danh sách document đầy đủ trong event `documents_json`) cùng span con theo stage (`raganything_query`/`rerank`/`resolve`). Worker không trace (ingest chỉ có metric Prometheus) và FastAPI telemetry tự động bị tắt (`FastAPI(telemetry={"auto_configure": False, "tracing": False, "metrics": False, "logs": False})` trong `ami_rag/api/main.py`), nếu không FastAPI ≥ 0.142 đọc `OTEL_EXPORTER_OTLP_ENDPOINT` rồi trace mọi route và đẩy metrics/logs tới `/v1/metrics`, `/v1/logs` (Tempo trả `404`: log `Failed to export metrics batch code: 404`). Tempo dùng riêng cho service này: release helm `tempo-multimodal-rag` (`monitoring/helm/tempo-multimodal-rag-values.yaml`, chart `grafana/tempo` 1.24.4, tách khỏi Tempo của conversational-agent); datasource Grafana uid `multimodal_rag_log` nạp qua sidecar bằng ConfigMap `monitoring/helm/grafana-datasource-tempo-multimodal-rag.yaml`; URL Grafana `http://tempo-multimodal-rag.monitoring.svc.cluster.local:3200`; `OTEL_EXPORTER_OTLP_ENDPOINT` trỏ tới NodePort OTLP/HTTP 4318 của service (`kubectl get svc tempo-multimodal-rag -n monitoring`).

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
| `stage_duration_seconds` | Histogram | `stage` (download/parse/upload_assets/insert/verify/delete) |
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

Một số test gốc stub module `lightrag` trong `sys.modules`, nên các test tích hợp trong `tests/ami_service/test_integration_*.py` tự skip khi chạy chung hai bộ; chạy `tests/ami_service` riêng để chúng chạy với `lightrag` thật. Phần còn lại của `tests/ami_service` dùng fake (không cần dịch vụ ngoài). `test_index_check.py` phủ index check của worker, `inspect_document`/`repair_document` (đường re-embed) và việc `reindex` bỏ qua doc đã `processed`; đường trích entity của `repair` cần Gemini thật nên chưa có test tự động.

## 11. Phiên bản

`pyproject.toml` ghim `lightrag-hku>=1.4.9,<1.5` (cài từ PyPI, không dùng checkout local). Extra `service` thêm FastAPI, uvicorn, pydantic-settings, redis, pymongo, qdrant-client, httpx, minio, prometheus-client, OpenTelemetry, openai, google-genai. Entry point: `ami-rag-api`, `ami-rag-worker`, `ami-rag`.

`mineru[core]>=3.4.1,<4`: MinerU 4.x đổi CLI (`mineru parse <path>`, `-p` = pages) nên không tương thích với lệnh `mineru -p <file> -o <dir> -m ...` mà `raganything/parser.py` gọi; bắt buộc ghim `<4`. MinerU 3.4.x mặc định backend `hybrid-engine` (nặng VRAM) khi không truyền `-b`, và chọn thiết bị bằng biến môi trường `MINERU_DEVICE_MODE` (không có cờ `-d`); `MINERU_VIRTUAL_VRAM_SIZE` (GB) buộc MinerU chọn batch size như thể GPU có chừng đó VRAM. Đo VRAM thực tế bằng `notebooks/mineru_vram_check.ipynb` (Colab).

Thay đổi liên quan trong thư viện `raganything`: chunk multimodal lưu field có cấu trúc và `RAGAnything.aquery_data` làm giàu chunk; xem `docs/architecture.md` và `docs/api_reference.md`.
