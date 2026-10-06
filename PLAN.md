# PLAN — Migration RAG-Anything sang vector pipeline thuần

> Cập nhật: 06/10/2026. Fork `ami_rag` (service RAG-Anything) migrate từ LightRAG KG + Gemini embedding sang **vector pipeline thuần** với `Qwen/Qwen3-VL-Embedding-2B` chạy remote trên máy B (HTTP, port 8007).

## Objective
- Pipeline (parse → describe → chunk → embed → indexed → query) chạy trên máy A, KHÔNG cần GPU, KHÔNG gọi LLM ngoài stage describe (tuỳ chọn).
- Embedding chạy trên máy B qua HTTP (`Qwen/Qwen3-VL-Embedding-2B`, dim 2048, MRL 64–2048, last-token pooling, L2 normalize).
- API chỉ giữ `POST /rag` + `/rag/stream` (v2-only); drop v1/filters/include_kg/mode.
- CLI (`ami-rag`) chỉ còn status/retry/reindex.
- Xóa LightRAG KG: không entity, không graph, không neo4j.

## Trạng thái phase
| Phase | Nội dung | Trạng thái |
|-------|----------|-----------|
| 0 | Inventory codebase (`docs/migration_inventory.md`) | ✅ commit `7221759` |
| 1 | Thiết kế tổng thể (đã duyệt) | ✅ |
| 2 | Qwen embed server (repo riêng) + client RemoteEmbedder + cache SQLite | ✅ commit `394c8df` |
| 3 | CLI + DocStatusStore + lockfile + PipelineRunner protocol | ✅ commit `32bf44c` |
| 4 | **VectorPipeline + wiring factory/worker/API + strip LightRAG** | 🔴 **ĐANG LÀM** |
| 5 | Reindex docs cũ (scan legacy → pending) | ⏳ |
| 6 | Dọn docs + xóa LightRAG còn sót | ⏳ |

---

## Phase 2 — ✅ (commit `394c8df`)
- Repo mới `D:\Code\Python\Ami\RAG\qwen-embedding-server\` (git `31a5393`, 20 tests pass/1 skip): FastAPI server, vendored `app/qwen3_vl_embedding.py`, Dockerfile `pytorch:2.8.0-cuda12.6`, docker-compose, `.env.example` (EMB_*), README. Port 8007.
- Client trong RAG-Anything:
  - `ami_rag/core/embedder.py`: protocol + errors (`EmbedServerUnreachable/EmbedModelMismatch/EmbedServerOOM/EmbedInputTooLong/EmbedAuthError/EmbedCircuitOpen`), `collection_name()`.
  - `ami_rag/core/remote_embedder.py`: `RemoteEmbedder` + `build_remote_embedder` (handshake, per-response check, batching items+bytes, semaphore, circuit breaker 5 fail liên tiếp, backoff_base).
  - `ami_rag/core/embedding_cache.py`: SQLite, key = `model|dim|instruction_ns|sha256(text)`.
- Settings block EMBED_*; bỏ GEMINI_*/LLM_PROFILE/QWEN_EMBED_*/REPAIR_CONCURRENCY; bỏ dep google-genai.
- Factory: `_build_llm_func`/`_build_vision_func` qwen-only; `_build_embedding_func` wrap `get_remote_embedder()` thành LightRAG EmbeddingFunc; singleton + `close_rag`.
- 16 client tests pass (`tests/test_remote_embedder.py`, httpx.MockTransport).

## Phase 3 — ✅ (commit `32bf44c`)
- `ami_rag/cli.py`: cmd_status/cmd_retry/cmd_reindex, deps injectable, LOCK_PATH `.ami_rag_cli.lock`, exit codes 0/1/2.
- `ami_rag/storage/doc_status.py`: `DocStatusStore` — schema stages parse→describe→chunk→embed→indexed; statuses pending/processing/indexed/failed/stale; legacy `processed` đọc là `stale`; `effective_status`, `begin_attempt/release_attempt`, `mark_stage/mark_indexed` rich meta (source, parser, file_path, counts, page_count, assets, document_type, title, organization_unit_id, owner_id, meta, content_hash, embed_model/embed_dim/chunker_version/chunk_count, error/error_stage/attempts).
- `ami_rag/core/lockfile.py`: pid-alive (ctypes win32 / os.kill posix), STALE_LOCK_AGE 6h.
- `ami_rag/core/pipeline.py`: `PipelineRunner` protocol (`preflight_embed()`, `run(doc_id, from_stage=None, dry_run=False) -> StageOutcome`), `validate_stage` (reject "indexed"), `create_runner` (đang raise RuntimeError — wire ở Phase 4).
- Tests: 30 pass (test_cli 21 + test_doc_status_store 9). `docs/ami_service.md` cập nhật (config EMBED_*, §5.2 embed server, §7 CLI, §7.1 Docker, §8 runbook, §11). Makefile: status|retry|reindex.

## Phase 4 — 🔴 ĐANG LÀM
### Đã xong
- ✅ `ami_rag/core/vector_store.py`: `VectorStore` protocol + `ChunkRecord` + `SearchHit` + `QdrantVectorStore` (ensure_collection dim check, upsert uuid5(NAMESPACE_URL, chunk_id), query_points search, delete_by_doc FilterSelector, count; `VectorStoreError`).
- ✅ `ami_rag/storage/doc_status.py`: bổ sung begin_attempt/release_attempt + mark_stage/mark_indexed rich meta (10 tests pass).
- ✅ Khảo sát: `assets.py` (MinioAssetStore: fetch/upload_content_list_assets/save_content_list/load_content_list/presign/delete_doc_assets, CONTENT_LIST_NAME="content_list.json"), modalprocessors (BaseModalProcessor.__init__ lightrag-coupled 371-427; generate_description_only 454/930/1156/1358/1546 pure; generate_chunk_sections ~501 pure; storage-writing callers 1053/1087, 1260/1289, 1455/1477, 1634/1645 sẽ bypass), api/schemas.py (RAGRequest cần simplify v2-only), routes/rag.py (get_rag, _search/_run_search 221-244), routes/admin.py (pipeline_status dùng rag.lightrag.doc_status:56, reprocess_failed dùng last_hash), api/main.py lifespan, raganything/base.py (DocStatus enum).
- 🔴 `ami_rag/core/vector_pipeline.py`: **VỪA VIẾT, CHƯA CLEAN** (715 dòng, syntax OK). Docstring thiết kế:
  - Stages: parse → describe → chunk → embed → indexed; ghi DocStatusStore ở MỖI stage.
  - parse: qua `parse_document`, content_list lưu MinIO.
  - describe: modal processor cho multimodal, lưu `{doc_id}/descriptions.json`; chỗ DUY NHẤT gọi LLM (tuỳ chọn).
  - chunk: text plain splitter; multimodal template; lưu `{doc_id}/chunks.json`; không gọi LLM.
  - embed: RemoteEmbedder + cache; luôn delete_by_doc trước upsert (idempotent); KHÔNG entity/graph.
  - indexed: mark_indexed + verify count Qdrant == số chunk.
  - dry_run: GET /info (model/dim/instruction) + đếm cache hits; KHÔNG /embed, KHÔNG ghi gì.

### Việc cần làm NGAY với `vector_pipeline.py`
1. **Sửa `run()`**: flow nonlocal `doc`/`content_list`/`descriptions` lỗi (do_parse dùng `doc` trước khi gán; truyền `row` cho `_stage_parse` không đúng shape). Viết lại tuần tự:
   - load row (DocStatusStore.get) → load doc (docs_repo) → parse (nếu cần) → describe (load content_list nếu thiếu) → chunk (load descriptions nếu thiếu) → embed (load chunks.json nếu thiếu, else build lại) → mark_indexed.
2. **Xóa junk**: `Midclass` (đổi tên → `ArtifactPaths`), `_file_rel_path`, `_content_hash` (dùng lại hay bỏ tùy), `_sheet_ok`, `_describe_fn`, `_require_content_list` (nonsense — xử lý inline trong run), import thừa (`urllib.error`? check).
3. **Verify `asset_store.doc_prefix(doc_id)`** tồn tại trong `ami_rag/storage/assets.py` — nếu không, dùng `f"{prefix}/{doc_id}/"`.
4. **Thêm `RemoteEmbedder.count_cache_hits(texts)`** vào `ami_rag/core/remote_embedder.py` (lookup EmbeddingCache với instruction_ns, KHÔNG gọi HTTP) — `_dry_run` đang gọi method này.
5. **`_stage_parse(doc_id, doc)`**: bỏ fallback từ row; nếu `doc is None` (không có docs_repo) → raise lỗi rõ ràng.

### Còn lại của Phase 4
- Wire `create_runner` trong `ami_rag/core/pipeline.py` → VectorPipeline (deps: embedder via `build_remote_embedder`, QdrantVectorStore, MinioAssetStore, DocStatusStore, get_parser).
- `factory.py`: build_pipeline/get_pipeline thay get_raganything/get_rag.
- Worker ingest rewrite: pipeline thay `rag_anything.lightrag`/`_verify_index`; `_build_default_deps`; count_content_list (line 52).
- API v2-only: routes/rag.py (get_rag → factory mới, _search embed+search+rerank), routes/admin.py (pipeline_status không dùng rag.lightrag.doc_status, reprocess_failed), routes/main.py lifespan (get_pipeline + IngestWorker + close), schemas.py (RAGRequest: chỉ messages/top_k/include_references).
- modalprocessors.py strip: BaseModalProcessor lightrag=None tolerant, xóa storage attrs (389-392, 402-403) + storage-writing methods (callers 1053/1087/1260/1289/1455/1477/1634/1645); giữ generate_description_only/generate_chunk_sections.
- processor.py: giữ parse_document (387); xóa multimodal graph paths (630-1732).
- query.py rewrite: aquery_data → embed+search+rerank; aquery → answer_func; xóa aquery_vlm_enhanced/multimodal paths (460, 306, 629-691, 920-925).
- raganything.py rewrite; utils.py trim.
- `tests/fixtures/`: sample questions; pipeline tests (fake embedder/vector_store/asset_store trong `tests/ami_service/` — extend fakes.py với FakeAssetStore descriptions/chunks).
- pytest + ruff + `gitnexus_detect_changes()` → commit Phase 4 → STOP xin duyệt.

## Phase 5 — ⏳ Reindex docs cũ
- Scan legacy records (`processed`) → tạo pending records trong DocStatusStore mới; chạy `ami-rag reindex`.

## Phase 6 — ⏳ Dọn dẹp
- Xóa LightRAG còn sót (deps, docs, code dead); cập nhật `docs/ami_service.md` cuối cùng.

---

## Quy tắc làm việc (bắt buộc)
- **MỖI phase commit riêng, STOP xin duyệt sau mỗi phase.** Trước commit: `gitnexus_detect_changes()`.
- Trước khi sửa symbol: `gitnexus_impact` trước (xem AGENTS.md gitnexus block).
- Test: `pytest` cần `$env:PYTHONIOENCODING="utf-8"`; lint ruff 0.16.0 global.
- GitNexus CLI dùng bản global 1.6.5 (`C:\Users\phanb\AppData\Local\pnpm\bin\gitnexus.CMD`), KHÔNG `npx` latest (DB mismatch v42 vs v40).
- File có tiếng Việt: chỉ dùng tool write/edit, KHÔNG Add-Content/Set-Content (cp1252 mojibake).
- Không gọi LLM/API ngoài ý muốn; describe là stage duy nhất gọi LLM (tuỳ chọn).
- Repo Qwen server nằm RIÊNG: `D:\Code\Python\Ami\RAG\qwen-embedding-server\`.

## Quyết định đã duyệt
- API v2-only: `POST /rag` + `/rag/stream`; drop v1/filters/include_kg/mode.
- Xóa `reproduce/` + LightRAG examples.
- Backend: Qdrant; Mongo cho DocStatusStore (extend registry `multimodal_rag_documents`).
- Dim 2048 qua env (MRL 64–2048); cache embedding ON mặc định (SQLite); `EMB_MAX_INPUT_TOKENS=8192`; port 8007.
- Qwen server là repo MỚI ngoài RAG-Anything (không đụng `embedding-server` BGE-M3 cũ).
- Giữ parser (MinerU) + parser config; metadata multimodal giữ nguyên.

## Môi trường / sự cố đã biết
- Env thiếu module `lightrag` → 18 collection errors + 6 observability failures trong test suite là PRE-EXISTING (đã verify qua git stash trên HEAD sạch).
- Collection naming: `{prefix}__{embed_model_slug}__{chunker_version}` (`collection_name()` trong `ami_rag/core/embedder.py`).
- Circuit breaker: 5 fail liên tiếp → `EmbedCircuitOpen`; retry chỉ transient (timeout/5xx/429/connect), không bao giờ 4xx.
