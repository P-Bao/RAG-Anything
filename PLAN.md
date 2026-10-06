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
| 4 | VectorPipeline + wiring factory/worker/API + strip LightRAG | ✅ commit `b2f8828` |
| 5 | Reindex docs cũ (scan legacy → pending) | ✅ commit `787ff0d` |
| 6 | Dọn docs + xóa LightRAG còn sót | 🔴 **CÒN LẠI** |

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

## Phase 4 — ✅ (commit `b2f8828`)
- `ami_rag/core/vector_store.py`: `VectorStore` protocol + `ChunkRecord` + `SearchHit` + `QdrantVectorStore` (ensure_collection dim check, upsert uuid5(NAMESPACE_URL, chunk_id), query_points search, delete_by_doc FilterSelector, count; `VectorStoreError`).
- `ami_rag/core/vector_pipeline.py` (clean): `run()` tuần tự load row → load doc (docs_repo) → parse → describe → chunk → embed → mark_indexed; ghi DocStatusStore ở MỖI stage; resume từ MinIO artifacts (`content_list.json`/`descriptions.json`/`chunks.json` qua `ArtifactPaths`); `_load_or_build_chunks` (chunks.json reuse, build lại nếu thiếu); `_stage_parse(doc_id, doc)` raise khi doc None (không fallback từ row); embed luôn delete_by_doc trước upsert (idempotent) + verify count; indexed: mark_indexed meta từ row refresh + doc link fields; dry_run (handshake + đếm chunk/cache hits, không embed/không ghi); `delete_doc` (vector + assets + registry).
- `MinioAssetStore.save_json/load_json` (assets.py) + `RemoteEmbedder.count_cache_hits(texts)` (lookup cache SQLite, không HTTP).
- Wiring: `create_runner` (pipeline.py) → `build_pipeline`; `factory.py` không còn LightRAG: `build_pipeline/get_pipeline/close_pipeline` (+ `close_rag` alias), `_build_llm_func/_build_vision_func` thuần OpenAI-compatible qua `openai` lib, `_build_modal_processors` (lightrag=None, audio/video gate theo deps_available).
- `ingest_worker.py` rewrite: `IngestWorker(runner=...)` — created/updated → ensure_pending + `runner.run(doc_id, from_stage="parse")`; unchanged (indexed cùng content_hash + embed model/chunker) skip; deleted → `runner.delete_doc`; DocStatusStore làm state repo (thay RagDocumentsRepo); xóa `index_check.py` + `storage/rag_documents.py`.
- API v2-only: `RAGRequest` = messages/top_k/include_references (drop version/mode/filters/include_kg, v1 schemas); `/rag` = `embed_query` → `vector_store.search` → rerank → `_build_documents` (resolve qua `DocResolver.resolve_doc_id` từ chunk payload `doc_id`); references dedup theo document; `/rag/stream` NDJSON giữ nguyên format; admin `pipeline_status`/`reprocess_failed`/`document_status` dùng DocStatusStore.
- raganything thin (không import lightrag): `query.py` (aquery → llm_model_func; aquery_data → embedder+vector_store), `processor.py` (parse-only: parse_document + content-based doc_id, bỏ cache KV + insert + multimodal graph paths), `raganything.py` (parser + modal processors + query, fields embedder/vector_store/collection), `utils.py` (bỏ `insert_text_content*`, logger local), `config.py` (`get_env_value` local).
- `modalprocessors.py` strip: `BaseModalProcessor.__init__(lightrag=None, modal_caption_func, context_extractor, tokenizer, global_config)` tolerant; xóa `_create_entity_and_chunk`/`_process_chunk_for_extraction`/`process_multimodal_content` (mọi file gồm audio/video); `compute_mdhash_id` local; giữ `generate_description_only`/`generate_chunk_sections` (pure).
- Tests: 480 pass khi commit Phase 4; xóa 15 test file old-pipeline (integration lightrag, insert_content_list, multimodal query key/cache, doc_status_creation, modal_chunk_metadata, ...); rewrite `test_worker.py`/`test_api.py`/`test_observability.py` cho pipeline mới; thêm `test_vector_pipeline.py` (15 test, fakes trong conftest: FakeEmbedder/FakeVectorStore/FakeParser/FakeModalProcessor/FakeDocStatusStore/FakeCLIAssetStore); **thêm `tests/__init__.py`** (bắt buộc: site-packages có package `tests` shadow local dir → `tests.ami_service` import lỗi nếu thiếu); pyproject: per-file-ignores thêm `ami_rag/core/**` (BLE001/S110), `ami_rag/cli.py` thêm S110.

## Phase 5 — ✅ (commit `787ff0d`)
- `ami_rag/core/scan.py`: `scan_pending` (scan backend `documents` active → ensure_pending cho docs thiếu registry row; chỉ đọc Mongo, 0 LLM/0 embed) + `reindex_candidates` (pending + stale, gồm legacy `processed`; indexed/failed loại trừ).
- CLI: `ami-rag reindex --scan` (scan rồi chọn pending+stale); from_stage per-doc (`_from_stage_for`: "chunk" khi content_list đã có trong MinIO, None=full parse khi thiếu; explicit `--from-stage` override tất).
- Tests: 3 test scan mới (`test_cli.py` 23 tests). Full suite: 483 pass, 2 skip (pre-existing: reportlab + lightrag); ruff clean trên `ami_rag`.

## Phase 6 — 🔴 CÒN LẠI
- Xóa `reproduce/` + LightRAG examples (`examples/`), deps lightrag còn sót (pyproject `[project.optional-dependencies]`?), code dead (`raganything/batch.py`, `resilience.py`, callback paths còn dùng?, `notebooks/`?).
- Kiểm tra còn import `lightrag` ở đâu: `rg "lightrag" raganything/ ami_rag/` (parser.py? batch.py?).
- Cập nhật `docs/ami_service.md` cuối cùng (kiến trúc vector pipeline, CLI --scan, collection naming, runbook reindex).
- Sample questions `tests/fixtures/` nếu cần.

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
- Env thiếu module `lightrag` → test suite cũ có 18 collection errors + 6 observability failures PRE-EXISTING; sau Phase 4 (xóa old-pipeline tests + `tests/__init__.py`) suite là 483 pass / 2 skip (reportlab + lightrag importorskip).
- Site-packages có package `tests` (regular) shadow local dir → `tests/ami_service` import lỗi nếu thiếu `tests/__init__.py` (regular package local thắng nhờ sys.path cwd trước site-packages khi `python -m pytest`).
- Collection naming: `{prefix}__{embed_model_slug}__{chunker_version}` (`collection_name()` trong `ami_rag/core/embedder.py`).
- Circuit breaker: 5 fail liên tiếp → `EmbedCircuitOpen`; retry chỉ transient (timeout/5xx/429/connect), không bao giờ 4xx.
