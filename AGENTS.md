<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **RAG-Anything** (5926 symbols, 11096 relationships, 297 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
| 6 | Dọn docs + xóa LightRAG còn sót | 🔴 **CÒN LẠI** |

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

## Phase 6 — 🔴 còn lại
- Xóa `reproduce/` + LightRAG examples (`examples/`), deps lightrag còn sót (pyproject), code dead (batch.py? resilience? callback paths?), docs cũ.
- Cập nhật `docs/ami_service.md` cuối cùng; sample questions `tests/fixtures/` nếu cần.

## Quy tắc (bắt buộc)
- **MỖI phase commit riêng, STOP xin duyệt sau mỗi phase.** Trước commit: `gitnexus_detect_changes()`.
- Trước khi sửa symbol: `gitnexus_impact` trước (AGENTS.md gitnexus block ở trên).
- Test: `pytest` cần `$env:PYTHONIOENCODING="utf-8"`; lint ruff 0.16.0 global (`ami_rag/**` đã per-file-ignores BLE001/S110/B008).
- GitNexus CLI phải dùng bản global 1.6.5 (`C:\Users\phanb\AppData\Local\pnpm\bin\gitnexus.CMD`), KHÔNG `npx` latest.
- File có tiếng Việt: chỉ dùng tool write/edit, KHÔNG Add-Content/Set-Content (cp1252 mojibake).
- Không gọi LLM/API ngoài ý muốn; describe là stage duy nhất gọi LLM (tuỳ chọn).
- Repo Qwen server nằm RIÊNG ngoài RAG-Anything: `D:\Code\Python\Ami\RAG\qwen-embedding-server\` (git `31a5393`, 20 tests pass/1 skip).
