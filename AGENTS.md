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
| 4 | **VectorPipeline + wiring factory/worker/API + strip LightRAG** | 🔴 **ĐANG LÀM** |
| 5 | Reindex docs cũ (scan legacy → pending) | ⏳ |
| 6 | Dọn docs + xóa LightRAG còn sót | ⏳ |

## Phase 4 — đang làm
- ✅ `ami_rag/core/vector_store.py`: QdrantVectorStore (ensure_collection dim check, upsert uuid5, query_points, delete_by_doc, count).
- ✅ `ami_rag/storage/doc_status.py`: begin_attempt/release_attempt + mark_stage/mark_indexed rich meta (10 tests pass).
- 🔴 `ami_rag/core/vector_pipeline.py`: **VỪA VIẾT, CHƯA CLEAN** (715 dòng, syntax OK). Cần:
  1. Sửa `run()`: flow nonlocal `doc`/`content_list`/`descriptions` lỗi (do_parse dùng `doc` trước khi gán; truyền `row` cho `_stage_parse` không đúng shape). Viết lại tuần tự: load row → load doc (docs_repo) → parse (nếu cần) → describe (load content_list nếu thiếu) → chunk (load descriptions nếu thiếu) → embed (load chunks.json nếu thiếu, else build lại) → mark_indexed.
  2. Xóa junk: `Midclass` (đổi tên → ArtifactPaths), `_file_rel_path`, `_content_hash` (dùng lại hay bỏ tùy), `_sheet_ok`, `_describe_fn`, `_require_content_list` (nonsense — xử lý inline trong run), import thừa (`urllib.error`?, check).
  3. Verify `asset_store.doc_prefix(doc_id)` có tồn tại trong `ami_rag/storage/assets.py` — nếu không, dùng `f"{prefix}/{doc_id}/"`.
  4. Thêm `RemoteEmbedder.count_cache_hits(texts)` vào `ami_rag/core/remote_embedder.py` (lookup EmbeddingCache với instruction_ns, KHÔNG gọi HTTP) — `_dry_run` đang gọi method này.
  5. `_stage_parse(doc_id, doc)`: bỏ fallback từ row; nếu `doc is None` (không có docs_repo) → raise lỗi rõ ràng.
- ⏳ Còn lại: wire `create_runner` trong `ami_rag/core/pipeline.py` → VectorPipeline; factory.py (build_pipeline/get_pipeline thay get_raganything); worker ingest rewrite; api routes rag.py/admin.py/main.py/schemas.py (v2-only); modalprocessors.py strip (BaseModalProcessor lightrag-tolerant, xóa storage attrs + storage-writing methods ~1053/1087/1260/1289/1455/1477/1634/1645); processor.py (giữ parse_document, xóa graph paths 630-1732); query.py rewrite (aquery_data → embed+search+rerank); raganything.py rewrite; utils.py trim; tests/fixtures sample questions; pipeline tests (fake embedder/vector_store/asset_store trong `tests/ami_service/`).

## Quy tắc (bắt buộc)
- **MỖI phase commit riêng, STOP xin duyệt sau mỗi phase.** Trước commit: `gitnexus_detect_changes()`.
- Trước khi sửa symbol: `gitnexus_impact` trước (AGENTS.md gitnexus block ở trên).
- Test: `pytest` cần `$env:PYTHONIOENCODING="utf-8"`; lint ruff 0.16.0 global.
- GitNexus CLI phải dùng bản global 1.6.5 (`C:\Users\phanb\AppData\Local\pnpm\bin\gitnexus.CMD`), KHÔNG `npx` latest.
- File có tiếng Việt: chỉ dùng tool write/edit, KHÔNG Add-Content/Set-Content (cp1252 mojibake).
- Không gọi LLM/API ngoài ý muốn; describe là stage duy nhất gọi LLM (tuỳ chọn).
- Repo Qwen server nằm RIÊNG ngoài RAG-Anything: `D:\Code\Python\Ami\RAG\qwen-embedding-server\` (git `31a5393`, 20 tests pass/1 skip).
