<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **RAG-Anything** (5180 symbols, 9720 relationships, 286 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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

# Task hiện hành — Crawl images: tải ảnh section 'Hình ảnh' về MinIO (12/10/2026)

Trạng thái: **HOÀN TẤT** (5 commit `c361475`..docs). Quyết định user: **luôn bật, không env tắt** (đã xoá `CRAWL_IMAGES_ENABLED`/`CRAWL_IMAGES_PREFIX`); timeout tải 60s/ảnh; ảnh crawl lưu dưới `ASSET_PREFIX` (`rag-assets/{doc_id}/…`) nên `delete_doc_assets` dọn theo doc.

## Nội dung
1. `c361475` — `crawl_images.py::split_image_section`: tách section `Hình ảnh` khỏi text crawl (heading fold không dấu nhận `**Hình ảnh:**`/`## HÌNH ẢNH`/`Hinh anh`; URL bullet/plain/markdown, lọc đuôi ảnh, dedupe; heading không có URL ảnh → giữ nguyên) + 12 test.
2. `80036b7` — `_stage_parse` nhánh `mongo_text` → `_crawl_text_content_list`: split → tải ảnh (semaphore 4, lỗi từng ảnh bỏ qua) → append item `image` kèm `asset_key`; text strip section, strip rỗng → bỏ text item; fail toàn bộ → giữ text gốc (URL không mất khỏi index). Ảnh crawl đi describe/chunk/embed như ảnh pdf.
3. `aff2b87` — lazy-build `CrawlImageWorker` trong pipeline; gỡ plumbing chết `image_worker` khỏi `IngestWorker`/`_build_default_deps` (giờ 4-tuple)/`main.py`; `build_pipeline(image_worker=)` cho test inject.
4. `4c813ca` — luôn bật: xoá setting, split-first (doc không section không dựng worker).
5. commit này — docs (`docs/ami_service.md` §1/§2/§5.3/runbook, `.env.ami.example`, AGENTS.md).

## Verify ảnh doc/pdf (đã kiểm tra code — không phải sửa)
MinerU `img_path` tuyệt đối → `upload_content_list_assets` gắn `asset_key` → `_materialize_images` (describe) → chunk modality `image` → embed `image_b64`. Test phủ: `test_describe_image_via_real_image_processor`, `test_prepare_embed_items_attaches_image_b64`. Còn checklist runtime server: parse 1 PDF có ảnh thật → `counts.image > 0` + chunk Qdrant có `asset_key`.

## Deploy
Doc crawl đã index trước tính năng: `ami-rag reindex --all --from-stage parse` (parse lại cả PDF/DOCX bằng MinerU — ưu tiên `--doc <id>` từng bài crawl). Bài crawl sửa sau này tự chạy đúng luồng, không cần env.

---

# Hiện trạng hệ thống (đã hoàn tất — chỉ giữ phần còn giá trị)

Chi tiết lịch sử: `git log` (`7221759`..`4c813ca`), `docs/ami_service.md`, `docs/migration_inventory.md`.

## Kiến trúc
- Pipeline vector thuần, **không LightRAG** (không entity/graph/neo4j; chỉ còn param compat `lightrag=None` trong `modalprocessors`/`factory`). Stages: `parse → describe → chunk → embed → indexed`, artifacts trung gian (`content_list.json`/`descriptions.json`/`chunks.json`) lưu MinIO nên retry chạy tiếp từ stage lỗi.
- **describe là stage DUY NHẤT có thể gọi LLM** (tuỳ chọn). Máy A không cần GPU, không cài torch/transformers.
- Parse: MinerU (ảnh + layout) + `PARSE_TEXT_SOURCE=backend_data` (pdfplumber/fitz local dựng bố cục text/bảng, nội dung chữ/bảng từ OCR service chính, fallback Vinter máy B :8090 khi lỗi/treo/rỗng). Doc crawl (`mongo_text`): ảnh section `Hình ảnh` **luôn tải** về MinIO (`split_image_section`, item `image` kèm `asset_key`, lỗi bỏ qua) — không có env tắt.
- API v2-only: `POST /rag` + `/rag/stream` (`RAGRequest` = messages/top_k/include_references); query chỉ text. CLI `ami-rag` chỉ còn status/retry/reindex/cleanup.
- Collection Qdrant: `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}` (`collection_name()` trong `ami_rag/core/embedder.py`). Đổi `EMBED_MODEL`/`CHUNKER_VERSION` → collection mới → `ami-rag reindex --scan`.

## Service máy B (HTTP)
| Port | Repo (ngoài RAG-Anything) | Vai trò |
|---|---|---|
| 8080 | `D:\Code\Python\Ami\RAG\nemotron-vl-vllm\` | gateway FastAPI trước 2 vLLM: embed `nvidia/llama-nemotron-embed-vl-1b-v2` + rerank `nvidia/llama-nemotron-rerank-vl-1b-v2` |
| 8007 | `D:\Code\Python\Ami\RAG\qwen-embedding-server\` | `Qwen/Qwen3-VL-Embedding-2B` (backend `custom`, `RemoteEmbedder`) |
| 8090 | `D:\Code\Python\Ami\RAG\vintern-vllm\` | OCR fallback `5CD-AI/Vintern-1B-v3_5` (đã build + 22 test; chưa git init; chạy thật chờ GPU máy B) |

- Contract gateway nemotron (client `OpenAIEmbedder` / `build_vllm_rerank_func` đã đối chiếu khớp): `GET /health`; `POST /v1/embeddings` với `{"input": [items], "input_type": "query"|"document"}` — item `str` | `{"text", "image"=<base64|URL>}` | `{"content": [parts]}`; `POST /v1/rerank` (alias `/rerank`) với `{query, documents, top_n}` → `{results: [{index, relevance_score}]}`, documents `str` hoặc `{"content": [text/image_url parts]}`.
- Contract gateway vintern (client `VinterOcrClient` đã đối chiếu khớp): `POST /v1/ocr` multipart `file` (PDF/ảnh) + `mode=text|table` + `pages` (`1-5`/`1,3,7`) + `dpi`; response `mode=text` → `{pages, text, page_texts: [{page, text}]}`, `mode=table` → `{pages, tables: [{page, html, markdown}]}`; `GET /health` → `{status, model, vllm}`; lỗi 400 (file/mode/trang), 401 (thiếu `Authorization: Bearer`), 502 (vLLM chết), 503 (OOM sau retry). Cần `templates/vintern-ocr.jinja` cho vLLM (tokenizer Vinter đóng template Qwen text-only, không sinh `<image>`).
- Backend dispatch: `EMBED_BACKEND` (`auto|custom|openai`; auto: prefix `Qwen/` → custom) + `RERANK_BACKEND` (`auto|legacy|vllm`; auto: rỗng/chứa `bge` → legacy BGE text-only) + `RERANK_MULTIMODAL` (default True). Resolver: `settings.resolve_embed_backend`/`resolve_rerank_backend`; factory `build_embedder`; cli `_build_embedder`.
- Multimodal: `_prepare_embed_items` gắn ảnh asset cho modality image/table/equation; `build_rerank_documents` làm tương tự cho rerank (thiếu ảnh → fallback text). Dim 2048 cho cả Nemotron và Qwen.
- URL default `EMBED_SERVER_URL`/`RERANK_BASE_URL` = `http://localhost:8080`; dùng Qwen thì phải đổi `EMBED_SERVER_URL` về 8007 qua env.

## Vận hành
- `ami-rag status [--check-embed-server]` — `_check_embed_server` backend-agnostic (`/health` + `embedder.verify()`); doc lỗi nằm ở `failed` kèm `error`/`error_stage`.
- `ami-rag reindex --scan` (doc legacy `processed` → pending) / `--all --from-stage <stage>` / `--dry-run` / `--doc <id>`; from_stage per-doc tự chọn (`chunk` khi `content_list` đã có trong MinIO, full parse khi thiếu). Doc crawl index trước tính năng ảnh crawl → `--from-stage parse` (docs §7 runbook).
- `ami-rag cleanup --stale-models` (xoá Qdrant `{WORKSPACE}__*` không phải collection hiện tại) + `--purge-cache-model "<model>"` (xoá cache SQLite theo prefix `{MODEL}|`). Helpers: `legacy_cleanup.find_stale_model_collections`, `EmbeddingCache.delete_by_prefix`.
- Embed client: handshake + probe xác minh model/dim, batch `EMBED_BATCH_SIZE` + `EMBED_MAX_CONCURRENCY`, cache SQLite (`model|dim|instruction_ns|sha256`), retry CHỈ transient (timeout/5xx/429/connect — không bao giờ 4xx), circuit breaker 5 fail liên tiếp → `EmbedCircuitOpen`.
- OCR service `.153:12006` đã treo toàn cục (accept TCP nhưng ReadTimeout mọi endpoint) → đã có fallback Vinter (`OCR_FALLBACK_*`, `docs/ami_service.md` §5.4): primary lỗi/timeout/rỗng → `VinterOcrClient` :8090, cả hai lỗi → text dùng PyMuPDF local / bảng dùng html pdfplumber. Log warning `chuyển sang fallback Vinter` cho biết service chính đang chết.
- Caption ảnh (`docs/ami_service.md` §5.3): `IMAGE_DESCRIBE_MODE` `auto|gemini|vision|none`; ảnh được materialize từ MinIO sang temp dir (`img_path`) trước khi gọi processor — đã fix bug vision đọc temp dir đã xoá. Deploy máy A cần `CAPTION_LLM_API_KEY` (env) để bật Gemini; mô tả ảnh đổi → `ami-rag reindex --all --from-stage describe`.

## Bẫy đã biết (bắt buộc nhớ)
- **`tests/__init__.py` phải tồn tại**: site-packages có package `tests` shadow local dir → `tests.ami_service` import lỗi nếu thiếu.
- GitNexus CLI phải dùng bản global 1.6.5 (`C:\Users\phanb\AppData\Local\pnpm\bin\gitnexus.CMD`), KHÔNG `npx` latest (DB mismatch v42 vs v40). Nếu GitNexus tool warns index STALE (thường sau khi đổi schema symbol) → chạy `gitnexus.CMD analyze` trước khi `gitnexus_impact`; nếu symbol vẫn không ra thì dùng `rg` thủ công và ghi rõ trong commit message.
- MinIO `PoolManager(maxsize=16)` ở cả 3 nơi tạo client (`storage/assets.py`, `api/resolver.py`, `workers/crawl_images.py`) — describe fetch nhiều ảnh song song; đừng hạ về 1.
- httpx gọi OCR: `connect` timeout tách riêng `min(10s, timeout)` để fail-fast; message lỗi phải phân biệt `ReadTimeout` (service treo) / `ConnectError` (unreachable-firewall) / `ConnectTimeout` (SYN drop).

# Quy tắc làm việc (bắt buộc)
- **MỖI phase commit riêng, STOP xin duyệt sau mỗi phase.** Trước commit: `gitnexus_detect_changes()`; trước khi sửa symbol: `gitnexus_impact({target, direction:"upstream", repo:"RAG-Anything"})`. HIGH/CRITICAL → cảnh báo user trước khi sửa.
- Test: `$env:PYTHONIOENCODING="utf-8"; uv run pytest -q tests/ami_service/` (service, `make test`) và `uv run pytest -q tests --ignore=tests/ami_service` (lib, `make test-lib`); lint `uv run ruff check ami_rag tests/ami_service` (ruff 0.16.0 global; `ami_rag/**` đã có per-file-ignores BLE001/S110/B008).
- File có tiếng Việt: chỉ dùng tool write/edit, KHÔNG `Add-Content`/`Set-Content` (cp1252 mojibake).
- Không gọi LLM/API ngoài ý muốn: test chỉ dùng fake + `httpx.MockTransport`/`ASGITransport`.
- Repo server máy B luôn nằm RIÊNG ngoài RAG-Anything (`qwen-embedding-server`, `nemotron-vl-vllm`, `vintern-vllm`). `nemotron-vl-vllm` KHÔNG phải git repo → hỏi user trước khi `git init` cho repo mới.
- Test infra sẵn có: `tests/ami_service/conftest.py` (`FakeAssetStore.get_bytes` → `b"\x89PNG-fake-bytes"`, `FakeParser`, `FakeModalProcessor`), `tests/test_gateway_parity.py` (pattern chạy gateway app THẬT qua ASGI + upstream `MockTransport`, skip nếu repo không có — copy cho `vintern-vllm`).
