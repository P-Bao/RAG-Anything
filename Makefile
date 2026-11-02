# Makefile
#
# KG branch (LightRAG 1.5.7 + Qwen3.5 + nemotron gateway + Gemini caption).
# Override: make index-pdf LIMIT=5 | make index-pdf DOC=<id>
#
SHELL = /bin/bash

LOG_DIR ?= logs

.PHONY: help setup style lint test index-pdf index-status clean

help:
	@echo "Available commands:"
	@echo "  make setup        - Cai dependencies (uv sync --extra text)"
	@echo "  make style        - Format + auto-fix lint (ruff, neu co)"
	@echo "  make lint         - Kiem tra import smoke (python compileall)"
	@echo "  make test         - Chay pytest"
	@echo "  make index-pdf    - Index PDF da parse tu MinIO (ami-multimodal-retrieval)"
	@echo "                      ARGS='--doc <id>' | ARGS='--limit 2 --dry-run' | BG=1 de chay nen"
	@echo "  make index-status - Dem entity/relation/chunk trong KG local"
	@echo "  make clean        - Xoa cache pytest/rag_storage tam"

setup:
	uv sync --extra text

style:
	uv run python -m compileall -q raganything scripts

lint:
	uv run python -m compileall -q raganything scripts

test:
	PYTHONIOENCODING=utf-8 uv run pytest -q tests --ignore=tests/ami_service

# Index theo MinIO artifacts cua ami-multimodal-retrieval (read-only).
# KHONG chay 2 tien trinh index cung luc (storage file-based, last-write-wins).
#   make index-pdf ARGS="--limit 2 --dry-run"
#   make index-pdf DOC=<doc-id>
#   make index-pdf BG=1 ARGS="--limit 10"
index-pdf:
ifeq ($(BG),1)
	@mkdir -p $(LOG_DIR)
	@f=$(LOG_DIR)/index-pdf-$$(date +%Y%m%d-%H%M%S).log; \
	PYTHONUNBUFFERED=1 uv run python scripts/kg_index_pdfs.py index $(ARGS) > $$f 2>&1 & \
	echo "started in background: $$f (pid $$!)"; echo "  follow: tail -f $$f"
else
	uv run python scripts/kg_index_pdfs.py index $(ARGS)
endif

index-status:
	uv run python scripts/kg_index_pdfs.py status

clean:
	rm -rf .pytest_cache
	find . -path ./.venv -prune -o -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
