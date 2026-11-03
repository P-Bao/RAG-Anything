# Makefile
#
# KG branch (LightRAG 1.5.7 + Qwen3.5 + nemotron gateway + Gemini caption).
# Override: make index-pdf LIMIT=5 | make index-pdf DOC=<id>
#
SHELL = /bin/bash

LOG_DIR ?= logs
COMPOSE_FILE ?= docker-compose.kg.yml

.PHONY: help setup style lint test index-pdf index-status index-mongo query clean \
	docker-build docker-index-pdf docker-index-status docker-index-mongo docker-query docker-clean

help:
	@echo "Available commands:"
	@echo "  make setup        - Cai dependencies (uv sync --extra text)"
	@echo "  make style        - Format + auto-fix lint (ruff, neu co)"
	@echo "  make lint         - Kiem tra import smoke (python compileall)"
	@echo "  make test         - Chay pytest"
	@echo "  make index-pdf    - Index PDF da parse tu MinIO (ami-multimodal-retrieval)"
	@echo "                      ARGS='--doc <id>' | ARGS='--limit 2 --dry-run' | BG=1 de chay nen"
	@echo "  make index-status - Dem entity/relation/chunk trong KG local"
	@echo "  make index-mongo  - Index theo danh sach Mongo (parse PDF thieu artifacts)"
	@echo "                      ARGS='--limit 5' | '--doc <ObjectId>' | '--dry-run'"
	@echo "  make query        - Query KG: make query Q='cau hoi' [MODE=mix]"
	@echo "  make docker-build       - Build image kg-rag"
	@echo "  make docker-index-pdf   - Index trong container (ARGS/BG nhu index-pdf)"
	@echo "  make docker-index-status- Trang thai KG trong volume rag-storage"
	@echo "  make docker-query       - Query trong container: Q='...' [MODE=mix]"
	@echo "  make docker-clean       - Xoa container + volume + image local"
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

# Query KG (mode mix: KG + vector, thinking ON qua role config).
#   make query Q="chuong trinh AI-RAG do ai phu track?"
query:
	@test -n "$(Q)" || { echo "usage: make query Q='<question>' [MODE=hybrid]"; exit 2; }
	uv run python scripts/kg_index_pdfs.py query "$(Q)" $(if $(MODE),--mode $(MODE))

clean:
	rm -rf .pytest_cache
	find . -path ./.venv -prune -o -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# --- Docker (mot-shot runner; KHONG chay 2 index song song cung volume) ---
# .env can ton tai (env_file). Host trong .env = ten container tren ami-network.

docker-build:
	docker compose -f $(COMPOSE_FILE) build

docker-index-pdf:
	docker compose -f $(COMPOSE_FILE) run --rm kg-rag python scripts/kg_index_pdfs.py index $(ARGS)

docker-index-status:
	docker compose -f $(COMPOSE_FILE) run --rm kg-rag python scripts/kg_index_pdfs.py status

index-mongo:
	uv run python scripts/kg_index_pdfs.py mongo $(ARGS)

docker-index-mongo:
	docker compose -f $(COMPOSE_FILE) run --rm kg-rag python scripts/kg_index_pdfs.py mongo $(ARGS)

docker-query:
	@test -n "$(Q)" || { echo "usage: make docker-query Q='<question>' [MODE=hybrid]"; exit 2; }
	docker compose -f $(COMPOSE_FILE) run --rm kg-rag python scripts/kg_index_pdfs.py query "$(Q)" $(if $(MODE),--mode $(MODE))

# Xoa container + volume + image local cua project nay (KHONG -a/--all).
docker-clean:
	docker compose -f $(COMPOSE_FILE) down --remove-orphans -v --rmi local
