# Makefile
#
# KG branch (LightRAG 1.5.7 + Qwen3.5 + nemotron gateway + Gemini caption).
# Override: make index-pdf LIMIT=5 | make index-pdf DOC=<id>
#
SHELL = /bin/bash

LOG_DIR ?= logs
COMPOSE_FILE ?= docker-compose.kg.yml

.PHONY: help setup style lint test api worker index-pdf index-status index-mongo query clean \
	network docker-build docker-up docker-down docker-logs docker-restart \
	docker-index-pdf docker-index-status docker-index-mongo docker-jobs docker-cli \
	docker-query docker-clean

help:
	@echo "Available commands:"
	@echo "  make setup        - Cai dependencies (uv sync --extra text --extra service)"
	@echo "  make style        - Format + auto-fix lint (ruff, neu co)"
	@echo "  make lint         - Kiem tra import smoke (python compileall)"
	@echo "  make test         - Chay pytest"
	@echo "  make api          - Chay API server local (port 8009, uvicorn)"
	@echo "  make worker       - Chay worker ingest local (an job tu Redis, can REDIS_URL)"
	@echo "  make network      - Tao docker network ami-network neu chua co"
	@echo "  make docker-up    - Build + chay kg-api (8009) + kg-rag worker (thuong tru)"
	@echo "  make docker-down  - Dung container (giu KG)"
	@echo "  make docker-logs  - Log ca 2 service"
	@echo "  make docker-restart - Down roi up lai"
	@echo "  make index-pdf    - Index PDF da parse tu MinIO (ami-multimodal-retrieval)"
	@echo "                      ARGS='--doc <id>' | ARGS='--limit 2 --dry-run' | BG=1 de chay nen"
	@echo "  make index-status - Dem entity/relation/chunk trong KG local"
	@echo "  make index-mongo  - Index theo danh sach Mongo (parse PDF thieu artifacts)"
	@echo "                      ARGS='--limit 5' | '--doc <ObjectId>' | '--dry-run'"
	@echo "  make query        - Query KG: make query Q='cau hoi' [MODE=mix]"
	@echo "  make docker-build       - Build image"
	@echo "  make docker-index-pdf   - Enqueue index MinIO qua API: [LIMIT=2] [DRY=1]"
	@echo "  make docker-index-mongo - Enqueue index Mongo qua API: [LIMIT=0] [DRY=1]"
	@echo "  make docker-jobs        - Trang thai job: [JOB=<id>] (khong JOB = 20 job gan nhat)"
	@echo "  make docker-index-status- Trang thai KG trong volume rag-storage (docs-only)"
	@echo "  make docker-cli         - CLI mot-shot: ARGS='status|query \"...\"|index --limit 2' [FORCE=1]"
	@echo "  make docker-query       - Query trong container: Q='...' [MODE=mix]"
	@echo "  make docker-clean       - Xoa container + volume + image local"
	@echo "  make clean        - Xoa cache pytest/rag_storage tam"

setup:
	uv sync --extra text --extra service

style:
	uv run python -m compileall -q raganything scripts

lint:
	uv run python -m compileall -q raganything scripts

test:
	PYTHONIOENCODING=utf-8 uv run pytest -q tests --ignore=tests/ami_service

# API server local (resident, port 8009 = port cu cua ami-multimodal-retrieval).
# Env: API_HOST/API_PORT/RAG_API_KEY/MONGO_URI/REDIS_URL/KG_STATE_COLLECTION.
api:
	uv run python -c "from raganything.api import run; run()"

# Worker ingest local (an job tu Redis). Can REDIS_URL; thieu Redis thi API tu
# chay job trong process (che do 1 container).
worker:
	uv run python -m raganything.worker

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

# --- Docker ---
# .env can ton tai (env_file). Host trong .env = ten container tren ami-network.
# kg-api = API thuong tru (8009); kg-rag = WORKER thuong tru (an job tu Redis).
# Index di qua API -> Redis -> worker: chi 1 process ghi KG.
# Compose khai bao network external nen phai ton tai truoc khi up.
network:
	@docker network inspect ami-network >/dev/null 2>&1 || docker network create ami-network

docker-build:
	docker compose -f $(COMPOSE_FILE) build

docker-up: network
	docker compose -f $(COMPOSE_FILE) up --build -d

docker-down:
	docker compose -f $(COMPOSE_FILE) down --remove-orphans

docker-logs:
	docker compose -f $(COMPOSE_FILE) logs -f --tail=100

docker-restart: network
	docker compose -f $(COMPOSE_FILE) down --remove-orphans
	docker compose -f $(COMPOSE_FILE) up --build -d

# --- Index qua API (enqueue vao Redis; worker kg-rag chay) ---
# Vi du: make docker-index-pdf LIMIT=0 DRY=1 | IDS=id1,id2 | SOURCE=minio
API_URL ?= http://localhost:8009
# -d '\r': .env soan tren Windows co the co CRLF, lam hong header bearer.
API_KEY ?= $(shell grep -E '^RAG_API_KEY=' .env 2>/dev/null | cut -d= -f2- | tr -d '\r')
AUTH := $(if $(API_KEY),-H "Authorization: Bearer $(API_KEY)")

docker-index-pdf:
	@curl -sS -X POST $(API_URL)/v2/rag/index -H "Content-Type: application/json" $(AUTH) \
	  -d '{"source":"minio","limit":$(if $(LIMIT),$(LIMIT),2),"dry_run":$(if $(DRY),true,false)}'
	@echo

docker-index-mongo:
	@curl -sS -X POST $(API_URL)/v2/rag/index -H "Content-Type: application/json" $(AUTH) \
	  -d '{"source":"mongo","limit":$(if $(LIMIT),$(LIMIT),0),"dry_run":$(if $(DRY),true,false)}'
	@echo

# Trang thai job: make docker-jobs            (20 job gan nhat)
#                make docker-jobs JOB=<id>    (tien do tung doc cua 1 job)
docker-jobs:
	@if [ -n "$(JOB)" ]; then \
	  curl -sS $(API_URL)/v2/rag/index/$(JOB) $(AUTH); \
	else \
	  curl -sS "$(API_URL)/v2/rag/index?limit=20" $(AUTH); \
	fi
	@echo

# Doc-only CLI mot-shot (status/query/artifact text). Cam ingest khi worker dang chay.
docker-index-status:
	docker compose -f $(COMPOSE_FILE) run --rm kg-rag python scripts/kg_index_pdfs.py status

docker-cli:
	@test -n "$(ARGS)" || { echo "usage: make docker-cli ARGS='status|query \"...\"|index --limit 2' [FORCE=1]"; exit 2; }
	@if [ -z "$(FORCE)" ] && [ -n "$$(docker compose -f $(COMPOSE_FILE) ps -q kg-rag)" ]; then \
	  echo "kg-rag worker dang chay: index mot-shot se ghi de KG (2 writer). Dung make docker-index-* hoac FORCE=1."; exit 1; \
	fi
	docker compose -f $(COMPOSE_FILE) run --rm kg-rag python scripts/kg_index_pdfs.py $(ARGS)

docker-query:
	@test -n "$(Q)" || { echo "usage: make docker-query Q='<question>' [MODE=hybrid]"; exit 2; }
	docker compose -f $(COMPOSE_FILE) run --rm kg-rag python scripts/kg_index_pdfs.py query "$(Q)" $(if $(MODE),--mode $(MODE))

index-mongo:
	uv run python scripts/kg_index_pdfs.py mongo $(ARGS)

# Xoa container + volume + image local cua project nay (KHONG -a/--all).
docker-clean:
	docker compose -f $(COMPOSE_FILE) down --remove-orphans -v --rmi local
