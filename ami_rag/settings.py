from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- LLM (qwen-selfhost, OpenAI-compatible): describe_func / answer_func ---
    QWEN_LLM_BASE_URL: str = "http://vllm:8000/v1"
    QWEN_LLM_MODEL: str = "Qwen/Qwen3-32B"
    QWEN_LLM_API_KEY: str = ""
    QWEN_VLM_MODEL: str = ""

    # --- Remote embedding server (máy B) ---
    # backend "custom": qwen-embedding-server (/info + /embed);
    # backend "openai": gateway nemotron-vl-vllm (máy B) — /health + /v1/embeddings
    # với {"input", "input_type"} cho nvidia/llama-nemotron-embed-vl-1b-v2.
    EMBED_SERVER_URL: str = "http://localhost:8080"
    # Chỉ từ env - không ghi token vào file config/commit
    EMBED_SERVER_TOKEN: str = ""
    EMBED_MODEL: str = "nvidia/llama-nemotron-embed-vl-1b-v2"
    # auto | custom | openai (auto: prefix "Qwen/" -> custom, còn lại -> openai)
    EMBED_BACKEND: str = "auto"
    # Dùng để xác minh với server lúc handshake (dim thực tế do server quyết định)
    EMBED_DIM: int = 2048
    EMBED_TIMEOUT: int = 60
    EMBED_BATCH_SIZE: int = 32
    EMBED_MAX_CONCURRENCY: int = 4
    EMBED_RETRIES: int = 3
    EMBED_CACHE_ENABLED: bool = True
    EMBED_CACHE_PATH: str = "./embed_cache.db"
    # Guard kích thước input embed (client-side): item vượt budget bị head-truncate
    # trước khi gửi (content đầy đủ vẫn lưu Qdrant). Default tính cho max_model_len
    # 8192 của gateway - server phải chạy 8192, còn không thì giảm qua env.
    EMBED_MAX_INPUT_TOKENS: int = 6000
    # Một ảnh Nemotron VL tốn tối đa ~1792 visual token (6 tile + thumbnail - model card)
    EMBED_IMAGE_TOKEN_RESERVE: int = 1792

    MONGO_URI: str = "mongodb://localhost:27017/?directConnection=true"
    # Shares organization_db with the backend; every RAG collection is prefixed `multimodal_`
    RAG_DB: str = "organization_db"
    RAG_DOCUMENTS_COLLECTION: str = "multimodal_rag_documents"
    ORG_DB: str = "organization_db"
    DOC_COLLECTION: str = "documents"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    # Giới hạn byte mỗi request upsert (server Qdrant mặc định max_request_size_mb=32)
    QDRANT_UPSERT_MAX_MB: int = 16

    WORKSPACE: str = "multimodal"

    PARSER: str = "mineru"
    PARSE_METHOD: str = "auto"
    PARSER_OUTPUT_DIR: str = "./output"

    # MinerU (only used when PARSER=mineru). Measured on Colab T4, 10-page PDF: `pipeline` peaks at
    # ~1.8 GiB with auto VRAM (1.1 GiB at MINERU_VIRTUAL_VRAM_SIZE=4, 3.3 GiB at 16) in ~50 s;
    # `hybrid-engine` (MinerU 3.4.x default when -b is omitted) peaks at ~14.5 GiB in ~286 s,
    # so always pass a backend.
    MINERU_BACKEND: str = "pipeline"
    MINERU_DEVICE: str = ""  # "" = auto; cuda | cuda:0 | cpu (-> env MINERU_DEVICE_MODE)
    MINERU_VIRTUAL_VRAM_SIZE: int = 0  # GB the parser may assume; 0 = auto (use GPU's real VRAM)
    MINERU_LANG: str = ""  # OCR language hint (ch, en, latin, ...); "" = MinerU default
    MINERU_SOURCE: str = ""  # model source: huggingface | modelscope | local; "" = default
    MINERU_TIMEOUT: int = 1800  # seconds per document; 0 = no limit

    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 100

    REDIS_URL: str = "redis://localhost:6379/0"
    RAG_STREAM: str = "rag:ingest"
    RAG_CONSUMER_GROUP: str = "ami-rag"
    RAG_CONSUMER_NAME: str = ""
    WORKER_ENABLED: bool = True
    WORKER_BATCH: int = 10
    WORKER_POLL_BLOCK_MS: int = 5000
    WORKER_MAX_DELIVERY: int = 3
    # A failed message is re-delivered once it has been pending this long (ms)
    WORKER_RETRY_IDLE_MS: int = 600000
    WORKER_METRICS_PORT: int = 9109

    # --- Pipeline / CLI ---
    # Quy ước collection: {WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}
    CHUNKER_VERSION: str = "v1"
    # doc `processing` quá lâu (tiến trình chết giữa chừng) được `status` báo là treo
    CLI_STUCK_PROCESSING_MINUTES: int = 60

    RERANK_BASE_URL: str = "http://localhost:8080"
    # Gateway nemotron-vl-vllm (máy B) phục vụ cả embed + rerank; vLLM serving
    # nvidia/llama-nemotron-rerank-vl-1b-v2 phía sau gateway.
    RERANK_MODEL: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    # auto | legacy | vllm (auto: model rỗng hoặc chứa "bge" -> legacy BGE, còn lại -> vllm)
    RERANK_BACKEND: str = "auto"
    # vLLM rerank kèm ảnh + text cho chunk image/table/equation (cần asset_key)
    RERANK_MULTIMODAL: bool = True
    RERANK_TOP_K: int = 5
    RERANK_TIMEOUT: int = 60
    # Guard kích thước text gửi lên rerank server (cùng vLLM 8192): document vượt
    # budget bị head-truncate (kèm ảnh -> trừ image token reserve).
    RERANK_MAX_INPUT_TOKENS: int = 6000
    RERANK_IMAGE_TOKEN_RESERVE: int = 1792
    RERANK_CALIBRATION_ENABLED: bool = False
    RERANK_CALIBRATION_PATH: str = "./rerank_calibration.json"

    # candidates fetched = final top_k * RETRIEVAL_OVERFETCH, so rerank always yields top_k docs
    RETRIEVAL_CHUNK_TOP_K: int = 40
    RETRIEVAL_OVERFETCH: int = 4
    # docs repaired in parallel by `reindex --repair` (low: avoids provider 429)

# --- Hai nhánh retrieve + rerank + fusion (thử nghiệm) ---
    # single: 1 pool chung như cũ (retrieve không filter -> rerank 1 lần -> sort).
    # calibrated | rrf | quota: tách đúng 2 nhánh, mỗi nhánh 1 lệnh rerank:
    #   • `text+table` — văn xuất + bảng cùng nhau, 1 lệnh rerank chung.
    #   • `image` — VL rerank riêng. Tách ở đúng chỗ này vì độ chênh lệch thang
    #     điểm chỉ tồn tại ở ảnh, không có ở bảng và text.
    # Cổng lọc ảnh: xem ami_rag.core.fusion.gate_image_hits.
    RETRIEVAL_FUSION_MODE: str = "single"
    # Pool = nhóm modality, phân cách nhau bằng `+`. Ví dụ: "text+table,image".
    # Modality không thuộc về pool nào -> pool "other".
    RETRIEVAL_FUSION_POOLS: str = "text+table,image"
    # Sâu retrieve của từng pool (key là TÊN pool, không phải nhóm modality).
    # Đo thực tế (28 case, scripts/run_fusion_bench.py): chunk đích của text
    # nằm trong top-11 vector. Trong pool gộp, table phải cạnh tranh với text:
    # `table_08` ở vector rank 44 nên cần depth >= 50 (depth 60 không thêm gì).
    RETRIEVAL_FUSION_POOL_SIZES: str = "text=50,image=15"
    # RRF: score = weight / (k + rank). Với pool rời nhau + weight bằng nhau thì
    # k không đổi thứ tự (RRF trở thành round-robin theo rank) -- giữ 60 theo
    # thông lệ.
    RETRIEVAL_FUSION_RRF_K: int = 60
    # quota: "pool=slots[@ngưỡng]" -- key là TÊN pool, không phải nhóm modality.
    # KHÔNG kèm ngưỡng là cấu hình đo tốt nhất (28 case: 22/28, table 7/9): reserve
    # slot cứng cho từng pool để modality yếu vẫn lên trang, rồi điền nốt. Có
    # ngưỡng thì slot bị chặn và rơi về đúng bằng `calibrated` (21/28).
    RETRIEVAL_FUSION_QUOTA: str = "text=3,image=2"
    # Pool được rerank KÈM ảnh render (VL reranker).
    # Mặc định LÀ "text,image", tức bảng cũng được gửi ảnh vào rerank chung.
    # Đo thực tế (28 case): bảng rerank TEXT-THUẦN -> table 3/9; bảng kèm ảnh ->
    # 5/9 @ depth 40, 6/9 @ depth 50. Điểm rerank của text và bảng đúng cùng
    # thang đo nên KHÔNG cần tách pool riêng, nhưng 1 chi phí nữa là phải chảy
    # qua calibrator. Đặt "image" để bảng rerank thuần text: nhanh hơn nhưng MẤT
    # 2 hit bảng.
    RETRIEVAL_FUSION_VL_POOLS: str = "text,image"
    # Cổng lọc ảnh: ảnh chỉ được vào nếu điểm >= điểm text tại đường cắt
    # (thiếu chỉ so điểm) để vào LLM. Tắt để so số ảnh bị loại điểm và
    # giữ nguyên số ảnh được mang vào context.
    RETRIEVAL_FUSION_IMAGE_GATE: bool = True

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "ami-data-documents"
    MINIO_SECURE: bool = False
    ASSET_PREFIX: str = "rag-assets"
    CRAWL_IMAGES_PREFIX: str = "crawl-images"
    CRAWL_IMAGES_ENABLED: bool = False
    MINIO_PRESIGN_EXPIRES: int = 3600

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8009
    RAG_API_KEY: str = ""


def resolve_embed_backend(settings) -> str:
    """Resolve EMBED_BACKEND: explicit value, else auto-infer từ EMBED_MODEL.

    - EMBED_BACKEND đặt rõ -> dùng luôn (custom | openai).
    - auto: model prefix "Qwen/" -> "custom" (qwen-embedding-server contract),
      còn lại -> "openai" (vLLM OpenAI-compatible, Nemotron VL embed).
    """
    backend = (getattr(settings, "EMBED_BACKEND", "auto") or "auto").lower()
    if backend != "auto":
        if backend not in ("custom", "openai"):
            raise ValueError(
                f"EMBED_BACKEND không hợp lệ: {backend} (custom | openai | auto)"
            )
        return backend
    model = getattr(settings, "EMBED_MODEL", "")
    return "custom" if model.startswith("Qwen/") else "openai"


def resolve_rerank_backend(settings) -> str:
    """Resolve RERANK_BACKEND: explicit value, else auto-infer từ RERANK_MODEL.

    - RERANK_BACKEND đặt rõ -> dùng luôn (legacy | vllm).
    - auto: RERANK_MODEL rỗng hoặc chứa "bge" -> "legacy" (BGE reranker,
      documents text-only), còn lại -> "vllm" (Nemotron VL rerank, multimodal).
    """
    backend = (getattr(settings, "RERANK_BACKEND", "auto") or "auto").lower()
    if backend != "auto":
        if backend not in ("legacy", "vllm"):
            raise ValueError(
                f"RERANK_BACKEND không hợp lệ: {backend} (legacy | vllm | auto)"
            )
        return backend
    model = (getattr(settings, "RERANK_MODEL", "") or "").lower()
    return "vllm" if model and "bge" not in model else "legacy"


def resolve_fusion(settings) -> dict:
    """Parse + validate the per-modality fusion settings into one ready-to-use dict.

    Returns `None` when fusion is off (`RETRIEVAL_FUSION_MODE=single`), i.e. the
    query path keeps the original single-pool retrieve -> rerank -> sort flow.
    Parsing delegates to `ami_rag.core.fusion` so there is one source of truth for
    the spec grammar, and an invalid spec fails here rather than mid-request.
    """
    from ami_rag.core.fusion import (
        STRATEGIES,
        parse_pool_groups,
        parse_pool_sizes,
        parse_quota,
    )

    mode = (getattr(settings, "RETRIEVAL_FUSION_MODE", "single") or "single").lower()
    if mode == "single":
        return None
    if mode not in STRATEGIES:
        raise ValueError(
            f"RETRIEVAL_FUSION_MODE không hợp lệ: {mode} (single | {' | '.join(STRATEGIES)})"
        )

    groups = parse_pool_groups(settings.RETRIEVAL_FUSION_POOLS)
    sizes = parse_pool_sizes(settings.RETRIEVAL_FUSION_POOL_SIZES)
    missing = [name for name in groups if name not in sizes]
    if missing:
        raise ValueError(
            f"RETRIEVAL_FUSION_POOL_SIZES thiếu depth cho pool: {', '.join(missing)}"
        )
    return {
        "mode": mode,
        "groups": groups,
        "sizes": sizes,
        "rrf_k": int(settings.RETRIEVAL_FUSION_RRF_K),
        "quotas": parse_quota(settings.RETRIEVAL_FUSION_QUOTA),
        "image_gate": bool(getattr(settings, "RETRIEVAL_FUSION_IMAGE_GATE", True)),
        "vl_pools": {
            name.strip().lower()
            for name in (settings.RETRIEVAL_FUSION_VL_POOLS or "").split(",")
            if name.strip()
        },
    }


def parser_kwargs(settings) -> dict:
    """Extra kwargs for `RAGAnything.parse_document` derived from MINERU_* settings.

    Empty for non-MinerU parsers. Device and VRAM cap go through the subprocess env
    (MINERU_DEVICE_MODE / MINERU_VIRTUAL_VRAM_SIZE), which is what the Colab VRAM sweep measured.
    """
    if getattr(settings, "PARSER", "mineru") != "mineru":
        return {}
    kwargs: dict = {}
    if backend := getattr(settings, "MINERU_BACKEND", ""):
        kwargs["backend"] = backend
    if lang := getattr(settings, "MINERU_LANG", ""):
        kwargs["lang"] = lang
    if source := getattr(settings, "MINERU_SOURCE", ""):
        kwargs["source"] = source
    if timeout := getattr(settings, "MINERU_TIMEOUT", 0):
        kwargs["timeout"] = int(timeout)
    env: dict[str, str] = {}
    if device := getattr(settings, "MINERU_DEVICE", ""):
        env["MINERU_DEVICE_MODE"] = device
    if vram := getattr(settings, "MINERU_VIRTUAL_VRAM_SIZE", 0):
        env["MINERU_VIRTUAL_VRAM_SIZE"] = str(int(vram))
    if env:
        kwargs["env"] = env
    return kwargs


@lru_cache
def get_settings() -> Settings:
    return Settings()
