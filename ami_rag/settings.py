from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    LLM_PROFILE: str = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_LLM_MODEL: str = "gemini-2.5-flash"
    GEMINI_VISION_MODEL: str = ""
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 1536

    QWEN_LLM_BASE_URL: str = "http://vllm:8000/v1"
    QWEN_LLM_MODEL: str = "Qwen/Qwen3-32B"
    QWEN_LLM_API_KEY: str = ""
    QWEN_VLM_MODEL: str = ""
    QWEN_EMBED_BASE_URL: str = "http://vllm:8000/v1"
    QWEN_EMBED_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    QWEN_EMBED_API_KEY: str = ""
    QWEN_EMBED_DIM: int = 1024

    MONGO_URI: str = "mongodb://localhost:27017/?directConnection=true"
    # Shares organization_db with the backend; every RAG collection is prefixed `multimodal_`
    # (LightRAG names Mongo collections "{WORKSPACE}_{namespace}").
    RAG_DB: str = "organization_db"
    RAG_DOCUMENTS_COLLECTION: str = "multimodal_rag_documents"
    ORG_DB: str = "organization_db"
    DOC_COLLECTION: str = "documents"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""

    WORKSPACE: str = "multimodal"
    WORKING_DIR: str = "./rag_storage"

    PARSER: str = "mineru"
    PARSE_METHOD: str = "auto"
    PARSER_OUTPUT_DIR: str = "./output"

    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 100
    MAX_GLEANING: int = 1
    SUMMARY_LANGUAGE: str = "Tiếng Việt"

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

    RERANK_BASE_URL: str = "http://localhost:8010"
    RERANK_TOP_K: int = 5
    RERANK_TIMEOUT: int = 60

    RETRIEVAL_TOP_K: int = 40
    RETRIEVAL_CHUNK_TOP_K: int = 40

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
