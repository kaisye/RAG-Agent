from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG PDF API"
    debug: bool = False

    cors_origins: list[str] = ["http://localhost:5173"]

    # LLM
    llm_provider: str = "nvidia"
    nvidia_api_key: str = ""
    # Verify the exact model IDs on https://build.nvidia.com/models before use
    nvidia_chat_model: str = "meta/llama-3.1-70b-instruct"
    # nv-embedqa-e5-v5 is the recommended text embedding model on NIM free tier
    nvidia_embed_model: str = "nvidia/nv-embedqa-e5-v5"
    nvidia_embed_batch_size: int = 16
    # Verify current model IDs at https://build.nvidia.com/models before use —
    # multimodal embed models are deprecated frequently; fallback to caption if empty/unavailable
    nvidia_multimodal_embed_model: str = ""
    # Vision LLM for caption fallback (also used if multimodal embed model unavailable)
    nvidia_vision_model: str = "meta/llama-3.2-90b-vision-instruct"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_chat_model: str = "qwen2.5"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"
    # Must match the embedding model's output dimension; 1024 for nv-embedqa-e5-v5
    qdrant_vector_size: int = 1024

    # Retrieval
    hybrid_search_enabled: bool = True   # set to false to use vector-only (ablation)
    rrf_k: int = 60                      # RRF smoothing constant
    reranker_enabled: bool = True        # set to false to skip rerank step (ablation)
    reranker_model: str = "BAAI/bge-reranker-base"  # local CrossEncoder model
    chat_supports_vision: bool = False   # set True only when using a vision-capable chat model

    # Query Transformations
    hyde_enabled: bool = False           # HyDE: embed a LLM-generated hypothetical doc instead of raw query
    query_decomposition_enabled: bool = False  # break complex queries into N sub-queries then merge
    query_decomposition_n: int = 3       # number of sub-queries to generate

    # Storage
    upload_dir: str = "storage/uploads"
    static_dir: str = "storage/static"
    max_upload_size_mb: int = 50

    # Database
    database_url: str = "sqlite+aiosqlite:///storage/rag_pdf.db"


settings = Settings()
