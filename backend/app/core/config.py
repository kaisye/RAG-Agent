from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG PDF API"
    debug: bool = False

    cors_origins: list[str] = ["http://localhost:5173"]

    # LLM
    llm_provider: str = "nvidia"
    nvidia_api_key: str = ""
    nvidia_chat_model: str = "meta/llama-3.1-70b-instruct"
    nvidia_embed_model: str = "nvidia/nv-embedqa-e5-v5"
    nvidia_embed_batch_size: int = 16
    nvidia_multimodal_embed_model: str = ""
    nvidia_vision_model: str = "meta/llama-3.2-90b-vision-instruct"
    nvidia_rerank_model: str = "nvidia/rerank-qa-mistral-4b"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_chat_model: str = "qwen2.5"

    # Chunking
    chunking_strategy: str = "recursive"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    semantic_threshold: float = 0.5
    min_chunk_size: int = 600
    max_chunk_size: int = 1024

    # HNSW (ChromaDB)
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef_search: int = 128

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"
    qdrant_vector_size: int = 1024

    # Retrieval
    hybrid_search_enabled: bool = True
    retrieval_strategy: str = "hybrid_rrf"
    top_k_retrieval: int = 10
    rrf_k: int = 60
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    chat_supports_vision: bool = False

    # Query Transformations
    hyde_enabled: bool = False
    query_decomposition_enabled: bool = False
    query_decomposition_n: int = 3

    # Reranking
    query_transform: str = "none"
    rerank_strategy: str = "cross_encoder"
    top_k_final: int = 5
    retrieve_k: int = 10
    mmr_lambda: float = 0.5

    # Evaluation
    testset_size: int = 50
    eval_max_workers: int = 4
    eval_timeout: int = 180

    # Storage
    upload_dir: str = "storage/uploads"
    static_dir: str = "storage/static"
    images_dir: str = "storage/images"
    markdown_dir: str = "storage/markdown"
    chroma_persist_dir: str = "./storage/chroma"
    max_upload_size_mb: int = 50

    # Database
    database_url: str = "sqlite+aiosqlite:///storage/rag_pdf.db"


settings = Settings()


@lru_cache
def get_settings() -> Settings:
    return Settings()


class PipelineConfig(BaseModel):
    """Per-request pipeline configuration (used by RAGPipeline and eval ablation)."""

    # Chunking
    chunking_strategy: str = "recursive"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    semantic_threshold: float = 0.5

    # HNSW
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef_search: int = 128

    # Retrieval
    retrieval_strategy: str = "hybrid_rrf"
    top_k_retrieval: int = 10
    rrf_k: int = 60

    # Query Transform
    query_transform: str = "none"

    # Reranking
    rerank_strategy: str = "cross_encoder"
    top_k_final: int = 5
    mmr_lambda: float = 0.5
