from functools import lru_cache
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Provider
    llm_provider: str = "nvidia"
    nvidia_api_key: str = ""
    nvidia_chat_model: str = ""
    nvidia_embed_model: str = ""
    nvidia_rerank_model: str = "nvidia/llama-3_2-nv-rerankqa-1b-v2"
    ollama_chat_model: str = "qwen2.5"
    ollama_base_url: str = "http://localhost:11434/v1"

    # Chunking
    chunking_strategy: str = "recursive"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    semantic_threshold: float = 0.5
    min_chunk_size: int = 600
    max_chunk_size: int = 1024

    # HNSW
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef_search: int = 128

    # Retrieval
    retrieval_strategy: str = "vector"
    top_k_retrieval: int = 10
    rrf_k: int = 60

    # Query Transform
    query_transform: str = "none"

    # Reranking
    rerank_strategy: str = "none"
    top_k_final: int = 3
    retrieve_k: int = 10
    mmr_lambda: float = 0.5

    # Evaluation
    testset_size: int = 50
    eval_max_workers: int = 4
    eval_timeout: int = 180

    # Storage
    database_url: str = "sqlite+aiosqlite:///./storage/rag.db"
    chroma_persist_dir: str = "./storage/chroma"
    upload_dir: str = "./storage/uploads"
    images_dir: str = "./storage/images"
    markdown_dir: str = "./storage/markdown"


class PipelineConfig(BaseModel):
    # Chunking
    chunking_strategy: str = "recursive"   # "recursive" | "semantic"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    semantic_threshold: float = 0.5

    # HNSW
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef_search: int = 128

    # Retrieval
    retrieval_strategy: str = "vector"     # "vector"|"bm25"|"hybrid_interleaving"|"hybrid_rrf"
    top_k_retrieval: int = 10
    rrf_k: int = 60

    # Query Transform
    query_transform: str = "none"          # "none"|"hyde"|"decomposition"

    # Reranking
    rerank_strategy: str = "none"          # "none"|"cross_encoder"|"mmr"
    top_k_final: int = 3
    mmr_lambda: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
