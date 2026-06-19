from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG PDF API"
    debug: bool = False

    cors_origins: list[str] = ["http://localhost:5173"]

    # LLM
    llm_provider: str = "nvidia"
    nvidia_api_key: str = ""
    # Verify the exact model ID on https://build.nvidia.com/models before use
    nvidia_chat_model: str = "meta/llama-3.1-70b-instruct"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_chat_model: str = "qwen2.5"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_pdf"

    # Storage
    upload_dir: str = "storage/uploads"
    static_dir: str = "storage/static"
    max_upload_size_mb: int = 50

    # Database
    database_url: str = "sqlite+aiosqlite:///storage/rag_pdf.db"


settings = Settings()
