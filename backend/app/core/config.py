from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG PDF API"
    debug: bool = False

    cors_origins: list[str] = ["http://localhost:5173"]

    # LLM
    llm_provider: str = "ollama"
    nvidia_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434/v1"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_pdf"

    # Storage
    upload_dir: str = "storage/uploads"
    static_dir: str = "storage/static"


settings = Settings()
