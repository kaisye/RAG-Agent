from app.core.config import get_settings
from app.services.chunking.base import BaseChunker
from app.services.chunking.recursive import RecursiveChunker


def get_chunker(strategy: str, embed_service=None) -> BaseChunker:
    settings = get_settings()
    if strategy == "recursive":
        return RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    if strategy == "semantic":
        if embed_service is None:
            raise ValueError("SemanticChunker requires embed_service")
        from app.services.chunking.semantic import SemanticChunker  # imported lazily — implemented in feature/semantic-chunking
        return SemanticChunker(
            embed_service=embed_service,
            threshold=settings.semantic_threshold,
            min_size=settings.min_chunk_size,
            max_size=settings.max_chunk_size,
            overlap=settings.chunk_overlap,
        )
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
