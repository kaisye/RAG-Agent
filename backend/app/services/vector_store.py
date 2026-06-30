import logging
import uuid

logger = logging.getLogger(__name__)


# ===========================================================================
# Qdrant-based store — used by retrieval.py (chat pipeline)
# ===========================================================================

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
    from app.core.config import settings as _settings

    _singleton: QdrantClient | None = None

    def _client() -> QdrantClient:
        global _singleton
        if _singleton is not None:
            return _singleton
        url = _settings.qdrant_url
        if url == ":memory:":
            _singleton = QdrantClient(location=":memory:")
        elif url.startswith("./") or url.startswith(".\\") or (len(url) > 1 and url[1] == ":"):
            from pathlib import Path
            Path(url).mkdir(parents=True, exist_ok=True)
            _singleton = QdrantClient(path=url)
        else:
            _singleton = QdrantClient(url=url)
        return _singleton

    def _point_id(logical_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, logical_id))

    def ensure_collection() -> None:
        client = _client()
        existing = {c.name for c in client.get_collections().collections}
        if _settings.qdrant_collection in existing:
            return
        client.create_collection(
            collection_name=_settings.qdrant_collection,
            vectors_config=VectorParams(size=_settings.qdrant_vector_size, distance=Distance.COSINE),
        )
        logger.info("created Qdrant collection '%s' (dim=%d)", _settings.qdrant_collection, _settings.qdrant_vector_size)

    def upsert_chunks(chunks: list[dict], images: list[dict]) -> int:
        client = _client()
        points: list[PointStruct] = []

        for chunk in chunks:
            vector = chunk.get("vector")
            if not vector:
                continue
            points.append(PointStruct(
                id=_point_id(chunk["chunk_id"]),
                vector=vector,
                payload={
                    "document_id": chunk["document_id"],
                    "page":        chunk["page"],
                    "chunk_id":    chunk["chunk_id"],
                    "type":        "text",
                    "content":     chunk["text"],
                    "source_path": None,
                },
            ))

        for img in images:
            vector = img.get("vector")
            if not vector:
                continue
            points.append(PointStruct(
                id=_point_id(img["image_id"]),
                vector=vector,
                payload={
                    "document_id": img["document_id"],
                    "page":        img["page"],
                    "chunk_id":    img["image_id"],
                    "type":        "image",
                    "content":     "",
                    "source_path": img.get("file_path"),
                },
            ))

        if not points:
            return 0

        client.upsert(collection_name=_settings.qdrant_collection, points=points)
        logger.info("upsert_chunks: %d points upserted to '%s'", len(points), _settings.qdrant_collection)
        return len(points)

    def _delete_qdrant_vectors(document_id: str) -> None:
        client = _client()
        client.delete(
            collection_name=_settings.qdrant_collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
        logger.info("delete_document_vectors: removed Qdrant points for document_id=%s", document_id)

    _QDRANT_AVAILABLE = True

except ImportError:
    logger.warning("qdrant-client not installed — Qdrant retrieval unavailable")
    _QDRANT_AVAILABLE = False

    def _client():
        raise RuntimeError("qdrant-client is not installed")

    def ensure_collection():
        logger.warning("ensure_collection: qdrant-client not installed, skipping")

    def upsert_chunks(chunks, images):
        raise RuntimeError("qdrant-client is not installed")

    def _delete_qdrant_vectors(document_id):
        pass


# ===========================================================================
# ChromaDB-based store — used by RAGPipeline (quiz/flashcard pipeline)
# ===========================================================================

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from app.core.config import get_settings as _get_settings

    class VectorStoreService:
        def __init__(self):
            cfg = _get_settings()
            self.client = chromadb.PersistentClient(
                path=cfg.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._settings = cfg

        def get_or_create_collection(self, document_id: str):
            s = self._settings
            return self.client.get_or_create_collection(
                name=f"doc_{document_id}",
                metadata={
                    "hnsw:space":           "cosine",
                    "hnsw:M":               s.hnsw_m,
                    "hnsw:construction_ef": s.hnsw_ef_construct,
                    "hnsw:search_ef":       s.hnsw_ef_search,
                },
            )

        def upsert_chunks(self, document_id: str, chunks: list[dict], embeddings: list[list[float]]) -> None:
            col = self.get_or_create_collection(document_id)
            col.upsert(
                ids=       [c["chunk_id"] for c in chunks],
                embeddings=embeddings,
                documents= [c["text"] for c in chunks],
                metadatas= [
                    {
                        "document_id": c["document_id"],
                        "page":        c["page"],
                        "strategy":    c.get("strategy", "recursive"),
                        "type":        c.get("type", "text"),
                    }
                    for c in chunks
                ],
            )
            logger.info("Upserted %d chunks into collection doc_%s", len(chunks), document_id)

        def delete_collection(self, document_id: str) -> None:
            try:
                self.client.delete_collection(f"doc_{document_id}")
                logger.info("Deleted ChromaDB collection doc_%s", document_id)
            except Exception:
                logger.debug("Collection doc_%s not found, skipping delete", document_id)

        def search(self, document_id: str, query_embedding: list[float], top_k: int) -> list[dict]:
            col = self.get_or_create_collection(document_id)
            results = col.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            chunks = []
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                chunks.append({
                    "chunk_id":    results["ids"][0][i],
                    "document_id": meta["document_id"],
                    "page":        meta["page"],
                    "text":        doc,
                    "score":       round(1.0 - dist, 6),
                    "strategy":    meta.get("strategy", "recursive"),
                    "type":        meta.get("type", "text"),
                })
            return chunks

    _CHROMA_AVAILABLE = True

except ImportError:
    logger.warning("chromadb not installed — RAGPipeline (quiz/flashcard) unavailable")
    _CHROMA_AVAILABLE = False

    class VectorStoreService:
        def __init__(self):
            raise RuntimeError("chromadb is not installed. Run: pip install chromadb")


# ===========================================================================
# Shared async wrapper (used by documents.py DELETE endpoint)
# ===========================================================================

async def delete_document_vectors(document_id: str) -> None:
    if _QDRANT_AVAILABLE:
        _delete_qdrant_vectors(document_id)
    if _CHROMA_AVAILABLE:
        VectorStoreService().delete_collection(document_id)
