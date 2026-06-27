import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self):
        settings = get_settings()
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._settings = settings

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def get_or_create_collection(self, document_id: str):
        s = self._settings
        return self.client.get_or_create_collection(
            name=f"doc_{document_id}",
            metadata={
                "hnsw:space":           "cosine",
                "hnsw:M":               s.hnsw_m,            # 16
                "hnsw:construction_ef": s.hnsw_ef_construct,  # 100
                "hnsw:search_ef":       s.hnsw_ef_search,     # 128
            },
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        document_id: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
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

    # ------------------------------------------------------------------
    # Delete — replaces the no-op stub from nhánh 2.2
    # ------------------------------------------------------------------

    def delete_collection(self, document_id: str) -> None:
        try:
            self.client.delete_collection(f"doc_{document_id}")
            logger.info("Deleted collection doc_%s", document_id)
        except Exception:
            logger.debug("Collection doc_%s not found, skipping delete", document_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        document_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
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
                "score":       round(1.0 - dist, 6),  # cosine distance → similarity
                "strategy":    meta.get("strategy", "recursive"),
                "type":        meta.get("type", "text"),
            })
        return chunks


# ---------------------------------------------------------------------------
# Module-level async wrapper used by routers/documents.py (DELETE endpoint).
# Wraps the sync VectorStoreService to keep the router interface unchanged.
# ---------------------------------------------------------------------------

async def delete_document_vectors(document_id: str) -> None:
    VectorStoreService().delete_collection(document_id)
