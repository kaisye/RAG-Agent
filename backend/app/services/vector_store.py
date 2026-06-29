import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


_singleton: QdrantClient | None = None


def _client() -> QdrantClient:
    """Return a process-wide singleton QdrantClient.

    `:memory:` — local in-process store (no Docker needed, data lost on restart).
    Any URL    — persistent remote Qdrant; reuse the same connection pool across
                 all requests instead of opening a new TCP connection each time.
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    url = settings.qdrant_url
    if url == ":memory:":
        _singleton = QdrantClient(location=":memory:")
    elif url.startswith("./") or url.startswith(".\\") or (len(url) > 1 and url[1] == ":"):
        # Local file path (e.g. ./qdrant_storage) — persistent, no Docker needed
        from pathlib import Path
        Path(url).mkdir(parents=True, exist_ok=True)
        _singleton = QdrantClient(path=url)
    else:
        _singleton = QdrantClient(url=url)
    return _singleton


def _point_id(logical_id: str) -> str:
    """Return a stable UUID5 string from a logical chunk/image ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, logical_id))


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def ensure_collection() -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    client = _client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection in existing:
        logger.info("collection '%s' already exists", settings.qdrant_collection)
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=settings.qdrant_vector_size,
            distance=Distance.COSINE,
        ),
    )
    logger.info(
        "created collection '%s' (dim=%d, cosine)",
        settings.qdrant_collection,
        settings.qdrant_vector_size,
    )


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(chunks: list[dict], images: list[dict]) -> int:
    """
    Upsert text chunks and image embeddings into Qdrant.

    Args:
        chunks: output of load_chunks() — each must have a 'vector' field.
        images: output of load_image_metadata() — each must have a 'vector' field.

    Returns:
        Total number of points upserted.
    """
    client = _client()
    points: list[PointStruct] = []

    for chunk in chunks:
        vector = chunk.get("vector")
        if not vector:
            logger.warning("chunk %s has no vector, skipping", chunk.get("chunk_id"))
            continue
        points.append(
            PointStruct(
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
            )
        )

    for img in images:
        vector = img.get("vector")
        if not vector:
            logger.warning("image %s has no vector, skipping", img.get("image_id"))
            continue
        points.append(
            PointStruct(
                id=_point_id(img["image_id"]),
                vector=vector,
                payload={
                    "document_id": img["document_id"],
                    "page":        img["page"],
                    "chunk_id":    img["image_id"],
                    "type":        "image",
                    "content":     "",     # caption injected by embed_image fallback is not stored here;
                                           # nhánh citation sẽ render thumbnail trực tiếp từ source_path
                    "source_path": img.get("file_path"),
                },
            )
        )

    if not points:
        logger.info("upsert_chunks: nothing to upsert")
        return 0

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info(
        "upsert_chunks: %d points upserted to '%s'",
        len(points),
        settings.qdrant_collection,
    )
    return len(points)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete_document_vectors(document_id: str) -> None:
    """Delete all points whose payload.document_id matches document_id."""
    client = _client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
    )
    logger.info(
        "delete_document_vectors: removed points for document_id=%s from '%s'",
        document_id,
        settings.qdrant_collection,
    )
