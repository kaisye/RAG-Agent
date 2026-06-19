import logging

from qdrant_client.models import FieldCondition, Filter, MatchValue, Query

from app.core.config import settings
from app.services.embedding import embed_texts
from app.services.vector_store import _client

logger = logging.getLogger(__name__)


def vector_search(
    query: str,
    top_k: int = 10,
    document_id: str | None = None,
) -> list[dict]:
    """
    Embed *query* with input_type='query' and search Qdrant for the top-k
    most similar points (text chunks and images combined).

    Args:
        query:       The user's question or search string.
        top_k:       Maximum number of results to return.
        document_id: If given, restrict the search to a single document.

    Returns:
        List of result dicts sorted by descending similarity score.
    """
    query_vector = embed_texts([query], input_type="query")[0]

    query_filter = None
    if document_id:
        query_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )

    response = _client().query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    hits = response.points

    results = [
        {
            "chunk_id":    hit.payload["chunk_id"],
            "document_id": hit.payload["document_id"],
            "page":        hit.payload["page"],
            "type":        hit.payload["type"],
            "content":     hit.payload["content"],
            "source_path": hit.payload.get("source_path"),
            "score":       hit.score,
        }
        for hit in hits
    ]

    logger.info(
        "vector_search: query=%r top_k=%d doc_filter=%s → %d results",
        query[:60], top_k, document_id, len(results),
    )
    return results
