import logging
import re
from typing import NamedTuple

from rank_bm25 import BM25Okapi
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings
from app.services.embedding import embed_texts
from app.services.vector_store import _client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase and strip punctuation so 'recurrence.' matches 'recurrence'."""
    return [t for t in re.sub(r"[^\w\s]", " ", text.lower()).split() if t]


class _BM25Entry(NamedTuple):
    index: BM25Okapi
    chunks: list[dict]   # full chunk dicts in corpus order


# In-process cache: document_id → _BM25Entry
# Only text chunks are indexed (images have empty content and are skipped).
_bm25_cache: dict[str, _BM25Entry] = {}


def build_bm25_index(document_id: str) -> _BM25Entry:
    """
    Build (or return cached) a BM25Okapi index for all text chunks of a document.

    Reads chunks from storage/parsed/{document_id}_chunks.json, filters to
    type=='text' (image entries have empty content and hurt precision), and
    tokenises by whitespace.

    The cache is process-local; it is rebuilt on server restart or when
    called explicitly after new ingestion.
    """
    if document_id in _bm25_cache:
        return _bm25_cache[document_id]

    from app.services.chunker import load_chunks  # late import — avoids circular at module load

    all_chunks = load_chunks(document_id)
    text_chunks = [c for c in all_chunks if c.get("type", "text") == "text" and c.get("text", "").strip()]

    if not text_chunks:
        raise ValueError(f"No text chunks found for document_id={document_id!r}")

    tokenized = [_tokenize(c["text"]) for c in text_chunks]
    entry = _BM25Entry(index=BM25Okapi(tokenized), chunks=text_chunks)
    _bm25_cache[document_id] = entry
    logger.info("build_bm25_index: %s → %d text chunks indexed", document_id, len(text_chunks))
    return entry


def bm25_search(
    query: str,
    document_id: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Keyword-based retrieval using BM25Okapi over the text chunks of *document_id*.

    Returns top-k results in the same shape as vector_search() so the two can
    be fused by hybrid-search.  Scores are raw BM25 scores (not normalised).
    Only text chunks are returned; images are not in the BM25 index.
    """
    entry = build_bm25_index(document_id)
    tokens = _tokenize(query)
    scores = entry.index.get_scores(tokens)

    # Pair each chunk with its BM25 score, sort descending, take top-k
    ranked = sorted(
        zip(entry.chunks, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )[:top_k]

    results = [
        {
            "chunk_id":    chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "page":        chunk["page"],
            "type":        "text",
            "content":     chunk["text"],
            "source_path": None,
            "score":       float(score),
        }
        for chunk, score in ranked
        if score > 0   # skip chunks with zero relevance
    ]

    logger.info(
        "bm25_search: query=%r doc=%s top_k=%d → %d results",
        query[:60], document_id, top_k, len(results),
    )
    return results
