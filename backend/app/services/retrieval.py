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


# ---------------------------------------------------------------------------
# Hybrid search (RRF fusion of vector + BM25)
# ---------------------------------------------------------------------------

def _rrf_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion over two ranked lists.

    RRF score = Σ 1/(k + rank + 1) across all lists.
    Uses only rank, not raw scores, so cosine and BM25 values need no normalisation.

    Returns list of (chunk_id, rrf_score) sorted by descending score.
    """
    rrf: dict[str, float] = {}
    for rank, result in enumerate(vector_results):
        cid = result["chunk_id"]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, result in enumerate(bm25_results):
        cid = result["chunk_id"]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    document_id: str,
    top_k: int = 20,
) -> list[dict]:
    """
    Retrieve top-k candidates by fusing vector search and BM25 with RRF.

    When settings.hybrid_search_enabled is False, falls back to vector_search
    only — this lets the evaluation nhánh run ablation experiments by toggling
    the HYBRID_SEARCH_ENABLED env var without changing code.

    Args:
        query:       User question or search string.
        document_id: Restrict retrieval to a single document.
        top_k:       Number of candidates to return (before reranking).

    Returns:
        List of result dicts (same shape as vector_search / bm25_search),
        sorted by descending RRF score.
    """
    if not settings.hybrid_search_enabled:
        logger.info("hybrid_search: disabled — falling back to vector_search only")
        return vector_search(query, top_k=top_k, document_id=document_id)

    # Retrieve a larger pool from each source so RRF has enough candidates
    pool = top_k * 2
    vec_results = vector_search(query, top_k=pool, document_id=document_id)
    bm25_results = bm25_search(query, document_id=document_id, top_k=pool)

    fused = _rrf_fusion(vec_results, bm25_results, k=settings.rrf_k)[:top_k]

    # Build a lookup by chunk_id from both result lists to recover full payload
    payload: dict[str, dict] = {r["chunk_id"]: r for r in vec_results}
    payload.update({r["chunk_id"]: r for r in bm25_results})

    results = [
        {**payload[cid], "score": rrf_score}
        for cid, rrf_score in fused
        if cid in payload
    ]

    logger.info(
        "hybrid_search: query=%r doc=%s top_k=%d → %d results (vec=%d bm25=%d)",
        query[:60], document_id, top_k, len(results), len(vec_results), len(bm25_results),
    )
    return results


# ---------------------------------------------------------------------------
# Reranker — cross-encoder score for exact query-chunk relevance
# ---------------------------------------------------------------------------

# Module-level singleton: the CrossEncoder is expensive to load (~400 MB),
# so we load it once per process and reuse it across all requests.
_cross_encoder: "CrossEncoder | None" = None  # type: ignore[name-defined]


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder  # lazy import — avoids load at startup
        _cross_encoder = CrossEncoder(settings.reranker_model)
        logger.info("Loaded CrossEncoder model: %s", settings.reranker_model)
    return _cross_encoder


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Re-score *candidates* using a local CrossEncoder and return the top-k.

    CrossEncoder evaluates each (query, chunk) pair jointly — more accurate
    than bi-encoder cosine similarity at the cost of being O(n) on candidates.
    Call after hybrid_search with a bounded candidate pool (20–30) so latency
    stays acceptable.

    The returned dicts have the same shape as hybrid_search() results, with
    'score' replaced by the CrossEncoder logit (higher = more relevant).
    """
    if not candidates:
        return []

    pairs = [(query, c["content"]) for c in candidates]
    scores = _get_cross_encoder().predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = [{**chunk, "score": float(score)} for chunk, score in ranked]

    logger.info(
        "rerank: query=%r candidates=%d → top_k=%d",
        query[:60], len(candidates), len(results),
    )
    return results


# ---------------------------------------------------------------------------
# retrieve() — main entry point for feature/chat
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    document_id: str,
    top_k_final: int = 5,
    hybrid_pool: int = 20,
) -> list[dict]:
    """
    Full retrieval pipeline: hybrid_search → (optional) rerank.

    Feature/chat calls this function exclusively — it never calls
    vector_search / bm25_search / rerank directly.

    Args:
        query:        User question.
        document_id:  Document to search within.
        top_k_final:  Number of chunks to return after reranking.
        hybrid_pool:  Candidates passed to reranker (before trimming to top_k_final).
                      Ignored when reranker is disabled.

    Returns:
        List of up to top_k_final result dicts, sorted by relevance.

    Ablation:
        HYBRID_SEARCH_ENABLED=false  → vector-only, then rerank
        RERANKER_ENABLED=false       → hybrid results, no rerank (BM25 order)
        Both false                   → pure vector search, no rerank
    """
    candidates = hybrid_search(query, document_id=document_id, top_k=hybrid_pool)

    if not settings.reranker_enabled:
        logger.info("retrieve: reranker disabled — returning hybrid top-%d", top_k_final)
        return candidates[:top_k_final]

    return rerank(query, candidates, top_k=top_k_final)
