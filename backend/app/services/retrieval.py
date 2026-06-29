import json as _json
import logging
import re
import time as _time
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
    document_ids: list[str] | None = None,
    input_type: str = "query",
) -> list[dict]:
    """
    Embed *query* with input_type='query' and search Qdrant for the top-k
    most similar points (text chunks and images combined).

    Args:
        query:        The text to embed and search with.
        top_k:        Maximum number of results to return.
        document_id:  If given, restrict the search to a single document (legacy).
        document_ids: If given, restrict the search to this set of documents.
                      Takes precedence over document_id when set.
        input_type:   Embedding input type: 'query' for short queries,
                      'passage' for longer hypothetical documents (HyDE).

    Returns:
        List of result dicts sorted by descending similarity score.
    """
    query_vector = embed_texts([query], input_type=input_type)[0]

    # Resolve effective document set (document_ids takes precedence)
    effective_ids: list[str] | None = document_ids or ([document_id] if document_id else None)

    query_filter = None
    if effective_ids:
        if len(effective_ids) == 1:
            query_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=effective_ids[0]))]
            )
        else:
            query_filter = Filter(
                should=[
                    FieldCondition(key="document_id", match=MatchValue(value=did))
                    for did in effective_ids
                ]
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
        query[:60], top_k, effective_ids, len(results),
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


def bm25_search_multi(
    query: str,
    document_ids: list[str],
    top_k: int = 10,
) -> list[dict]:
    """BM25 search across multiple documents — merges and re-ranks by score."""
    all_results: list[dict] = []
    for did in document_ids:
        try:
            all_results.extend(bm25_search(query, document_id=did, top_k=top_k))
        except ValueError:
            logger.warning("bm25_search_multi: no chunks for doc %s — skipping", did)
    return sorted(all_results, key=lambda r: r["score"], reverse=True)[:top_k]


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
    document_id: str | None = None,
    document_ids: list[str] | None = None,
    top_k: int = 20,
    hyde_text: str | None = None,
) -> list[dict]:
    """
    Retrieve top-k candidates by fusing vector search and BM25 with RRF.

    Accepts either a single document_id (legacy) or a list of document_ids
    (for project-level multi-document retrieval).

    Args:
        query:        User question or search string (always used for BM25).
        document_id:  Restrict retrieval to a single document (legacy).
        document_ids: Restrict retrieval to this set of documents (takes precedence).
        top_k:        Number of candidates to return (before reranking).
        hyde_text:    HyDE hypothetical document text.

    Returns:
        List of result dicts sorted by descending RRF score.
    """
    effective_ids: list[str] = document_ids or ([document_id] if document_id else [])

    # HyDE: use the hypothetical doc for vector search, original query for BM25
    vec_text = hyde_text if hyde_text else query
    vec_input_type = "passage" if hyde_text else "query"

    if not settings.hybrid_search_enabled:
        logger.info("hybrid_search: disabled — falling back to vector_search only")
        return vector_search(
            vec_text, top_k=top_k,
            document_ids=effective_ids or None,
            input_type=vec_input_type,
        )

    # Retrieve a larger pool from each source so RRF has enough candidates
    pool = top_k * 2
    vec_results = vector_search(
        vec_text, top_k=pool,
        document_ids=effective_ids or None,
        input_type=vec_input_type,
    )
    if len(effective_ids) == 1:
        bm25_results = bm25_search(query, document_id=effective_ids[0], top_k=pool)
    elif effective_ids:
        bm25_results = bm25_search_multi(query, document_ids=effective_ids, top_k=pool)
    else:
        bm25_results = []

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
        "hybrid_search: query=%r docs=%s top_k=%d → %d results (vec=%d bm25=%d)",
        query[:60], effective_ids, top_k, len(results), len(vec_results), len(bm25_results),
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
# HyDE — Hypothetical Document Embeddings
# ---------------------------------------------------------------------------

_HYDE_PROMPT = (
    "Write a concise passage (3-5 sentences) that directly and factually answers "
    "the question below. Write as if you are the source document itself, not an "
    "assistant. Be specific.\n\nQuestion: {query}\n\nPassage:"
)


def generate_hypothetical_doc(query: str) -> str:
    """
    Generate a hypothetical document that would answer *query* (HyDE technique).

    The generated text is embedded as a 'passage' and used for vector search
    instead of the raw query.  BM25 still uses the original query.
    Falls back to the original query on any LLM error so retrieval never breaks.
    """
    from app.services.llm.providers import get_llm_provider
    try:
        provider = get_llm_provider()
        resp = provider.chat(
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
            stream=False,
        )
        hyp = resp.choices[0].message.content.strip()
        logger.info("HyDE: generated %d-char hypothetical doc for query=%r", len(hyp), query[:60])
        return hyp
    except Exception:
        logger.exception("HyDE generation failed — falling back to original query")
        return query


# ---------------------------------------------------------------------------
# Query Decomposition
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = (
    "Break the following question into exactly {n} simpler, focused sub-questions "
    "that together fully cover the original question.\n"
    "Return ONLY a JSON array of strings — no explanation, no markdown.\n\n"
    "Question: {query}\n\nJSON array:"
)


def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex query into N simpler sub-queries.

    Each sub-query is searched independently; results are merged with RRF
    before reranking.  The original query is always included so simple
    questions still work without decomposition noise.
    Falls back to [query] on any LLM or parse error.
    """
    n = settings.query_decomposition_n
    from app.services.llm.providers import get_llm_provider
    try:
        provider = get_llm_provider()
        resp = provider.chat(
            messages=[{"role": "user", "content": _DECOMPOSE_PROMPT.format(n=n, query=query)}],
            stream=False,
        )
        raw = resp.choices[0].message.content.strip()
        s, e = raw.find("["), raw.rfind("]") + 1
        if s != -1 and e > 0:
            parts = _json.loads(raw[s:e])
            sub_qs = [str(p).strip() for p in parts if str(p).strip()][:n]
            if sub_qs:
                logger.info(
                    "decompose_query: %d sub-queries for %r: %s",
                    len(sub_qs), query[:60], sub_qs,
                )
                return sub_qs
    except Exception:
        logger.exception("Query decomposition failed — using original query")
    return [query]


# ---------------------------------------------------------------------------
# Multi-query RRF merge
# ---------------------------------------------------------------------------

def _multi_rrf(result_lists: list[list[dict]], k: int, pool: int) -> list[dict]:
    """
    Merge multiple independent ranked result lists using Reciprocal Rank Fusion.

    Each list contributes rank-based scores; chunks appearing in several lists
    accumulate higher scores.  Returns up to *pool* deduplicated results.
    """
    rrf: dict[str, float] = {}
    for ranked in result_lists:
        for rank, r in enumerate(ranked):
            cid = r["chunk_id"]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:pool]

    payload: dict[str, dict] = {}
    for ranked in result_lists:
        for r in ranked:
            payload.setdefault(r["chunk_id"], r)

    return [{**payload[cid], "score": score} for cid, score in sorted_ids if cid in payload]


# ---------------------------------------------------------------------------
# retrieve() — main entry point for feature/chat
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    document_id: str | None = None,
    document_ids: list[str] | None = None,
    top_k_final: int = 5,
    hybrid_pool: int = 20,
) -> list[dict]:
    """
    Full retrieval pipeline with optional Query Decomposition and HyDE.

    Accepts either document_id (single doc, legacy) or document_ids (multi-doc project).

    Pipeline:
        1. [Query Decomposition] split query → N sub-queries  (if enabled)
        2. [HyDE]                LLM generates hypothetical doc per sub-query (if enabled)
        3. hybrid_search         vector (+ optional HyDE text) + BM25, RRF-fused
        4. [multi-query RRF]     merge results from all sub-queries  (if >1 sub-query)
        5. [Reranker]            cross-encoder reranking with ORIGINAL query
    """
    effective_ids: list[str] = document_ids or ([document_id] if document_id else [])

    # Step 1: Query Decomposition
    if settings.query_decomposition_enabled:
        sub_queries = decompose_query(query)
    else:
        sub_queries = [query]

    # Step 2 & 3: HyDE + hybrid search for each sub-query
    per_query_pool = hybrid_pool * max(1, len(sub_queries))
    all_result_lists: list[list[dict]] = []

    for q in sub_queries:
        hyde_text = generate_hypothetical_doc(q) if settings.hyde_enabled else None
        results = hybrid_search(
            q,
            document_ids=effective_ids or None,
            top_k=per_query_pool,
            hyde_text=hyde_text,
        )
        if results:
            all_result_lists.append(results)

    if not all_result_lists:
        logger.warning("retrieve: all sub-queries returned empty results")
        return []

    # Step 4: Merge multiple result lists (only needed for decomposition)
    if len(all_result_lists) == 1:
        candidates = all_result_lists[0][:hybrid_pool]
    else:
        candidates = _multi_rrf(all_result_lists, k=settings.rrf_k, pool=hybrid_pool)
        logger.info(
            "retrieve: merged %d sub-query result lists → %d candidates",
            len(all_result_lists), len(candidates),
        )

    # Step 5: Rerank with the ORIGINAL query (not sub-queries or hypothetical docs)
    if not settings.reranker_enabled:
        logger.info("retrieve: reranker disabled — returning top-%d", top_k_final)
        return candidates[:top_k_final]

    return rerank(query, candidates, top_k=top_k_final)


# ---------------------------------------------------------------------------
# retrieve_debug() — same pipeline, also returns intermediate step data
# ---------------------------------------------------------------------------

def _to_debug_chunks(chunks: list[dict], n: int = 5) -> list[dict]:
    return [
        {
            "chunk_id": c["chunk_id"],
            "page":     c["page"],
            "score":    round(float(c.get("score", 0)), 4),
            "snippet":  (c.get("content") or "")[:120],
        }
        for c in chunks[:n]
    ]


def retrieve_debug(
    query: str,
    document_id: str | None = None,
    document_ids: list[str] | None = None,
    top_k_final: int = 5,
    hybrid_pool: int = 20,
) -> tuple[list[dict], dict]:
    """
    Identical to retrieve() but also returns a debug dict capturing every
    intermediate step: sub_queries, hyde_docs, vector/bm25/rrf/rerank hits,
    and per-step latency in milliseconds.
    """
    effective_ids: list[str] = document_ids or ([document_id] if document_id else [])
    latency: dict[str, int] = {}

    # Step 1: Query Decomposition + HyDE
    t0 = _time.monotonic()
    sub_queries = decompose_query(query) if settings.query_decomposition_enabled else [query]
    hyde_docs: list[str] = [generate_hypothetical_doc(q) for q in sub_queries] if settings.hyde_enabled else []
    latency["transform"] = int((_time.monotonic() - t0) * 1000)

    # Step 2 & 3: Vector + BM25 per sub-query
    t0 = _time.monotonic()
    per_query_pool = hybrid_pool * max(1, len(sub_queries))
    all_result_lists: list[list[dict]] = []
    first_vec_hits: list[dict] = []
    first_bm25_hits: list[dict] = []

    for i, q in enumerate(sub_queries):
        hyde_text = hyde_docs[i] if i < len(hyde_docs) else None
        vec_text = hyde_text if hyde_text else q
        vec_input_type = "passage" if hyde_text else "query"

        if settings.hybrid_search_enabled:
            pool = per_query_pool * 2
            vec_results = vector_search(
                vec_text, top_k=pool,
                document_ids=effective_ids or None,
                input_type=vec_input_type,
            )
            if len(effective_ids) == 1:
                bm25_results = bm25_search(q, document_id=effective_ids[0], top_k=pool)
            elif effective_ids:
                bm25_results = bm25_search_multi(q, document_ids=effective_ids, top_k=pool)
            else:
                bm25_results = []
            if i == 0:
                first_vec_hits  = vec_results
                first_bm25_hits = bm25_results
            fused = _rrf_fusion(vec_results, bm25_results, k=settings.rrf_k)[:per_query_pool]
            pm = {r["chunk_id"]: r for r in vec_results}
            pm.update({r["chunk_id"]: r for r in bm25_results})
            results = [{**pm[cid], "score": s} for cid, s in fused if cid in pm]
        else:
            results = vector_search(
                vec_text, top_k=per_query_pool,
                document_ids=effective_ids or None,
                input_type=vec_input_type,
            )
            if i == 0:
                first_vec_hits = results

        if results:
            all_result_lists.append(results)
    latency["search"] = int((_time.monotonic() - t0) * 1000)

    if not all_result_lists:
        return [], {
            "sub_queries": sub_queries, "hyde_docs": hyde_docs,
            "vector_hits": [], "bm25_hits": [], "rrf_candidates": [], "reranked": [],
            "latency_ms": latency,
        }

    # Step 4: Merge
    candidates = (
        all_result_lists[0][:hybrid_pool]
        if len(all_result_lists) == 1
        else _multi_rrf(all_result_lists, k=settings.rrf_k, pool=hybrid_pool)
    )

    # Step 5: Rerank
    t0 = _time.monotonic()
    final = rerank(query, candidates, top_k=top_k_final) if settings.reranker_enabled else candidates[:top_k_final]
    latency["rerank"] = int((_time.monotonic() - t0) * 1000)

    debug = {
        "sub_queries":    sub_queries,
        "hyde_docs":      hyde_docs,
        "vector_hits":    _to_debug_chunks(first_vec_hits),
        "bm25_hits":      _to_debug_chunks(first_bm25_hits),
        "rrf_candidates": _to_debug_chunks(candidates, n=10),
        "reranked":       _to_debug_chunks(final),
        "latency_ms":     latency,
    }
    return final, debug
