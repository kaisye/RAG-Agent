"""
Evaluation helper endpoint — used by scripts/eval_suite.py.

Keeps Qdrant access inside the backend process so the eval script
does not need to open the local storage file (which only allows 1 process).
"""
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings
from app.services.llm.providers import get_llm_provider

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/reindex")
def reindex_document(document_id: str):
    """Re-embed and upsert a document's chunks into Qdrant (fixes missing vectors)."""
    from app.services.chunker import load_chunks
    from app.services.embedding import embed_texts
    from app.services.vector_store import upsert_chunks

    chunks = load_chunks(document_id)
    text_chunks = [c for c in chunks if c.get("text")]
    if not text_chunks:
        return {"error": f"No chunks found for {document_id}"}

    texts = [c["text"] for c in text_chunks]
    vectors = embed_texts(texts, input_type="passage")
    for chunk, vec in zip(text_chunks, vectors):
        chunk["vector"] = vec

    n = upsert_chunks(text_chunks, [])
    return {"document_id": document_id, "upserted": n}


@router.get("/debug_vector")
def debug_vector(document_id: str, q: str = "What is FastAPI?"):
    """Temporary: compare query_points vs internal search."""
    from app.services.embedding import embed_texts
    from app.services.vector_store import _client
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    vec = embed_texts([q], input_type="query")[0]
    filt = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
    client = _client()

    # Try query_points with filter
    resp = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vec,
        query_filter=filt,
        limit=5,
        with_payload=True,
    )
    qp_with_filter = len(resp.points)

    # Try query_points WITHOUT filter
    resp2 = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vec,
        limit=5,
        with_payload=True,
    )
    qp_no_filter = len(resp2.points)

    # Sample a point to check payload structure
    sample, _ = client.scroll(settings.qdrant_collection, limit=1, with_payload=True, with_vectors=False)
    sample_payload = dict(sample[0].payload) if sample else {}

    # Collection info
    info = client.get_collection(settings.qdrant_collection)
    return {
        "points_count": info.points_count,
        "vec_dim": len(vec),
        "query_points_with_filter": qp_with_filter,
        "query_points_no_filter": qp_no_filter,
        "sample_payload_keys": list(sample_payload.keys()),
        "sample_doc_id": sample_payload.get("document_id", "?"),
    }


class RunQARequest(BaseModel):
    question: str
    document_id: str
    hybrid: bool = True
    reranker: bool = True
    hyde: bool = False
    decomposition: bool = False
    top_k: int = 5


class RunQAResponse(BaseModel):
    answer: str
    contexts: list[str]


_ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer using ONLY the provided context. "
    "Be concise and factual. "
    "If the answer is not in the context, say: 'I don't know based on the provided context.'"
)


def _retrieve_with_flags(
    query: str,
    document_id: str,
    hybrid: bool,
    reranker: bool,
    hyde: bool,
    decomposition: bool,
    top_k: int,
) -> list[dict]:
    from app.services.retrieval import (
        vector_search, hybrid_search, rerank,
        generate_hypothetical_doc, decompose_query, _multi_rrf,
    )

    pool = top_k * 4
    sub_queries = decompose_query(query) if decomposition else [query]
    hyde_docs = [generate_hypothetical_doc(q) for q in sub_queries] if hyde else []

    all_results: list[list[dict]] = []
    per_pool = pool * max(1, len(sub_queries))

    for i, q in enumerate(sub_queries):
        hyde_text = hyde_docs[i] if i < len(hyde_docs) else None
        if hybrid:
            results = hybrid_search(q, document_id=document_id, top_k=per_pool, hyde_text=hyde_text)
        else:
            vec_text = hyde_text if hyde_text else q
            vec_type = "passage" if hyde_text else "query"
            results = vector_search(vec_text, top_k=per_pool, document_id=document_id, input_type=vec_type)
        if results:
            all_results.append(results)

    if not all_results:
        return []

    candidates = (
        all_results[0][:pool]
        if len(all_results) == 1
        else _multi_rrf(all_results, k=settings.rrf_k, pool=pool)
    )

    if reranker:
        return rerank(query, candidates, top_k=top_k)
    return candidates[:top_k]


@router.post("/run_qa", response_model=RunQAResponse)
def run_qa(req: RunQARequest) -> RunQAResponse:
    """Retrieve context and generate an answer for one Q&A evaluation sample."""
    chunks = _retrieve_with_flags(
        query=req.question,
        document_id=req.document_id,
        hybrid=req.hybrid,
        reranker=req.reranker,
        hyde=req.hyde,
        decomposition=req.decomposition,
        top_k=req.top_k,
    )

    text_contexts = [
        c["content"] for c in chunks
        if c.get("type") == "text" and c.get("content")
    ]

    if not text_contexts:
        return RunQAResponse(answer="I don't know based on the provided context.", contexts=[])

    ctx_block = "\n\n".join(
        f"[Page {c.get('page', '?')}] {c['content']}"
        for c in chunks if c.get("type") == "text"
    )
    provider = get_llm_provider()
    resp = provider.chat(
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user",   "content": f"Context:\n{ctx_block}\n\nQuestion: {req.question}"},
        ],
        stream=False,
    )
    answer = resp.choices[0].message.content.strip()

    logger.info("eval/run_qa: q=%r → %d contexts, answer_len=%d", req.question[:60], len(text_contexts), len(answer))
    return RunQAResponse(answer=answer, contexts=text_contexts)
