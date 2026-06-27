from app.services.retrieval.base import BaseRetriever
from app.services.retrieval.vector import DenseRetriever


def get_retriever(
    strategy: str,
    chunks: list[dict],
    embed_service,
    vector_store,
) -> BaseRetriever:
    """
    Factory cho tất cả retrieval strategies.

    strategy:
      "vector"              — DenseRetriever (baseline, implemented here)
      "bm25"               — BM25Retriever (feature/bm25-retrieval)
      "hybrid_interleaving" — HybridInterleavingRetriever (feature/hybrid-interleaving)
      "hybrid_rrf"          — HybridRRFRetriever k=60 (feature/hybrid-rrf, production default)
    """
    dense = DenseRetriever(embed_service, vector_store)

    if strategy == "vector":
        return dense

    if strategy == "bm25":
        from app.services.retrieval.bm25 import BM25Retriever  # feature/bm25-retrieval
        return BM25Retriever(chunks)

    if strategy == "hybrid_interleaving":
        from app.services.retrieval.bm25 import BM25Retriever
        from app.services.retrieval.hybrid_interleaving import HybridInterleavingRetriever
        return HybridInterleavingRetriever(BM25Retriever(chunks), dense)

    if strategy == "hybrid_rrf":
        from app.core.config import get_settings
        from app.services.retrieval.bm25 import BM25Retriever
        from app.services.retrieval.hybrid_rrf import HybridRRFRetriever
        return HybridRRFRetriever(BM25Retriever(chunks), dense, rrf_k=get_settings().rrf_k)

    raise ValueError(f"Unknown retrieval strategy: {strategy!r}")
