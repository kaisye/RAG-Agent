import importlib.util
import pathlib
import sys

from app.services.retrieval.base import BaseRetriever
from app.services.retrieval.vector import DenseRetriever


def get_retriever(
    strategy: str,
    chunks: list[dict],
    embed_service,
    vector_store,
) -> BaseRetriever:
    """
    Factory for all retrieval strategies.

    strategy:
      "vector"               — DenseRetriever (baseline)
      "bm25"                 — BM25Retriever
      "hybrid_interleaving"  — HybridInterleavingRetriever
      "hybrid_rrf"           — HybridRRFRetriever (production default)
    """
    dense = DenseRetriever(embed_service, vector_store)

    if strategy == "vector":
        return dense

    if strategy == "bm25":
        from app.services.retrieval.bm25 import BM25Retriever
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


# ---------------------------------------------------------------------------
# Backwards-compat: re-export Qdrant-based retrieve_debug for app.routers.chat
# retrieval.py (single-file module) is shadowed by this package at Python import
# time, so we load it via importlib to keep its exports accessible.
# ---------------------------------------------------------------------------

_qdrant_file = pathlib.Path(__file__).parent.parent / "retrieval.py"
if _qdrant_file.exists():
    _spec = importlib.util.spec_from_file_location("app.services._qdrant_retrieval", _qdrant_file)
    if _spec and _spec.loader:
        _qdrant_mod = importlib.util.module_from_spec(_spec)
        sys.modules.setdefault("app.services._qdrant_retrieval", _qdrant_mod)
        try:
            _spec.loader.exec_module(_qdrant_mod)
            retrieve_debug = _qdrant_mod.retrieve_debug
            retrieve = _qdrant_mod.retrieve
        except Exception:
            pass
