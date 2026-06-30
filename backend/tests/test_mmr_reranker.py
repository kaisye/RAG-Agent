"""
Tests for MMRReranker — 8 test cases.
Dùng MockEmbeddingService để kiểm soát embedding, không cần NVIDIA API key.
"""
import math
import pytest

from app.services.reranking.base import BaseReranker
from app.services.reranking.mmr import MMRReranker
from app.services.reranking import get_reranker, IdentityReranker


# ---------------------------------------------------------------------------
# MockEmbeddingService — trả vector cố định để test có thể predict kết quả
# ---------------------------------------------------------------------------

class MockEmbeddingService:
    def __init__(self, vectors: dict[str, list[float]]):
        """vectors: map từ text → embedding vector."""
        self._vectors = vectors

    def embed_texts(self, texts: list[str], input_type: str) -> list[list[float]]:
        assert input_type == "query", f"MMRReranker phải dùng input_type='query', got {input_type!r}"
        return [self._vectors[t] for t in texts]


def _unit(v: list[float]) -> list[float]:
    """Normalize vector về unit length."""
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _make_reranker(vectors: dict[str, list[float]], top_k: int = 3, lambda_mult: float = 0.5):
    r = MMRReranker(top_k=top_k, lambda_mult=lambda_mult)
    r._embed_svc = MockEmbeddingService(vectors)
    return r


def _chunk(cid: str, text: str, score: float = 1.0) -> dict:
    return {"chunk_id": cid, "document_id": "doc1", "page": 1, "text": text, "score": score}


# ---------------------------------------------------------------------------
# Vectors cho test: query, 3 candidates
#   q  = (1, 0)
#   c0 = (1, 0)  — giống query nhất (relevance=1.0)
#   c1 = (0, 1)  — vuông góc với q và c0 (diversity cao)
#   c2 = (0.9, 0.4) norm — gần q, nhưng gần c0 hơn c1
# ---------------------------------------------------------------------------
Q  = "query"
T0 = "text_c0"
T1 = "text_c1"
T2 = "text_c2"

VECTORS = {
    Q:  _unit([1.0, 0.0]),
    T0: _unit([1.0, 0.0]),   # cos(q,c0)=1.0
    T1: _unit([0.0, 1.0]),   # cos(q,c1)=0.0, cos(c0,c1)=0.0 → diversity cao
    T2: _unit([0.9, 0.4]),   # cos(q,c2)≈0.91, cos(c0,c2)≈0.91 → gần giống c0
}

CANDIDATES = [
    _chunk("c0", T0, score=0.9),
    _chunk("c1", T1, score=0.5),
    _chunk("c2", T2, score=0.8),
]


# ---------------------------------------------------------------------------
# Test 1: Protocol conformance
# ---------------------------------------------------------------------------

def test_protocol_conformance():
    r = MMRReranker(top_k=3)
    assert isinstance(r, BaseReranker)


# ---------------------------------------------------------------------------
# Test 2: top_k được tôn trọng
# ---------------------------------------------------------------------------

def test_top_k_respected():
    r = _make_reranker(VECTORS, top_k=2)
    result = r.rerank(Q, CANDIDATES)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Test 3: candidates <= top_k → trả nguyên danh sách
# ---------------------------------------------------------------------------

def test_fewer_candidates_than_top_k():
    r = _make_reranker(VECTORS, top_k=10)
    result = r.rerank(Q, CANDIDATES)
    assert len(result) == len(CANDIDATES)


# ---------------------------------------------------------------------------
# Test 4: Vòng 1 — chọn chunk liên quan nhất với query
#   c0 có cos(q,c0)=1.0 → relevance cao nhất → phải được chọn đầu tiên
# ---------------------------------------------------------------------------

def test_first_selection_is_most_relevant():
    r = _make_reranker(VECTORS, top_k=1)
    result = r.rerank(Q, CANDIDATES)
    assert result[0]["chunk_id"] == "c0", f"Expected c0 first, got {result[0]['chunk_id']}"


# ---------------------------------------------------------------------------
# Test 5: Diversity — sau khi chọn c0, c1 đa dạng hơn c2
#   c2 gần với c0 (redundancy cao) → MMR ưu tiên c1 (cos(c0,c1)=0)
# ---------------------------------------------------------------------------

def test_diversity_over_redundancy():
    r = _make_reranker(VECTORS, top_k=2, lambda_mult=0.5)
    result = r.rerank(Q, CANDIDATES)
    assert result[0]["chunk_id"] == "c0"
    assert result[1]["chunk_id"] == "c1", (
        f"Expected c1 at pos 2 (diverse), got {result[1]['chunk_id']}. "
        "c1 vuông góc với c0 nên redundancy=0, MMR ưu tiên hơn c2 giống c0."
    )


# ---------------------------------------------------------------------------
# Test 6: lambda=1.0 → pure relevance, không quan tâm diversity
#   c0 rồi c2 (gần query nhất còn lại), không phải c1
# ---------------------------------------------------------------------------

def test_lambda_1_pure_relevance():
    r = _make_reranker(VECTORS, top_k=2, lambda_mult=1.0)
    result = r.rerank(Q, CANDIDATES)
    assert result[0]["chunk_id"] == "c0"
    # c2 cos≈0.91 > c1 cos=0.0, nên c2 xếp thứ 2
    assert result[1]["chunk_id"] == "c2", (
        f"lambda=1.0 should pick c2 (relevance≈0.91) over c1 (relevance=0.0), "
        f"got {result[1]['chunk_id']}"
    )


# ---------------------------------------------------------------------------
# Test 7: EmbeddingService phải dùng input_type="query" — mock sẽ assert
# ---------------------------------------------------------------------------

def test_embed_uses_query_input_type():
    # MockEmbeddingService assert input_type=="query" và raise nếu sai
    r = _make_reranker(VECTORS, top_k=2)
    r.rerank(Q, CANDIDATES)  # không raise → input_type đúng


# ---------------------------------------------------------------------------
# Test 8: factory get_reranker("mmr") trả MMRReranker
# ---------------------------------------------------------------------------

def test_factory_mmr():
    r = get_reranker("mmr", top_k=3)
    assert isinstance(r, MMRReranker)
