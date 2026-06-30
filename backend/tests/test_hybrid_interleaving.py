"""
Tests for HybridInterleavingRetriever — 6 test cases.
"""
from app.services.retrieval.base import BaseRetriever
from app.services.retrieval.hybrid_interleaving import HybridInterleavingRetriever


# ---------------------------------------------------------------------------
# Stub retrievers (no API needed)
# ---------------------------------------------------------------------------

class StubRetriever:
    def __init__(self, results):
        self._results = results

    def search(self, query, document_id, top_k=10):
        return self._results[:top_k]


def _chunk(cid, score=1.0):
    return {"chunk_id": cid, "document_id": "doc1", "page": 1, "text": f"text {cid}", "score": score}


# ---------------------------------------------------------------------------
# Test 1: Protocol conformance
# ---------------------------------------------------------------------------

def test_protocol_conformance():
    r = HybridInterleavingRetriever(StubRetriever([]), StubRetriever([]))
    assert isinstance(r, BaseRetriever)


# ---------------------------------------------------------------------------
# Test 2: Xen kẽ đúng thứ tự BM25 trước, Vector sau
# ---------------------------------------------------------------------------

def test_interleaving_order():
    bm25 = StubRetriever([_chunk("b0"), _chunk("b1")])
    vec  = StubRetriever([_chunk("v0"), _chunk("v1")])
    r = HybridInterleavingRetriever(bm25, vec)
    results = r.search("query", "doc1", top_k=4)
    ids = [x["chunk_id"] for x in results]
    # Thứ tự phải là: b0, v0, b1, v1
    assert ids == ["b0", "v0", "b1", "v1"], f"Got {ids}"


# ---------------------------------------------------------------------------
# Test 3: Dedup theo chunk_id — không có bản sao
# ---------------------------------------------------------------------------

def test_dedup_by_chunk_id():
    shared = _chunk("shared")
    bm25 = StubRetriever([shared, _chunk("b1")])
    vec  = StubRetriever([shared, _chunk("v1")])
    r = HybridInterleavingRetriever(bm25, vec)
    results = r.search("query", "doc1", top_k=10)
    ids = [x["chunk_id"] for x in results]
    assert len(ids) == len(set(ids)), f"Duplicate chunk_ids found: {ids}"
    assert ids.count("shared") == 1


# ---------------------------------------------------------------------------
# Test 4: top_k được tôn trọng
# ---------------------------------------------------------------------------

def test_top_k_respected():
    bm25 = StubRetriever([_chunk(f"b{i}") for i in range(5)])
    vec  = StubRetriever([_chunk(f"v{i}") for i in range(5)])
    r = HybridInterleavingRetriever(bm25, vec)
    results = r.search("query", "doc1", top_k=3)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# Test 5: BM25 rỗng → chỉ trả vector results
# ---------------------------------------------------------------------------

def test_bm25_empty_fallback():
    bm25 = StubRetriever([])
    vec  = StubRetriever([_chunk("v0"), _chunk("v1")])
    r = HybridInterleavingRetriever(bm25, vec)
    results = r.search("query", "doc1", top_k=5)
    ids = [x["chunk_id"] for x in results]
    assert ids == ["v0", "v1"]


# ---------------------------------------------------------------------------
# Test 6: Cả hai rỗng → list rỗng
# ---------------------------------------------------------------------------

def test_both_empty():
    r = HybridInterleavingRetriever(StubRetriever([]), StubRetriever([]))
    assert r.search("query", "doc1", top_k=5) == []
