"""
Tests for HybridRRFRetriever — 8 test cases.
"""
from app.services.retrieval.base import BaseRetriever
from app.services.retrieval.hybrid_rrf import HybridRRFRetriever


# ---------------------------------------------------------------------------
# Stub retrievers
# ---------------------------------------------------------------------------

class StubRetriever:
    def __init__(self, results):
        self._results = results

    def search(self, query, document_id, top_k=10):
        return self._results[:top_k]


def _chunk(cid, score=1.0, page=1):
    return {"chunk_id": cid, "document_id": "doc1", "page": page,
            "text": f"text {cid}", "score": score}


# ---------------------------------------------------------------------------
# Test 1: Protocol conformance
# ---------------------------------------------------------------------------

def test_protocol_conformance():
    r = HybridRRFRetriever(StubRetriever([]), StubRetriever([]))
    assert isinstance(r, BaseRetriever)


# ---------------------------------------------------------------------------
# Test 2: RRF score — chunk đồng thuận từ cả hai retriever xếp cao hơn
# ---------------------------------------------------------------------------

def test_consensus_chunk_ranks_higher():
    # "shared" xuất hiện ở hạng 3 cả BM25 lẫn Vector
    # "top_vector" chỉ xuất hiện ở hạng 1 Vector
    bm25   = StubRetriever([_chunk("b0"), _chunk("b1"), _chunk("shared")])
    vector = StubRetriever([_chunk("top_vector"), _chunk("v1"), _chunk("shared")])
    r = HybridRRFRetriever(bm25, vector, rrf_k=60)
    results = r.search("query", "doc1", top_k=5)
    ids = [x["chunk_id"] for x in results]

    # "shared" có 1/63 + 1/63 ≈ 0.0317 > "top_vector" 1/61 ≈ 0.0164
    shared_pos  = ids.index("shared")
    top_vec_pos = ids.index("top_vector")
    assert shared_pos < top_vec_pos, (
        f"Chunk đồng thuận 'shared' phải xếp trước 'top_vector' đơn lẻ. "
        f"Got shared={shared_pos}, top_vector={top_vec_pos}. ids={ids}"
    )


# ---------------------------------------------------------------------------
# Test 3: Score được gán và là float
# ---------------------------------------------------------------------------

def test_score_is_float():
    bm25   = StubRetriever([_chunk("a"), _chunk("b")])
    vector = StubRetriever([_chunk("a"), _chunk("c")])
    r = HybridRRFRetriever(bm25, vector, rrf_k=60)
    results = r.search("q", "doc1", top_k=5)
    for res in results:
        assert isinstance(res["score"], float), f"score phải là float, got {type(res['score'])}"
        assert res["score"] > 0


# ---------------------------------------------------------------------------
# Test 4: Không có bản sao chunk_id
# ---------------------------------------------------------------------------

def test_no_duplicate_chunk_ids():
    bm25   = StubRetriever([_chunk("x"), _chunk("y")])
    vector = StubRetriever([_chunk("x"), _chunk("z")])
    r = HybridRRFRetriever(bm25, vector, rrf_k=60)
    results = r.search("q", "doc1", top_k=10)
    ids = [res["chunk_id"] for res in results]
    assert len(ids) == len(set(ids)), f"Duplicate ids: {ids}"


# ---------------------------------------------------------------------------
# Test 5: top_k được tôn trọng
# ---------------------------------------------------------------------------

def test_top_k_respected():
    bm25   = StubRetriever([_chunk(f"b{i}") for i in range(10)])
    vector = StubRetriever([_chunk(f"v{i}") for i in range(10)])
    r = HybridRRFRetriever(bm25, vector, rrf_k=60)
    results = r.search("q", "doc1", top_k=3)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# Test 6: Kết quả giảm dần theo score
# ---------------------------------------------------------------------------

def test_results_sorted_descending():
    bm25   = StubRetriever([_chunk(f"b{i}") for i in range(5)])
    vector = StubRetriever([_chunk(f"v{i}") for i in range(5)])
    r = HybridRRFRetriever(bm25, vector, rrf_k=60)
    results = r.search("q", "doc1", top_k=8)
    scores = [res["score"] for res in results]
    assert scores == sorted(scores, reverse=True), f"Scores không giảm dần: {scores}"


# ---------------------------------------------------------------------------
# Test 7: rrf_k ảnh hưởng đúng — k nhỏ hơn → score cao hơn
# ---------------------------------------------------------------------------

def test_rrf_k_effect():
    chunks = [_chunk("a")]
    bm25   = StubRetriever(chunks)
    vector = StubRetriever(chunks)

    r60 = HybridRRFRetriever(bm25, vector, rrf_k=60)
    r10 = HybridRRFRetriever(bm25, vector, rrf_k=10)

    score60 = r60.search("q", "doc1", top_k=1)[0]["score"]
    score10 = r10.search("q", "doc1", top_k=1)[0]["score"]

    # k=10: 2/(10+1) ≈ 0.182 > k=60: 2/(60+1) ≈ 0.033
    assert score10 > score60, f"k=10 phải cho score cao hơn k=60. Got {score10} vs {score60}"


# ---------------------------------------------------------------------------
# Test 8: Một retriever rỗng → vẫn trả kết quả từ retriever còn lại
# ---------------------------------------------------------------------------

def test_one_empty_retriever():
    bm25   = StubRetriever([])
    vector = StubRetriever([_chunk("v0"), _chunk("v1")])
    r = HybridRRFRetriever(bm25, vector, rrf_k=60)
    results = r.search("q", "doc1", top_k=5)
    assert len(results) == 2
    ids = [x["chunk_id"] for x in results]
    assert "v0" in ids and "v1" in ids
