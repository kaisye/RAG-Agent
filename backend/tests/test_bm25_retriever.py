"""
Tests for BM25Retriever — 7 test cases.
Không cần NVIDIA API key vì BM25 là sparse retrieval thuần túy.
"""
import logging
import pytest

from app.services.retrieval.bm25 import BM25Retriever
from app.services.retrieval.base import BaseRetriever


CHUNKS = [
    {
        "chunk_id": "doc1_rc_0000",
        "document_id": "doc1",
        "page": 1,
        "text": "Học máy là một lĩnh vực của trí tuệ nhân tạo.",
        "strategy": "recursive",
        "type": "text",
    },
    {
        "chunk_id": "doc1_rc_0001",
        "document_id": "doc1",
        "page": 2,
        "text": "Mạng nơ-ron nhân tạo được lấy cảm hứng từ não người.",
        "strategy": "recursive",
        "type": "text",
    },
    {
        "chunk_id": "doc1_rc_0002",
        "document_id": "doc1",
        "page": 3,
        "text": "Deep learning sử dụng nhiều lớp nơ-ron để học đặc trưng.",
        "strategy": "recursive",
        "type": "text",
    },
    {
        "chunk_id": "doc1_rc_0003",
        "document_id": "doc1",
        "page": 4,
        "text": "Xử lý ngôn ngữ tự nhiên giúp máy tính hiểu tiếng Việt.",
        "strategy": "recursive",
        "type": "text",
    },
]


# ---------------------------------------------------------------------------
# Test 1: Protocol conformance
# ---------------------------------------------------------------------------

def test_protocol_conformance():
    retriever = BM25Retriever(CHUNKS)
    assert isinstance(retriever, BaseRetriever), (
        "BM25Retriever phải conform BaseRetriever Protocol"
    )


# ---------------------------------------------------------------------------
# Test 2: Kết quả trả về đúng cấu trúc
# ---------------------------------------------------------------------------

def test_result_structure():
    retriever = BM25Retriever(CHUNKS)
    results = retriever.search("học máy", "doc1", top_k=3)
    assert isinstance(results, list)
    for r in results:
        assert "chunk_id" in r
        assert "document_id" in r
        assert "page" in r
        assert "text" in r
        assert "score" in r
        assert isinstance(r["score"], float)


# ---------------------------------------------------------------------------
# Test 3: Top-1 là chunk liên quan nhất
# ---------------------------------------------------------------------------

def test_top1_relevance():
    retriever = BM25Retriever(CHUNKS)
    results = retriever.search("học máy trí tuệ nhân tạo", "doc1", top_k=4)
    assert len(results) > 0
    # Chunk 0 chứa "Học máy" và "trí tuệ nhân tạo" — phải là top-1
    assert results[0]["chunk_id"] == "doc1_rc_0000", (
        f"Expected doc1_rc_0000 at top-1, got {results[0]['chunk_id']}"
    )


# ---------------------------------------------------------------------------
# Test 4: top_k được tôn trọng
# ---------------------------------------------------------------------------

def test_top_k_respected():
    retriever = BM25Retriever(CHUNKS)
    results = retriever.search("học nơ-ron deep learning", "doc1", top_k=2)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# Test 5: Score > 0 cho tất cả kết quả trả về
# ---------------------------------------------------------------------------

def test_scores_positive():
    retriever = BM25Retriever(CHUNKS)
    results = retriever.search("ngôn ngữ tiếng Việt", "doc1", top_k=4)
    for r in results:
        assert r["score"] > 0, f"Score phải > 0, got {r['score']}"


# ---------------------------------------------------------------------------
# Test 6: Query không khớp → trả về list rỗng (score=0 bị lọc)
# ---------------------------------------------------------------------------

def test_no_match_returns_empty():
    retriever = BM25Retriever(CHUNKS)
    results = retriever.search("zzz_không_tồn_tại_xyz_999", "doc1", top_k=5)
    assert results == [], f"Expected [], got {results}"


# ---------------------------------------------------------------------------
# Test 7: Cảnh báo log khi khởi tạo BM25Retriever
# ---------------------------------------------------------------------------

def test_warning_logged_on_init(caplog):
    with caplog.at_level(logging.WARNING, logger="app.services.retrieval.bm25"):
        BM25Retriever(CHUNKS)
    assert any("standalone" in record.message or "hybrid" in record.message
               for record in caplog.records), (
        "BM25Retriever phải log WARNING về việc không dùng đơn lẻ"
    )
