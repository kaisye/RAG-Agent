"""
Test CrossEncoderReranker với NVIDIA /v1/ranking API.
Yêu cầu: .env có NVIDIA_API_KEY và NVIDIA_RERANK_MODEL.
Không dùng torch, GPU, hay model local.

Chạy: cd backend && python scripts/test_reranker.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.reranking import get_reranker

QUERY = "Hybrid RRF retrieval cải thiện Faithfulness như thế nào?"

CANDIDATES = [
    {
        "chunk_id": "doc_rc_0000",
        "document_id": "doc1",
        "page": 1,
        "text": (
            "Hybrid RRF (Reciprocal Rank Fusion) kết hợp BM25 sparse retrieval "
            "và dense vector retrieval bằng công thức Score(d) = Σ 1/(k + rank_i(d)) "
            "với k=60. Kết quả: Faithfulness tăng từ 0.73 lên 0.86."
        ),
        "score": 0.91,
    },
    {
        "chunk_id": "doc_rc_0001",
        "document_id": "doc1",
        "page": 2,
        "text": (
            "BM25 là phương pháp sparse retrieval dựa trên tần suất từ. "
            "Khi dùng đơn lẻ, BM25 giảm Faithfulness xuống 0.44 (-0.29 so với baseline). "
            "Không nên dùng BM25 đơn lẻ trong production."
        ),
        "score": 0.73,
    },
    {
        "chunk_id": "doc_rc_0002",
        "document_id": "doc1",
        "page": 3,
        "text": (
            "SemanticChunker dùng cosine similarity để tách văn bản thành chunk. "
            "Ngưỡng threshold=0.5, min_size=600, max_size=1024 ký tự. "
            "Cải thiện Faithfulness +0.08 so với RecursiveChunker."
        ),
        "score": 0.55,
    },
    {
        "chunk_id": "doc_rc_0003",
        "document_id": "doc1",
        "page": 4,
        "text": (
            "HNSW (Hierarchical Navigable Small World) là cấu trúc dữ liệu đồ thị "
            "dùng trong ChromaDB để tìm kiếm vector gần nhất. "
            "Tham số M=16, ef_construct=100, ef_search=128."
        ),
        "score": 0.42,
    },
    {
        "chunk_id": "doc_rc_0004",
        "document_id": "doc1",
        "page": 5,
        "text": (
            "Query Decomposition tách câu hỏi phức tạp thành ≤3 câu con đơn giản. "
            "Mỗi câu con được retrieve riêng rồi gộp và dedup theo chunk_id. "
            "Kết quả: Faithfulness +0.20 — cải thiện cao nhất trong tất cả strategies."
        ),
        "score": 0.38,
    },
]


def main():
    print(f"Query: {QUERY}\n")
    print(f"Input: {len(CANDIDATES)} candidates")
    for i, c in enumerate(CANDIDATES):
        print(f"  [{i}] score={c['score']:.2f} | {c['text'][:60]}...")

    print("\n--- Reranking với NVIDIA /v1/ranking ---")
    reranker = get_reranker("cross_encoder", top_k=3)
    results = reranker.rerank(QUERY, CANDIDATES, top_k=3)

    print(f"\nTop {len(results)} sau rerank:")
    for rank, r in enumerate(results, 1):
        print(
            f"  [{rank}] rerank_score={r['rerank_score']:.4f} "
            f"(original_score={r['score']:.2f}) | {r['text'][:70]}..."
        )

    print("\n--- Assertions ---")
    assert len(results) <= 3, f"Expected <= 3, got {len(results)}"
    assert all("rerank_score" in r for r in results), "Missing rerank_score field"
    assert all("chunk_id" in r for r in results), "Missing chunk_id field"
    assert results == sorted(results, key=lambda r: r["rerank_score"], reverse=True), \
        "Results not sorted by rerank_score desc"

    # Chunk liên quan nhất (chunk 0 về Hybrid RRF) phải ở top
    assert results[0]["chunk_id"] == "doc_rc_0000", (
        f"Expected doc_rc_0000 at top-1, got {results[0]['chunk_id']}"
    )

    print("OK — CrossEncoderReranker hoạt động đúng (no torch, no GPU)")


if __name__ == "__main__":
    main()
