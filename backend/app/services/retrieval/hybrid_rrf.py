import logging

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion — production default cho RAG pipeline này.
# Kết quả thực nghiệm (tốt nhất): Faithfulness=0.86, AR=0.57, CP=0.80, CR=0.80.
#
# Công thức: Score(d) = Σ  1 / (k + rank_i(d))
#   rank_i(d) = vị trí của document d trong danh sách kết quả của retriever i (0-indexed → +1)
#   k = 60 (hằng số làm mượt):
#     - k nhỏ (ví dụ 1): hạng 1 chiếm ưu thế tuyệt đối → 1/2 vs 1/3
#     - k lớn (ví dụ 1000): san phẳng quá mức → mọi hạng gần như bằng nhau
#     - k=60: đủ nhỏ để phân biệt hạng, đủ lớn để không để hạng 1 áp đảo.
#       Ví dụ: hạng 1 → 1/61 ≈ 0.0164; hạng 10 → 1/70 ≈ 0.0143 (chênh lệch có ý nghĩa)
#       Doc nằm ở hạng 2 cả BM25 lẫn Vector → 1/62+1/62=0.0323 > Top1 của một retriever 1/61≈0.0164.
#       Đây là lý do RRF ưu tiên "đồng thuận" hơn "hạng 1 tuyệt đối".


class HybridRRFRetriever:
    """
    Hybrid retrieval dùng Reciprocal Rank Fusion (k=60).

    Kết hợp BM25 (sparse) và DenseRetriever (dense) theo công thức RRF.
    Chunk được ưu tiên nếu xuất hiện cao trong CẢ HAI danh sách.
    """

    def __init__(self, bm25_retriever, dense_retriever, rrf_k: int = 60):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, document_id: str, top_k: int = 10) -> list[dict]:
        # Lấy nhiều ứng viên hơn top_k để RRF có đủ nguyên liệu tái xếp hạng
        candidate_k = max(top_k * 2, 20)
        bm25_results = self.bm25.search(query, document_id, candidate_k)
        vector_results = self.dense.search(query, document_id, candidate_k)

        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, chunk in enumerate(vector_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[cid] = chunk

        for rank, chunk in enumerate(bm25_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[cid] = chunk

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for cid in sorted_ids[:top_k]:
            chunk = dict(doc_map[cid])
            chunk["score"] = round(rrf_scores[cid], 6)
            results.append(chunk)

        logger.debug(
            "HybridRRFRetriever: query=%r k=%d -> %d results (bm25=%d, vector=%d)",
            query[:60], self.rrf_k, len(results), len(bm25_results), len(vector_results),
        )
        return results
