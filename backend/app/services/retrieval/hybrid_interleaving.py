import logging

logger = logging.getLogger(__name__)

# Kết quả thực nghiệm: CP giảm -0.24 so với baseline vector.
# Faithfulness=0.79, AR=0.55, CP=0.57, CR=0.73.
# Cơ chế xen kẽ máy móc — BM25 kém làm loãng kết quả Vector.
# Chỉ để ablation study (Config C_interleaving), không dùng production.


class HybridInterleavingRetriever:
    """
    Lấy xen kẽ kết quả BM25 và DenseRetriever, dedup theo chunk_id.

    Thứ tự ưu tiên: BM25[0], Vector[0], BM25[1], Vector[1], ...
    Chunk đã thấy (theo chunk_id) bị bỏ qua.
    """

    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

    def search(self, query: str, document_id: str, top_k: int = 10) -> list[dict]:
        bm25_results = self.bm25.search(query, document_id, top_k)
        vector_results = self.dense.search(query, document_id, top_k)

        merged = []
        seen = set()
        for i in range(max(len(bm25_results), len(vector_results))):
            for source in (bm25_results, vector_results):
                if i < len(source):
                    cid = source[i]["chunk_id"]
                    if cid not in seen:
                        merged.append(source[i])
                        seen.add(cid)
            if len(merged) >= top_k:
                break

        result = merged[:top_k]
        logger.debug(
            "HybridInterleavingRetriever: query=%r -> %d results (bm25=%d, vector=%d)",
            query[:60], len(result), len(bm25_results), len(vector_results),
        )
        return result
