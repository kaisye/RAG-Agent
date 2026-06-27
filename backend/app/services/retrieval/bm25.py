import logging

import numpy as np
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

logger = logging.getLogger(__name__)

# BM25 đơn lẻ giảm Faithfulness -0.29 so với DenseRetriever.
# Chỉ dùng BM25Retriever trong HybridInterleaving hoặc HybridRRF, không bao giờ đứng độc lập.


class BM25Retriever:
    """
    Sparse retrieval dùng BM25Okapi + underthesea word_tokenize.

    CẢNH BÁO: BM25 đơn lẻ: Faithfulness=0.44 (-0.29 so với baseline 0.73).
    Chỉ dùng kết hợp trong hybrid_interleaving hoặc hybrid_rrf.

    Lý do bắt buộc word_tokenize thay vì .split():
    "học máy" → word_tokenize → ["học máy"]  (1 token, đúng)
    "học máy" → .split()     → ["học", "máy"] (2 token, sai với từ ghép tiếng Việt)
    """

    def __init__(self, chunks: list[dict]):
        logger.warning(
            "BM25Retriever khởi tạo độc lập — "
            "BM25 standalone giảm Faithfulness -0.29. "
            "Chỉ dùng trong hybrid (hybrid_interleaving hoặc hybrid_rrf)."
        )
        self.chunks = chunks
        tokenized = [word_tokenize(c["text"].lower()) for c in chunks]
        self.index = BM25Okapi(tokenized)

    def search(self, query: str, document_id: str, top_k: int = 10) -> list[dict]:
        tokens = word_tokenize(query.lower())
        scores = self.index.get_scores(tokens)
        top_ids = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_ids:
            if scores[i] > 0:
                results.append({
                    **self.chunks[i],
                    "score": float(scores[i]),
                })
        logger.debug(
            "BM25Retriever: query=%r tokens=%r -> %d results",
            query[:60], tokens[:5], len(results),
        )
        return results
