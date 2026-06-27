import logging

import numpy as np

from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

# MMR — Maximal Marginal Relevance (Mục VI.2 tài liệu AIO2026)
#
# Vấn đề: CrossEncoder chọn top-3 giống nhau → lãng phí context window với thông tin lặp.
# Giải pháp: MMR vừa chọn chunk liên quan vừa tối đa hóa sự khác biệt với các chunk đã chọn.
#
# Công thức: Score = λ·Sim(query, doc) - (1-λ)·max(Sim(doc, selected))
#   λ=0.5: cân bằng relevance/diversity (default)
#   λ→1.0: gần giống CrossEncoder (pure relevance)
#   λ→0.0: pure diversity (không dùng trong thực tế)
#
# Kết quả thực nghiệm: Faithfulness +0.04, AR -0.04, CR -0.07.
# Đánh đổi: đa dạng hơn nhưng có thể bỏ sót thông tin tập trung.
# Dùng cho câu hỏi tổng quan nhiều khía cạnh, không phải câu hỏi kỹ thuật cụ thể.


class MMRReranker:
    """
    Reranker dùng Maximal Marginal Relevance.

    Mỗi vòng chọn một chunk tối đa hóa:
      λ·Sim(query, doc) - (1-λ)·max(Sim(doc, đã_chọn))
    cho đến khi đủ top_k candidates.
    """

    def __init__(self, top_k: int = 3, lambda_mult: float = 0.5):
        self.top_k = top_k
        self.lambda_mult = lambda_mult
        self._embed_svc = None  # lazy init để tránh khởi tạo khi factory chưa cần

    @property
    def embed_svc(self):
        if self._embed_svc is None:
            self._embed_svc = EmbeddingService()
        return self._embed_svc

    def _cosine(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / (denom + 1e-10))

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        k = top_k if top_k is not None else self.top_k
        if not candidates:
            return []
        if len(candidates) <= k:
            return candidates

        # Embed query + tất cả candidates trong 1 batch
        texts = [query] + [c["text"] for c in candidates]
        embs = self.embed_svc.embed_texts(texts, input_type="query")
        q_emb = embs[0]
        d_embs = embs[1:]

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        while len(selected) < k and remaining:
            best_score = -float("inf")
            best_idx = remaining[0]

            for idx in remaining:
                relevance = self._cosine(q_emb, d_embs[idx])
                redundancy = (
                    max(self._cosine(d_embs[idx], d_embs[s]) for s in selected)
                    if selected else 0.0
                )
                score = self.lambda_mult * relevance - (1 - self.lambda_mult) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        result = [candidates[i] for i in selected]
        logger.debug(
            "MMRReranker: query=%r lambda=%.2f %d->%d candidates",
            query[:60], self.lambda_mult, len(candidates), len(result),
        )
        return result
