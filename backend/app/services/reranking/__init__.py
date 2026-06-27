from app.services.reranking.base import BaseReranker


class IdentityReranker:
    """Pass-through reranker cho strategy="none". Cắt top_k từ candidates."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        k = top_k if top_k is not None else self.top_k
        return candidates[:k]


def get_reranker(strategy: str, top_k: int = 3) -> BaseReranker:
    """
    Factory cho reranking strategies.

    strategy:
      "none"          — IdentityReranker (cắt top_k, không rerank)
      "cross_encoder" — CrossEncoderReranker (NVIDIA /v1/ranking, requests)
      "mmr"           — MMRReranker (feature/mmr-rerank)
    """
    if strategy == "none":
        return IdentityReranker(top_k=top_k)

    if strategy == "cross_encoder":
        from app.services.reranking.cross_encoder import CrossEncoderReranker
        return CrossEncoderReranker(top_k=top_k)

    if strategy == "mmr":
        from app.services.reranking.mmr import MMRReranker  # feature/mmr-rerank
        return MMRReranker(top_k=top_k)

    raise ValueError(f"Unknown rerank strategy: {strategy!r}")
