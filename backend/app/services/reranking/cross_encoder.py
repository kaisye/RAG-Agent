import logging
import time

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# CrossEncoderReranker gọi NVIDIA NeMo Retriever Reranking API qua HTTP.
# KHÔNG dùng torch, transformers, hay bất kỳ model local nào.
# Model chạy trên NVIDIA H100 phía cloud — máy không cần GPU.
#
# Kết quả thực nghiệm: Faithfulness +0.03, AR +0.08, CP +0.01, CR +0.04.
# Pipeline phễu: Retriever → top 10 candidates → CrossEncoder → top 3 → LLM.


class CrossEncoderReranker:
    """
    Reranker dùng NVIDIA /v1/ranking endpoint.

    Khác với /v1/chat/completions — phải dùng requests, không phải OpenAI SDK.
    Payload format: query + passages, mỗi item có {"role":"user","content":"..."}.
    Response: {"rankings": [{"index": N, "logit": float}]}.
    """

    # Endpoint đã đổi từ /v1/ranking sang /v1/retrieval/nvidia/reranking (2024+).
    # Payload format cũng thay đổi: query/passages dùng {"text": "..."} thay vì {"role":"user","content":"..."}.
    RANKING_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
    MAX_PASSAGES = 50  # giới hạn NVIDIA per request

    def __init__(self, top_k: int = 3):
        settings = get_settings()
        self.top_k = top_k
        self.model = settings.nvidia_rerank_model
        self.headers = {
            "Authorization": f"Bearer {settings.nvidia_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        k = top_k if top_k is not None else self.top_k
        if not candidates:
            return []

        if len(candidates) <= self.MAX_PASSAGES:
            return self._call_api(query, candidates, k)

        # Chunking khi vượt giới hạn 50 passages — chia batch, gộp top candidates, rerank lần 2
        logger.info(
            "CrossEncoderReranker: %d candidates > %d, chunking into batches",
            len(candidates), self.MAX_PASSAGES,
        )
        intermediate: list[dict] = []
        for i in range(0, len(candidates), self.MAX_PASSAGES):
            batch = candidates[i:i + self.MAX_PASSAGES]
            batch_top_k = min(k * 2, len(batch))
            intermediate.extend(self._call_api(query, batch, batch_top_k))

        if len(intermediate) <= self.MAX_PASSAGES:
            return self._call_api(query, intermediate, k)

        # Fallback: sắp xếp theo rerank_score từ vòng 1
        return sorted(intermediate, key=lambda c: c.get("rerank_score", 0.0), reverse=True)[:k]

    def _call_api(self, query: str, candidates: list[dict], top_k: int, retries: int = 4) -> list[dict]:
        payload = {
            "model":    self.model,
            "query":    {"text": query},
            "passages": [{"text": c["text"]} for c in candidates],
            "truncate": "END",
        }
        resp = None
        for attempt in range(retries):
            resp = requests.post(
                self.RANKING_URL,
                headers=self.headers,
                json=payload,
                timeout=30,
            )
            if resp.status_code == 429:
                delay = 2 * (2 ** attempt)
                logger.warning("CrossEncoderReranker: 429 rate limit, retry in %ds", delay)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            break
        else:
            # Tất cả retries đều 429
            resp.raise_for_status()

        rankings = sorted(resp.json()["rankings"], key=lambda r: r["logit"], reverse=True)
        result = []
        for r in rankings[:top_k]:
            chunk = dict(candidates[r["index"]])
            chunk["rerank_score"] = r["logit"]
            result.append(chunk)

        logger.debug(
            "CrossEncoderReranker: query=%r %d->%d candidates",
            query[:60], len(candidates), len(result),
        )
        return result
