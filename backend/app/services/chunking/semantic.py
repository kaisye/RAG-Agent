import logging

import numpy as np
from underthesea import sent_tokenize  # KHÔNG dùng NLTK — sai boundary với tiếng Việt

from app.services.chunking.base import BaseChunker

logger = logging.getLogger(__name__)

# Độ dài tối thiểu để một câu được đưa vào chunking
_MIN_SENTENCE_LEN = 10


class SemanticChunker:
    """
    Thuật toán từ tài liệu AIO2026 (Mục III.1).

    Thứ tự ưu tiên override (từ cao xuống thấp):
      1. len >= max_size  → buộc ngắt ngay dù similarity cao
      2. len < min_size   → tiếp tục gộp dù similarity thấp
      3. sim < threshold  → ngắt theo ngữ nghĩa

    Kết quả thực nghiệm: Faithfulness ↑0.08, Answer Relevancy ↑0.08 so với RecursiveChunker.
    """

    def __init__(
        self,
        embed_service,
        threshold: float = 0.5,
        min_size: int = 600,
        max_size: int = 1024,
        overlap: int = 128,
    ):
        self.embed_service = embed_service
        self.threshold = threshold
        self.min_size = min_size
        self.max_size = max_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, text_blocks: list[dict], document_id: str) -> list[dict]:
        # Gộp blocks theo page để tách câu xuyên block trong cùng trang
        pages: dict[int, list[str]] = {}
        for block in text_blocks:
            pages.setdefault(block["page"], []).append(block["text"])

        chunks: list[dict] = []
        for page_num, texts in sorted(pages.items()):
            merged_text = " ".join(texts)
            sentences = self._split_sentences(merged_text)
            if not sentences:
                continue

            # Embed tất cả câu trong 1 batch — tiết kiệm API calls
            # input_type="passage": đây là lúc ingest, không phải query
            embeddings = self.embed_service.embed_texts(sentences, input_type="passage")

            raw_chunks = self._build_chunks(sentences, embeddings)

            for idx, chunk_text in enumerate(raw_chunks):
                # Thêm overlap từ cuối chunk trước (nếu có)
                if idx > 0 and self.overlap > 0 and chunks:
                    tail = chunks[-1]["text"][-self.overlap:]
                    chunk_text = tail + " " + chunk_text

                chunks.append({
                    "chunk_id": f"{document_id}_sc_{len(chunks):04d}",
                    "document_id": document_id,
                    "page": page_num,
                    "text": chunk_text.strip(),
                    "strategy": "semantic",
                })

        logger.info(
            "SemanticChunker: %d blocks → %d chunks (threshold=%.2f, min=%d, max=%d)",
            len(text_blocks), len(chunks), self.threshold, self.min_size, self.max_size,
        )
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        # underthesea hiểu ranh giới câu tiếng Việt; NLTK không làm được điều này
        return [s.strip() for s in sent_tokenize(text) if len(s.strip()) >= _MIN_SENTENCE_LEN]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / (denom + 1e-10))

    def _build_chunks(self, sentences: list[str], embeddings: list[list[float]]) -> list[str]:
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            current_text = " ".join(current)
            current_len = len(current_text)
            sim = self._cosine_sim(embeddings[i - 1], embeddings[i])

            if current_len >= self.max_size:
                # Override 1: buộc ngắt — chunk đã đủ lớn
                chunks.append(current_text)
                current = [sentences[i]]
            elif current_len < self.min_size:
                # Override 2: tiếp tục gộp — chunk còn quá nhỏ
                current.append(sentences[i])
            elif sim >= self.threshold:
                # Câu tiếp theo đồng ngữ nghĩa → gộp vào
                current.append(sentences[i])
            else:
                # Breakpoint ngữ nghĩa phát hiện → ngắt chunk
                chunks.append(current_text)
                current = [sentences[i]]

        if current:
            chunks.append(" ".join(current))

        return chunks
