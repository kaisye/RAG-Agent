import logging

logger = logging.getLogger(__name__)

# HyDE — Hypothetical Document Embeddings (Mục V.1 tài liệu AIO2026)
#
# Vấn đề: vector của câu hỏi ngắn cách xa tài liệu dài trong không gian embedding.
# Giải pháp: LLM sinh "câu trả lời giả" văn phong chuyên ngành (~100-150 từ)
#            → vector của nó gần với tài liệu thật hơn → search chính xác hơn.
#
# QUAN TRỌNG: embed hypothetical_doc để search,
#             truyền original_question cho reranker và LLM generation.
#             Không bao giờ sinh câu trả lời từ hypothetical_doc.
#
# Kết quả thực nghiệm: Faithfulness +0.08, AR -0.03.

_HYDE_PROMPT = (
    "Hãy viết một đoạn văn khoảng 100-150 từ trả lời câu hỏi sau. "
    "Viết như thể đây là đoạn trích từ tài liệu chuyên ngành. "
    "Đoạn văn nên chứa thuật ngữ kỹ thuật liên quan. "
    "Không bắt đầu bằng 'Theo tài liệu' hay cụm tương tự.\n\n"
    "Câu hỏi: {question}\n\nĐoạn văn:"
)


class HyDETransformer:
    """
    Sinh hypothetical document để cải thiện vector search.

    transform(question) → (hypothetical_doc, original_question)
      - hypothetical_doc: embed để search ChromaDB
      - original_question: truyền cho reranker và LLM generation
    """

    def __init__(self, llm_provider):
        self.llm = llm_provider

    def transform(self, question: str) -> tuple[str, str]:
        prompt = _HYDE_PROMPT.format(question=question)
        response = self.llm.chat([{"role": "user", "content": prompt}])
        hypothetical = response.choices[0].message.content.strip()
        logger.debug(
            "HyDE: question=%r -> hypothetical_doc=%d chars",
            question[:60], len(hypothetical),
        )
        return hypothetical, question
