import logging
import re

logger = logging.getLogger(__name__)

# Query Decomposition (Mục V.2 tài liệu AIO2026)
#
# Vấn đề: câu hỏi so sánh / đa chủ đề → vector nằm lửng lơ giữa các chủ đề → search không trúng.
# Giải pháp: LLM tách thành ≤3 câu con đơn giản → retrieve riêng từng câu → gộp và dedup.
#
# Kết quả thực nghiệm: Faithfulness +0.20 (CAO NHẤT!), AR +0.12.
# CP giảm -0.13 vì retrieve nhiều ứng viên hơn — đánh đổi chấp nhận được.
#
# Lưu ý tích hợp (RAGPipeline nhánh 8.1):
#   candidates  = retriever.search(original_query, top_k=10)   # câu gốc PHẢI được search
#   for sq in transformer.sub_questions:
#       candidates += retriever.search(sq, top_k=3)
#   candidates = deduplicate_by_chunk_id(candidates)
# KHÔNG bỏ qua câu gốc — sub-questions bổ sung, không thay thế.

_DECOMPOSE_PROMPT = (
    "Hãy phân tách câu hỏi sau thành các câu hỏi con đơn giản hơn.\n"
    "Mỗi câu hỏi con tập trung vào một khía cạnh cụ thể.\n"
    "Trả về tối đa 3 câu, mỗi câu trên một dòng.\n"
    "Nếu câu hỏi đã đơn giản, chỉ trả lại chính câu hỏi đó.\n\n"
    "Câu hỏi gốc: {question}\n\nCâu hỏi con:"
)

_LEADING_NUMBER = re.compile(r"^\d+[.\)]\s*")


class DecompositionTransformer:
    """
    Tách câu hỏi phức tạp thành ≤3 câu con để retrieve riêng từng phần.

    Sau khi gọi transform(), đọc self.sub_questions để lấy danh sách câu con
    dùng trong bước retrieve của RAGPipeline.
    """

    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.sub_questions: list[str] = []

    def decompose(self, question: str) -> list[str]:
        prompt = _DECOMPOSE_PROMPT.format(question=question)
        response = self.llm.chat([{"role": "user", "content": prompt}])
        raw = response.choices[0].message.content.strip()
        lines = [
            _LEADING_NUMBER.sub("", line.strip())
            for line in raw.split("\n")
            if line.strip()
        ]
        result = [l for l in lines if l][:3]
        logger.debug(
            "Decomposition: question=%r -> %d sub-questions: %s",
            question[:60], len(result), result,
        )
        return result

    def transform(self, question: str) -> tuple[str, str]:
        self.sub_questions = self.decompose(question)
        # Trả câu gốc để RAGPipeline search câu gốc trước, sau đó search từng sub_question
        return question, question
