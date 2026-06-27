class IdentityTransformer:
    """
    Pass-through transformer cho strategy="none".
    Trả (question, question) — search và generate đều dùng câu hỏi gốc.
    """

    def transform(self, question: str) -> tuple[str, str]:
        return question, question
