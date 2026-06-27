from app.services.query_transform.base import BaseTransformer
from app.services.query_transform.identity import IdentityTransformer


def get_transformer(mode: str, llm_provider=None) -> BaseTransformer:
    """
    Factory cho query transform strategies.

    mode:
      "none"          — IdentityTransformer (pass-through)
      "hyde"          — HyDETransformer (embed hypothetical doc, generate từ câu gốc)
      "decomposition" — DecompositionTransformer (feature/query-decomposition)
    """
    if mode == "none":
        return IdentityTransformer()

    if mode == "hyde":
        if llm_provider is None:
            raise ValueError("HyDETransformer cần llm_provider")
        from app.services.query_transform.hyde import HyDETransformer
        return HyDETransformer(llm_provider)

    if mode == "decomposition":
        if llm_provider is None:
            raise ValueError("DecompositionTransformer cần llm_provider")
        from app.services.query_transform.decomposition import DecompositionTransformer  # feature/query-decomposition
        return DecompositionTransformer(llm_provider)

    raise ValueError(f"Unknown query_transform mode: {mode!r}")
