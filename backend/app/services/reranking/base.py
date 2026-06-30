from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseReranker(Protocol):
    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]: ...
