from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseRetriever(Protocol):
    def search(self, query: str, document_id: str, top_k: int = 10) -> list[dict]: ...
