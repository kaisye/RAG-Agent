from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseTransformer(Protocol):
    def transform(self, question: str) -> tuple[str, str]: ...
