from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseChunker(Protocol):
    def split(self, text_blocks: list[dict], document_id: str) -> list[dict]: ...
