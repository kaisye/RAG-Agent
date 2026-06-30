from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    def chat(self, messages: list, stream: bool = False, tools: list | None = None): ...
