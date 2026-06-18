from typing import Protocol


class LLMProvider(Protocol):
    def chat(
        self,
        messages: list,
        stream: bool = False,
        tools: list | None = None,
    ): ...
