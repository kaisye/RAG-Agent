from openai import OpenAI
from app.core.config import get_settings


class OllamaProvider:
    """Fallback offline — chỉ dùng khi không có mạng hoặc NVIDIA API không khả dụng."""

    def __init__(self):
        settings = get_settings()
        self.model = settings.ollama_chat_model
        self.client = OpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",
        )

    def chat(self, messages: list, stream: bool = False, tools: list | None = None):
        kwargs = dict(
            model=self.model,
            messages=messages,
            stream=stream,
        )
        if tools:
            kwargs["tools"] = tools
        return self.client.chat.completions.create(**kwargs)
