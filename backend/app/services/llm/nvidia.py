from openai import OpenAI
from app.core.config import get_settings


class NvidiaProvider:
    def __init__(self):
        settings = get_settings()
        self.model = settings.nvidia_chat_model
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=settings.nvidia_api_key,
        )

    def chat(self, messages: list, stream: bool = False, tools: list | None = None):
        kwargs = dict(
            model=self.model,
            messages=messages,
            stream=stream,
            max_tokens=2048,
        )
        if tools:
            kwargs["tools"] = tools
        return self.client.chat.completions.create(**kwargs)
