import logging

from openai import OpenAI

from .retry import call_with_retry

logger = logging.getLogger(__name__)


class NvidiaProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
        self.model = model
        logger.info("NvidiaProvider ready — model: %s", self.model)

    def chat(self, messages: list, stream: bool = False, tools: list | None = None):
        kwargs = {"model": self.model, "messages": messages, "stream": stream}
        if tools:
            kwargs["tools"] = tools
        return call_with_retry(lambda: self.client.chat.completions.create(**kwargs))


class OllamaProvider:
    def __init__(self, base_url: str, model: str) -> None:
        # Ollama's OpenAI-compatible endpoint doesn't need a real key
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.model = model
        logger.info("OllamaProvider ready — base_url: %s, model: %s", base_url, self.model)

    def chat(self, messages: list, stream: bool = False, tools: list | None = None):
        kwargs = {"model": self.model, "messages": messages, "stream": stream}
        if tools:
            kwargs["tools"] = tools
        return self.client.chat.completions.create(**kwargs)


def get_llm_provider() -> NvidiaProvider | OllamaProvider:
    from app.core.config import settings  # late import avoids circular deps at module load

    provider = settings.llm_provider.lower()
    if provider == "nvidia":
        if not settings.nvidia_api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Add it to your .env file or set the environment variable."
            )
        return NvidiaProvider(api_key=settings.nvidia_api_key, model=settings.nvidia_chat_model)
    if provider == "ollama":
        return OllamaProvider(base_url=settings.ollama_base_url, model=settings.ollama_chat_model)
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Valid values: 'nvidia', 'ollama'.")
