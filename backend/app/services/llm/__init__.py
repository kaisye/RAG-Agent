from app.core.config import get_settings
from app.services.llm.base import LLMProvider
from app.services.llm.nvidia import NvidiaProvider
from app.services.llm.ollama import OllamaProvider


def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "nvidia":
        return NvidiaProvider()
    return OllamaProvider()
