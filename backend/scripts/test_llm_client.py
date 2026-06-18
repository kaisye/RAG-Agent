"""
Smoke-test script for the LLM provider layer.

Run from ANY directory:
    python backend/scripts/test_llm_client.py
    python backend/scripts/test_llm_client.py --stream
    python backend/scripts/test_llm_client.py --model meta/llama-3.1-70b-instruct
    python backend/scripts/test_llm_client.py "your question here"
"""

import argparse
import os
import sys
from pathlib import Path

# Resolve backend/ regardless of cwd, then cd into it so pydantic-settings
# finds .env in the expected location before Settings() is instantiated.
BACKEND_DIR = Path(__file__).resolve().parent.parent
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402 — must come after chdir
from app.services.llm import get_llm_provider  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the configured LLM provider.")
    parser.add_argument(
        "question",
        nargs="?",
        default="What does RAG stand for in the context of AI? Answer in one sentence.",
    )
    parser.add_argument("--stream", action="store_true", help="Use streaming output")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override the chat model (NVIDIA only). "
            "Copy the exact ID from https://build.nvidia.com/models "
            "e.g. --model meta/llama-3.1-70b-instruct"
        ),
    )
    args = parser.parse_args()

    is_nvidia = settings.llm_provider.lower() == "nvidia"
    effective_model = args.model or (settings.nvidia_chat_model if is_nvidia else settings.ollama_chat_model)

    print(f"Provider : {settings.llm_provider.upper()}")
    if is_nvidia:
        print(f"Model    : {effective_model}")
        key_preview = settings.nvidia_api_key[:8] + "..." if settings.nvidia_api_key else "(not set)"
        print(f"API key  : {key_preview}")
    else:
        print(f"Model    : {effective_model}  (base: {settings.ollama_base_url})")
    print(f"Stream   : {args.stream}")
    print(f"Question : {args.question}")
    print("-" * 60)

    provider = get_llm_provider()

    # Allow --model to override without touching config
    if args.model:
        provider.model = args.model

    messages = [{"role": "user", "content": args.question}]

    try:
        if args.stream:
            print("Response (streaming):")
            response = provider.chat(messages, stream=True)
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
            print()
        else:
            print("Response:")
            response = provider.chat(messages, stream=False)
            print(response.choices[0].message.content)
            if getattr(response, "usage", None):
                u = response.usage
                print(
                    f"\nTokens — prompt: {u.prompt_tokens}, "
                    f"completion: {u.completion_tokens}, "
                    f"total: {u.total_tokens}"
                )
    except ValueError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 60)
    print("OK")


if __name__ == "__main__":
    main()
