"""
Smoke-test script for the LLM provider layer.

Run from backend/ directory:
    python scripts/test_llm_client.py
    python scripts/test_llm_client.py --stream
    python scripts/test_llm_client.py --model meta/llama-3.1-70b-instruct
    python scripts/test_llm_client.py "your question here"
"""

import argparse
import math
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.services.llm import get_llm_provider  # noqa: E402
from app.services.embedding import EmbeddingService  # noqa: E402


def test_chat(args) -> None:
    is_nvidia = settings.llm_provider.lower() == "nvidia"
    effective_model = args.model or (settings.nvidia_chat_model if is_nvidia else settings.ollama_chat_model)

    print(f"Provider : {settings.llm_provider.upper()}")
    print(f"Model    : {effective_model}")
    if is_nvidia:
        key_preview = settings.nvidia_api_key[:8] + "..." if settings.nvidia_api_key else "(not set)"
        print(f"API key  : {key_preview}")
    print(f"Stream   : {args.stream}")
    print(f"Question : {args.question}")
    print("-" * 60)

    provider = get_llm_provider()
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
                print(f"\nTokens — prompt: {u.prompt_tokens}, completion: {u.completion_tokens}, total: {u.total_tokens}")
    except ValueError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


def test_embedding() -> None:
    print("\n=== Embedding ===")
    service = EmbeddingService()

    passages = ["Học máy là một nhánh của trí tuệ nhân tạo.", "RAG kết hợp retrieval và generation."]
    vecs = service.embed_texts(passages, input_type="passage")
    print(f"Passages: {len(vecs)} embeddings, dim={len(vecs[0])}")

    query_vecs = service.embed_texts(["RAG là gì?"], input_type="query")
    print(f"Query: {len(query_vecs)} embedding, dim={len(query_vecs[0])}")

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b)))

    q = query_vecs[0]
    for i, (text, vec) in enumerate(zip(passages, vecs)):
        print(f"  sim(query, passage[{i}]) = {cosine(q, vec):.4f}  | {text[:40]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the configured LLM provider.")
    parser.add_argument("question", nargs="?", default="What does RAG stand for in the context of AI? Answer in one sentence.")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--embedding", action="store_true", help="Also test embedding service")
    args = parser.parse_args()

    test_chat(args)
    if args.embedding:
        test_embedding()

    print("-" * 60)
    print("OK")


if __name__ == "__main__":
    main()
