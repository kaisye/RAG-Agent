"""
Chạy từ thư mục backend/:
    python scripts/test_llm_client.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.services.llm import get_llm_provider
from app.services.embedding import EmbeddingService


def test_chat():
    print("=== LLM Chat ===")
    provider = get_llm_provider()
    response = provider.chat(
        messages=[{"role": "user", "content": "Trả lời ngắn gọn: RAG là gì?"}],
        stream=False,
    )
    print("Response:", response.choices[0].message.content)
    print()


def test_embedding():
    print("=== Embedding ===")
    service = EmbeddingService()

    passages = ["Học máy là một nhánh của trí tuệ nhân tạo.", "RAG kết hợp retrieval và generation."]
    vecs = service.embed_texts(passages, input_type="passage")
    print(f"Passages: {len(vecs)} embeddings, dim={len(vecs[0])}")

    query_vecs = service.embed_texts(["RAG là gì?"], input_type="query")
    print(f"Query: {len(query_vecs)} embedding, dim={len(query_vecs[0])}")

    # Cosine similarity giữa query và 2 passages
    import math
    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x ** 2 for x in a))
        nb = math.sqrt(sum(x ** 2 for x in b))
        return dot / (na * nb)

    q = query_vecs[0]
    for i, (text, vec) in enumerate(zip(passages, vecs)):
        print(f"  sim(query, passage[{i}]) = {cosine(q, vec):.4f}  | {text[:40]}")


if __name__ == "__main__":
    test_chat()
    test_embedding()
