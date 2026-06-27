"""
Test RAGPipeline với 2 config khác nhau.
Yêu cầu: .env có NVIDIA_API_KEY, NVIDIA_CHAT_MODEL, NVIDIA_EMBED_MODEL.
Yêu cầu: đã upload ít nhất 1 PDF và lấy document_id từ GET /documents.

Chạy: cd backend && python scripts/test_pipeline.py <document_id>
      cd backend && python scripts/test_pipeline.py  (dùng DOCUMENT_ID trong .env)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import PipelineConfig
from app.services.pipeline import RAGPipeline

QUERY = "Hybrid RRF cải thiện kết quả retrieval như thế nào so với vector search đơn thuần?"

# Lấy document_id từ arg hoặc env
DOCUMENT_ID = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TEST_DOCUMENT_ID", "")
if not DOCUMENT_ID:
    print("ERROR: Truyền document_id qua arg hoặc set TEST_DOCUMENT_ID trong .env")
    print("  python scripts/test_pipeline.py <document_id>")
    sys.exit(1)


def run_config(label: str, config: PipelineConfig):
    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"  retrieval={config.retrieval_strategy}, transform={config.query_transform}, "
          f"rerank={config.rerank_strategy}, top_k_final={config.top_k_final}")
    print(f"{'='*60}")

    pipeline = RAGPipeline(config)

    # --- Retrieve ---
    print(f"\nQuery: {QUERY}")
    print("\n[RETRIEVE]")
    contexts = pipeline.retrieve(QUERY, DOCUMENT_ID)
    print(f"  {len(contexts)} chunks retrieved:")
    for i, c in enumerate(contexts, 1):
        score_field = "rerank_score" if "rerank_score" in c else "score"
        print(f"  [{i}] page={c['page']} {score_field}={c.get(score_field, 0):.4f} | {c['text'][:80]}...")

    assert len(contexts) > 0, "Retrieve phải trả ít nhất 1 chunk"
    assert all("chunk_id" in c and "text" in c and "page" in c for c in contexts), \
        "Mỗi chunk phải có chunk_id, text, page"

    # --- Generate ---
    print("\n[GENERATE]")
    full_response = ""
    for token in pipeline.generate(QUERY, contexts):
        print(token, end="", flush=True)
        full_response += token
    print()

    assert len(full_response) > 50, f"Response quá ngắn: {len(full_response)} chars"
    print(f"\n  => {len(full_response)} chars generated")
    print("  OK")


def main():
    print(f"Document ID: {DOCUMENT_ID}")

    # Config 1: Baseline — vector search, no transform, no rerank
    run_config(
        "Baseline (vector + none + none)",
        PipelineConfig(
            retrieval_strategy="vector",
            query_transform="none",
            rerank_strategy="none",
            top_k_retrieval=5,
            top_k_final=3,
        ),
    )

    # Config 2: Best — hybrid_rrf + HyDE + cross_encoder
    run_config(
        "Best config (hybrid_rrf + hyde + cross_encoder)",
        PipelineConfig(
            retrieval_strategy="hybrid_rrf",
            query_transform="hyde",
            rerank_strategy="cross_encoder",
            top_k_retrieval=10,
            top_k_final=3,
            rrf_k=60,
        ),
    )

    print("\n" + "="*60)
    print("ALL PIPELINE TESTS PASSED")


if __name__ == "__main__":
    main()
