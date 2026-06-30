"""
evaluate_pipeline() — Đo 4 RAGAS metrics cho một PipelineConfig.

Quy trình:
  testset.json → RAGPipeline.retrieve() + generate() → Dataset → ragas.evaluate()
  → {"config": ..., "scores": {faithfulness, answer_relevancy, context_precision, context_recall}}

Rate limit: NVIDIA free tier ~40 req/phút → time.sleep(1.5) giữa mỗi câu hỏi.

RAGAS metrics cần 5 cột:
  user_input, response, retrieved_contexts, reference, reference_contexts
"""
import json
import time
import logging
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import PipelineConfig, get_settings
from app.services.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_NVIDIA_URL = "https://integrate.api.nvidia.com/v1"
_RESULTS_DIR = Path("evaluation/results")

# ragas.evaluate() cần LLM + embeddings riêng (không dùng RAGPipeline internals)
# để tính metrics — wrap lại NVIDIA NIM giống generate_testset.py


def _ragas_llm():
    settings = get_settings()
    return ChatOpenAI(
        model=settings.nvidia_chat_model,
        base_url=_NVIDIA_URL,
        api_key=settings.nvidia_api_key,
        max_tokens=4096,
        temperature=0,
    )


def _ragas_embed():
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.nvidia_embed_model,
        base_url=_NVIDIA_URL,
        api_key=settings.nvidia_api_key,
        model_kwargs={"input_type": "query", "truncate": "END"},
    )


async def evaluate_pipeline(
    config: PipelineConfig,
    document_id: str,
    testset_path: str = "evaluation/testset.json",
    sleep_between: float = 1.5,
) -> dict:
    """
    Chạy đánh giá RAGAS cho một PipelineConfig.

    Args:
        config: PipelineConfig xác định retrieval/transform/rerank strategy.
        document_id: ID tài liệu đã được ingest vào ChromaDB.
        testset_path: Đường dẫn tới testset.json (output của generate_testset.py).
        sleep_between: Giây nghỉ giữa các câu hỏi để tránh rate limit.

    Returns:
        dict với keys: config, scores (4 metrics), num_samples, document_id.
    """
    with open(testset_path, encoding="utf-8") as f:
        testset = json.load(f)

    if not testset:
        raise ValueError(f"Testset rỗng: {testset_path}")

    pipeline = RAGPipeline(config)
    records = []

    logger.info(
        "evaluate_pipeline: %d samples, doc=%s, strategy=%s+%s+%s",
        len(testset),
        document_id,
        config.retrieval_strategy,
        config.query_transform,
        config.rerank_strategy,
    )

    for i, item in enumerate(testset):
        question = item["user_input"]
        logger.debug("Sample %d/%d: %s", i + 1, len(testset), question[:60])

        try:
            contexts = pipeline.retrieve(question, document_id)

            # Không stream — thu thập toàn bộ response trước khi tính metrics
            response_tokens = list(pipeline.generate(question, contexts))
            response = "".join(response_tokens)

            records.append({
                "user_input":         question,
                "response":           response,
                "retrieved_contexts": [c["text"] for c in contexts],
                "reference":          item.get("reference", ""),
                "reference_contexts": item.get("reference_contexts", []),
            })
        except Exception as exc:
            logger.warning("Sample %d failed: %s — skipping", i + 1, exc)

        # Rate limit: NVIDIA free tier ~40 req/phút → ~1 req/1.5s là an toàn
        if i < len(testset) - 1:
            time.sleep(sleep_between)

    if not records:
        raise RuntimeError("Tất cả samples đều thất bại — không thể tính metrics.")

    dataset = Dataset.from_list(records)

    logger.info("Running ragas.evaluate() on %d records...", len(records))
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=_ragas_llm(),
        embeddings=_ragas_embed(),
        run_config=RunConfig(max_workers=2, timeout=180, max_retries=1),
        raise_exceptions=False,
    )

    scores = {
        "faithfulness":      _safe_float(result["faithfulness"]),
        "answer_relevancy":  _safe_float(result["answer_relevancy"]),
        "context_precision": _safe_float(result["context_precision"]),
        "context_recall":    _safe_float(result["context_recall"]),
    }

    output = {
        "config":      config.model_dump(),
        "scores":      scores,
        "num_samples": len(records),
        "document_id": document_id,
    }

    # Lưu kết quả ngay — tránh mất khi ablation bị interrupt
    _save_result(output, config)

    return output


def _safe_float(val) -> float:
    """Chuyển NaN/None thành -1 để dễ phát hiện trong báo cáo."""
    try:
        v = float(val)
        import math
        return v if not math.isnan(v) else -1.0
    except (TypeError, ValueError):
        return -1.0


def _save_result(result: dict, config: PipelineConfig) -> Path:
    """Lưu kết quả vào evaluation/results/{strategy_label}.json."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    label = f"{config.chunking_strategy}_{config.retrieval_strategy}_{config.query_transform}_{config.rerank_strategy}"
    out_path = _RESULTS_DIR / f"{label}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Result saved: %s", out_path)
    return out_path
