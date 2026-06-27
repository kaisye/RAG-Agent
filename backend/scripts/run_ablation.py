"""
Ablation study: chạy 8 cấu hình pipeline, đo RAGAS, xuất bảng so sánh.

Quy trình:
  8 PipelineConfig × evaluate_pipeline() → evaluation/results/config_{i}.json
  → ablation_summary.csv + ablation_summary.md (delta so với baseline)

Chạy: cd backend && python scripts/run_ablation.py --document-id <id> [--testset PATH]
Lưu ý: 8 configs × 50 câu × 4 RAGAS calls ≈ 1600 API calls (~40 phút free tier).
        Kết quả từng config được lưu ngay — interrupt an toàn.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import PipelineConfig
from app.services.evaluation.evaluator import evaluate_pipeline

RESULTS_DIR = Path("evaluation/results")

# ---------------------------------------------------------------------------
# 8 configs chuẩn theo tài liệu AIO2026 Bảng 3
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: list[tuple[str, PipelineConfig]] = [
    (
        "1. Baseline (Recursive+Vector)",
        PipelineConfig(
            chunking_strategy="recursive",
            retrieval_strategy="vector",
            query_transform="none",
            rerank_strategy="none",
        ),
    ),
    (
        "2. Semantic Chunking",
        PipelineConfig(
            chunking_strategy="semantic",
            retrieval_strategy="vector",
            query_transform="none",
            rerank_strategy="none",
        ),
    ),
    (
        "3. Hybrid+RRF",
        PipelineConfig(
            chunking_strategy="semantic",
            retrieval_strategy="hybrid_rrf",
            query_transform="none",
            rerank_strategy="none",
        ),
    ),
    (
        "4. HyDE",
        PipelineConfig(
            chunking_strategy="semantic",
            retrieval_strategy="hybrid_rrf",
            query_transform="hyde",
            rerank_strategy="none",
        ),
    ),
    (
        "5. Decomposition",
        PipelineConfig(
            chunking_strategy="semantic",
            retrieval_strategy="hybrid_rrf",
            query_transform="decomposition",
            rerank_strategy="none",
        ),
    ),
    (
        "6. Cross-Encoder",
        PipelineConfig(
            chunking_strategy="semantic",
            retrieval_strategy="hybrid_rrf",
            query_transform="none",
            rerank_strategy="cross_encoder",
        ),
    ),
    (
        "7. MMR",
        PipelineConfig(
            chunking_strategy="semantic",
            retrieval_strategy="hybrid_rrf",
            query_transform="none",
            rerank_strategy="mmr",
        ),
    ),
    (
        "8. Full Pipeline",
        PipelineConfig(
            chunking_strategy="semantic",
            retrieval_strategy="hybrid_rrf",
            query_transform="decomposition",
            rerank_strategy="cross_encoder",
        ),
    ),
]

CONFIG_LABELS = [label for label, _ in ABLATION_CONFIGS]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

async def run_full_ablation(
    document_id: str,
    testset_path: str = "evaluation/testset.json",
    sleep_between: float = 1.5,
    resume: bool = True,
) -> list[dict]:
    """
    Chạy tuần tự 8 configs, lưu từng config ngay sau khi xong.
    resume=True: bỏ qua config đã có file kết quả (tránh chạy lại khi interrupt).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    baseline_scores: dict | None = None

    for i, (label, config) in enumerate(ABLATION_CONFIGS):
        out_path = RESULTS_DIR / f"config_{i}.json"

        if resume and out_path.exists():
            print(f"\n[{i+1}/8] SKIP (đã có): {label}")
            result = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            print(f"\n{'='*60}")
            print(f"[{i+1}/8] {label}")
            print(f"{'='*60}")

            result = await evaluate_pipeline(
                config, document_id, testset_path, sleep_between
            )
            # Lưu ngay — tránh mất khi bị interrupt
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Saved: {out_path}")

        all_results.append(result)
        scores = result["scores"]
        print(f"  Faith={scores['faithfulness']:.2f}  AR={scores['answer_relevancy']:.2f}"
              f"  CP={scores['context_precision']:.2f}  CR={scores['context_recall']:.2f}")

        if i == 0:
            baseline_scores = scores

    _save_summary_table(all_results, baseline_scores)
    return all_results


# ---------------------------------------------------------------------------
# Tạo bảng tổng hợp
# ---------------------------------------------------------------------------

def _save_summary_table(results: list[dict], baseline: dict | None) -> Path:
    """Xuất ablation_summary.csv + .md với cột delta Δ so với baseline."""
    import pandas as pd

    if baseline is None and results:
        baseline = results[0]["scores"]

    rows = []
    for label, result in zip(CONFIG_LABELS[: len(results)], results):
        s = result["scores"]
        b = baseline or {}

        def delta(key: str) -> float:
            if b.get(key, -1) < 0 or s.get(key, -1) < 0:
                return float("nan")
            return round(s[key] - b[key], 2)

        rows.append({
            "Thực nghiệm":       label,
            "Faithfulness":      round(s.get("faithfulness", float("nan")), 2),
            "Δ Faith":           delta("faithfulness"),
            "Answer Relevancy":  round(s.get("answer_relevancy", float("nan")), 2),
            "Δ AR":              delta("answer_relevancy"),
            "Context Precision": round(s.get("context_precision", float("nan")), 2),
            "Δ CP":              delta("context_precision"),
            "Context Recall":    round(s.get("context_recall", float("nan")), 2),
            "Δ CR":              delta("context_recall"),
        })

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "ablation_summary.csv"
    md_path = RESULTS_DIR / "ablation_summary.md"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    md_text = df.to_markdown(index=False, floatfmt=".2f")
    md_path.write_text(md_text, encoding="utf-8")

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(md_text)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")
    return csv_path


def load_summary_table() -> list[dict]:
    """Đọc ablation_summary.csv → list[dict] để trả qua API."""
    csv_path = RESULTS_DIR / "ablation_summary.csv"
    if not csv_path.exists():
        return []
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run RAGAS ablation study — 8 configs")
    parser.add_argument("--document-id", required=True, help="ID tài liệu đã ingest vào ChromaDB")
    parser.add_argument("--testset", default="evaluation/testset.json",
                        help="Đường dẫn testset.json")
    parser.add_argument("--sleep", type=float, default=1.5,
                        help="Giây nghỉ giữa các câu (rate limit, default=1.5)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Không bỏ qua config đã có — chạy lại từ đầu")
    args = parser.parse_args()

    print("=" * 60)
    print("RAGAS Ablation Study — 8 Configs")
    print(f"Document: {args.document_id}")
    print(f"Testset:  {args.testset}")
    print("=" * 60)

    asyncio.run(
        run_full_ablation(
            document_id=args.document_id,
            testset_path=args.testset,
            sleep_between=args.sleep,
            resume=not args.no_resume,
        )
    )


if __name__ == "__main__":
    main()
