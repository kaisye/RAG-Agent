#!/usr/bin/env python3
"""
RAGAS evaluation for RAG PDF Project.

Runs three steps automatically:
  1. Generate synthetic Q&A pairs from document chunks (via the LLM).
  2. Run the full RAG pipeline (hybrid-search + rerank + LLM) for each question.
  3. Score with four RAGAS metrics and save results to JSON.

Usage (run from backend/ directory):
    python scripts/evaluate_ragas.py --document-id <uuid> [--n-questions 10]
    python scripts/evaluate_ragas.py --list-documents
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

# ── bootstrap: make 'app.*' imports work from any cwd ────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # .../backend
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                  # config reads .env relative to backend/

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from app.core.config import settings  # noqa: E402 — needs env loaded first

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _list_document_ids() -> list[str]:
    """Discover ingested document IDs from storage/parsed/."""
    parsed_dir = Path(settings.upload_dir).parent / "parsed"
    if not parsed_dir.exists():
        return []
    return [p.stem.replace("_chunks", "") for p in sorted(parsed_dir.glob("*_chunks.json"))]


def _llm_complete(messages: list[dict]) -> str:
    """Single synchronous call to the configured LLM provider."""
    from app.services.llm.providers import get_llm_provider
    provider = get_llm_provider()
    resp = provider.chat(messages, stream=False)
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Synthetic Q&A generation
# ─────────────────────────────────────────────────────────────────────────────

_QA_SYSTEM = (
    "You are a QA dataset generator for RAG evaluation. "
    "Your task is to create question-answer pairs from document passages."
)

_QA_USER = """\
Read the passage below and generate exactly {n} specific, diverse questions \
whose answers appear ONLY in the passage.
For every question, also write the ground-truth answer (1–3 factual sentences).

Passage:
{passage}

Return a JSON array ONLY — no commentary, no markdown fences:
[
  {{"question": "...", "ground_truth": "..."}},
  ...
]"""


def generate_qa_pairs(document_id: str, n: int = 10) -> list[dict]:
    from app.services.chunker import load_chunks

    chunks = load_chunks(document_id)
    text_chunks = [c for c in chunks if len(c.get("text", "")) > 200]

    if not text_chunks:
        raise ValueError(f"No usable text chunks for document {document_id!r}")

    random.shuffle(text_chunks)
    qa_pairs: list[dict] = []
    batch_size = 3  # chunks per LLM call → ~3 questions per call

    for start in range(0, len(text_chunks), batch_size):
        if len(qa_pairs) >= n:
            break
        batch = text_chunks[start: start + batch_size]
        passage = "\n\n---\n\n".join(c["text"] for c in batch)
        per_call = min(batch_size, n - len(qa_pairs))

        messages = [
            {"role": "system", "content": _QA_SYSTEM},
            {"role": "user", "content": _QA_USER.format(n=per_call, passage=passage[:3500])},
        ]
        try:
            raw = _llm_complete(messages)
            s, e = raw.find("["), raw.rfind("]") + 1
            if s == -1 or e == 0:
                logger.warning("LLM did not return a JSON array; skipping batch %d", start)
                continue
            pairs = json.loads(raw[s:e])
            for p in pairs:
                if isinstance(p, dict) and "question" in p and "ground_truth" in p:
                    qa_pairs.append({"question": str(p["question"]), "ground_truth": str(p["ground_truth"])})
        except Exception as exc:
            logger.warning("QA generation batch %d failed: %s", start, exc)

    return qa_pairs[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — RAG pipeline execution
# ─────────────────────────────────────────────────────────────────────────────

_ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer using ONLY the provided context. "
    "Be concise and factual. "
    "If the answer is not in the context, say: 'I don't know based on the provided context.'"
)


def run_rag(question: str, document_id: str) -> tuple[str, list[str]]:
    """Return (answer, [retrieved_context_texts])."""
    from app.services.retrieval import retrieve

    chunks = retrieve(question, document_id=document_id)
    text_contexts = [c["content"] for c in chunks if c.get("type") == "text" and c.get("content")]

    context_block = "\n\n".join(
        f"[Page {c.get('page', '?')}] {c['content']}"
        for c in chunks if c.get("type") == "text" and c.get("content")
    )
    messages = [
        {"role": "system", "content": _ANSWER_SYSTEM},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
    ]
    answer = _llm_complete(messages)
    return answer, text_contexts


def build_eval_rows(qa_pairs: list[dict], document_id: str) -> list[dict]:
    rows: list[dict] = []
    n = len(qa_pairs)
    for i, pair in enumerate(qa_pairs):
        q, gt = pair["question"], pair["ground_truth"]
        print(f"  [{i + 1}/{n}] {q[:80]}...")
        try:
            answer, contexts = run_rag(q, document_id)
            if not contexts:
                print(f"         [!] No contexts retrieved -- skipped")
                continue
            rows.append({"question": q, "answer": answer, "contexts": contexts, "ground_truth": gt})
        except Exception as exc:
            print(f"         [!] RAG failed: {exc} -- skipped")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — RAGAS scoring
# ─────────────────────────────────────────────────────────────────────────────

def _make_ragas_llm():
    """Return a ragas-compatible LLM wrapper backed by the configured provider."""
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    if settings.llm_provider == "nvidia":
        chat = ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=settings.nvidia_api_key,
            model=settings.nvidia_chat_model,
            temperature=0,
        )
    else:
        chat = ChatOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",
            model=settings.ollama_chat_model,
            temperature=0,
        )
    return LangchainLLMWrapper(chat)


def _make_ragas_embeddings():
    """Return a ragas-compatible embeddings wrapper.

    Uses NVIDIA NIM embedding API when llm_provider=='nvidia', otherwise
    falls back to a local sentence-transformers model so Ollama users don't
    need a second API key.
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if settings.llm_provider == "nvidia" and settings.nvidia_api_key and settings.nvidia_embed_model:
        from langchain_openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=settings.nvidia_api_key,
            model=settings.nvidia_embed_model,
        )
    else:
        # Local sentence-transformers fallback — already installed by requirements.txt
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
            emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return LangchainEmbeddingsWrapper(emb)


def run_ragas_evaluation(rows: list[dict]) -> tuple[dict, object]:
    """Run RAGAS and return (metric_scores_dict, full_result_object)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in rows],
        "answer":       [r["answer"]       for r in rows],
        "contexts":     [r["contexts"]     for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    ragas_llm = _make_ragas_llm()
    ragas_emb = _make_ragas_embeddings()

    # Assign judge LLM/embeddings to each metric
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        raise_exceptions=False,
    )
    scores = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    return scores, result


# ─────────────────────────────────────────────────────────────────────────────
# Pretty output
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_LABELS = {
    "faithfulness":      "Faithfulness       — answer grounded in retrieved context",
    "answer_relevancy":  "Answer Relevancy   — answer is on-topic for the question",
    "context_precision": "Context Precision  — retrieved chunks are relevant to query",
    "context_recall":    "Context Recall     — context covers the ground-truth answer",
}


def _print_results(scores: dict, n_rows: int) -> None:
    print()
    print("+" + "-" * 72 + "+")
    print("|  RAGAS Evaluation Results" + " " * 46 + "|")
    print("|" + f"  Samples evaluated: {n_rows}" + " " * (70 - len(str(n_rows)) - 20) + "|")
    print("+" + "-" * 72 + "+")
    for key, label in _METRIC_LABELS.items():
        val = scores.get(key)
        if val is None:
            print(f"|  {label}")
            print(f"|    Score: N/A")
        else:
            filled = int(val * 20)
            bar = "#" * filled + "." * (20 - filled)
            print(f"|  {label}")
            print(f"|    Score: {val:.4f}  [{bar}]  ({val * 100:.1f}%)")
        print("|")
    print("+" + "-" * 72 + "+")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG PDF pipeline with RAGAS metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--document-id", help="UUID of the ingested document to evaluate")
    parser.add_argument("--list-documents", action="store_true", help="Print available document IDs and exit")
    parser.add_argument("--n-questions", type=int, default=10, help="Synthetic test questions to generate (default: 10)")
    parser.add_argument("--output", default="ragas_results.json", help="JSON output path (default: ragas_results.json)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    # --list-documents
    if args.list_documents:
        docs = _list_document_ids()
        if not docs:
            print("No ingested documents found in storage/parsed/")
            print("Upload a PDF via the UI first.")
        else:
            print("Available document IDs:")
            for d in docs:
                print(f"  {d}")
        return

    # Resolve document ID
    if not args.document_id:
        docs = _list_document_ids()
        if not docs:
            print("No documents found. Upload a PDF first, then re-run.")
            sys.exit(1)
        args.document_id = docs[0]
        print(f"No --document-id given; auto-selected: {args.document_id}")

    model = settings.nvidia_chat_model if settings.llm_provider == "nvidia" else settings.ollama_chat_model

    print()
    print("=" * 66)
    print("  RAGAS Evaluation — RAG PDF Project")
    print("=" * 66)
    print(f"  Document     : {args.document_id}")
    print(f"  N Questions  : {args.n_questions}")
    print(f"  LLM Provider : {settings.llm_provider.upper()}  ({model})")
    print(f"  Reranker     : {'enabled' if settings.reranker_enabled else 'disabled'}  ({settings.reranker_model})")
    print(f"  Hybrid search: {'enabled' if settings.hybrid_search_enabled else 'disabled'}")
    print()

    # ── Step 1: synthetic Q&A ─────────────────────────────────────────────────
    print("* [1/3]  Generating synthetic Q&A pairs from document chunks...")
    qa_pairs = generate_qa_pairs(args.document_id, n=args.n_questions)
    print(f"         {len(qa_pairs)} pair(s) generated\n")

    if not qa_pairs:
        print("ERROR: Could not generate Q&A pairs. Check LLM provider config.")
        sys.exit(1)

    # ── Step 2: RAG pipeline ──────────────────────────────────────────────────
    print("* [2/3]  Running RAG pipeline for each question...")
    rows = build_eval_rows(qa_pairs, args.document_id)
    print(f"\n         {len(rows)} / {len(qa_pairs)} question(s) evaluated\n")

    if not rows:
        print("ERROR: Empty dataset after RAG pipeline. Check retrieval config.")
        sys.exit(1)

    # ── Step 3: RAGAS metrics ─────────────────────────────────────────────────
    print("* [3/3]  Computing RAGAS metrics (may take several minutes)...")
    scores, _raw = run_ragas_evaluation(rows)

    _print_results(scores, n_rows=len(rows))

    # ── Save ──────────────────────────────────────────────────────────────────
    output_data = {
        "document_id":   args.document_id,
        "n_evaluated":   len(rows),
        "llm_provider":  settings.llm_provider,
        "chat_model":    model,
        "reranker":      settings.reranker_enabled,
        "hybrid_search": settings.hybrid_search_enabled,
        "metrics":       scores,
        "samples": [
            {
                "question":     r["question"],
                "answer":       r["answer"],
                "ground_truth": r["ground_truth"],
                "n_contexts":   len(r["contexts"]),
                "contexts":     r["contexts"],
            }
            for r in rows
        ],
    }

    out_path = ROOT / args.output
    out_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
