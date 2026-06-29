#!/usr/bin/env python3
"""
eval_suite.py — Comprehensive RAG evaluation & ablation suite.

Subcommands
-----------
  prepare   Generate & save a test Q&A set (run once per document).
  run       Evaluate a single named pipeline configuration.
  ablation  Evaluate all 5 configurations and compare side-by-side.
  report    Re-generate HTML report from saved results (no re-scoring needed).
  list      Show available document IDs.

Pipeline configurations
-----------------------
  vector_only    — vector search only, no hybrid, no reranker
  hybrid         — vector + BM25 + RRF, no reranker
  hybrid_rerank  — vector + BM25 + RRF + cross-encoder reranker
  hyde           — hybrid_rerank + HyDE hypothetical document
  full           — all features including query decomposition

RAGAS metrics (0.1.21)
-----------------------
  faithfulness      — answer grounded in retrieved context (anti-hallucination)
  answer_relevancy  — answer addresses the question
  context_precision — retrieved chunks are relevant to the query
  context_recall    — context covers the ground-truth answer

Usage
-----
  cd backend
  python scripts/eval_suite.py prepare --document-id <uuid> --n 15
  python scripts/eval_suite.py ablation --document-id <uuid>
  python scripts/eval_suite.py report --document-id <uuid>

Results are saved to backend/eval_results/<document-id>/.
"""

import argparse
import io
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import requests

# Force UTF-8 output on Windows (cp1252 can't encode LLM-generated Unicode)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # not installed — RAGAS may hang on Windows without it

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from app.core.config import settings  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline configurations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    name: str
    label: str
    hybrid: bool = True
    reranker: bool = True
    hyde: bool = False
    decomposition: bool = False


ABLATION_CONFIGS: list[PipelineConfig] = [
    PipelineConfig("vector_only",   "[1] Vector Only",              hybrid=False, reranker=False),
    PipelineConfig("hybrid",        "[2] Hybrid Search (BM25+RRF)", hybrid=True,  reranker=False),
    PipelineConfig("hybrid_rerank", "[3] Hybrid + Rerank",          hybrid=True,  reranker=True),
    PipelineConfig("hyde",          "[4] Hybrid + Rerank + HyDE",   hybrid=True,  reranker=True,  hyde=True),
    PipelineConfig("full",          "[5] Full Pipeline (+Decomp)",  hybrid=True,  reranker=True,  hyde=True, decomposition=True),
]


# ─────────────────────────────────────────────────────────────────────────────
# Storage helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_dir(document_id: str) -> Path:
    d = ROOT / "eval_results" / document_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _testset_path(document_id: str) -> Path:
    return _eval_dir(document_id) / "testset.json"


def _result_path(document_id: str, config_name: str) -> Path:
    return _eval_dir(document_id) / f"{config_name}.json"


def _list_document_ids() -> list[str]:
    parsed_dir = Path(settings.upload_dir).parent / "parsed"
    if not parsed_dir.exists():
        return []
    return [p.stem.replace("_chunks", "") for p in sorted(parsed_dir.glob("*_chunks.json"))]


# ─────────────────────────────────────────────────────────────────────────────
# LLM helper
# ─────────────────────────────────────────────────────────────────────────────

def _llm(messages: list[dict]) -> str:
    from app.services.llm.providers import get_llm_provider
    return get_llm_provider().chat(messages, stream=False).choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Prepare & persist test set
# ─────────────────────────────────────────────────────────────────────────────

_QA_SYSTEM = (
    "You are a QA dataset generator for RAG evaluation. "
    "Create question-answer pairs from document passages."
)
_QA_USER = """\
Read the passage and generate exactly {n} specific, diverse questions whose answers \
appear ONLY in the passage. For each question, write a concise factual ground-truth \
answer (1–3 sentences).

Passage:
{passage}

Return ONLY a JSON array — no commentary, no markdown fences:
[{{"question": "...", "ground_truth": "..."}}, ...]"""


def prepare_testset(document_id: str, n: int = 15, seed: int = 42) -> list[dict]:
    """Generate Q&A pairs from document chunks and save to disk.

    If a testset already exists for this document, loads it instead of
    regenerating — preserving reproducibility across ablation runs.
    """
    path = _testset_path(document_id)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        print(f"  Loaded {len(existing)} Q&A pairs from cache: {path}")
        return existing

    from app.services.chunker import load_chunks
    random.seed(seed)

    chunks = [c for c in load_chunks(document_id) if len(c.get("text", "")) > 200]
    if not chunks:
        raise ValueError(f"No usable chunks for document {document_id!r}")
    random.shuffle(chunks)

    pairs: list[dict] = []
    for start in range(0, len(chunks), 3):
        if len(pairs) >= n:
            break
        batch = chunks[start : start + 3]
        passage = "\n\n---\n\n".join(c["text"] for c in batch)
        per_call = min(3, n - len(pairs))
        try:
            raw = _llm([
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user",   "content": _QA_USER.format(n=per_call, passage=passage[:3500])},
            ])
            s, e = raw.find("["), raw.rfind("]") + 1
            if s == -1:
                continue
            for p in json.loads(raw[s:e]):
                if isinstance(p, dict) and "question" in p and "ground_truth" in p:
                    pairs.append({
                        "question":     str(p["question"]),
                        "ground_truth": str(p["ground_truth"]),
                    })
        except Exception as exc:
            logger.warning("QA generation batch %d failed: %s", start, exc)

    pairs = pairs[:n]
    path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved {len(pairs)} Q&A pairs -> {path}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Retrieval via backend HTTP endpoint (avoids Qdrant single-process lock)
# ─────────────────────────────────────────────────────────────────────────────

_BACKEND_URL = os.getenv("EVAL_BACKEND_URL", "http://localhost:8000")


def _run_qa_via_http(
    question: str,
    document_id: str,
    config: PipelineConfig,
    top_k: int = 5,
) -> tuple[str, list[str]]:
    """POST to /eval/run_qa — backend holds the Qdrant lock, we just call HTTP."""
    resp = requests.post(
        f"{_BACKEND_URL}/eval/run_qa",
        json={
            "question":      question,
            "document_id":   document_id,
            "hybrid":        config.hybrid,
            "reranker":      config.reranker,
            "hyde":          config.hyde,
            "decomposition": config.decomposition,
            "top_k":         top_k,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["answer"], data["contexts"]


def _run_rag_for_config(
    qa_pairs: list[dict],
    document_id: str,
    config: PipelineConfig,
) -> list[dict]:
    rows: list[dict] = []
    n = len(qa_pairs)
    for i, pair in enumerate(qa_pairs):
        q, gt = pair["question"], pair["ground_truth"]
        print(f"    [{i+1}/{n}] {q[:72]}...")
        try:
            answer, contexts = _run_qa_via_http(q, document_id, config)
            if not contexts:
                print("           [!] No context retrieved — skipped")
                continue
            rows.append({"question": q, "answer": answer, "contexts": contexts, "ground_truth": gt})
        except Exception as exc:
            print(f"           [!] Failed: {exc} — skipped")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — RAGAS scoring
# ─────────────────────────────────────────────────────────────────────────────

def _make_ragas_llm():
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    if settings.llm_provider == "nvidia":
        chat = ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=settings.nvidia_api_key,
            model=settings.nvidia_chat_model,
            temperature=0,
            timeout=60,
            max_retries=1,
        )
    else:
        chat = ChatOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",
            model=settings.ollama_chat_model,
            temperature=0,
            timeout=60,
            max_retries=1,
        )
    return LangchainLLMWrapper(chat)


def _make_ragas_embeddings():
    """Always use sentence-transformers for RAGAS scoring.

    NVIDIA NIM embedding API requires a non-standard `input_type` parameter that
    the generic LangchainEmbeddingsWrapper does not pass, causing answer_relevancy
    to return NaN.  The local all-MiniLM-L6-v2 model is already installed, is
    fast, and produces reliable cosine-similarity scores for RAGAS.
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(emb)


def _score_with_ragas(rows: list[dict]) -> tuple[dict, list[dict]]:
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
    faithfulness.llm       = ragas_llm
    answer_relevancy.llm   = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_precision.llm  = ragas_llm
    context_recall.llm     = ragas_llm

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        raise_exceptions=False,
    )
    scores = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}

    per_sample: list[dict] = []
    try:
        df = result.to_pandas()
        for _, row in df.iterrows():
            per_sample.append({
                "question": str(row.get("question", "")),
                "answer":   str(row.get("answer",   ""))[:300],
                "scores":   {m: float(row.get(m, 0)) for m in METRICS if m in row},
            })
    except Exception:
        pass

    return scores, per_sample


# ─────────────────────────────────────────────────────────────────────────────
# Run a single config (with caching)
# ─────────────────────────────────────────────────────────────────────────────

def run_config(document_id: str, config: PipelineConfig) -> dict:
    out_path = _result_path(document_id, config.name)
    if out_path.exists():
        cached = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"  Cached result for '{config.name}' loaded from {out_path.name}")
        return cached

    testset = prepare_testset(document_id)
    if not testset:
        raise ValueError("No test set. Run 'prepare' first.")

    print(f"\n  Config: {config.label}")
    t0 = time.monotonic()
    rows = _run_rag_for_config(testset, document_id, config)
    if not rows:
        print(f"  [!] No rows to score for '{config.name}'")
        return {}

    print(f"\n  Scoring {len(rows)} rows with RAGAS (this may take a few minutes)...")
    scores, per_sample = _score_with_ragas(rows)
    elapsed = int(time.monotonic() - t0)

    result = {
        "config_name":  config.name,
        "config_label": config.label,
        "config_flags": asdict(config),
        "document_id":  document_id,
        "n_evaluated":  len(rows),
        "elapsed_sec":  elapsed,
        "timestamp":    datetime.now().isoformat(),
        "scores":       scores,
        "per_sample":   per_sample,
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved -> {out_path}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HTML Report
# ─────────────────────────────────────────────────────────────────────────────

def _hsl(v: float) -> str:
    if v != v:  # NaN check
        return "hsl(0, 0%, 93%)"
    return f"hsl({int(v * 120)}, 60%, 88%)"


def _bar_html(v: float, width: int = 100) -> str:
    h = int(v * 120)
    pct = int(v * 100)
    return (
        f'<div style="display:inline-block;width:{width}px;height:9px;'
        f'background:#e0e0e0;border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:5px">'
        f'<div style="width:{pct}%;height:100%;background:hsl({h},55%,48%);border-radius:4px"></div>'
        f'</div>'
    )


def generate_html_report(document_id: str) -> Path | None:
    results = []
    for cfg in ABLATION_CONFIGS:
        p = _result_path(document_id, cfg.name)
        if p.exists():
            results.append(json.loads(p.read_text(encoding="utf-8")))

    if not results:
        return None

    report_path = _eval_dir(document_id) / "report.html"

    # ── Comparison table ──────────────────────────────────────────────────────
    th_metrics = "".join(f'<th>{METRIC_LABELS[m]}</th>' for m in METRICS) + "<th>Avg</th>"

    def _flag_chip(label: str, active: bool) -> str:
        color = "#2e7d32" if active else "#bdbdbd"
        return f'<span style="font-size:10px;color:{color}">{"✓" if active else "✗"} {label}</span>'

    comparison_rows = ""
    all_avgs = []
    for r in results:
        sc = r.get("scores", {})
        vals = [sc.get(m, 0) for m in METRICS]
        avg = sum(vals) / len(vals)
        all_avgs.append(avg)

    best_avg = max(all_avgs) if all_avgs else 0

    for r, avg in zip(results, all_avgs):
        sc = r.get("scores", {})
        vals = [sc.get(m, 0) for m in METRICS]
        flags = r.get("config_flags", {})
        is_best = abs(avg - best_avg) < 0.001

        chip_row = " &nbsp; ".join([
            _flag_chip("Hybrid",   flags.get("hybrid",       False)),
            _flag_chip("Rerank",   flags.get("reranker",     False)),
            _flag_chip("HyDE",     flags.get("hyde",         False)),
            _flag_chip("Decomp",   flags.get("decomposition",False)),
        ])
        label_cell = (
            f'<strong>{r["config_label"]}</strong>'
            f'{"<br><span style=\'font-size:11px;background:#fff9c4;padding:1px 5px;border-radius:3px\'>⭐ best</span>" if is_best else ""}'
            f'<br><span style="font-size:11px">{chip_row}</span>'
            f'<br><span style="font-size:11px;color:#9e9e9e">{r.get("n_evaluated",0)} samples · {r.get("elapsed_sec",0)}s</span>'
        )
        metric_cells = "".join(
            f'<td style="background:{_hsl(v) if v==v else _hsl(0)};text-align:center;padding:10px 8px">'
            f'{_bar_html(v) if v==v else ""}<strong>{"N/A" if v!=v else f"{v:.3f}"}</strong></td>'
            for v in vals
        )
        avg_cell = (
            f'<td style="background:{_hsl(avg) if avg==avg else _hsl(0)};text-align:center;padding:10px 8px;'
            f'font-weight:700;{"border:2px solid #43a047;" if is_best else ""}">'
            f'{"N/A" if avg!=avg else f"{_bar_html(avg)}<strong>{avg:.3f}</strong>"}</td>'
        )
        comparison_rows += f'<tr><td style="padding:10px 12px">{label_cell}</td>{metric_cells}{avg_cell}</tr>'

    # ── Per-sample breakdown ──────────────────────────────────────────────────
    sample_section = ""
    for r in results:
        samples = r.get("per_sample", [])
        if not samples:
            continue
        sample_rows = ""
        for s in samples:
            sc = s.get("scores", {})
            avg_s = sum(sc.values()) / max(len(sc), 1)
            row_bg = "#fff9c4" if avg_s < 0.5 else "#fff"
            q_text = s["question"][:110] + ("…" if len(s["question"]) > 110 else "")
            metric_cells = "".join(
                f'<td style="text-align:center;color:{"#c62828" if sc.get(m,0)<0.5 else "#388e3c"};padding:7px">'
                f'{sc.get(m,0):.2f}</td>'
                for m in METRICS
            )
            sample_rows += (
                f'<tr style="background:{row_bg}">'
                f'<td style="font-size:12px;padding:7px 10px">{q_text}</td>'
                f'{metric_cells}'
                f'<td style="text-align:center;font-weight:700;padding:7px">{avg_s:.2f}</td>'
                f'</tr>'
            )
        th_m = "".join(f'<th>{METRIC_LABELS[m]}</th>' for m in METRICS)
        sample_section += f"""
        <h2 style="margin:32px 0 12px">Per-sample Breakdown
          <span style="font-weight:400;font-size:12px;color:#9e9e9e">— {r['config_label']}</span>
        </h2>
        <div class="card" style="overflow-x:auto">
          <p style="font-size:12px;color:#9e9e9e;margin-bottom:10px">
            <span style="background:#fff9c4;padding:2px 7px;border-radius:3px">Vàng</span>
            = avg &lt; 0.5 — câu hỏi cần chú ý
          </p>
          <table>
            <thead><tr><th style="min-width:260px">Question</th>{th_m}<th>Avg</th></tr></thead>
            <tbody>{sample_rows}</tbody>
          </table>
        </div>"""
        break  # show breakdown for first config that has data

    # ── Metric legend ─────────────────────────────────────────────────────────
    legend_items = "".join(
        f'<div style="padding:10px 0;border-bottom:1px solid #f0f0f0">'
        f'<strong style="display:inline-block;width:170px">{METRIC_LABELS[m]}</strong>'
        f'<span style="color:#616161;font-size:13px">{desc}</span></div>'
        for m, desc in [
            ("faithfulness",      "Câu trả lời có được hỗ trợ bởi retrieved context không? (chống hallucination)"),
            ("answer_relevancy",  "Câu trả lời có đúng chủ đề của câu hỏi không?"),
            ("context_precision", "Các chunks được retrieve có thực sự liên quan đến query không?"),
            ("context_recall",    "Context có đủ thông tin để trả lời ground-truth không?"),
        ]
    )

    model = settings.nvidia_chat_model if settings.llm_provider == "nvidia" else settings.ollama_chat_model
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>RAGAS Evaluation Report</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#f8f9fa; color:#212121; padding:32px 16px 64px; line-height:1.6; }}
  .container {{ max-width:1140px; margin:0 auto; }}
  h1 {{ font-size:1.6rem; font-weight:700; margin-bottom:6px; }}
  h2 {{ font-size:.9rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#616161; }}
  .meta {{ font-size:13px; color:#757575; margin:8px 0 32px; }}
  .card {{ background:#fff; border:1px solid #e0e0e0; border-radius:12px; padding:20px 24px; margin-bottom:16px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#f5f5f5; padding:8px 10px; text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#616161; border-bottom:2px solid #e0e0e0; white-space:nowrap; }}
  td {{ border-bottom:1px solid #f0f0f0; vertical-align:middle; }}
  tr:last-child td {{ border-bottom:none; }}
  .section-header {{ margin:36px 0 14px; display:flex; align-items:center; gap:10px; }}
  .section-header::after {{ content:""; flex:1; height:1px; background:#e0e0e0; }}
</style>
</head>
<body>
<div class="container">

  <h1>RAGAS Evaluation Report</h1>
  <div class="meta">
    Document: <code>{document_id[:16]}…</code> &nbsp;·&nbsp;
    Model: <code>{model}</code> &nbsp;·&nbsp;
    {len(results)} config(s) &nbsp;·&nbsp;
    Generated: {ts}
  </div>

  <div class="section-header"><h2>Metrics</h2></div>
  <div class="card">{legend_items}</div>

  <div class="section-header"><h2>Pipeline Comparison</h2></div>
  <div class="card" style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th style="min-width:240px">Configuration</th>
          {th_metrics}
        </tr>
      </thead>
      <tbody>{comparison_rows}</tbody>
    </table>
  </div>

  {sample_section}

  <div style="margin-top:48px;text-align:center;font-size:12px;color:#bdbdbd">
    RAG PDF Project &nbsp;·&nbsp; RAGAS 0.1.21 &nbsp;·&nbsp; AIO 2026
  </div>

</div>
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Terminal output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_scores(scores: dict, label: str) -> None:
    print(f"\n  {label}")
    print("  " + "-" * 58)
    for m in METRICS:
        v = scores.get(m)
        if v is not None:
            bar = "#" * int(v * 20) + "." * (20 - int(v * 20))
            print(f"  {METRIC_LABELS[m]:<22} {v:.4f}  {bar}  ({v*100:.1f}%)")
    print()


def _print_comparison(results: list[dict]) -> None:
    valid = [r for r in results if r.get("scores")]
    if not valid:
        return
    avgs = [sum(r["scores"].get(m, 0) for m in METRICS) / len(METRICS) for r in valid]
    best = max(avgs)

    print()
    print("=" * 80)
    print("  ABLATION COMPARISON")
    print("=" * 80)
    print(f"  {'Configuration':<32} {'Faith':>6} {'AnsRel':>7} {'CtxPre':>7} {'CtxRec':>7} {'Avg':>6}")
    print("  " + "-" * 70)
    for r, avg in zip(valid, avgs):
        sc = r["scores"]
        vals = [sc.get(m, 0) for m in METRICS]
        marker = "  <- best" if abs(avg - best) < 0.001 else ""
        print(f"  {r['config_label']:<32} {vals[0]:>6.3f} {vals[1]:>7.3f} {vals[2]:>7.3f} {vals[3]:>7.3f} {avg:>6.3f}{marker}")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Evaluation & Ablation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # prepare
    p = sub.add_parser("prepare", help="Generate & save Q&A test set")
    p.add_argument("--document-id")
    p.add_argument("--n", type=int, default=15, help="Number of Q&A pairs (default: 15)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Regenerate even if cache exists")

    # run
    p = sub.add_parser("run", help="Evaluate a single pipeline config")
    p.add_argument("--document-id")
    p.add_argument("--config", choices=[c.name for c in ABLATION_CONFIGS], default="hybrid_rerank")
    p.add_argument("--force", action="store_true", help="Re-run even if cached")

    # ablation
    p = sub.add_parser("ablation", help="Evaluate all 5 configs (or a subset)")
    p.add_argument("--document-id")
    p.add_argument("--configs", nargs="*", choices=[c.name for c in ABLATION_CONFIGS],
                   help="Subset of configs to run (default: all 5)")
    p.add_argument("--force", action="store_true", help="Re-run even if cached")

    # report
    p = sub.add_parser("report", help="Generate HTML report from saved results")
    p.add_argument("--document-id")

    # list
    sub.add_parser("list", help="List available document IDs")

    args = parser.parse_args()

    if args.cmd == "list":
        docs = _list_document_ids()
        if not docs:
            print("No ingested documents found. Upload a PDF via the UI first.")
        for d in docs:
            ts = _testset_path(d)
            n_qa = len(json.loads(ts.read_text(encoding="utf-8"))) if ts.exists() else 0
            cached = [c.name for c in ABLATION_CONFIGS if _result_path(d, c.name).exists()]
            print(f"  {d}  |  testset: {n_qa} Q&A  |  cached: {cached or 'none'}")
        return

    # Resolve document ID
    if not args.document_id:
        docs = _list_document_ids()
        if not docs:
            sys.exit("No documents found. Upload a PDF first, then re-run.")
        args.document_id = docs[0]
        print(f"Auto-selected document: {args.document_id}")

    model = settings.nvidia_chat_model if settings.llm_provider == "nvidia" else settings.ollama_chat_model

    if args.cmd == "prepare":
        if getattr(args, "force", False):
            _testset_path(args.document_id).unlink(missing_ok=True)
        print(f"\nPreparing test set ({args.n} Q&A pairs)  —  model: {model}")
        pairs = prepare_testset(args.document_id, n=args.n, seed=args.seed)
        print(f"\nDone: {len(pairs)} Q&A pairs ready at {_testset_path(args.document_id)}")

    elif args.cmd == "run":
        cfg = next(c for c in ABLATION_CONFIGS if c.name == args.config)
        if getattr(args, "force", False):
            _result_path(args.document_id, cfg.name).unlink(missing_ok=True)
        print(f"\nEvaluating config '{cfg.label}'  —  model: {model}")
        result = run_config(args.document_id, cfg)
        if result:
            _print_scores(result.get("scores", {}), cfg.label)

    elif args.cmd == "ablation":
        chosen = [c for c in ABLATION_CONFIGS if not args.configs or c.name in args.configs]
        if getattr(args, "force", False):
            for c in chosen:
                _result_path(args.document_id, c.name).unlink(missing_ok=True)
        print(f"\nAblation: {len(chosen)} configurations  —  model: {model}")
        print(f"Document: {args.document_id}\n")
        all_results = [run_config(args.document_id, cfg) for cfg in chosen]
        _print_comparison([r for r in all_results if r])
        report_path = generate_html_report(args.document_id)
        if report_path:
            print(f"\nHTML report -> {report_path}")

    elif args.cmd == "report":
        path = generate_html_report(args.document_id)
        if path:
            print(f"Report -> {path}")
        else:
            print("No results found. Run 'ablation' first.")


if __name__ == "__main__":
    main()
