"""
Sinh testset tự động cho RAGAS evaluation.
Quy trình: PDF → Markdown → LangChain Docs → TestsetGenerator → testset.json

Chạy: cd backend && python scripts/generate_testset.py [--size N] [--pdf-dir DIR] [--out PATH]
Yêu cầu: NVIDIA_API_KEY, NVIDIA_CHAT_MODEL, NVIDIA_EMBED_MODEL trong .env
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Phải load settings trước khi import langchain/ragas để lru_cache bắt đúng .env
from app.core.config import get_settings

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


# ---------------------------------------------------------------------------
# Bước 1: PDF → Markdown
# ---------------------------------------------------------------------------

def step1_convert_pdfs(pdf_dir: str, md_dir: str) -> int:
    """
    PDF → Markdown.
    Ưu tiên pymupdf4llm.to_markdown() để giữ cấu trúc heading/table.
    Nếu pymupdf4llm thất bại (onnxruntime tensor mismatch), fallback sang
    fitz text extraction thuần — đủ cho TestsetGenerator.
    """
    import fitz  # PyMuPDF — luôn available

    pdf_path = Path(pdf_dir)
    md_path = Path(md_dir)
    md_path.mkdir(parents=True, exist_ok=True)

    converted = 0
    for pdf_file in sorted(pdf_path.glob("*.pdf")):
        out_file = md_path / f"{pdf_file.stem}.md"
        if out_file.exists():
            print(f"  Skip (exists): {pdf_file.name}")
            continue
        print(f"  Converting: {pdf_file.name} ...", end="", flush=True)

        md = None
        try:
            import pymupdf4llm
            md = pymupdf4llm.to_markdown(str(pdf_file))
        except Exception as e:
            print(f"\n  [pymupdf4llm failed: {e!s:.80}] fallback to fitz text extraction")

        if md is None:
            # Fallback: extract raw text page-by-page và format thành Markdown
            doc = fitz.open(str(pdf_file))
            parts = []
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    parts.append(f"## Page {i}\n\n{text}")
            doc.close()
            md = "\n\n".join(parts)

        out_file.write_text(md, encoding="utf-8")
        print(f" {len(md)} chars")
        converted += 1

    print(f"Step 1 done: {converted} files converted → {md_path}")
    return converted


# ---------------------------------------------------------------------------
# Bước 2: Load Markdown → LangChain Documents
# ---------------------------------------------------------------------------

def step2_load_docs(md_dir: str):
    """Load .md files với UTF-8 cho tiếng Việt."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    loader = DirectoryLoader(
        md_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"Step 2 done: {len(docs)} documents loaded from {md_dir}")
    if not docs:
        raise ValueError(f"Không tìm thấy .md files trong {md_dir}. Chạy step1 trước.")
    return docs


# ---------------------------------------------------------------------------
# Bước 3: TestsetGenerator → testset.json
# ---------------------------------------------------------------------------

class _NvidiaEmbeddingsWrapper:
    """
    Langchain-compatible Embeddings wrapper dùng EmbeddingService của project.
    Tránh lỗi 500 từ OpenAIEmbeddings async batch path với NVIDIA API.
    Cần có method embed_documents(texts) và embed_query(text).
    """

    def __init__(self, embed_service):
        self._svc = embed_service

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._svc.embed_texts(texts, input_type="passage")

    def embed_query(self, text: str) -> list[float]:
        return self._svc.embed_texts([text], input_type="query")[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def step3_generate(docs, output_path: str, size: int = 50) -> int:
    """
    Dùng NVIDIA NIM qua ChatOpenAI (LLM) + EmbeddingService (embed).
    RAGAS 0.1.x API: TestsetGenerator.from_langchain() + distributions dict.
    Sinh testset với 3 loại câu hỏi (ragas.testset.evolutions):
      50% simple       (single-hop specific)
      25% reasoning    (multi-hop abstract)
      25% multi_context (multi-hop specific)
    """
    from langchain_openai import ChatOpenAI
    from ragas.testset.evolutions import multi_context, reasoning, simple
    from ragas.testset.generator import TestsetGenerator

    from app.services.embedding import EmbeddingService

    settings = get_settings()

    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY chưa được set trong .env")
    if not settings.nvidia_chat_model:
        raise ValueError("NVIDIA_CHAT_MODEL chưa được set trong .env")
    if not settings.nvidia_embed_model:
        raise ValueError("NVIDIA_EMBED_MODEL chưa được set trong .env")

    print(f"  LLM: {settings.nvidia_chat_model}")
    print(f"  Embed: {settings.nvidia_embed_model}")
    print(f"  Size: {size} samples")

    llm = ChatOpenAI(
        model=settings.nvidia_chat_model,
        base_url=NVIDIA_BASE_URL,
        api_key=settings.nvidia_api_key,
        max_tokens=8192,
        temperature=0,
    )
    embed = _NvidiaEmbeddingsWrapper(EmbeddingService())

    # ragas 0.1.x: from_langchain() nhận generator_llm + critic_llm + embeddings
    generator = TestsetGenerator.from_langchain(
        generator_llm=llm,
        critic_llm=llm,
        embeddings=embed,
    )

    distributions = {
        simple:        0.50,  # single-hop specific
        reasoning:     0.25,  # multi-hop abstract
        multi_context: 0.25,  # multi-hop specific
    }

    print(f"  Generating testset ({size} samples)... này có thể mất vài phút")
    t0 = time.time()
    dataset = generator.generate_with_langchain_docs(
        docs,
        test_size=size,
        distributions=distributions,
        raise_exceptions=False,
    )
    elapsed = time.time() - t0
    print(f"  Generated in {elapsed:.1f}s")

    # Chuyển sang list[dict] với schema chuẩn cho RAGAS evaluate
    df = dataset.to_pandas()
    records = []
    for _, row in df.iterrows():
        record = {
            # ragas 0.1.x dùng "question" / "contexts" / "ground_truth"
            "user_input":         str(row.get("question", row.get("user_input", ""))),
            "reference_contexts": _to_list(row.get("contexts", row.get("reference_contexts", []))),
            "reference":          str(row.get("ground_truth", row.get("reference", ""))),
            "synthesizer_name":   str(row.get("evolution_type", row.get("synthesizer_name", ""))),
        }
        records.append(record)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)

    print(f"Step 3 done: {len(records)} samples → {output_path}")
    return len(records)


def _to_list(val) -> list[str]:
    """Normalize reference_contexts — có thể là list hoặc string tuỳ phiên bản RAGAS."""
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val]
    return [str(val)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate RAGAS testset from PDF documents")
    parser.add_argument("--pdf-dir", default="evaluation/sample_docs",
                        help="Thư mục chứa PDF gốc")
    parser.add_argument("--md-dir", default="storage/markdown",
                        help="Thư mục lưu Markdown trung gian")
    parser.add_argument("--out", default="evaluation/testset.json",
                        help="File output testset.json")
    parser.add_argument("--size", type=int, default=50,
                        help="Số câu hỏi cần sinh (default 50)")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Bỏ qua bước convert PDF nếu Markdown đã tồn tại")
    args = parser.parse_args()

    print("=" * 60)
    print("RAGAS Testset Generator")
    print("=" * 60)

    if not args.skip_convert:
        print("\n[Step 1] PDF → Markdown")
        step1_convert_pdfs(args.pdf_dir, args.md_dir)
    else:
        print("\n[Step 1] Skipped (--skip-convert)")

    print(f"\n[Step 2] Load Markdown from {args.md_dir}")
    docs = step2_load_docs(args.md_dir)

    print(f"\n[Step 3] Generate testset ({args.size} samples)")
    n = step3_generate(docs, args.out, size=args.size)

    print(f"\n{'=' * 60}")
    print(f"DONE: {n} test samples saved to {args.out}")
    if n < 5:
        print(f"WARNING: Chỉ có {n} samples (< 5). Thêm PDF lớn hơn vào {args.pdf_dir}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
