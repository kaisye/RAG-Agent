import logging
from pathlib import Path

import pymupdf4llm

logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(pdf_path: str, output_dir: str) -> str:
    """
    Chuyển PDF → Markdown giữ nguyên heading, table, list.
    CHỈ dùng cho RAGAS TestsetGenerator — không dùng để chunk embedding.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    out_path = out_dir / f"{pdf_path.stem}.md"
    out_path.write_text(md_text, encoding="utf-8")
    logger.info("Converted %s → %s (%d chars)", pdf_path.name, out_path, len(md_text))
    return str(out_path)


def convert_all_documents(upload_dir: str, markdown_dir: str) -> list[dict]:
    """
    Batch convert tất cả PDF trong upload_dir.
    Bỏ qua file đã có .md tương ứng.
    Return list[{"pdf": str, "markdown": str}].
    """
    results = []
    for pdf_path in sorted(Path(upload_dir).glob("*.pdf")):
        md_path = Path(markdown_dir) / f"{pdf_path.stem}.md"
        if md_path.exists():
            logger.debug("Skipping %s (already converted)", pdf_path.name)
            results.append({"pdf": str(pdf_path), "markdown": str(md_path)})
            continue
        try:
            out = convert_pdf_to_markdown(str(pdf_path), markdown_dir)
            results.append({"pdf": str(pdf_path), "markdown": out})
        except Exception:
            logger.exception("Failed to convert %s", pdf_path.name)
    return results
