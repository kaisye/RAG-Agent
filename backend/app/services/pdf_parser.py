"""
Parse text blocks from a PDF using PyMuPDF.

Each block has the shape:
    {"page": int, "text": str, "bbox": [x0, y0, x1, y1]}

Header/footer detection: a text string that appears verbatim on more than
half the pages AND is located in the top-10% or bottom-10% of the page
height is treated as a repeating header/footer and filtered out.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from app.core.config import settings

logger = logging.getLogger(__name__)


def _load_raw_blocks(pdf_path: str) -> tuple[list[dict], int]:
    """Return (raw_blocks, page_count). Keeps only text blocks (type 0)."""
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    raw: list[dict] = []

    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
            if block_type != 0:
                continue
            stripped = text.strip()
            if not stripped:
                continue
            raw.append(
                {
                    "page": page_num,
                    "text": stripped,
                    "bbox": [x0, y0, x1, y1],
                    "_page_height": page_height,
                }
            )

    doc.close()
    return raw, page_count


def _detect_repeating_headers_footers(raw_blocks: list[dict], page_count: int) -> set[str]:
    """
    Return a set of text strings that appear on more than half the pages
    AND sit within the top or bottom 10% of their page.
    """
    if page_count < 2:
        return set()

    text_pages: dict[str, set[int]] = {}
    text_position: dict[str, str] = {}  # "top" | "bottom" | "middle"

    for b in raw_blocks:
        t = b["text"]
        ph = b["_page_height"]
        y0, y1 = b["bbox"][1], b["bbox"][3]
        mid_y = (y0 + y1) / 2

        if mid_y < ph * 0.10:
            pos = "top"
        elif mid_y > ph * 0.90:
            pos = "bottom"
        else:
            pos = "middle"

        text_pages.setdefault(t, set()).add(b["page"])
        # keep the most extreme position seen for this text
        if t not in text_position or pos != "middle":
            text_position[t] = pos

    threshold = page_count / 2
    return {
        t
        for t, pages in text_pages.items()
        if len(pages) > threshold and text_position.get(t) != "middle"
    }


def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Parse a PDF and return a list of text blocks with page and bbox.
    Repeating header/footer strings are removed.
    """
    raw_blocks, page_count = _load_raw_blocks(pdf_path)
    noise = _detect_repeating_headers_footers(raw_blocks, page_count)

    blocks = [
        {"page": b["page"], "text": b["text"], "bbox": b["bbox"]}
        for b in raw_blocks
        if b["text"] not in noise
    ]

    logger.info(
        "parse_pdf: %s → %d blocks across %d pages (%d header/footer strings removed)",
        pdf_path,
        len(blocks),
        page_count,
        len(noise),
    )
    return blocks


def save_parsed_blocks(document_id: str, blocks: list[dict]) -> Path:
    """
    Persist blocks as JSON. Assigns a stable block_id to each block so that
    downstream steps (image-extraction, semantic-chunking) can reference them.
    """
    parsed_dir = Path(settings.static_dir).parent / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    stamped = [
        {**b, "block_id": f"{document_id}_block_{i:04d}"}
        for i, b in enumerate(blocks)
    ]

    out_path = parsed_dir / f"{document_id}.json"
    out_path.write_text(json.dumps(stamped, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_parsed_blocks(document_id: str) -> list[dict]:
    parsed_path = Path(settings.static_dir).parent / "parsed" / f"{document_id}.json"
    return json.loads(parsed_path.read_text(encoding="utf-8"))
