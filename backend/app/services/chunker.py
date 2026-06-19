import json
import logging
from pathlib import Path

import nltk

from app.core.config import settings

logger = logging.getLogger(__name__)

_TARGET_TOKENS = 400   # mid-point of 300-500 range
_MAX_TOKENS = 500
_OVERLAP_RATIO = 0.12  # 12% overlap


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _count_tokens(text: str) -> int:
    """Whitespace-based token count — lightweight and consistent."""
    return len(text.split())


def _is_heading(text: str) -> bool:
    """
    Heuristic: a block is a heading when it is short, has no sentence-ending
    punctuation at the end, and contains no newline in the middle.
    """
    stripped = text.strip()
    if len(stripped) > 120:
        return False
    if "\n" in stripped.rstrip("\n"):
        return False
    if stripped.endswith((".", "?", "!", ":")):
        return False
    words = stripped.split()
    if len(words) < 2:
        return False
    # Require most words to start with uppercase (title-case heuristic)
    upper_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
    return upper_ratio >= 0.5


# ---------------------------------------------------------------------------
# Core chunking
# ---------------------------------------------------------------------------

def _build_image_index(images: list[dict]) -> dict[str, list[str]]:
    """Map block_id -> [image_id, ...] for fast lookup during chunking."""
    idx: dict[str, list[str]] = {}
    for img in images:
        for bid in img.get("nearby_block_ids", []):
            idx.setdefault(bid, []).append(img["image_id"])
    return idx


def _sentences_from_blocks(blocks: list[dict]) -> list[dict]:
    """
    Split each block into sentences, propagating page and block_id.
    Returns list of {text, page, block_id}.
    """
    _ensure_nltk()
    sentences: list[dict] = []
    for block in blocks:
        raw = block["text"].replace("\n", " ").strip()
        if not raw:
            continue
        for sent in nltk.sent_tokenize(raw):
            sent = sent.strip()
            if sent:
                sentences.append(
                    {"text": sent, "page": block["page"], "block_id": block["block_id"]}
                )
    return sentences


def chunk_blocks(
    document_id: str,
    blocks: list[dict],
    images: list[dict],
) -> list[dict]:
    """
    Group blocks into semantic chunks with sentence-level overlap.

    Args:
        document_id: used for chunk IDs.
        blocks:      output of load_parsed_blocks() (includes block_id).
        images:      output of load_image_metadata() (includes nearby_block_ids).

    Returns:
        List of chunk dicts matching the contract.
    """
    image_index = _build_image_index(images)
    overlap_tokens = int(_TARGET_TOKENS * _OVERLAP_RATIO)

    # Group blocks by heading sections first, then split into sentences
    sections: list[list[dict]] = []
    current_section: list[dict] = []

    for block in blocks:
        if _is_heading(block["text"]) and current_section:
            sections.append(current_section)
            current_section = []
        current_section.append(block)
    if current_section:
        sections.append(current_section)

    chunks: list[dict] = []
    chunk_idx = 0

    # Accumulate sentences into chunks within each section
    carry_sentences: list[dict] = []  # overlap from previous chunk

    for section_blocks in sections:
        sentences = _sentences_from_blocks(section_blocks)
        window = carry_sentences + sentences
        carry_sentences = []

        buf: list[dict] = []
        buf_tokens = 0

        for sent in window:
            sent_tokens = _count_tokens(sent["text"])

            if buf_tokens + sent_tokens > _MAX_TOKENS and buf:
                # Emit current chunk
                chunk_text = " ".join(s["text"] for s in buf)
                pages = sorted({s["page"] for s in buf})
                block_ids_in_chunk = {s["block_id"] for s in buf}
                img_ids = list(dict.fromkeys(
                    iid
                    for bid in block_ids_in_chunk
                    for iid in image_index.get(bid, [])
                ))

                chunks.append(
                    {
                        "chunk_id": f"{document_id}_chunk_{chunk_idx:04d}",
                        "document_id": document_id,
                        "page": pages[0],
                        "pages": pages,
                        "text": chunk_text,
                        "image_ids": img_ids,
                    }
                )
                chunk_idx += 1

                # Carry overlap: keep trailing sentences whose total ≤ overlap budget
                overlap_buf: list[dict] = []
                overlap_count = 0
                for s in reversed(buf):
                    tc = _count_tokens(s["text"])
                    if overlap_count + tc > overlap_tokens:
                        break
                    overlap_buf.insert(0, s)
                    overlap_count += tc
                carry_sentences = overlap_buf
                buf = list(overlap_buf)
                buf_tokens = overlap_count

            buf.append(sent)
            buf_tokens += sent_tokens

        # Flush remaining sentences as a final chunk for this section
        if buf:
            chunk_text = " ".join(s["text"] for s in buf)
            pages = sorted({s["page"] for s in buf})
            block_ids_in_chunk = {s["block_id"] for s in buf}
            img_ids = list(dict.fromkeys(
                iid
                for bid in block_ids_in_chunk
                for iid in image_index.get(bid, [])
            ))
            chunks.append(
                {
                    "chunk_id": f"{document_id}_chunk_{chunk_idx:04d}",
                    "document_id": document_id,
                    "page": pages[0],
                    "pages": pages,
                    "text": chunk_text,
                    "image_ids": img_ids,
                }
            )
            chunk_idx += 1
            carry_sentences = []

    logger.info("chunker: %s → %d chunks", document_id, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _parsed_dir() -> Path:
    p = Path(settings.static_dir).parent / "parsed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_chunks(document_id: str, chunks: list[dict]) -> Path:
    out = _parsed_dir() / f"{document_id}_chunks.json"
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_chunks(document_id: str) -> list[dict]:
    p = _parsed_dir() / f"{document_id}_chunks.json"
    return json.loads(p.read_text(encoding="utf-8"))
