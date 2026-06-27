import json
import logging
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_blocks(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    raw: list[dict] = []

    for page_num, page in enumerate(doc):
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, _block_no, block_type = block
            if block_type != 0 or not text.strip():
                continue
            raw.append({
                "page": page_num + 1,
                "text": text.strip(),
                "bbox": [x0, y0, x1, y1],
            })

    doc.close()
    return _filter_repeated_blocks(raw, num_pages)


def _filter_repeated_blocks(blocks: list[dict], num_pages: int) -> list[dict]:
    # Header/footer: same text prefix appears on >70% of pages
    text_count = Counter(b["text"][:80] for b in blocks)
    repeated = {t for t, c in text_count.items() if c > num_pages * 0.7}
    filtered = [b for b in blocks if b["text"][:80] not in repeated]
    removed = len(blocks) - len(filtered)
    if removed:
        logger.debug("Filtered %d repeated header/footer blocks", removed)
    return filtered


def extract_images(pdf_path: str, document_id: str, output_dir: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    out_dir = Path(output_dir) / document_id
    out_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict] = []

    for page_num, page in enumerate(doc):
        for idx, img_info in enumerate(page.get_image_info(xrefs=True)):
            xref = img_info.get("xref", 0)
            if not xref:
                continue
            bbox = img_info["bbox"]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < 50 or h < 50:
                continue  # skip icons / watermarks

            base = doc.extract_image(xref)
            if not base:
                continue

            fname = f"p{page_num + 1}_{idx}.{base['ext']}"
            (out_dir / fname).write_bytes(base["image"])

            images.append({
                "image_id": f"{document_id}_p{page_num + 1}_{idx}",
                "document_id": document_id,
                "page": page_num + 1,
                "bbox": list(bbox),
                "file_path": str(out_dir / fname),
            })

    doc.close()
    logger.info("Extracted %d images from %s", len(images), pdf_path)
    return images


def parse_document(pdf_path: str, document_id: str, images_dir: str) -> dict:
    logger.info("Parsing document %s", document_id)
    text_blocks = extract_text_blocks(pdf_path)
    images = extract_images(pdf_path, document_id, images_dir)
    logger.info("Parsed %d text blocks, %d images", len(text_blocks), len(images))
    return {"text_blocks": text_blocks, "images": images}
