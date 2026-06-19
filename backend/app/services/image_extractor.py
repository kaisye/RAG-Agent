import json
import logging
from pathlib import Path

import fitz  # PyMuPDF

from app.core.config import settings

logger = logging.getLogger(__name__)

_MIN_SIDE_PX = 50        # images smaller than this on either axis are decorative
_NEARBY_MAX_DIST = 80    # points; same heuristic as described in skill


def _find_nearby_block_ids(
    img_bbox: list[float],
    page_blocks: list[dict],
) -> list[str]:
    """Return block_ids whose y-range is within _NEARBY_MAX_DIST of the image."""
    img_y0, img_y1 = img_bbox[1], img_bbox[3]
    result = []
    for block in page_blocks:
        b_y0, b_y1 = block["bbox"][1], block["bbox"][3]
        distance = min(abs(b_y1 - img_y0), abs(img_y1 - b_y0))
        if distance <= _NEARBY_MAX_DIST:
            result.append(block["block_id"])
    return result


def extract_images(
    document_id: str,
    pdf_path: str,
    text_blocks: list[dict],
) -> list[dict]:
    images_dir = Path(settings.static_dir).parent / "images" / document_id
    images_dir.mkdir(parents=True, exist_ok=True)

    # Group text blocks by page for fast lookup
    blocks_by_page: dict[int, list[dict]] = {}
    for b in text_blocks:
        blocks_by_page.setdefault(b["page"], []).append(b)

    doc = fitz.open(pdf_path)
    seen_xrefs: set[int] = set()   # deduplicate images referenced on multiple pages
    results: list[dict] = []

    for page_num, page in enumerate(doc):
        for img_index, img_info in enumerate(page.get_image_info(xrefs=True)):
            xref = img_info.get("xref", 0)
            if xref == 0 or xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            bbox = list(img_info["bbox"])   # [x0, y0, x1, y1] in page coords
            width = img_info.get("width", 0)
            height = img_info.get("height", 0)

            if min(width, height) < _MIN_SIDE_PX:
                logger.debug(
                    "skip tiny image xref=%d page=%d (%dx%d)", xref, page_num, width, height
                )
                continue

            base_image = doc.extract_image(xref)
            if not base_image:
                continue

            ext = base_image["ext"]
            filename = f"p{page_num}_{img_index}.{ext}"
            file_path = images_dir / filename
            file_path.write_bytes(base_image["image"])

            storage_rel = f"storage/images/{document_id}/{filename}"
            image_id = f"{document_id}_img_p{page_num}_{img_index}"

            nearby_ids = _find_nearby_block_ids(
                bbox, blocks_by_page.get(page_num, [])
            )

            results.append(
                {
                    "image_id": image_id,
                    "document_id": document_id,
                    "page": page_num,
                    "bbox": bbox,
                    "file_path": storage_rel,
                    "nearby_block_ids": nearby_ids,
                }
            )
            logger.debug(
                "extracted %s (%dx%d) nearby_blocks=%s",
                image_id, width, height, nearby_ids,
            )

    doc.close()
    logger.info(
        "image-extraction: %s → %d images saved", document_id, len(results)
    )
    return results


def save_image_metadata(document_id: str, images: list[dict]) -> Path:
    parsed_dir = Path(settings.static_dir).parent / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    out_path = parsed_dir / f"{document_id}_images.json"
    out_path.write_text(json.dumps(images, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_image_metadata(document_id: str) -> list[dict]:
    path = Path(settings.static_dir).parent / "parsed" / f"{document_id}_images.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
