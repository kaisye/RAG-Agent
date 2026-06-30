import os
from pathlib import Path

from app.core.config import get_settings

_SNIPPET_LEN = 150


def enrich_chunk(chunk: dict) -> dict:
    """
    Bổ sung citation fields vào một context chunk từ retriever.

    Input fields đã có: chunk_id, document_id, page, text, score, strategy, type (optional)
    Output bổ sung:     snippet, type ("text"|"image"), thumbnail_url (nếu ảnh)

    thumbnail_url trỏ đến /static/images/{document_id}/p{page}_{idx}.{ext}
    — mount từ storage/images/ qua StaticFiles("/static") trong main.py.
    """
    settings = get_settings()
    chunk_type = chunk.get("type", "text")

    enriched = dict(chunk)
    enriched["snippet"] = chunk.get("text", "")[:_SNIPPET_LEN]
    enriched["type"] = chunk_type

    if chunk_type == "image":
        # Tìm file ảnh theo pattern p{page}_*.* trong thư mục document
        doc_id = chunk.get("document_id", "")
        page = chunk.get("page", 1)
        img_dir = Path(settings.images_dir) / doc_id
        thumbnail_url = None
        if img_dir.is_dir():
            for f in sorted(img_dir.iterdir()):
                if f.name.startswith(f"p{page}_"):
                    # URL tương đối từ static mount root
                    rel = os.path.relpath(f, "storage").replace("\\", "/")
                    thumbnail_url = f"/static/{rel}"
                    break
        enriched["thumbnail_url"] = thumbnail_url
    else:
        enriched["thumbnail_url"] = None

    return enriched


def enrich_chunks(chunks: list[dict]) -> list[dict]:
    return [enrich_chunk(c) for c in chunks]
