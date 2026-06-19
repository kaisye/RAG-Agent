import logging

from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.services.chunker import chunk_blocks, save_chunks
from app.services.image_extractor import extract_images, load_image_metadata, save_image_metadata
from app.services.pdf_parser import load_parsed_blocks, parse_pdf, save_parsed_blocks

logger = logging.getLogger(__name__)


async def _set_status(document_id: str, status: str) -> None:
    async with AsyncSessionLocal() as db:
        doc = await db.get(Document, document_id)
        if doc:
            doc.status = status
            await db.commit()


async def run_ingestion_pipeline(document_id: str, file_path: str) -> None:
    logger.info("ingestion start — document_id=%s", document_id)

    # Step 1: pdf-parser
    await _set_status(document_id, "parsing")
    try:
        blocks = parse_pdf(file_path)
        save_parsed_blocks(document_id, blocks)
        logger.info("pdf-parser done — %d blocks", len(blocks))
    except Exception:
        logger.exception("pdf-parser failed — document_id=%s", document_id)
        await _set_status(document_id, "failed")
        return

    # Step 2: image-extraction
    try:
        stamped_blocks = load_parsed_blocks(document_id)
        images = extract_images(document_id, file_path, stamped_blocks)
        save_image_metadata(document_id, images)
        logger.info("image-extraction done — %d images", len(images))
    except Exception:
        logger.exception("image-extraction failed — document_id=%s", document_id)
        await _set_status(document_id, "failed")
        return

    # Step 3: semantic-chunking
    await _set_status(document_id, "chunking")
    try:
        images_meta = load_image_metadata(document_id)
        chunks = chunk_blocks(document_id, stamped_blocks, images_meta)
        save_chunks(document_id, chunks)
        logger.info("semantic-chunking done — %d chunks", len(chunks))
    except Exception:
        logger.exception("semantic-chunking failed — document_id=%s", document_id)
        await _set_status(document_id, "failed")
        return

    await _set_status(document_id, "ready")
    logger.info("ingestion complete — document_id=%s", document_id)
