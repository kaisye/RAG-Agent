import logging

from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.services.pdf_parser import parse_pdf, save_parsed_blocks

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

    await _set_status(document_id, "ready")
    logger.info("ingestion complete — document_id=%s", document_id)
