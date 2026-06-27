import json
import logging
from pathlib import Path

from sqlalchemy import update

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.services.parsing import parse_document

logger = logging.getLogger(__name__)

_PARSED_DIR = "storage/parsed"


async def _update_status(document_id: str, status: str, error: str | None = None) -> None:
    async with AsyncSessionLocal() as session:
        values: dict = {"status": status}
        if error is not None:
            values["error_message"] = error
        await session.execute(
            update(Document).where(Document.id == document_id).values(**values)
        )
        await session.commit()


async def run_ingestion_pipeline(document_id: str, file_path: str) -> None:
    logger.info("Pipeline starting for %s", document_id)
    settings = get_settings()

    try:
        # Step 1 — parse
        await _update_status(document_id, "parsing")
        result = parse_document(file_path, document_id, settings.images_dir)

        # Persist intermediate JSON
        parsed_dir = Path(_PARSED_DIR)
        parsed_dir.mkdir(parents=True, exist_ok=True)
        parsed_path = parsed_dir / f"{document_id}.json"
        parsed_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved parsed JSON to %s", parsed_path)

        # Subsequent steps (chunking, embedding, indexing) added in later branches
        logger.info("Pipeline step 1 (parsing) complete for %s — %d blocks, %d images",
                    document_id,
                    len(result["text_blocks"]),
                    len(result["images"]))

    except Exception as exc:
        logger.exception("Pipeline failed for %s", document_id)
        await _update_status(document_id, "failed", error=str(exc))
        raise
