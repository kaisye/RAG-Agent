import json
import logging
from pathlib import Path

from sqlalchemy import update

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.services.chunking import get_chunker
from app.services.parsing import parse_document

logger = logging.getLogger(__name__)

_PARSED_DIR = "storage/parsed"
_CHUNKS_DIR = "storage/chunks"


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
        # ------------------------------------------------------------------ #
        # Step 1 — Parse                                                       #
        # ------------------------------------------------------------------ #
        await _update_status(document_id, "parsing")
        result = parse_document(file_path, document_id, settings.images_dir)

        parsed_dir = Path(_PARSED_DIR)
        parsed_dir.mkdir(parents=True, exist_ok=True)
        (parsed_dir / f"{document_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        text_blocks = result["text_blocks"]
        logger.info("Step 1 done: %d text blocks, %d images", len(text_blocks), len(result["images"]))

        # ------------------------------------------------------------------ #
        # Step 2 — Chunk                                                       #
        # ------------------------------------------------------------------ #
        await _update_status(document_id, "chunking")
        chunker = get_chunker(settings.chunking_strategy)
        chunks = chunker.split(text_blocks, document_id)

        chunks_dir = Path(_CHUNKS_DIR)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        (chunks_dir / f"{document_id}.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Persist chunk count to DB
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(num_chunks=len(chunks))
            )
            await session.commit()

        logger.info("Step 2 done: %d chunks (strategy=%s)", len(chunks), settings.chunking_strategy)

        # Embedding + indexing added in later branches
        await _update_status(document_id, "ready")
        logger.info("Pipeline complete for %s", document_id)

    except Exception as exc:
        logger.exception("Pipeline failed for %s", document_id)
        await _update_status(document_id, "failed", error=str(exc))
        raise
