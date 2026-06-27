import json
import logging
from pathlib import Path

from sqlalchemy import update

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.services.chunking import get_chunker
from app.services.embedding import EmbeddingService
from app.services.parsing import parse_document
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

_PARSED_DIR = "storage/parsed"
_CHUNKS_DIR = "storage/chunks"


async def _update_status(
    document_id: str,
    status: str,
    num_chunks: int | None = None,
    error: str | None = None,
) -> None:
    async with AsyncSessionLocal() as session:
        values: dict = {"status": status}
        if num_chunks is not None:
            values["num_chunks"] = num_chunks
        if error is not None:
            values["error_message"] = error
        await session.execute(
            update(Document).where(Document.id == document_id).values(**values)
        )
        await session.commit()


async def run_ingestion_pipeline(document_id: str, file_path: str) -> None:
    logger.info("[%s] Pipeline start", document_id)
    settings = get_settings()

    try:
        # ------------------------------------------------------------------ #
        # Step 1 — Parse: text blocks + images                                #
        # ------------------------------------------------------------------ #
        await _update_status(document_id, "parsing")
        logger.info("[%s] Step 1/4: parsing %s", document_id, file_path)

        result = parse_document(file_path, document_id, settings.images_dir)
        text_blocks = result["text_blocks"]

        parsed_dir = Path(_PARSED_DIR)
        parsed_dir.mkdir(parents=True, exist_ok=True)
        (parsed_dir / f"{document_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("[%s] Step 1 done: %d blocks, %d images",
                    document_id, len(text_blocks), len(result["images"]))

        # ------------------------------------------------------------------ #
        # Step 2 — Chunk                                                       #
        # ------------------------------------------------------------------ #
        await _update_status(document_id, "chunking")
        logger.info("[%s] Step 2/4: chunking (strategy=%s)", document_id, settings.chunking_strategy)

        chunker = get_chunker(settings.chunking_strategy)
        chunks = chunker.split(text_blocks, document_id)

        chunks_dir = Path(_CHUNKS_DIR)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        (chunks_dir / f"{document_id}.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("[%s] Step 2 done: %d chunks", document_id, len(chunks))

        # ------------------------------------------------------------------ #
        # Step 3 — Embed                                                       #
        # ------------------------------------------------------------------ #
        await _update_status(document_id, "embedding")
        logger.info("[%s] Step 3/4: embedding %d chunks", document_id, len(chunks))

        texts = [c["text"] for c in chunks]
        # input_type="passage" khi ingest — không được nhầm sang "query"
        embeddings = EmbeddingService().embed_texts(texts, input_type="passage")
        logger.info("[%s] Step 3 done: %d embeddings (dim=%d)",
                    document_id, len(embeddings), len(embeddings[0]) if embeddings else 0)

        # ------------------------------------------------------------------ #
        # Step 4 — Index                                                       #
        # ------------------------------------------------------------------ #
        await _update_status(document_id, "indexing")
        logger.info("[%s] Step 4/4: indexing into ChromaDB", document_id)

        VectorStoreService().upsert_chunks(document_id, chunks, embeddings)
        logger.info("[%s] Step 4 done: upserted to collection doc_%s", document_id, document_id)

        # ------------------------------------------------------------------ #
        # Done                                                                 #
        # ------------------------------------------------------------------ #
        await _update_status(document_id, "ready", num_chunks=len(chunks))
        logger.info("[%s] Pipeline complete: status=ready, num_chunks=%d", document_id, len(chunks))

    except Exception as exc:
        logger.exception("[%s] Pipeline failed: %s", document_id, exc)
        await _update_status(document_id, "failed", error=str(exc))
        raise
