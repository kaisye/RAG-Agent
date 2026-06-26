import logging

logger = logging.getLogger(__name__)


async def delete_document_vectors(document_id: str) -> None:
    # Placeholder — feature/chroma-vector-store sẽ implement xoá collection doc_{document_id}
    logger.info("delete_document_vectors called for %s (no-op until chroma branch)", document_id)
