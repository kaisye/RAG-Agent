import logging

logger = logging.getLogger(__name__)


async def run_ingestion_pipeline(document_id: str, file_path: str) -> None:
    logger.info("Pipeline starting for %s", document_id)
