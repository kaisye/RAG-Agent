import logging

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(document_id: str, file_path: str) -> None:
    # TODO: chưa implement — các nhánh pdf-parser, image-extraction,
    # semantic-chunking, embedding-service, qdrant sẽ lần lượt bổ sung vào đây.
    logger.info("TODO: chưa implement — document_id=%s file_path=%s", document_id, file_path)
