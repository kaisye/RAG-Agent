import logging

logger = logging.getLogger(__name__)


def delete_document_vectors(document_id: str) -> None:
    # TODO: chưa implement — nhánh feature/qdrant sẽ xoá toàn bộ vector
    # theo filter document_id trong Qdrant collection.
    logger.info("TODO: chưa implement — delete_document_vectors document_id=%s", document_id)
