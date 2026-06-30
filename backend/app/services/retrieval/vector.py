import logging

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Vector (dense) retrieval dùng NVIDIA embedding + ChromaDB cosine search.

    QUAN TRỌNG: input_type="query" khi search — khác với "passage" lúc ingest.
    Nhầm hai giá trị không gây lỗi runtime nhưng làm cosine similarity sai lệch âm thầm.

    Kết quả thực nghiệm (baseline): Faithfulness=0.73, AR=0.53, CP=0.81, CR=0.67
    """

    def __init__(self, embed_service, vector_store):
        self.embed_service = embed_service
        self.vector_store = vector_store

    def search(self, query: str, document_id: str, top_k: int = 10) -> list[dict]:
        # input_type="query" khi embed câu hỏi — không được nhầm sang "passage"
        query_vec = self.embed_service.embed_texts([query], input_type="query")[0]
        results = self.vector_store.search(document_id, query_vec, top_k)
        logger.debug("DenseRetriever: query=%r doc=%s top_k=%d -> %d results",
                     query[:60], document_id, top_k, len(results))
        return results
