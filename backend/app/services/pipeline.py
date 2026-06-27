import logging
from typing import Generator

from app.core.config import PipelineConfig
from app.services.embedding import EmbeddingService
from app.services.llm import get_llm_provider
from app.services.query_transform import get_transformer
from app.services.query_transform.decomposition import DecompositionTransformer
from app.services.reranking import get_reranker
from app.services.retrieval import get_retriever
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp. "
    "Chỉ sử dụng thông tin trong các đoạn tài liệu dưới đây để trả lời. "
    "Nếu tài liệu không đủ thông tin, hãy nói rõ điều đó."
)

_USER_TEMPLATE = """\
[TÀI LIỆU]
{context}

[CÂU HỎI]
{question}

[TRẢ LỜI]"""


class _LazyBM25Retriever:
    """
    Wrapper cho BM25/hybrid retrievers khi RAGPipeline khởi tạo.

    BM25Okapi cần corpus để build index — không thể build lúc __init__ vì chưa có chunks.
    Lấy chunks từ ChromaDB khi search() được gọi lần đầu cho mỗi document_id.
    Cache index theo document_id để tránh re-build trong cùng session.
    """

    def __init__(self, strategy: str, dense_retriever, vector_store, rrf_k: int = 60):
        self.strategy = strategy
        self.dense = dense_retriever
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self._cache: dict[str, object] = {}  # document_id → built retriever

    def _get_retriever(self, document_id: str):
        if document_id in self._cache:
            return self._cache[document_id]

        # Fetch tất cả chunks từ collection ChromaDB
        col = self.vector_store.get_or_create_collection(document_id)
        count = col.count()
        if count == 0:
            logger.warning("_LazyBM25Retriever: collection doc_%s is empty", document_id)
            self._cache[document_id] = self.dense
            return self.dense

        raw = col.get(include=["documents", "metadatas"])
        chunks = [
            {
                "chunk_id":    raw["ids"][i],
                "document_id": document_id,
                "page":        raw["metadatas"][i].get("page", 1),
                "text":        raw["documents"][i],
                "score":       1.0,
                "strategy":    raw["metadatas"][i].get("strategy", "recursive"),
            }
            for i in range(len(raw["ids"]))
        ]
        logger.info("_LazyBM25Retriever: fetched %d chunks for doc_%s", len(chunks), document_id)

        from app.services.retrieval.bm25 import BM25Retriever
        bm25 = BM25Retriever(chunks)

        if self.strategy == "bm25":
            retriever = bm25
        elif self.strategy == "hybrid_interleaving":
            from app.services.retrieval.hybrid_interleaving import HybridInterleavingRetriever
            retriever = HybridInterleavingRetriever(bm25, self.dense)
        elif self.strategy == "hybrid_rrf":
            from app.services.retrieval.hybrid_rrf import HybridRRFRetriever
            retriever = HybridRRFRetriever(bm25, self.dense, rrf_k=self.rrf_k)
        else:
            retriever = self.dense

        self._cache[document_id] = retriever
        return retriever

    def search(self, query: str, document_id: str, top_k: int = 10) -> list[dict]:
        return self._get_retriever(document_id).search(query, document_id, top_k)


def _deduplicate(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for c in chunks:
        cid = c["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            result.append(c)
    return result


class RAGPipeline:
    """
    Orchestrator kết nối tất cả RAG strategies theo PipelineConfig.

    Luồng:
      retrieve(query, doc_id)
        → transformer.transform(query)         # "none"|"hyde"|"decomposition"
        → retriever.search(transformed, ...)   # "vector"|"bm25"|"hybrid_rrf"|...
        → [decomp sub-question retrieval]      # chỉ khi strategy="decomposition"
        → reranker.rerank(original, candidates) # "none"|"cross_encoder"|"mmr"
      generate(query, contexts)
        → llm.chat(stream=True)                # yield delta tokens
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._embed_svc = EmbeddingService()
        self._vector_store = VectorStoreService()
        self._llm = get_llm_provider()
        self._transformer = self._build_transformer()
        self._retriever = self._build_retriever()
        self._reranker = self._build_reranker()

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_retriever(self):
        strategy = self.config.retrieval_strategy
        dense = get_retriever("vector", chunks=[], embed_service=self._embed_svc, vector_store=self._vector_store)

        if strategy == "vector":
            return dense

        # BM25/hybrid cần chunks thực tế từ ChromaDB — dùng lazy wrapper fetch khi search()
        return _LazyBM25Retriever(strategy, dense, self._vector_store, self.config.rrf_k)

    def _build_transformer(self):
        return get_transformer(self.config.query_transform, llm_provider=self._llm)

    def _build_reranker(self):
        return get_reranker(
            self.config.rerank_strategy,
            top_k=self.config.top_k_final,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, document_id: str) -> list[dict]:
        """
        Retrieve top_k_final chunks cho query từ document_id.

        Trả về list[dict] mỗi item có: chunk_id, text, page, score, ...
        """
        # Step 1: transform query
        search_query, original_query = self._transformer.transform(query)
        logger.info(
            "RAGPipeline.retrieve: doc=%s strategy=%s transform=%s",
            document_id, self.config.retrieval_strategy, self.config.query_transform,
        )

        # Step 2: retrieve với transformed query (hypothetical_doc nếu HyDE)
        candidates = self._retriever.search(
            search_query, document_id, top_k=self.config.top_k_retrieval
        )

        # Step 3: Decomposition — thêm retrieve cho từng sub-question + câu gốc
        if isinstance(self._transformer, DecompositionTransformer) and self._transformer.sub_questions:
            for sq in self._transformer.sub_questions:
                candidates += self._retriever.search(sq, document_id, top_k=3)
            # Câu gốc phải được search nếu transformer đã return search_query=original
            # (DecompositionTransformer.transform() trả original_query nên search_query==original_query đã có ở step 2)
            candidates = _deduplicate(candidates)
            logger.info(
                "Decomposition: %d sub-questions -> %d candidates after dedup",
                len(self._transformer.sub_questions), len(candidates),
            )

        # Step 4: rerank với original query (không phải hypothetical_doc)
        results = self._reranker.rerank(original_query, candidates, top_k=self.config.top_k_final)
        logger.info(
            "RAGPipeline.retrieve: %d candidates -> %d final chunks",
            len(candidates), len(results),
        )
        return results

    def generate(self, query: str, contexts: list[dict]) -> Generator[str, None, None]:
        """
        Sinh câu trả lời streaming từ contexts.

        Yields từng delta token string. Caller dùng trong SSE hoặc gộp thành string.
        """
        context_text = "\n\n---\n\n".join(
            f"[Trang {c['page']}]\n{c['text']}" for c in contexts
        )
        user_content = _USER_TEMPLATE.format(context=context_text, question=query)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        logger.info(
            "RAGPipeline.generate: query=%r contexts=%d",
            query[:60], len(contexts),
        )

        response = self._llm.chat(messages, stream=True)
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    def run(self, query: str, document_id: str) -> Generator[str, None, None]:
        """
        Convenience: retrieve + generate trong một lần gọi.
        Yields (contexts, generator) — contexts để build citation payload ở nhánh 9.2.
        Caller: contexts, gen = pipeline.run(query, doc_id)
        """
        contexts = self.retrieve(query, document_id)
        return contexts, self.generate(query, contexts)
