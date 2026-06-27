"""
Unit tests for RAGPipeline — 7 test cases.
Dùng Mock hoàn toàn, không cần NVIDIA API, ChromaDB, hay underthesea.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import PipelineConfig
from app.services.pipeline import RAGPipeline, _deduplicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(cid: str, text: str = "text", page: int = 1) -> dict:
    return {"chunk_id": cid, "document_id": "doc1", "page": page,
            "text": text, "score": 0.9}


def _make_delta(content: str):
    delta = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _mock_llm(reply: str = "mocked answer", stream_tokens=None):
    llm = MagicMock()
    # non-streaming
    msg = SimpleNamespace(content=reply)
    choice = SimpleNamespace(message=msg)
    llm.chat.return_value = SimpleNamespace(choices=[choice])
    return llm


def _mock_stream_llm(tokens: list[str]):
    llm = MagicMock()
    llm.chat.return_value = [_make_delta(t) for t in tokens]
    return llm


# ---------------------------------------------------------------------------
# Patch all external dependencies so tests don't hit APIs or disk
# ---------------------------------------------------------------------------

PATCH_TARGETS = {
    "app.services.pipeline.EmbeddingService": MagicMock,
    "app.services.pipeline.VectorStoreService": MagicMock,
    "app.services.pipeline.get_llm_provider": None,  # set per test
}


def _build_pipeline(config: PipelineConfig, llm=None, retriever_results=None, transformer_result=None):
    """Build RAGPipeline with all external deps mocked."""
    mock_llm = llm or _mock_llm()

    with patch("app.services.pipeline.EmbeddingService"), \
         patch("app.services.pipeline.VectorStoreService"), \
         patch("app.services.pipeline.get_llm_provider", return_value=mock_llm), \
         patch("app.services.pipeline.get_transformer") as mock_transform_factory, \
         patch("app.services.pipeline.get_retriever") as mock_retriever_factory, \
         patch("app.services.pipeline.get_reranker") as mock_reranker_factory:

        # Mock transformer
        mock_transformer = MagicMock()
        mock_transformer.transform.return_value = transformer_result or ("query", "query")
        mock_transformer.sub_questions = []
        mock_transform_factory.return_value = mock_transformer

        # Mock retriever (dense only via "vector" path)
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = retriever_results or [_chunk("c1"), _chunk("c2")]
        mock_retriever_factory.return_value = mock_retriever

        # Mock reranker — returns candidates unchanged
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = lambda q, cands, top_k=None: cands[:top_k or 3]
        mock_reranker_factory.return_value = mock_reranker

        pipeline = RAGPipeline(config)
        # inject mocks for post-init use
        pipeline._transformer = mock_transformer
        pipeline._retriever = mock_retriever
        pipeline._reranker = mock_reranker
        pipeline._llm = mock_llm

    return pipeline


# ---------------------------------------------------------------------------
# Test 1: _deduplicate utility
# ---------------------------------------------------------------------------

def test_deduplicate():
    chunks = [_chunk("a"), _chunk("b"), _chunk("a"), _chunk("c")]
    result = _deduplicate(chunks)
    assert [c["chunk_id"] for c in result] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Test 2: retrieve() — vector strategy, no transform, no rerank
# ---------------------------------------------------------------------------

def test_retrieve_vector_baseline():
    config = PipelineConfig(retrieval_strategy="vector", query_transform="none",
                            rerank_strategy="none", top_k_retrieval=5, top_k_final=3)
    pipeline = _build_pipeline(config)
    pipeline._transformer.transform.return_value = ("question", "question")
    pipeline._retriever.search.return_value = [_chunk(f"c{i}") for i in range(5)]
    pipeline._reranker.rerank.side_effect = lambda q, c, top_k=None: c[:3]

    results = pipeline.retrieve("question", "doc1")
    assert len(results) == 3
    pipeline._retriever.search.assert_called_once_with("question", "doc1", top_k=5)


# ---------------------------------------------------------------------------
# Test 3: retrieve() — HyDE embeds hypothetical_doc, reranks with original
# ---------------------------------------------------------------------------

def test_retrieve_hyde_uses_hypothetical_for_search_original_for_rerank():
    config = PipelineConfig(retrieval_strategy="vector", query_transform="hyde",
                            rerank_strategy="none", top_k_retrieval=5, top_k_final=3)
    pipeline = _build_pipeline(config)
    pipeline._transformer.transform.return_value = ("hypothetical doc text", "original question")
    pipeline._retriever.search.return_value = [_chunk("c1")]
    pipeline._reranker.rerank.side_effect = lambda q, c, top_k=None: c[:top_k]

    pipeline.retrieve("original question", "doc1")

    # search phải nhận hypothetical_doc
    pipeline._retriever.search.assert_called_once_with("hypothetical doc text", "doc1", top_k=5)
    # rerank phải nhận original_question
    call_args = pipeline._reranker.rerank.call_args
    assert call_args[0][0] == "original question"


# ---------------------------------------------------------------------------
# Test 4: retrieve() — Decomposition retrieves sub-questions + dedup
# ---------------------------------------------------------------------------

def test_retrieve_decomposition_sub_questions():
    from app.services.query_transform.decomposition import DecompositionTransformer

    config = PipelineConfig(retrieval_strategy="vector", query_transform="decomposition",
                            rerank_strategy="none", top_k_retrieval=5, top_k_final=10)
    pipeline = _build_pipeline(config)

    # Setup DecompositionTransformer mock
    mock_decomp = MagicMock(spec=DecompositionTransformer)
    mock_decomp.transform.return_value = ("original", "original")
    mock_decomp.sub_questions = ["sub A", "sub B"]
    pipeline._transformer = mock_decomp

    # Main search returns c1,c2; sub_question searches return c2(dup),c3; c4
    call_count = [0]
    def search_side_effect(q, doc_id, top_k=10):
        call_count[0] += 1
        if call_count[0] == 1:  return [_chunk("c1"), _chunk("c2")]  # original
        if call_count[0] == 2:  return [_chunk("c2"), _chunk("c3")]  # sub A (c2 dup)
        return [_chunk("c4")]                                          # sub B
    pipeline._retriever.search.side_effect = search_side_effect
    pipeline._reranker.rerank.side_effect = lambda q, c, top_k=None: c

    results = pipeline.retrieve("original", "doc1")

    # 3 searches: original + 2 sub-questions
    assert pipeline._retriever.search.call_count == 3
    # dedup: c1, c2, c3, c4 (c2 dup removed)
    ids = [r["chunk_id"] for r in results]
    assert len(set(ids)) == len(ids), f"Duplicates found: {ids}"
    assert "c1" in ids and "c2" in ids and "c3" in ids and "c4" in ids


# ---------------------------------------------------------------------------
# Test 5: generate() — yields tokens from streaming LLM
# ---------------------------------------------------------------------------

def test_generate_yields_tokens():
    config = PipelineConfig()
    pipeline = _build_pipeline(config)
    pipeline._llm = _mock_stream_llm(["Hello", " world", "!"])

    tokens = list(pipeline.generate("câu hỏi", [_chunk("c1", "context text")]))
    assert tokens == ["Hello", " world", "!"]


# ---------------------------------------------------------------------------
# Test 6: generate() — prompt chứa context + question
# ---------------------------------------------------------------------------

def test_generate_prompt_contains_context_and_question():
    config = PipelineConfig()
    pipeline = _build_pipeline(config)
    pipeline._llm = _mock_stream_llm([])

    list(pipeline.generate("câu hỏi thử", [_chunk("c1", "đoạn văn context")]))

    call_args = pipeline._llm.chat.call_args
    messages = call_args[0][0]
    full_text = " ".join(m["content"] for m in messages)
    assert "đoạn văn context" in full_text
    assert "câu hỏi thử" in full_text


# ---------------------------------------------------------------------------
# Test 7: run() — trả (contexts, generator)
# ---------------------------------------------------------------------------

def test_run_returns_contexts_and_generator():
    config = PipelineConfig()
    pipeline = _build_pipeline(config)
    pipeline._transformer.transform.return_value = ("q", "q")
    pipeline._retriever.search.return_value = [_chunk("c1")]
    pipeline._reranker.rerank.side_effect = lambda q, c, top_k=None: c
    pipeline._llm = _mock_stream_llm(["token1"])

    contexts, gen = pipeline.run("query", "doc1")
    assert isinstance(contexts, list)
    assert len(contexts) > 0
    tokens = list(gen)
    assert tokens == ["token1"]
