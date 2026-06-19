"""
Tests for BM25 retrieval in app/services/retrieval.py.

BM25 is fully local (no API calls), so these tests run without NVIDIA_API_KEY.
We build a synthetic in-memory corpus using the same chunk format the pipeline
produces, so the tests are self-contained and fast.
"""

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers: build a minimal chunk corpus on disk so load_chunks() can read it
# ---------------------------------------------------------------------------

CORPUS = [
    {
        "chunk_id":    "doc_test_chunk_0000",
        "document_id": "doc_test",
        "page":        0,
        "pages":       [0],
        "text":        "Convolutional neural networks use spatial feature hierarchies for image classification.",
        "image_ids":   [],
    },
    {
        "chunk_id":    "doc_test_chunk_0001",
        "document_id": "doc_test",
        "page":        1,
        "pages":       [1],
        "text":        "Recurrent neural networks process sequential data with hidden state transitions.",
        "image_ids":   [],
    },
    {
        "chunk_id":    "doc_test_chunk_0002",
        "document_id": "doc_test",
        "page":        2,
        "pages":       [2],
        "text":        "Transformer architectures rely on self-attention mechanisms instead of recurrence.",
        "image_ids":   [],
    },
]


@pytest.fixture(scope="module")
def corpus_on_disk(tmp_path_factory):
    """Write CORPUS to a temp parsed dir and redirect settings for the test module."""
    from app.core import config as cfg

    storage = tmp_path_factory.mktemp("storage")
    parsed = storage / "parsed"
    parsed.mkdir()
    (parsed / "doc_test_chunks.json").write_text(
        json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8"
    )

    original_static_dir = cfg.settings.static_dir
    cfg.settings.static_dir = str(storage / "static")

    # Clear any cached BM25 index from previous test runs
    from app.services import retrieval
    retrieval._bm25_cache.pop("doc_test", None)

    yield parsed

    # Restore original setting so other test modules are not affected
    cfg.settings.static_dir = original_static_dir
    retrieval._bm25_cache.pop("doc_test", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_index_returns_entry(corpus_on_disk):
    from app.services.retrieval import build_bm25_index
    entry = build_bm25_index("doc_test")
    assert entry.index is not None
    assert len(entry.chunks) == 3


def test_index_is_cached(corpus_on_disk):
    from app.services.retrieval import build_bm25_index, _bm25_cache
    build_bm25_index("doc_test")
    first_obj = id(_bm25_cache["doc_test"].index)
    build_bm25_index("doc_test")
    assert id(_bm25_cache["doc_test"].index) == first_obj, "Index was rebuilt on second call"


def test_exact_keyword_match(corpus_on_disk):
    """'convolutional' should rank the first chunk highest."""
    from app.services.retrieval import bm25_search
    results = bm25_search("convolutional spatial feature", "doc_test", top_k=3)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "doc_test_chunk_0000"


def test_rare_term_match(corpus_on_disk):
    """'recurrence' is unique to chunk_0002; it should surface that chunk."""
    from app.services.retrieval import bm25_search
    results = bm25_search("recurrence attention", "doc_test", top_k=3)
    top_ids = [r["chunk_id"] for r in results]
    assert "doc_test_chunk_0002" in top_ids


def test_scores_descending(corpus_on_disk):
    from app.services.retrieval import bm25_search
    results = bm25_search("neural network sequential hidden state", "doc_test", top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_zero_score_results_excluded(corpus_on_disk):
    """A query that matches no tokens should return an empty list."""
    from app.services.retrieval import bm25_search
    results = bm25_search("zzz_nonexistent_token_xyz", "doc_test", top_k=3)
    assert all(r["score"] > 0 for r in results)


def test_result_fields_present(corpus_on_disk):
    from app.services.retrieval import bm25_search
    results = bm25_search("transformer self-attention", "doc_test", top_k=3)
    required = {"chunk_id", "document_id", "page", "type", "content", "source_path", "score"}
    for r in results:
        assert required <= r.keys()


def test_top_k_respected(corpus_on_disk):
    from app.services.retrieval import bm25_search
    results = bm25_search("neural network", "doc_test", top_k=1)
    assert len(results) <= 1


def test_content_contains_query_term(corpus_on_disk):
    """The top BM25 result for an exact term should contain that term."""
    from app.services.retrieval import bm25_search
    results = bm25_search("recurrent sequential", "doc_test", top_k=1)
    assert len(results) > 0
    assert "recurrent" in results[0]["content"].lower()
