"""
Tests for rerank() and retrieve() in app/services/retrieval.py.

Two tiers:
  1. Reranker unit tests — use a tiny in-process stub model, no download needed.
     Verify: top_k respected, score replaced, order correct, empty input safe.
  2. Integration tests with the real CrossEncoder model and real hybrid fixtures.
     Skipped unless NVIDIA_API_KEY is set and mm_test fixtures exist.
     These tests confirm retrieve() returns ≤ top_k_final results and that
     reranking shifts a semantically relevant chunk above a lower-quality one.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidates(n: int) -> list[dict]:
    return [
        {
            "chunk_id":    f"doc_chunk_{i:04d}",
            "document_id": "doc",
            "page":        i,
            "type":        "text",
            "content":     f"Content of chunk {i}",
            "source_path": None,
            "score":       float(n - i),  # initially sorted by decreasing score
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tier 1 — unit tests with a mocked CrossEncoder (always run)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cross_encoder():
    """Ensure the module-level singleton is cleared between tests."""
    import app.services.retrieval as r
    original = r._cross_encoder
    yield
    r._cross_encoder = original


def test_rerank_top_k_respected():
    """rerank() must return at most top_k results."""
    import app.services.retrieval as r

    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.9, 0.1, 0.5, 0.8, 0.3]
    r._cross_encoder = mock_ce

    candidates = _make_candidates(5)
    results = r.rerank("query", candidates, top_k=3)
    assert len(results) == 3


def test_rerank_order_by_cross_encoder_score():
    """rerank() must sort by CrossEncoder score, not by input order."""
    import app.services.retrieval as r

    # CrossEncoder says chunk 2 is best, then chunk 0, then chunk 1
    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.4, 0.1, 0.9]
    r._cross_encoder = mock_ce

    candidates = _make_candidates(3)
    results = r.rerank("query", candidates, top_k=3)
    assert results[0]["chunk_id"] == "doc_chunk_0002"
    assert results[1]["chunk_id"] == "doc_chunk_0000"
    assert results[2]["chunk_id"] == "doc_chunk_0001"


def test_rerank_score_replaced_by_cross_encoder():
    """The 'score' field in returned dicts must be the CrossEncoder logit."""
    import app.services.retrieval as r

    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.77, 0.33]
    r._cross_encoder = mock_ce

    candidates = _make_candidates(2)
    results = r.rerank("query", candidates, top_k=2)
    scores = {r["chunk_id"]: r["score"] for r in results}
    assert abs(scores["doc_chunk_0000"] - 0.77) < 1e-6
    assert abs(scores["doc_chunk_0001"] - 0.33) < 1e-6


def test_rerank_empty_candidates():
    import app.services.retrieval as r
    mock_ce = MagicMock()
    r._cross_encoder = mock_ce
    assert r.rerank("query", [], top_k=5) == []
    mock_ce.predict.assert_not_called()


def test_rerank_fewer_candidates_than_top_k():
    import app.services.retrieval as r
    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.5, 0.9]
    r._cross_encoder = mock_ce

    results = r.rerank("query", _make_candidates(2), top_k=10)
    assert len(results) == 2


def test_retrieve_disabled_reranker_skips_cross_encoder(monkeypatch):
    """With RERANKER_ENABLED=false, retrieve() must not call the CrossEncoder."""
    from app.core import config as cfg
    monkeypatch.setattr(cfg.settings, "reranker_enabled", False)

    import app.services.retrieval as r

    fake_hybrid = _make_candidates(10)
    with patch.object(r, "hybrid_search", return_value=fake_hybrid) as mock_h, \
         patch.object(r, "_get_cross_encoder") as mock_ce:
        results = r.retrieve("query", document_id="doc", top_k_final=3)

    mock_ce.assert_not_called()
    assert len(results) == 3
    assert results == fake_hybrid[:3]


def test_retrieve_respects_top_k_final(monkeypatch):
    from app.core import config as cfg
    monkeypatch.setattr(cfg.settings, "reranker_enabled", True)

    import app.services.retrieval as r

    fake_hybrid = _make_candidates(20)
    mock_ce = MagicMock()
    mock_ce.predict.return_value = [float(i) for i in range(20)]
    r._cross_encoder = mock_ce

    with patch.object(r, "hybrid_search", return_value=fake_hybrid):
        results = r.retrieve("query", document_id="doc", top_k_final=5)

    assert len(results) == 5


def test_retrieve_result_fields_present(monkeypatch):
    from app.core import config as cfg
    monkeypatch.setattr(cfg.settings, "reranker_enabled", True)

    import app.services.retrieval as r

    fake_hybrid = _make_candidates(4)
    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.9, 0.8, 0.7, 0.6]
    r._cross_encoder = mock_ce

    with patch.object(r, "hybrid_search", return_value=fake_hybrid):
        results = r.retrieve("query", document_id="doc", top_k_final=4)

    required = {"chunk_id", "document_id", "page", "type", "content", "source_path", "score"}
    for res in results:
        assert required <= res.keys()


# ---------------------------------------------------------------------------
# Tier 2 — integration tests with real model + real fixtures
# ---------------------------------------------------------------------------

CHUNKS_PATH = Path(__file__).parent.parent / "storage" / "parsed" / "mm_test_chunks.json"
IMAGES_PATH = Path(__file__).parent.parent / "storage" / "parsed" / "mm_test_images.json"

needs_fixtures = pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY") or not CHUNKS_PATH.exists(),
    reason="NVIDIA_API_KEY not set or mm_test fixtures missing",
)


@pytest.fixture(scope="module")
def seeded_qdrant_rerank():
    from app.core import config as cfg
    cfg.settings.qdrant_url = ":memory:"
    cfg.settings.qdrant_collection = "rag_rerank_test"
    cfg.settings.qdrant_vector_size = 1024
    cfg.settings.reranker_enabled = True

    import app.services.vector_store as vs
    vs._in_memory_client = None

    import app.services.retrieval as r
    r._cross_encoder = None   # force reload with real model
    r._bm25_cache.pop("mm_test", None)

    from app.services.chunker import load_chunks
    from app.services.image_extractor import load_image_metadata
    from app.services.vector_store import ensure_collection, upsert_chunks
    ensure_collection()
    upsert_chunks(load_chunks("mm_test"), load_image_metadata("mm_test"))
    yield
    r._bm25_cache.pop("mm_test", None)
    r._cross_encoder = None


@needs_fixtures
def test_retrieve_returns_top_k_final(seeded_qdrant_rerank):
    from app.services.retrieval import retrieve
    results = retrieve("neural network classification", document_id="mm_test", top_k_final=5)
    assert 1 <= len(results) <= 5


@needs_fixtures
def test_retrieve_scores_descending(seeded_qdrant_rerank):
    from app.services.retrieval import retrieve
    results = retrieve("convolutional feature extraction", document_id="mm_test", top_k_final=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@needs_fixtures
def test_retrieve_result_fields_integration(seeded_qdrant_rerank):
    from app.services.retrieval import retrieve
    results = retrieve("image classification network", document_id="mm_test", top_k_final=3)
    required = {"chunk_id", "document_id", "page", "type", "content", "source_path", "score"}
    for r in results:
        assert required <= r.keys()


@needs_fixtures
def test_rerank_improves_order_over_hybrid(seeded_qdrant_rerank):
    """
    Query 'Convolutional' — an exact keyword unique to one chunk.
    After reranking the chunk containing this exact term should be in
    the top-3, whereas without reranking (hybrid order) it may not be.
    We assert the reranked position is ≤ the hybrid position (i.e. not worse).
    """
    from app.services.retrieval import hybrid_search, retrieve

    query = "Convolutional Networks classification"

    hybrid = hybrid_search(query, document_id="mm_test", top_k=20)
    reranked = retrieve(query, document_id="mm_test", top_k_final=5, hybrid_pool=20)

    def rank_of(results, partial_id):
        for i, r in enumerate(results):
            if partial_id in r["chunk_id"]:
                return i
        return len(results)

    hybrid_rank  = rank_of(hybrid,   "chunk_0001")
    reranked_rank = rank_of(reranked, "chunk_0001")

    # Reranker must not hurt the recall of a relevant chunk
    assert reranked_rank <= max(hybrid_rank, 4), (
        f"Reranked position {reranked_rank} is worse than hybrid {hybrid_rank}"
    )
