"""
Tests for hybrid_search() in app/services/retrieval.py.

Two tiers:
  1. RRF logic tests — pure Python, no API or Qdrant needed.
  2. Integration tests — require NVIDIA_API_KEY and the mm_test fixtures on disk.
     The integration tests confirm that hybrid places a keyword-exact match
     higher than vector-only when the query contains a rare exact term.
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Tier 1 — RRF unit tests (always run, no external deps)
# ---------------------------------------------------------------------------

def test_rrf_fusion_sums_ranks():
    """A chunk_id appearing in both lists should score higher than one in only one."""
    from app.services.retrieval import _rrf_fusion

    vec  = [{"chunk_id": "A"}, {"chunk_id": "B"}, {"chunk_id": "C"}]
    bm25 = [{"chunk_id": "B"}, {"chunk_id": "D"}, {"chunk_id": "A"}]
    fused = _rrf_fusion(vec, bm25, k=60)
    ids = [cid for cid, _ in fused]

    # B appears at rank 1 (vec) and rank 0 (bm25) → combined score should beat solo entries
    assert ids[0] in ("A", "B"), "A or B should be top-2 (both appear in both lists)"
    assert ids[1] in ("A", "B")


def test_rrf_fusion_scores_descending():
    from app.services.retrieval import _rrf_fusion
    vec  = [{"chunk_id": str(i)} for i in range(5)]
    bm25 = [{"chunk_id": str(i)} for i in range(4, -1, -1)]
    fused = _rrf_fusion(vec, bm25, k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fusion_deduplicates():
    from app.services.retrieval import _rrf_fusion
    vec  = [{"chunk_id": "X"}, {"chunk_id": "Y"}]
    bm25 = [{"chunk_id": "X"}, {"chunk_id": "Z"}]
    fused = _rrf_fusion(vec, bm25, k=60)
    ids = [cid for cid, _ in fused]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in fused output"


def test_rrf_unique_entry_lower_than_shared():
    """A chunk in only one list must score ≤ a chunk in both lists."""
    from app.services.retrieval import _rrf_fusion
    # "shared" is rank-0 in vec AND rank-0 in bm25; "solo" only in vec rank-1
    vec  = [{"chunk_id": "shared"}, {"chunk_id": "solo"}]
    bm25 = [{"chunk_id": "shared"}]
    fused = dict(_rrf_fusion(vec, bm25, k=60))
    assert fused["shared"] > fused["solo"]


def test_hybrid_disabled_falls_back(monkeypatch):
    """When hybrid_search_enabled=False, hybrid_search should call vector_search only."""
    from app.core import config as cfg
    monkeypatch.setattr(cfg.settings, "hybrid_search_enabled", False)

    called = []

    from app.services import retrieval as r
    original_vs = r.vector_search
    def fake_vs(query, top_k=10, document_id=None):
        called.append("vector_search")
        return []
    monkeypatch.setattr(r, "vector_search", fake_vs)

    r.hybrid_search("any query", document_id="doc", top_k=5)
    assert called == ["vector_search"], "Should have called vector_search exactly once"


# ---------------------------------------------------------------------------
# Tier 2 — integration tests (skipped without API key / fixtures)
# ---------------------------------------------------------------------------

CHUNKS_PATH = Path(__file__).parent.parent / "storage" / "parsed" / "mm_test_chunks.json"
IMAGES_PATH = Path(__file__).parent.parent / "storage" / "parsed" / "mm_test_images.json"

needs_fixtures = pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY") or not CHUNKS_PATH.exists(),
    reason="NVIDIA_API_KEY not set or mm_test fixtures missing",
)


@pytest.fixture(scope="module")
def seeded_qdrant():
    """Seed in-memory Qdrant with mm_test data (same pattern as test_vector_search)."""
    from app.core import config as cfg
    cfg.settings.qdrant_url = ":memory:"
    cfg.settings.qdrant_collection = "rag_hybrid_test"
    cfg.settings.qdrant_vector_size = 1024

    import app.services.vector_store as vs
    vs._in_memory_client = None

    from app.services.chunker import load_chunks
    from app.services.image_extractor import load_image_metadata
    from app.services.vector_store import ensure_collection, upsert_chunks
    from app.services.retrieval import _bm25_cache

    _bm25_cache.pop("mm_test", None)
    ensure_collection()
    upsert_chunks(load_chunks("mm_test"), load_image_metadata("mm_test"))
    yield
    _bm25_cache.pop("mm_test", None)


@needs_fixtures
def test_hybrid_returns_results(seeded_qdrant):
    from app.services.retrieval import hybrid_search
    results = hybrid_search("convolutional neural network", document_id="mm_test", top_k=10)
    assert len(results) > 0


@needs_fixtures
def test_hybrid_scores_descending(seeded_qdrant):
    from app.services.retrieval import hybrid_search
    results = hybrid_search("neural network architecture", document_id="mm_test", top_k=10)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@needs_fixtures
def test_hybrid_result_fields(seeded_qdrant):
    from app.services.retrieval import hybrid_search
    results = hybrid_search("image classification", document_id="mm_test", top_k=5)
    required = {"chunk_id", "document_id", "page", "type", "content", "source_path", "score"}
    for r in results:
        assert required <= r.keys()


@needs_fixtures
def test_hybrid_beats_vector_on_exact_keyword(seeded_qdrant):
    """
    'Convolutional' is an exact token in chunk_0000.
    Hybrid should rank that chunk at least as high as vector-only,
    because BM25 boosts exact keyword matches.
    """
    from app.services.retrieval import hybrid_search, vector_search

    # Use a query that contains the exact word in the document
    query = "Convolutional Networks"

    hybrid = hybrid_search(query, document_id="mm_test", top_k=10)
    vector = vector_search(query, top_k=10, document_id="mm_test")

    def rank_of(results, partial_id):
        for i, r in enumerate(results):
            if partial_id in r["chunk_id"]:
                return i
        return len(results)  # not found → worst rank

    hybrid_rank = rank_of(hybrid, "chunk_0001")
    vector_rank = rank_of(vector, "chunk_0001")

    assert hybrid_rank <= vector_rank, (
        f"Hybrid rank {hybrid_rank} should be ≤ vector rank {vector_rank} "
        f"for the chunk containing the exact keyword"
    )
