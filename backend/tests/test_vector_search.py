import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Skip guard — skip whole module if no API key or no pre-embedded fixtures
# ---------------------------------------------------------------------------
CHUNKS_PATH = Path(__file__).parent.parent / "storage" / "parsed" / "mm_test_chunks.json"
IMAGES_PATH = Path(__file__).parent.parent / "storage" / "parsed" / "mm_test_images.json"

if not os.getenv("NVIDIA_API_KEY"):
    pytest.skip("NVIDIA_API_KEY not set", allow_module_level=True)
if not CHUNKS_PATH.exists() or not IMAGES_PATH.exists():
    pytest.skip("mm_test embedded fixtures not found — run the pipeline manually first", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seeded_collection():
    """
    Configure settings for in-memory Qdrant, seed the collection with mm_test
    data, and return the total point count.
    """
    from app.core import config as cfg
    cfg.settings.qdrant_url = ":memory:"
    cfg.settings.qdrant_collection = "rag_chunks_test"
    cfg.settings.qdrant_vector_size = 1024

    # Reset singleton so it picks up the new settings
    import app.services.vector_store as vs
    vs._in_memory_client = None

    from app.services.chunker import load_chunks
    from app.services.image_extractor import load_image_metadata
    from app.services.vector_store import ensure_collection, upsert_chunks

    ensure_collection()
    chunks = load_chunks("mm_test")
    images = load_image_metadata("mm_test")
    total = upsert_chunks(chunks, images)
    return total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_collection_has_points(seeded_collection):
    assert seeded_collection > 0, "No points were upserted — check pipeline fixtures"


def test_vector_search_returns_results(seeded_collection):
    from app.services.retrieval import vector_search
    results = vector_search("neural network architecture", top_k=10)
    assert len(results) > 0


def test_scores_descending(seeded_collection):
    from app.services.retrieval import vector_search
    results = vector_search("convolutional layers feature learning", top_k=10)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results are not sorted by descending score"


def test_result_fields_present(seeded_collection):
    from app.services.retrieval import vector_search
    results = vector_search("diagram architecture", top_k=5)
    required = {"chunk_id", "document_id", "page", "type", "content", "source_path", "score"}
    for r in results:
        assert required <= r.keys(), f"Missing fields in result: {r.keys()}"


def test_type_values_valid(seeded_collection):
    from app.services.retrieval import vector_search
    results = vector_search("image diagram", top_k=10)
    for r in results:
        assert r["type"] in ("text", "image"), f"Unexpected type: {r['type']}"


def test_document_id_filter(seeded_collection):
    from app.services.retrieval import vector_search
    results_all = vector_search("neural network", top_k=10)
    results_filtered = vector_search("neural network", top_k=10, document_id="mm_test")
    # All filtered results must belong to mm_test
    for r in results_filtered:
        assert r["document_id"] == "mm_test"
    # Filtering should not return more results than unfiltered
    assert len(results_filtered) <= len(results_all)


def test_image_results_have_source_path(seeded_collection):
    from app.services.retrieval import vector_search
    results = vector_search("image figure diagram", top_k=10)
    for r in results:
        if r["type"] == "image":
            assert r["source_path"] is not None, "Image result missing source_path"


def test_text_results_have_content(seeded_collection):
    from app.services.retrieval import vector_search
    results = vector_search("convolutional network layer", top_k=10)
    for r in results:
        if r["type"] == "text":
            assert r["content"].strip(), "Text result has empty content"
