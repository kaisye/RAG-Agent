"""
Tests for query_transform module — 9 test cases.
HyDETransformer dùng MockLLM, không cần NVIDIA API key.
"""
import pytest
from types import SimpleNamespace

from app.services.query_transform.base import BaseTransformer
from app.services.query_transform.identity import IdentityTransformer
from app.services.query_transform.hyde import HyDETransformer
from app.services.query_transform import get_transformer


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

def _make_llm(reply: str):
    """Tạo mock LLM trả về reply cố định."""
    msg = SimpleNamespace(content=reply)
    choice = SimpleNamespace(message=msg)
    response = SimpleNamespace(choices=[choice])

    class MockLLM:
        def chat(self, messages, stream=False, tools=None):
            return response

    return MockLLM()


# ---------------------------------------------------------------------------
# Test 1 & 2: IdentityTransformer
# ---------------------------------------------------------------------------

def test_identity_protocol():
    t = IdentityTransformer()
    assert isinstance(t, BaseTransformer)


def test_identity_returns_question_twice():
    t = IdentityTransformer()
    search_q, gen_q = t.transform("Học máy là gì?")
    assert search_q == "Học máy là gì?"
    assert gen_q    == "Học máy là gì?"


# ---------------------------------------------------------------------------
# Test 3-6: HyDETransformer
# ---------------------------------------------------------------------------

def test_hyde_protocol():
    t = HyDETransformer(_make_llm("hypothetical"))
    assert isinstance(t, BaseTransformer)


def test_hyde_returns_tuple():
    t = HyDETransformer(_make_llm("Đây là đoạn văn giả về học máy."))
    result = t.transform("Học máy là gì?")
    assert isinstance(result, tuple) and len(result) == 2


def test_hyde_search_query_is_hypothetical():
    hypo = "Học máy là lĩnh vực trí tuệ nhân tạo cho phép máy tính học từ dữ liệu."
    t = HyDETransformer(_make_llm(hypo))
    search_q, gen_q = t.transform("Học máy là gì?")
    # search dùng hypothetical doc (embed để tìm kiếm)
    assert search_q == hypo
    # generate dùng câu gốc
    assert gen_q == "Học máy là gì?"


def test_hyde_strips_whitespace():
    t = HyDETransformer(_make_llm("  có khoảng trắng  "))
    search_q, _ = t.transform("câu hỏi")
    assert search_q == "có khoảng trắng"


# ---------------------------------------------------------------------------
# Test 7-9: factory get_transformer()
# ---------------------------------------------------------------------------

def test_factory_none_returns_identity():
    t = get_transformer("none")
    assert isinstance(t, IdentityTransformer)


def test_factory_hyde_requires_llm():
    with pytest.raises(ValueError, match="llm_provider"):
        get_transformer("hyde")


def test_factory_hyde_with_llm():
    t = get_transformer("hyde", llm_provider=_make_llm("hypo"))
    assert isinstance(t, HyDETransformer)
    search_q, gen_q = t.transform("câu hỏi thử")
    assert search_q == "hypo"
    assert gen_q    == "câu hỏi thử"
