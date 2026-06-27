"""
Tests for DecompositionTransformer — 8 test cases.
Dùng MockLLM, không cần NVIDIA API key.
"""
from types import SimpleNamespace

import pytest

from app.services.query_transform.base import BaseTransformer
from app.services.query_transform.decomposition import DecompositionTransformer
from app.services.query_transform import get_transformer


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

def _make_llm(reply: str):
    msg = SimpleNamespace(content=reply)
    choice = SimpleNamespace(message=msg)
    response = SimpleNamespace(choices=[choice])

    class MockLLM:
        def chat(self, messages, stream=False, tools=None):
            return response

    return MockLLM()


# ---------------------------------------------------------------------------
# Test 1: Protocol conformance
# ---------------------------------------------------------------------------

def test_protocol_conformance():
    t = DecompositionTransformer(_make_llm(""))
    assert isinstance(t, BaseTransformer)


# ---------------------------------------------------------------------------
# Test 2: transform() trả (original, original)
# ---------------------------------------------------------------------------

def test_transform_returns_original_question():
    t = DecompositionTransformer(_make_llm("1. Câu con A\n2. Câu con B"))
    search_q, gen_q = t.transform("Câu hỏi gốc?")
    assert search_q == "Câu hỏi gốc?"
    assert gen_q    == "Câu hỏi gốc?"


# ---------------------------------------------------------------------------
# Test 3: sub_questions được lưu sau transform()
# ---------------------------------------------------------------------------

def test_sub_questions_stored_after_transform():
    t = DecompositionTransformer(_make_llm("1. Câu A\n2. Câu B\n3. Câu C"))
    t.transform("Câu hỏi phức tạp?")
    assert len(t.sub_questions) == 3
    assert t.sub_questions[0] == "Câu A"
    assert t.sub_questions[1] == "Câu B"
    assert t.sub_questions[2] == "Câu C"


# ---------------------------------------------------------------------------
# Test 4: Xoá số thứ tự đầu dòng (1. / 1) / dòng trống)
# ---------------------------------------------------------------------------

def test_strips_leading_numbers():
    t = DecompositionTransformer(_make_llm("1. Câu hỏi thứ nhất\n2) Câu hỏi thứ hai"))
    subs = t.decompose("gốc")
    assert subs[0] == "Câu hỏi thứ nhất"
    assert subs[1] == "Câu hỏi thứ hai"


# ---------------------------------------------------------------------------
# Test 5: Tối đa 3 câu dù LLM trả nhiều hơn
# ---------------------------------------------------------------------------

def test_max_3_sub_questions():
    reply = "1. A\n2. B\n3. C\n4. D\n5. E"
    t = DecompositionTransformer(_make_llm(reply))
    subs = t.decompose("câu hỏi dài")
    assert len(subs) <= 3


# ---------------------------------------------------------------------------
# Test 6: Câu hỏi đơn giản — LLM trả lại đúng câu gốc
# ---------------------------------------------------------------------------

def test_simple_question_passthrough():
    t = DecompositionTransformer(_make_llm("Học máy là gì?"))
    subs = t.decompose("Học máy là gì?")
    assert subs == ["Học máy là gì?"]


# ---------------------------------------------------------------------------
# Test 7: Bỏ qua dòng trống trong output LLM
# ---------------------------------------------------------------------------

def test_empty_lines_filtered():
    reply = "1. Câu A\n\n\n2. Câu B\n\n"
    t = DecompositionTransformer(_make_llm(reply))
    subs = t.decompose("gốc")
    assert len(subs) == 2
    assert "" not in subs


# ---------------------------------------------------------------------------
# Test 8: factory get_transformer("decomposition") + llm_provider bắt buộc
# ---------------------------------------------------------------------------

def test_factory_decomposition_requires_llm():
    with pytest.raises(ValueError, match="llm_provider"):
        get_transformer("decomposition")


def test_factory_decomposition_with_llm():
    t = get_transformer("decomposition", llm_provider=_make_llm("1. Sub A\n2. Sub B"))
    assert isinstance(t, DecompositionTransformer)
    t.transform("câu hỏi?")
    assert t.sub_questions == ["Sub A", "Sub B"]
