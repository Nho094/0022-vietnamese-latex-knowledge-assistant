"""Hồi quy các lỗi dữ liệu và luồng xử lý đầu-cuối được phát hiện khi audit."""
from __future__ import annotations

import io
import json
import re
from unittest.mock import patch

import pytest

from cli import ka
from core.config import DEFAULT_INTENTS_PATH
from core.pipeline import TfidfRanker, render_latex_template
from core.preprocess import canonicalize_retrieval_numbers
from core.slots import extract_slots, normalize_math
from core.validator import validate_intents


@pytest.fixture(scope="module")
def bank() -> dict[str, dict[str, object]]:
    return {item["intent"]: item for item in json.loads(DEFAULT_INTENTS_PATH.read_text())}


def test_math_normalizer_preserves_latex_commands_and_fraction():
    assert normalize_math(r"\sin(x)/x") == r"\sin(x)/x"
    assert normalize_math(r"\text{a b} + x") == r"\text{a b}+x"
    with pytest.raises(ValueError, match="không an toàn"):
        normalize_math(r"\input{secret}")
    intent = {
        "intent": "GENERIC_LIMIT",
        "slots": {"expr": {"type": "math", "regex": r"(?:biểu thức)\s+(.+)", "required": True}},
    }
    result = extract_slots(r"biểu thức \sin(x)/x", intent)
    assert result.slots["expr"] == r"\sin(x)/x"
    assert result.is_complete


@pytest.mark.parametrize(
    ("intent", "query", "expected"),
    [
        ("MATH_MATH_BASIC_TONG", "tổng từ i bằng 1 đến n của i^2", {"from": "i=1", "to": "n", "expr": "i^2"}),
        ("MATH_MATH_BASIC_CAN", "căn bậc hai của x", {"expr": "x"}),
        ("MATH_MATH_BASIC_PHAN_SO", "phân số tử số 2 mẫu số 3", {"num": "2", "den": "3"}),
        ("MATH_MATH_BASIC_GIOI_HAN", "giới hạn khi x tiến tới 0 của sin(x)/x", {"condition": r"x\to0", "expr": "sin(x)/x"}),
        ("MATH_MATH_BASIC_TICH_PHAN", "tích phân từ 0 đến 1 của x bình phương theo x", {"from": "0", "to": "1", "expr": "x^2"}),
        ("MATH_MATRIX_MA_TRAN_2X2", "ma trận 2 hàng 2 cột gồm 1, 2, 3, 4", {"a": "1", "b": "2", "c": "3", "d": "4"}),
    ],
)
def test_math_knowledge_extracts_typed_slots(bank, intent, query, expected):
    result = extract_slots(query, bank[intent], bank)
    assert result.is_complete, result.errors or result.missing
    for key, value in expected.items():
        assert result.slots[key] == value


def test_retrieval_distinguishes_matrix_from_tabular(bank):
    ranked = TfidfRanker(bank).rank("tạo ma trận 2 hàng 2 cột", top_k=3)
    assert ranked[0].intent == "MATH_MATRIX_MA_TRAN_2X2"
    assert canonicalize_retrieval_numbers("x^2 + 2x + 0.5") == "x^2 + 2x + 0.5"


def test_quadratic_and_fraction_render_through_templates(bank):
    from core.pipeline import render_latex_template

    fraction, fraction_ok, fraction_errors = render_latex_template(
        bank["MATH_MATH_BASIC_PHAN_SO"], {"num": "2", "den": "3"}
    )
    assert fraction_ok, fraction_errors
    assert r"\frac{2}{3}" in fraction
    quadratic, quadratic_ok, quadratic_errors = render_latex_template(
        bank["QUADRATIC_EQUATION"], {"a": "1", "b": "-3", "c": "2"}
    )
    assert quadratic_ok, quadratic_errors
    assert "x^{2} - 3x + 2" in quadratic


def test_templates_use_real_line_breaks_and_bullet_list_is_latex(bank):
    rendered, passed, errors = render_latex_template(bank["LATEX_DANH_SACH_KHONG_THU_TU"], {})
    assert passed, errors
    assert r"\item" in rendered and r"\n" not in rendered
    for item in bank.values():
        for template in item.get("templates", {}).values():
            if isinstance(template, str):
                assert not re.search(r"\\n(?![A-Za-z])", template), item["intent"]


def test_canceling_repl_returns_controlled_exit(monkeypatch):
    monkeypatch.setattr(ka, "create_pipeline", lambda **_: object())
    monkeypatch.setattr("builtins.input", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    stderr = io.StringIO()
    with patch("sys.stderr", stderr):
        assert ka.main(["repl", "--offline-mock"]) == 130
    assert "đã hủy" in stderr.getvalue().lower()


def test_new_intent_data_renders_without_core_change():
    data = {
        "intent": "USER_ADDED_EXAMPLE",
        "category": "tex_math",
        "describe": "Ví dụ lời chào tùy chỉnh.",
        "aliases": [
            "render lời chào màu tím",
            "tạo lời chào tím",
            "viết lời chào màu tím latex",
            "chèn lời chào tím",
            "mẫu lời chào màu tím",
        ],
        "templates": {"latex": r"\texttt{<<value>>}"},
        "slots": {
            "value": {
                "type": "text",
                "regex": r"(?:nội dung|value)\s+(.+)",
                "required": True,
            }
        },
        "kind": "snippet",
        "requires": [],
        "domain": "latex",
    }
    assert validate_intents([data]) == {}
    candidates = TfidfRanker({data["intent"]: data}).rank("render lời chào màu tím")
    assert candidates[0].intent == data["intent"]
    rendered, passed, errors = render_latex_template(data, {"value": "hello"})
    assert rendered == r"\texttt{hello}"
    assert passed, errors


def test_missing_math_slots_remain_unfilled(bank):
    result = extract_slots("tạo phân số", bank["MATH_MATH_BASIC_PHAN_SO"], bank)
    assert not result.is_complete
    assert result.missing == ["num", "den"]
