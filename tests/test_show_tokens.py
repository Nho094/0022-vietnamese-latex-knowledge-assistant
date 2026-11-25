"""
Kiểm thử tự động cho script phân tích tokenization PhoBERT (data/show_tokens.py).

Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Kiểm tra tính đầy đủ của 5 mẫu truy vấn đại diện (biểu thức toán, lệnh LaTeX, tiếng Việt đàm thoại).
- Kiểm tra quá trình phân đoạn từ, bóc tách BPE subword và kiểm định ngưỡng tối đa 64 tokens.
- Kiểm tra tính đúng đắn của logic xác định tuân thủ (compliance verification) và định dạng bảng báo cáo.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.config import PHOBERT_MAX_SEQUENCE_LIMIT
from data.show_tokens import (
    SAMPLE_QUERIES,
    TokenInspectionResult,
    format_detailed_cards,
    format_inspection_table,
    inspect_query_tokens,
    inspect_sample_queries,
    load_phobert_tokenizer,
)


def test_sample_queries_structure() -> None:
    """Kiểm tra 5 mẫu truy vấn đại diện bao phủ các khía cạnh yêu cầu."""
    assert len(SAMPLE_QUERIES) == 5

    categories = [q["category"] for q in SAMPLE_QUERIES]
    assert "math_expression" in categories
    assert "latex_command" in categories
    assert "colloquial_vietnamese" in categories

    for item in SAMPLE_QUERIES:
        assert "query" in item
        assert len(item["query"].strip()) > 0
        assert "category" in item
        assert "description" in item


def test_inspect_query_tokens_with_mock() -> None:
    """Kiểm tra logic phân tích token và kiểm định giới hạn 64 tokens với mock tokenizer."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = ["giải", "phương_trình", "bậc", "hai"]

    res = inspect_query_tokens(
        raw_text="giải phương trình bậc hai",
        category="math_expression",
        tokenizer=mock_tokenizer,
        limit=64,
    )

    assert isinstance(res, TokenInspectionResult)
    assert res.raw_text == "giải phương trình bậc hai"
    assert res.category == "math_expression"
    assert "phương_trình" in res.segmented_text
    assert res.subword_count == 4
    assert res.is_compliant is True
    assert res.limit == 64


def test_compliance_failure_with_exceeded_length() -> None:
    """Kiểm tra phát hiện vi phạm khi số lượng subwords vượt ngưỡng giới hạn quy định."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = [f"tok_{i}" for i in range(70)]

    res = inspect_query_tokens(
        raw_text="một câu truy vấn rất dài vượt quá độ dài chuỗi",
        category="general",
        tokenizer=mock_tokenizer,
        limit=64,
    )

    assert res.subword_count == 70
    assert res.is_compliant is False


def test_format_inspection_table() -> None:
    """Kiểm tra tạo bảng tổng hợp phân phối subword."""
    results = [
        TokenInspectionResult(
            raw_text="2x² = 4",
            category="math_expression",
            segmented_text="2x^2 = 4",
            subwords=["2@@", "x@@", "^@@", "2", "=", "4"],
            subword_count=6,
            is_compliant=True,
            limit=64,
        ),
        TokenInspectionResult(
            raw_text=r"\textbf{chào}",
            category="latex_command",
            segmented_text=r"\textbf{chào}",
            subwords=["\\@@", "text@@", "bf", "{", "chào", "}"],
            subword_count=6,
            is_compliant=True,
            limit=64,
        ),
    ]

    table_output = format_inspection_table(results)
    assert "math_expression" in table_output
    assert "latex_command" in table_output
    assert "ĐẠT (PASS)" in table_output
    assert "6" in table_output


def test_format_detailed_cards() -> None:
    """Kiểm tra định dạng card view chi tiết cho từng mẫu truy vấn."""
    results = [
        TokenInspectionResult(
            raw_text="tạo bảng 3 cột",
            category="colloquial_vietnamese",
            segmented_text="tạo bảng 3 cột",
            subwords=["tạo", "bảng", "3", "cột"],
            subword_count=4,
            is_compliant=True,
            limit=64,
        )
    ]

    card_str = format_detailed_cards(results)
    assert "MẪU THỬ NGHIỆM #1 [COLLOQUIAL_VIETNAMESE]" in card_str
    assert "tạo bảng 3 cột" in card_str
    assert "ĐẠT YÊU CẦU" in card_str
    assert "'tạo', 'bảng', '3', 'cột'" in card_str


def test_end_to_end_inspection_with_real_tokenizer() -> None:
    """Kiểm thử tích hợp đầu cuối với PhoBERT tokenizer thực tế."""
    try:
        tokenizer = load_phobert_tokenizer()
    except (RuntimeError, OSError, ValueError):
        pytest.skip("Bỏ qua kiểm thử tích hợp nếu tokenizer PhoBERT chưa khả dụng.")

    results = inspect_sample_queries(
        SAMPLE_QUERIES, tokenizer=tokenizer, limit=PHOBERT_MAX_SEQUENCE_LIMIT
    )
    assert len(results) == 5

    for res in results:
        # Xác minh toàn bộ 5 mẫu đại diện đều tuân thủ chặt chẽ giới hạn 64 tokens
        assert res.is_compliant is True
        assert res.subword_count <= PHOBERT_MAX_SEQUENCE_LIMIT
        assert len(res.subwords) == res.subword_count
        assert len(res.segmented_text) > 0
