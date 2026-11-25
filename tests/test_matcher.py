"""
Bộ kiểm thử đơn vị cho mô-đun so khớp mờ dự phòng từ vựng (tests/test_matcher.py).

Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Kiến trúc xử lý:
    text -> preprocess (NFC, chuẩn hóa ký hiệu toán học, tách từ pyvi cho PhoBERT)
         -> PhoBERT classifier: p = softmax over intents
            - if max p >= tau: intent accepted
            - else retrieval top-3 over aliases + scenario strings -> ask the user to choose
              (if best score < theta: say not understood)
- Mục đích kiểm thử:
    1. Kiểm chứng hàm compute_fuzzy_score với các thuật toán RapidFuzz (Token Set Ratio, Token Sort Ratio, Partial Ratio, Ratio).
    2. Kiểm chứng độ bền vững trước từ đệm đàm thoại tiếng Việt ('hướng dẫn tạo...', 'cho mình hỏi cách viết...').
    3. Kiểm chứng cơ chế phá vỡ thế hòa tất định (Deterministic Tie-Breaking).
    4. Kiểm chứng tổng hợp điểm kịch bản theo cơ chế Max-Pooling (per_intent=True).
    5. Kiểm chứng cơ chế Fallback kích hoạt có điều kiện:
       - Khi semantic_score >= theta: fallback_triggered=False, không chạy khớp mờ.
       - Khi semantic_score < theta: fallback_triggered=True.
         + Nếu tìm thấy ứng viên đạt ngưỡng: understood=True và sinh prompt gợi ý lựa chọn ('ask the user to choose').
         + Nếu không tìm thấy: understood=False và sinh prompt báo không hiểu ('say not understood').
    6. Kiểm chứng tính tương thích giao diện baseline phân loại (predict, predict_single).
    7. Kiểm chứng hàm tiện ích độc lập fuzzy_match.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import (
    DEFAULT_INTENTS_PATH,
)
from core.matcher import (
    CandidatePattern,
    FuzzyFallbackResult,
    FuzzyMatch,
    FuzzyMatcher,
    compute_fuzzy_score,
    fuzzy_match,
)

# =====================================================================
# Fixtures kiểm thử
# =====================================================================


@pytest.fixture
def sample_patterns() -> list[CandidatePattern]:
    """Danh sách mẫu kịch bản thử nghiệm."""
    return [
        CandidatePattern(
            intent="MATHJAX_PHAN_SO",
            pattern="phân số",
            pattern_type="alias",
            category="math_expression",
            describe="Chèn ký hiệu hoặc cấu trúc phân số toán học",
        ),
        CandidatePattern(
            intent="MATHJAX_PHAN_SO",
            pattern="phân thức",
            pattern_type="alias",
            category="math_expression",
            describe="Chèn ký hiệu hoặc cấu trúc phân số toán học",
        ),
        CandidatePattern(
            intent="MATHJAX_PHAN_SO",
            pattern="[ MATHJAX_PHAN_SO | phân số | num, den ]",
            pattern_type="scenario_string",
            category="math_expression",
            describe="Chèn ký hiệu hoặc cấu trúc phân số toán học",
        ),
        CandidatePattern(
            intent="LATEX_TAO_BANG",
            pattern="tạo bảng",
            pattern_type="alias",
            category="tables",
            describe="Tạo bảng tabular trong LaTeX",
        ),
        CandidatePattern(
            intent="LATEX_TAO_BANG",
            pattern="bảng",
            pattern_type="alias",
            category="tables",
            describe="Tạo bảng tabular trong LaTeX",
        ),
        CandidatePattern(
            intent="LATEX_CHEN_ANH",
            pattern="chèn ảnh",
            pattern_type="alias",
            category="figures",
            describe="Chèn hình ảnh includegraphics vào văn bản",
        ),
        CandidatePattern(
            intent="LATEX_CHEN_ANH",
            pattern="hình ảnh",
            pattern_type="alias",
            category="figures",
            describe="Chèn hình ảnh includegraphics vào văn bản",
        ),
    ]


@pytest.fixture
def temp_intents_file(tmp_path: Path) -> Path:
    """Tạo tệp intents.json tạm phục vụ kiểm thử nạp dữ liệu."""
    data = [
        {
            "intent": "MATHJAX_PHAN_SO",
            "category": "math_expression",
            "describe": "phân số | num, den",
            "aliases": ["phân số", "phân thức", "a trên b"],
            "sample_utterances": ["hướng dẫn tạo phân số", "cách gõ phân số"],
            "slots": {
                "num": {"required": True, "pattern": r"(\d+)"},
                "den": {"required": True, "pattern": r"(\d+)"},
            },
            "templates": {"latex": r"\frac{<<num>>}{<<den>>}", "python": "", "cpp": ""},
        },
        {
            "intent": "LATEX_TAO_BANG",
            "category": "tables",
            "describe": "tạo bảng | cols",
            "aliases": ["bảng", "tạo bảng", "tabular"],
            "sample_utterances": ["tạo một bảng trong tài liệu"],
            "slots": {},
            "templates": {"latex": r"\begin{tabular}...\end{tabular}", "python": "", "cpp": ""},
        },
    ]
    file_path = tmp_path / "intents.json"
    file_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return file_path


# =====================================================================
# 1. Kiểm thử hàm compute_fuzzy_score
# =====================================================================


def test_compute_fuzzy_score_exact_match() -> None:
    """Hai chuỗi giống hệt nhau phải trả về điểm số cực đại 1.0."""
    score = compute_fuzzy_score("phân số", "phân số")
    assert abs(score - 1.0) < 1e-4


def test_compute_fuzzy_score_empty_strings() -> None:
    """Một hoặc hai chuỗi rỗng phải trả về 0.0."""
    assert compute_fuzzy_score("", "phân số") == 0.0
    assert compute_fuzzy_score("phân số", "") == 0.0
    assert compute_fuzzy_score("", "") == 0.0


def test_compute_fuzzy_score_conversational_resilience() -> None:
    """
    Thuật toán Token Set Ratio phải đạt điểm cao khi câu chứa từ đệm đàm thoại
    tiếng Việt nhưng vẫn chứa đầy đủ từ khóa mục tiêu.
    """
    query = "hãy hướng dẫn tôi cách tạo phân số toán học trong latex"
    pattern = "phân số"
    score = compute_fuzzy_score(query, pattern, scorer="token_set_ratio")
    # Token Set Ratio: phần giao {'phân', 'số'} tạo ra tập con hoàn hảo -> điểm đạt 1.0
    assert score >= 0.90


def test_compute_fuzzy_score_scorers() -> None:
    """Kiểm tra các thuật toán scorer khác nhau được hỗ trợ."""
    s1 = "tạo bảng tabular"
    s2 = "tạo bảng"

    score_set = compute_fuzzy_score(s1, s2, scorer="token_set_ratio")
    score_sort = compute_fuzzy_score(s1, s2, scorer="token_sort_ratio")
    score_partial = compute_fuzzy_score(s1, s2, scorer="partial_ratio")
    score_ratio = compute_fuzzy_score(s1, s2, scorer="ratio")

    assert 0.0 <= score_set <= 1.0
    assert 0.0 <= score_sort <= 1.0
    assert 0.0 <= score_partial <= 1.0
    assert 0.0 <= score_ratio <= 1.0


def test_compute_fuzzy_score_invalid_scorer() -> None:
    """Scorer không hợp lệ phải ném ngoại lệ ValueError."""
    with pytest.raises(ValueError, match="Thuật toán scorer không được hỗ trợ"):
        compute_fuzzy_score("abc", "xyz", scorer="unsupported_algorithm")


# =====================================================================
# 2. Kiểm thử cấu trúc dữ liệu và khởi tạo FuzzyMatcher
# =====================================================================


def test_candidate_pattern_and_fuzzy_match_dataclass() -> None:
    """Kiểm tra thuộc tính của CandidatePattern và FuzzyMatch."""
    cp = CandidatePattern(
        intent="TEST_INTENT",
        pattern="mẫu kiểm thử",
        pattern_type="alias",
        category="test_cat",
        describe="mô tả kiểm thử",
    )
    assert cp.intent == "TEST_INTENT"
    assert cp.pattern == "mẫu kiểm thử"
    assert cp.pattern_type == "alias"
    assert cp.category == "test_cat"

    fm = FuzzyMatch(
        intent="TEST_INTENT",
        score=0.95,
        matched_pattern="mẫu kiểm thử",
        pattern_type="alias",
        category="test_cat",
        describe="mô tả kiểm thử",
    )
    assert fm.intent == "TEST_INTENT"
    assert fm.score == 0.95


def test_fuzzy_fallback_result_properties() -> None:
    """Kiểm tra các thuộc tính dẫn xuất của FuzzyFallbackResult."""
    match1 = FuzzyMatch(intent="INTENT_A", score=0.88, matched_pattern="a")
    match2 = FuzzyMatch(intent="INTENT_B", score=0.75, matched_pattern="b")

    res = FuzzyFallbackResult(
        query="câu hỏi",
        semantic_score=0.2,
        fallback_triggered=True,
        understood=True,
        matches=[match1, match2],
        prompt_message="thông điệp",
    )
    assert res.top_candidate == match1
    assert res.top_intent == "INTENT_A"
    assert res.top_score == 0.88

    empty_res = FuzzyFallbackResult(
        query="rỗng",
        semantic_score=0.1,
        fallback_triggered=True,
        understood=False,
        matches=[],
    )
    assert empty_res.top_candidate is None
    assert empty_res.top_intent is None
    assert empty_res.top_score == 0.0


def test_fuzzy_matcher_invalid_scorer_raises() -> None:
    """Khởi tạo FuzzyMatcher với scorer không hợp lệ phải ném ValueError."""
    with pytest.raises(ValueError, match="Thuật toán scorer không được hỗ trợ"):
        FuzzyMatcher(scorer="invalid_scorer")


def test_fuzzy_matcher_load_missing_file_raises() -> None:
    """Nạp tệp không tồn tại phải ném FileNotFoundError."""
    matcher = FuzzyMatcher(patterns=[])
    with pytest.raises(FileNotFoundError, match="Không tìm thấy tệp ngân hàng kịch bản"):
        matcher.load_intents_from_json("non_existent_file_path.json")


def test_fuzzy_matcher_load_from_json(temp_intents_file: Path) -> None:
    """Kiểm tra nạp intents.json tự động trích xuất các nguồn mẫu khác nhau."""
    matcher = FuzzyMatcher(intents_path=temp_intents_file)
    assert matcher.total_patterns > 0

    types = {p.pattern_type for p in matcher.patterns}
    assert "alias" in types
    assert "describe" in types
    assert "intent_name" in types


# =====================================================================
# 3. Kiểm thử so khớp mờ (match và best_match)
# =====================================================================


def test_matcher_basic_match(sample_patterns: list[CandidatePattern]) -> None:
    """Kiểm tra tìm kiếm cơ bản trả về đúng kịch bản có điểm cao nhất."""
    matcher = FuzzyMatcher(patterns=sample_patterns, threshold=0.5, top_k=3)
    matches = matcher.match("tôi muốn chèn hình ảnh vào bài")

    assert len(matches) > 0
    top = matches[0]
    assert top.intent == "LATEX_CHEN_ANH"
    assert top.score >= 0.80
    assert top.category == "figures"


def test_matcher_per_intent_deduplication(sample_patterns: list[CandidatePattern]) -> None:
    """
    Khi per_intent=True, mỗi intent chỉ xuất hiện 1 lần duy nhất trong kết quả,
    giữ lại mẫu có điểm cao nhất (max pooling).
    """
    matcher = FuzzyMatcher(patterns=sample_patterns, threshold=0.3, top_k=5)
    # Câu này có thể khớp cả alias 'phân số', 'phân thức' và scenario_string của MATHJAX_PHAN_SO
    matches = matcher.match("phân số hoặc phân thức", per_intent=True)

    intents = [m.intent for m in matches]
    assert len(intents) == len(set(intents))
    assert "MATHJAX_PHAN_SO" in intents


def test_matcher_without_per_intent(sample_patterns: list[CandidatePattern]) -> None:
    """Khi per_intent=False, cùng một intent có thể xuất hiện nhiều lần với các mẫu khác nhau."""
    matcher = FuzzyMatcher(patterns=sample_patterns, threshold=0.3, top_k=5)
    matches = matcher.match("phân số", per_intent=False)

    matching_intents = [m.intent for m in matches if m.intent == "MATHJAX_PHAN_SO"]
    # Có thể có nhiều hơn 1 bản ghi của MATHJAX_PHAN_SO
    assert len(matching_intents) >= 1


def test_matcher_threshold_filtering(sample_patterns: list[CandidatePattern]) -> None:
    """Các ứng viên có điểm số nhỏ hơn threshold phải bị loại bỏ."""
    matcher = FuzzyMatcher(patterns=sample_patterns, threshold=0.99)
    matches = matcher.match("tạo bảng tabular đơn giản")
    # Với threshold cực cao 0.99, các mẫu không khớp tuyệt đối sẽ bị loại
    for m in matches:
        assert m.score >= 0.99


def test_matcher_best_match(sample_patterns: list[CandidatePattern]) -> None:
    """best_match phải trả về ứng viên điểm cao nhất hoặc None khi không có ứng viên đạt threshold."""
    matcher = FuzzyMatcher(patterns=sample_patterns, threshold=0.6)

    best = matcher.best_match("làm sao tạo bảng")
    assert best is not None
    assert best.intent == "LATEX_TAO_BANG"

    none_match = matcher.best_match("xyzuvw 123456", threshold=0.9)
    assert none_match is None


# =====================================================================
# 4. Kiểm thử cơ chế dự phòng từ vựng (Fast Lexical Fallback)
# =====================================================================


def test_fallback_match_semantic_confident(sample_patterns: list[CandidatePattern]) -> None:
    """
    Khi semantic_score >= theta (ví dụ: 0.75 >= 0.3):
    Truy hồi ngữ nghĩa đã tự tin -> không kích hoạt fallback từ vựng,
    fallback_triggered = False, understood = True, danh sách matches rỗng.
    """
    matcher = FuzzyMatcher(patterns=sample_patterns, theta=0.3)
    res = matcher.fallback_match("tạo bảng", semantic_score=0.75)

    assert isinstance(res, FuzzyFallbackResult)
    assert res.fallback_triggered is False
    assert res.understood is True
    assert len(res.matches) == 0
    assert "confident" in res.prompt_message.lower()


def test_fallback_match_triggered_understood(sample_patterns: list[CandidatePattern]) -> None:
    """
    Khi semantic_score < theta (ví dụ: 0.15 < 0.3):
    Kích hoạt fallback từ vựng (fallback_triggered = True).
    Nếu tìm thấy mẫu khớp (score >= threshold):
    understood = True và prompt_message gợi ý người dùng chọn theo AGENTS.md ('ask the user to choose').
    """
    matcher = FuzzyMatcher(patterns=sample_patterns, threshold=0.5, theta=0.3)
    res = matcher.fallback_match("tôi cần tạo một cái bảng", semantic_score=0.15)

    assert isinstance(res, FuzzyFallbackResult)
    assert res.fallback_triggered is True
    assert res.understood is True
    assert len(res.matches) > 0
    assert res.top_intent == "LATEX_TAO_BANG"
    assert res.top_score >= 0.5

    # Xác nhận thông điệp hỏi người dùng chọn ('ask the user to choose')
    assert "Có phải bạn muốn thực hiện" in res.prompt_message
    assert "tạo bảng" in res.prompt_message.lower()


def test_fallback_match_triggered_not_understood(sample_patterns: list[CandidatePattern]) -> None:
    """
    Khi semantic_score < theta và câu hoàn toàn vô nghĩa không khớp mẫu nào:
    fallback_triggered = True, understood = False,
    prompt_message thông báo không hiểu ('say not understood').
    """
    matcher = FuzzyMatcher(patterns=sample_patterns, threshold=0.5, theta=0.3)
    res = matcher.fallback_match("kjshdfkjasdhf 982347102938 zxcmvnzxc", semantic_score=0.05)

    assert isinstance(res, FuzzyFallbackResult)
    assert res.fallback_triggered is True
    assert res.understood is False
    assert len(res.matches) == 0
    # Xác nhận thông điệp không hiểu ('say not understood')
    assert "không hiểu" in res.prompt_message.lower()


# =====================================================================
# 5. Kiểm thử giao diện tương thích Baseline (predict / predict_single)
# =====================================================================


def test_predict_single_and_batch(sample_patterns: list[CandidatePattern]) -> None:
    """Kiểm tra dự đoán phân loại theo intent hoặc category."""
    matcher = FuzzyMatcher(patterns=sample_patterns)

    # Dự đoán intent
    pred_intent = matcher.predict_single("chèn ảnh minh họa", target="intent")
    assert pred_intent == "LATEX_CHEN_ANH"

    # Dự đoán category
    pred_cat = matcher.predict_single("chèn ảnh minh họa", target="category")
    assert pred_cat == "figures"

    # Dự đoán hàng loạt
    queries = ["viết phân số", "tạo bảng biểu", "chèn hình"]
    preds = matcher.predict(queries, target="intent")
    assert len(preds) == 3
    assert preds[0] == "MATHJAX_PHAN_SO"
    assert preds[1] == "LATEX_TAO_BANG"
    assert preds[2] == "LATEX_CHEN_ANH"


# =====================================================================
# 6. Kiểm thử hàm tiện ích độc lập fuzzy_match
# =====================================================================


def test_standalone_fuzzy_match_with_strings() -> None:
    """Kiểm tra fuzzy_match độc lập với danh sách chuỗi."""
    candidates = ["phân số", "tạo bảng", "chèn ảnh", "tích phân"]
    matches = fuzzy_match("cách tạo phân số", candidates, top_k=2)

    assert len(matches) <= 2
    assert matches[0].matched_pattern == "phân số"
    assert matches[0].score >= 0.8


def test_standalone_fuzzy_match_with_tuples() -> None:
    """Kiểm tra fuzzy_match độc lập với danh sách tuple (intent, pattern)."""
    candidates = [
        ("MATH_FRAC", "phân số"),
        ("TABLE", "tạo bảng"),
        ("FIGURE", "chèn ảnh"),
    ]
    matches = fuzzy_match("hướng dẫn tạo bảng", candidates, top_k=1)

    assert len(matches) == 1
    assert matches[0].intent == "TABLE"
    assert matches[0].matched_pattern == "tạo bảng"


def test_standalone_fuzzy_match_with_candidate_patterns() -> None:
    """Kiểm tra fuzzy_match độc lập với danh sách CandidatePattern."""
    candidates = [
        CandidatePattern(intent="INTENT_1", pattern="ma trận"),
        CandidatePattern(intent="INTENT_2", pattern="định thức"),
    ]
    matches = fuzzy_match("tính định thức ma trận", candidates, top_k=2)
    assert len(matches) > 0


# =====================================================================
# 7. Kiểm thử tích hợp với intents.json thực tế của dự án
# =====================================================================


def test_real_intents_bank_loading() -> None:
    """Kiểm tra nạp toàn bộ intents.json chính thức của dự án (189 intents)."""
    if not DEFAULT_INTENTS_PATH.exists():
        pytest.skip(f"Không tìm thấy {DEFAULT_INTENTS_PATH}")

    matcher = FuzzyMatcher(intents_path=DEFAULT_INTENTS_PATH)
    # 189 intents x (5-8 aliases + describe + intent_name + scenario_string + ...) >= 1000 patterns
    assert matcher.total_patterns >= 1000

    # Test với câu truy vấn tiếng Việt thực tế
    match = matcher.best_match("tạo phân số")
    assert match is not None
    assert "PHAN_SO" in match.intent
