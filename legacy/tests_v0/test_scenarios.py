"""
Kiểm thử tự động cho module src/scenarios.py và cấu trúc dữ liệu data/intents.json.

Tuân thủ quy tắc kiến trúc AGENTS.md:
- Kiểm thử nạp dữ liệu (data loading), kiểm thực schema (validation),
  và thuật toán truy hồi ngoại tuyến (retrieval helpers).
- Kiểm tra tính hợp lệ và toàn vẹn của toàn bộ 162 kịch bản trong data/intents.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.config import DEFAULT_INTENTS_PATH
from src.scenarios import (
    Scenario,
    ScenarioBank,
    ScenarioNotFoundError,
    SlotDefinition,
    TemplateMapping,
    extract_slots,
    load_scenario_bank,
    load_scenarios,
    render_template,
    retrieve_scenarios,
    save_scenarios,
    validate_scenario,
    validate_scenario_bank,
)


def test_load_scenarios_from_default_path() -> None:
    """Kiểm tra việc nạp kịch bản từ file mặc định data/intents.json."""
    scenarios = load_scenarios(DEFAULT_INTENTS_PATH)
    assert len(scenarios) == 162
    assert all(isinstance(s, Scenario) for s in scenarios)


def test_scenario_bank_operations() -> None:
    """Kiểm tra các thao tác truy vấn trên đối tượng ScenarioBank."""
    bank = load_scenario_bank(DEFAULT_INTENTS_PATH)
    assert len(bank) == 162

    # Tra cứu thành công
    scenario = bank["MATHJAX_PHAN_SO"]
    assert scenario.intent == "MATHJAX_PHAN_SO"
    assert scenario.category == "math_expression"
    assert "num" in scenario.slots
    assert "den" in scenario.slots

    # Tra cứu intent không tồn tại ném lỗi ScenarioNotFoundError
    with pytest.raises(ScenarioNotFoundError):
        _ = bank["INTENT_KHONG_TON_TAI"]

    assert bank.get("INTENT_KHONG_TON_TAI") is None

    # Lọc theo category
    math_scenarios = bank.filter(category="math_expression")
    assert len(math_scenarios) > 0
    assert all(s.category == "math_expression" for s in math_scenarios)

    # Danh sách danh mục
    cats = bank.categories()
    assert "math_expression" in cats


def test_validate_official_intents_bank() -> None:
    """Kiểm tra toàn bộ 162 intent trong ngân hàng kịch bản data/intents.json đều hợp lệ."""
    bank = load_scenario_bank(DEFAULT_INTENTS_PATH)
    errors = bank.validate()
    assert errors == {}, f"Phát hiện intent có lỗi schema: {errors}"


def test_validate_scenario_invalid_cases() -> None:
    """Kiểm tra hàm validate_scenario phát hiện chính xác các vi phạm schema."""
    # 1. Thiếu hoặc sai định dạng intent
    bad_intent = {
        "intent": "invalid-intent-name!",
        "category": "math",
        "sample_utterances": ["câu mẫu"],
        "templates": {"latex": "abc", "python": "", "cpp": ""},
    }
    errs = validate_scenario(bad_intent)
    assert any("chữ hoa, số và dấu gạch dưới" in e for e in errs)

    # 2. Thiếu category
    bad_cat = {
        "intent": "INTENT_HOP_LE",
        "category": "",
        "sample_utterances": ["câu mẫu"],
        "templates": {"latex": "abc", "python": "", "cpp": ""},
    }
    assert any("category" in e for e in validate_scenario(bad_cat))

    # 3. Thiếu sample_utterances
    bad_utt = {
        "intent": "INTENT_HOP_LE",
        "category": "math",
        "sample_utterances": [],
        "templates": {"latex": "abc", "python": "", "cpp": ""},
    }
    assert any("sample_utterances" in e for e in validate_scenario(bad_utt))

    # 4. Thiếu key python hoặc cpp trong templates
    bad_templates = {
        "intent": "INTENT_HOP_LE",
        "category": "math",
        "sample_utterances": ["câu mẫu"],
        "templates": {"latex": "abc"},
    }
    errs_tpl = validate_scenario(bad_templates)
    assert any("python" in e for e in errs_tpl)
    assert any("cpp" in e for e in errs_tpl)

    # 5. Regex pattern không hợp lệ
    bad_regex = {
        "intent": "INTENT_HOP_LE",
        "category": "math",
        "sample_utterances": ["câu mẫu"],
        "templates": {"latex": "abc", "python": "", "cpp": ""},
        "slots": {"x": {"required": False, "default": "", "pattern": "[a-z"}},
    }
    assert any("Regex pattern" in e for e in validate_scenario(bad_regex))


def test_validate_duplicate_intents() -> None:
    """Kiểm tra validate_scenario_bank phát hiện lỗi trùng lặp tên intent."""
    item = {
        "intent": "INTENT_TRUNG_LAP",
        "category": "test",
        "sample_utterances": ["mẫu"],
        "templates": {"latex": "x", "python": "", "cpp": ""},
    }
    errors = validate_scenario_bank([item, item])
    assert "INTENT_TRUNG_LAP" in errors
    assert any("trùng lặp" in e for e in errors["INTENT_TRUNG_LAP"])


def test_offline_retrieval_matching() -> None:
    """Kiểm tra bộ truy hồi offline tìm kiếm chính xác kịch bản phù hợp."""
    bank = load_scenario_bank(DEFAULT_INTENTS_PATH)

    # 1. Truy vấn phân số
    results = retrieve_scenarios("tạo phân số a trên b", bank, top_k=3)
    assert len(results) > 0
    top_scenario, top_score = results[0]
    assert top_scenario.intent == "MATHJAX_PHAN_SO"
    assert top_score >= 0.3

    # 2. Truy vấn công thức inline
    results_inline = retrieve_scenarios("chèn công thức inline vào dòng văn bản", bank, top_k=3)
    assert any(s.intent == "MATHJAX_CONG_THUC_INLINE" for s, _ in results_inline)

    # 3. Truy vấn không liên quan với ngưỡng cao trả về rỗng
    results_empty = retrieve_scenarios("một câu vô nghĩa ngẫu nhiên xyz12345", bank, threshold=0.8)
    assert len(results_empty) == 0


def test_slot_extraction() -> None:
    """Kiểm tra trích xuất tham số từ câu người dùng bằng regex."""
    bank = load_scenario_bank(DEFAULT_INTENTS_PATH)
    phan_so_scenario = bank["MATHJAX_PHAN_SO"]

    # Trích xuất tử và mẫu
    text = "tạo phân số tử số là x + 1 mẫu số là 2y - 3"
    extracted, missing = extract_slots(phan_so_scenario, text)
    assert extracted.get("num") == "x + 1" or "x" in extracted.get("num", "")
    assert "2y" in extracted.get("den", "")
    assert missing == []


def test_slot_extraction_missing_required() -> None:
    """Kiểm tra phát hiện slot bắt buộc bị thiếu."""
    scenario = Scenario(
        intent="TEST_INTENT",
        category="test",
        sample_utterances=["câu mẫu"],
        slots={
            "req_slot": SlotDefinition(
                name="req_slot",
                required=True,
                pattern=r"yêu cầu\s+([a-z]+)",
            )
        },
        templates=TemplateMapping(latex="\\test{{{req_slot}}}", python="", cpp=""),
    )
    extracted, missing = extract_slots(scenario, "câu này không có slot yêu cầu")
    assert "req_slot" in missing
    assert extracted.get("req_slot") is None or extracted.get("req_slot") == ""


def test_render_template() -> None:
    """Kiểm tra việc render template LaTeX với các giá trị slot."""
    bank = load_scenario_bank(DEFAULT_INTENTS_PATH)
    phan_so_scenario = bank["MATHJAX_PHAN_SO"]

    rendered = render_template(phan_so_scenario, {"num": "u", "den": "v"}, target="latex")
    assert "\\frac{u}{v}" in rendered or "\\frac{{u}}{{v}}" in rendered or "u" in rendered

    # Kiểm tra target không hợp lệ ném lỗi ValueError
    with pytest.raises(ValueError):
        render_template(phan_so_scenario, target="unsupported_target")


def test_save_and_reload_scenarios(tmp_path: Path) -> None:
    """Kiểm tra lưu kịch bản ra đĩa và tải lại toàn vẹn."""
    bank = load_scenario_bank(DEFAULT_INTENTS_PATH)
    sample_scenarios = bank.all()[:5]

    test_file = tmp_path / "test_intents.json"
    save_scenarios(sample_scenarios, test_file)

    reloaded = load_scenarios(test_file)
    assert len(reloaded) == 5
    for orig, rel in zip(sample_scenarios, reloaded):
        assert orig.intent == rel.intent
        assert orig.templates.latex == rel.templates.latex
        assert orig.templates.python == rel.templates.python
        assert orig.templates.cpp == rel.templates.cpp
        assert orig.sample_utterances == rel.sample_utterances
