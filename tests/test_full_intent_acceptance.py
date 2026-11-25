"""Regression tests độc lập cho audit chấp nhận đủ 189 intent."""

from __future__ import annotations

import json
import re

from core.config import DEFAULT_INTENTS_PATH, MODE_RETRIEVAL_ONLY, SUPPORTED_TARGETS
from core.pipeline import Pipeline
from core.slots import extract_slots


FIXTURE_PATH = DEFAULT_INTENTS_PATH.parent.parent / "tests" / "fixtures" / "intent_acceptance_cases.json"


def _audit_inputs() -> tuple[list[dict], dict[str, dict]]:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    bank = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))
    return cases, {item["intent"]: item for item in bank}


def test_acceptance_fixture_covers_all_189_intents_and_slots() -> None:
    cases, bank = _audit_inputs()
    assert len(cases) == 189
    assert len({case["intent"] for case in cases}) == 189
    assert {case["intent"] for case in cases} == set(bank)

    for case in cases:
        assert case["utterance"].strip()
        assert case["expected_intent"] == case["intent"]
        assert case["source"]
        assert set(case["required_slots"] + case["optional_slots"]) <= set(case["slot_inputs"])
        assert set(case["required_slots"]) <= set(case["missing_slot_cases"])


def test_obsidian_is_a_supported_pipeline_target() -> None:
    pipeline = Pipeline(
        mode=MODE_RETRIEVAL_ONLY,
        intents_path=DEFAULT_INTENTS_PATH,
        target="obsidian",
        lazy_load=True,
    )
    assert pipeline.target == "obsidian"


def test_each_supported_intent_renders_from_its_schema_sample() -> None:
    cases, bank = _audit_inputs()
    pipeline = Pipeline(
        mode=MODE_RETRIEVAL_ONLY,
        intents_path=DEFAULT_INTENTS_PATH,
        lazy_load=True,
    )

    for case in cases:
        if case["expected_target"] is None:
            assert case["intent"] == "OUT_OF_SCOPE"
            assert bank[case["intent"]]["category"] == "tex_misc"
            continue
        assert case["expected_target"] in SUPPORTED_TARGETS
        pipeline.target = case["expected_target"]
        result = pipeline._accept_intent_and_extract_slots(
            intent=case["intent"],
            query=case["slot_utterance"],
            confidence=1.0,
            category=bank[case["intent"]]["category"],
            custom_slots=case["slot_inputs"],
        )
        assert result.status == "ok", (case["intent"], result.status, result.slot_errors)
        assert result.latex_code
        assert not re.search(r"\{\{\s*[A-Za-z_]\w*\s*\}\}|<<[A-Za-z_]\w*>>", result.latex_code)


def test_required_slot_absence_is_reported_for_each_fixture_case() -> None:
    cases, bank = _audit_inputs()
    pipeline = Pipeline(
        mode=MODE_RETRIEVAL_ONLY,
        intents_path=DEFAULT_INTENTS_PATH,
        lazy_load=True,
    )
    exceptions: list[tuple[str, str]] = []

    for case in cases:
        if case["expected_target"] not in SUPPORTED_TARGETS:
            continue
        pipeline.target = case["expected_target"]
        for missing_name in case["missing_slot_cases"]:
            supplied = {
                name: value
                for name, value in case["slot_inputs"].items()
                if name != missing_name
            }
            result = pipeline._accept_intent_and_extract_slots(
                intent=case["intent"],
                query=case["utterance"],
                confidence=1.0,
                category=bank[case["intent"]]["category"],
                custom_slots=supplied,
            )
            if result.status != "need_slots" or missing_name not in result.missing_slots:
                exceptions.append((case["intent"], missing_name))

    assert exceptions == [("LATEX_NAP_PACKAGE_CO_TUY_CHON", "package_name")]


def test_optional_defaults_are_applied_without_manual_values() -> None:
    cases, bank = _audit_inputs()
    pipeline = Pipeline(
        mode=MODE_RETRIEVAL_ONLY,
        intents_path=DEFAULT_INTENTS_PATH,
        lazy_load=True,
    )
    checked = 0

    for case in cases:
        defaults = {
            name: value
            for name, value in case["defaults"].items()
            if name in case["optional_slots"]
        }
        if not defaults or case["expected_target"] not in SUPPORTED_TARGETS:
            continue
        checked += 1
        pipeline.target = case["expected_target"]
        supplied = {
            name: value
            for name, value in case["slot_inputs"].items()
            if name not in defaults
        }
        result = pipeline._accept_intent_and_extract_slots(
            intent=case["intent"],
            query=case["utterance"],
            confidence=1.0,
            category=bank[case["intent"]]["category"],
            custom_slots=supplied,
        )
        assert result.status == "ok", (case["intent"], result.status, result.slot_errors)
        assert all(result.slots.get(name) for name in defaults), case["intent"]

    assert checked == 11


def test_manual_slot_correction_replaces_invalid_extraction_for_same_slot() -> None:
    """Giá trị người dùng sửa phải gỡ lỗi cũ của chính slot đã nhập tay."""
    bank = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))
    intent = next(item for item in bank if item["intent"] == "MATH_MATH_BASIC_PHAN_SO")
    pipeline = Pipeline(
        mode=MODE_RETRIEVAL_ONLY,
        intents_path=DEFAULT_INTENTS_PATH,
        lazy_load=True,
    )

    result = pipeline._accept_intent_and_extract_slots(
        intent=intent["intent"],
        query="Tạo phân số có tử số là 2 và mẫu số là 5",
        confidence=1.0,
        category=intent["category"],
        custom_slots={"num": "2"},
    )

    assert result.status == "ok"
    assert result.missing_slots == []
    assert result.slots["num"] == "2"
    assert result.slots["den"] == "5"
    assert "\\frac{2}{5}" in (result.latex_code or "")


def test_manual_math_slot_values_go_through_the_declared_normalizer() -> None:
    """Slot nhập tay không được bỏ qua chuẩn hóa và allowlist lệnh toán học."""
    pipeline = Pipeline(
        mode=MODE_RETRIEVAL_ONLY,
        intents_path=DEFAULT_INTENTS_PATH,
        lazy_load=True,
    )

    result = pipeline.choose(
        "MATH_MATH_BASIC_PHAN_SO",
        query="Tạo phân số",
        slots={"num": r"\input{secret}", "den": "5"},
    )

    assert result.status == "need_slots"
    assert result.missing_slots == ["num"]
    assert "\\input" not in (result.latex_code or "")
    assert result.slot_errors


def test_manual_number_slot_rejects_non_numeric_values() -> None:
    """Manual values phải tuân cùng kiểu number với giá trị bắt bằng regex."""
    pipeline = Pipeline(
        mode=MODE_RETRIEVAL_ONLY,
        intents_path=DEFAULT_INTENTS_PATH,
        lazy_load=True,
    )

    result = pipeline.choose(
        "QUADRATIC_EQUATION",
        query="Tạo phương trình bậc hai",
        slots={"a": "abc", "b": "3", "c": "4"},
    )

    assert result.status == "need_slots"
    assert result.missing_slots == ["a"]
    assert result.slot_errors


def test_spoken_division_is_normalized_before_limit_slot_matching() -> None:
    """Từ đọc phép chia phải tới regex slot dưới dạng toán học chuẩn."""
    result = extract_slots(
        "Tạo giới hạn khi x tiến đến 0 của sin x chia x",
        "MATH_MATH_BASIC_GIOI_HAN",
    )

    assert result.slots.get("expr") == "sin x/x"
