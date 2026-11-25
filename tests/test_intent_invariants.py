from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.config import DEFAULT_INTENTS_PATH
from core.latex_tools import POST_ACTIONS
from core.pipeline import Pipeline, PipelineStatus, render_latex_template
from core.preprocess import normalize_spoken_math
from core.slots import extract_slots
from core.validator import validate_intents_file


def _bank() -> list[dict]:
    return json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))


def _dummy_slots(item: dict) -> dict[str, str]:
    values: dict[str, str] = {}
    for name, definition in item.get("slots", {}).items():
        if definition.get("default") is not None:
            values[name] = str(definition["default"])
        elif definition.get("type") == "number":
            values[name] = "1"
        elif definition.get("type") == "math":
            values[name] = "x+1"
        elif definition.get("type") == "filename":
            values[name] = "image.png"
        elif definition.get("type") == "label":
            values[name] = "lbl"
        else:
            values[name] = "abc"
    return values


def test_every_latex_template_renders_safely_or_is_explicitly_unsupported() -> None:
    for item in _bank():
        code, valid, errors = render_latex_template(item, _dummy_slots(item), target="latex")
        templates = item.get("templates", {})
        supported = bool(templates.get("latex") or templates.get("mathjax"))
        if item["intent"] == "OUT_OF_SCOPE":
            continue
        if supported:
            assert code.strip(), item["intent"]
            assert "<<" not in code, item["intent"]
            assert valid, (item["intent"], errors)
        else:
            result = Pipeline(mode="retrieval_only", lazy_load=True).choose(item["intent"], query="test")
            assert result.status == PipelineStatus.UNSUPPORTED_TARGET.value
            assert not result.latex_code


def test_ok_status_never_has_empty_code() -> None:
    result = Pipeline(mode="retrieval_only").choose("LATEX_BOLD", query="in đậm")
    assert result.status != "ok" or result.latex_code.strip()


@pytest.mark.parametrize(
    "query",
    ["tạo giúp tôi cái này", "viết cho tôi một đoạn mã", "làm thế nào để dùng nó",
     "cho tôi ví dụ ngắn", "hello world"],
)
def test_neutral_text_never_fills_required_slots_without_evidence(query: str) -> None:
    for item in _bank():
        result = extract_slots(query, item)
        for name, definition in item.get("slots", {}).items():
            if definition.get("required") and definition.get("default") is None:
                assert name not in result.slots or not result.slots[name].strip(), (item["intent"], name)


def test_validator_bank_regex_rules_pass() -> None:
    assert validate_intents_file(DEFAULT_INTENTS_PATH) == {}


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [("cộng", "+"), ("trừ", "-"), ("- 3x", "-3x"), ("âm ba x", "-3x"),
     ("x bình phương", "x^2"), ("x mũ ba", "x^3"), ("bằng không", "= 0"),
     ("mười một", "11"), ("hai mươi mốt", "21"),
     ("ba mươi lăm", "35"), ("chín mươi chín", "99"), ("0,5", "0.5")],
)
def test_spoken_math_normalization(spoken: str, expected: str) -> None:
    assert normalize_spoken_math(spoken) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [("+ -4", "- 4"), ("- -4", "+ 4"), ("1x^{2}", "x^{2}"),
     ("-1x", "-x"), ("10x", "10x"), ("1.5x", "1.5x"),
     ("x^{1}", "x^{1}"), (r"\frac{1}{2}", r"\frac{1}{2}")],
)
def test_tidy_math_is_conservative(source: str, expected: str) -> None:
    assert POST_ACTIONS["tidy_math"](source) == expected


@pytest.mark.parametrize(
    ("query", "slots", "latex"),
    [
        ("x bình phương cộng 3x trừ 4 bằng 0", {"a": "1", "b": "3", "c": "-4"}, r"x^{2} + 3x - 4 = 0"),
        ("x^2 - 3x + 2 = 0", {"a": "1", "b": "-3", "c": "2"}, r"x^{2} - 3x + 2 = 0"),
        ("x bình phương trừ 3x cộng 2 bằng 0", {"a": "1", "b": "-3", "c": "2"}, r"x^{2} - 3x + 2 = 0"),
        ("trừ x bình phương cộng x cộng 1 bằng 0", {"a": "-1", "b": "1", "c": "1"}, r"-x^{2} + x + 1 = 0"),
        ("2x bình phương cộng 5 bằng 0", {"a": "2", "b": "0", "c": "5"}, r"2x^{2} + 5 = 0"),
        ("x bình phương trừ 4 bằng 0", {"a": "1", "b": "0", "c": "-4"}, r"x^{2} - 4 = 0"),
        ("hai x bình phương cộng ba x trừ bốn bằng không", {"a": "2", "b": "3", "c": "-4"}, r"2x^{2} + 3x - 4 = 0"),
        ("0.5x^2 - 1.5x = 0", {"a": "0.5", "b": "-1.5", "c": "0"}, r"0.5x^{2} - 1.5x = 0"),
    ],
)
def test_quadratic_generic_math_slots(query: str, slots: dict[str, str], latex: str) -> None:
    item = next(x for x in _bank() if x["intent"] == "QUADRATIC_EQUATION")
    extracted = extract_slots(query, item)
    assert extracted.slots == slots
    assert not extracted.missing
    rendered, valid, errors = render_latex_template(item, extracted.slots)
    assert valid, errors
    assert latex in rendered
    result = Pipeline(mode="retrieval_only").choose("QUADRATIC_EQUATION", query=query)
    assert result.status == "ok"
    assert latex in result.latex_code


def test_quadratic_edge_cases_ask_or_reject() -> None:
    bank = _bank()
    item = next(x for x in bank if x["intent"] == "QUADRATIC_EQUATION")
    assert extract_slots("tạo phương trình bậc hai", item).missing
    invalid = extract_slots("0x^2 + 3x + 1 = 0", item)
    assert invalid.errors


def test_caption_stops_at_other_slot_anchor_and_optional_package_regexes_work() -> None:
    bank = {x["intent"]: x for x in _bank()}
    figure = extract_slots(
        "chú thích là con mèo nhãn fig:cat rộng 0.5", bank["LATEX_FIGURE"]
    )
    assert figure.slots["caption"] == "con mèo"
    assert figure.slots["label"] == "fig-cat"
    assert figure.slots["width"] == "0.5"
    package = extract_slots(
        "nạp gói amsmath với tùy chọn utf8", bank["LATEX_NAP_PACKAGE_CO_TUY_CHON"]
    )
    assert package.slots["package_name"] == "amsmath"
    assert package.slots["options"] == "utf8"

    author_title = extract_slots(
        "tiêu đề là Kết quả tháng 5 tác giả Nguyễn An", bank["LATEX_TIEU_E_TAC_GIA_NGAY_THANG"]
    )
    assert author_title.slots["title"] == "Kết quả tháng 5"
    assert author_title.slots["author"] == "Nguyễn An"


@pytest.mark.parametrize("method", ["retrieval", "fuzzy", "tfidf"])
def test_candidate_sources_filter_unsupported_targets(method: str) -> None:
    bank = _bank()
    unsupported = next(
        item for item in bank
        if item["intent"] != "OUT_OF_SCOPE"
        and not (item.get("templates", {}).get("latex") or item.get("templates", {}).get("mathjax"))
    )
    supported = next(item for item in bank if item.get("templates", {}).get("latex"))
    candidates = [
        SimpleNamespace(intent=unsupported["intent"], score=0.99, describe="", category=unsupported["category"]),
        SimpleNamespace(intent=supported["intent"], score=0.8, describe="", category=supported["category"]),
    ]
    pipeline = Pipeline(mode="retrieval_only", top_k=1, lazy_load=True)
    if method == "retrieval":
        pipeline.retriever = SimpleNamespace(
            retrieve=lambda _query, top_k: SimpleNamespace(candidates=candidates[:top_k])
        )
        results = pipeline._run_retrieval("query")
    elif method == "fuzzy":
        pipeline.matcher = SimpleNamespace(
            match=lambda _query, top_k, threshold: candidates[:top_k]
        )
        results = pipeline._run_fuzzy("query")
    else:
        pipeline.tfidf_ranker = SimpleNamespace(
            rank=lambda _query, top_k, category: candidates[:top_k]
        )
        results = pipeline._run_tfidf("query")
    assert results
    assert all(candidate.intent != unsupported["intent"] for candidate in results)


def test_python_target_uses_data_template_without_latex_escaping(tmp_path) -> None:
    toy = {
        "intent": "TARGET_PYTHON_TOY", "category": "tex_misc",
        "describe": "Mã Python", "aliases": [], "slots": {
            "value": {"type": "text", "regex": "(?:giá trị)\\s+(.+)", "required": True}
        }, "kind": "snippet", "requires": [],
        "templates": {"latex": "", "python": "print(<<value>>)", "cpp": ""},
    }
    path = tmp_path / "intents.json"
    path.write_text(json.dumps([toy], ensure_ascii=False), encoding="utf-8")
    pipeline = Pipeline(mode="retrieval_only", intents_path=path, target="python")
    result = pipeline.choose("TARGET_PYTHON_TOY", query="giá trị a_b")
    assert result.status == "ok"
    assert result.latex_code == "print(a_b)"


def test_validator_rejects_bad_regex_and_unknown_post_action() -> None:
    from core.validator import validate_intent

    base = {
        "intent": "BAD_REGEX_TOY", "category": "tex_misc", "describe": "Mẫu thử",
        "aliases": ["mẫu thử một", "mẫu thử hai", "mẫu thử ba", "mẫu thử bốn", "mẫu thử năm"],
        "slots": {"value": {"type": "text", "regex": r"([A-Za-z]+)", "required": True}},
        "kind": "snippet", "requires": [],
        "templates": {"latex": "<<value>>", "python": "", "cpp": ""},
    }
    errors = validate_intent(base)
    assert any("không có từ khóa" in error for error in errors)
    base["slots"]["value"]["regex"] = r"(?:giá trị)\\s+(.+)"
    errors = validate_intent(base)
    assert any("escape kép" in error for error in errors)
    base["slots"]["value"]["regex"] = r"(?:giá trị)\s+(.+)"
    base["post"] = "unknown_action"
    assert any("chưa được đăng ký" in error for error in validate_intent(base))
