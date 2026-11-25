"""
Kiểm thử tự động cho module core/validator.py (JSON Schema Validator).

Tuân thủ quy tắc kiến trúc AGENTS.md:
- Ràng buộc cấu trúc schema: unique UPPER_SNAKE_CASE, category whitelist,
  describe format, 5-8 short Vietnamese aliases, slot specifications with
  single-group regex and types (number, text, label, filename), <<slot>> placeholders,
  kind, requires, and post attributes.
"""

from __future__ import annotations

import pytest

from core.config import DEFAULT_INTENTS_PATH
from core.validator import (
    CATEGORY_WHITELIST,
    IntentValidationError,
    IntentValidator,
    is_valid_intent,
    slugify_intent_name,
    validate_intent,
    validate_intents,
    validate_intents_file,
)


@pytest.fixture
def valid_intent_dict() -> dict:
    """Fixture cung cấp một intent mẫu hoàn toàn hợp lệ."""
    return {
        "intent": "LATEX_TEST_INTENT_CHUAN",
        "category": "tex_struct",
        "describe": "Mô tả kiểm thử kịch bản tài liệu mẫu tiếng Việt.",
        "aliases": [
            "tạo tài liệu kiểm thử",
            "mẫu văn bản test",
            "kịch bản cấu trúc tài liệu",
            "chèn tài liệu mẫu",
            "khung tex thử nghiệm",
            "cú pháp test intent",
        ],
        "slots": {
            "title": {
                "type": "text",
                "regex": r"tiêu đề ([A-Za-z0-9_\s]+)",
                "required": False,
                "default": "Tiêu đề mặc định",
                "desc": "Tiêu đề tài liệu",
            },
            "page_num": {
                "type": "number",
                "regex": r"trang ([0-9]+)",
                "required": True,
                "default": None,
                "desc": "Số trang",
            },
        },
        "templates": {
            "latex": "\\title{<<title>>}\n\\setcounter{page}{<<page_num>>}",
            "python": "",
            "cpp": "",
        },
        "kind": "snippet",
        "requires": ["amsmath"],
        "post": "tidy_math",
        "compilable": True,
        "domain": "latex",
    }


def test_valid_intent_passes_validation(valid_intent_dict: dict) -> None:
    """Kiểm tra intent chuẩn thỏa mãn tất cả quy tắc schema và không có lỗi."""
    errors = validate_intent(valid_intent_dict)
    assert errors == [], f"Intent hợp lệ không được có lỗi: {errors}"
    assert is_valid_intent(valid_intent_dict) is True


def test_intent_name_upper_snake_case(valid_intent_dict: dict) -> None:
    """Kiểm tra quy tắc tên intent phải là UPPER_SNAKE_CASE."""
    # Tên chữ thường
    bad1 = dict(valid_intent_dict, intent="latex_bad_lowercase")
    assert any("UPPER_SNAKE_CASE" in e for e in validate_intent(bad1))

    # Tên chứa ký tự đặc biệt / dấu cách
    bad2 = dict(valid_intent_dict, intent="LATEX-BAD-NAME!")
    assert any("UPPER_SNAKE_CASE" in e for e in validate_intent(bad2))

    # Tên rỗng hoặc None
    bad3 = dict(valid_intent_dict, intent="")
    assert any("chuỗi ký tự không rỗng" in e for e in validate_intent(bad3))


def test_intent_uniqueness(valid_intent_dict: dict) -> None:
    """Kiểm tra phát hiện trùng lặp tên intent thông qua seen_intents."""
    seen: set[str] = set()
    errors1 = validate_intent(valid_intent_dict, seen_intents=seen)
    assert errors1 == []
    assert "LATEX_TEST_INTENT_CHUAN" in seen

    # Validate lại cùng intent với seen_intents -> báo lỗi trùng lặp
    errors2 = validate_intent(valid_intent_dict, seen_intents=seen)
    assert any("trùng lặp" in e for e in errors2)


def test_category_whitelist(valid_intent_dict: dict) -> None:
    """Kiểm tra ràng buộc danh mục phải nằm trong category whitelist."""
    # Danh mục hợp lệ trong whitelist
    for cat in CATEGORY_WHITELIST:
        intent_copy = dict(valid_intent_dict, category=cat)
        assert validate_intent(intent_copy) == []

    # Danh mục không hợp lệ
    bad_cat = dict(valid_intent_dict, category="invalid_unknown_category")
    errs = validate_intent(bad_cat)
    assert any("không thuộc whitelist" in e for e in errs)


def test_describe_format(valid_intent_dict: dict) -> None:
    """Kiểm tra trường describe phải là chuỗi không rỗng."""
    bad_desc1 = dict(valid_intent_dict, describe="")
    assert any("không được để rỗng" in e for e in validate_intent(bad_desc1))

    bad_desc2 = dict(valid_intent_dict, describe="   \n  ")
    assert any("không được để rỗng" in e for e in validate_intent(bad_desc2))

    bad_desc3 = dict(valid_intent_dict, describe=12345)
    assert any("không được để rỗng" in e for e in validate_intent(bad_desc3))


def test_aliases_count_and_format(valid_intent_dict: dict) -> None:
    """Kiểm tra số lượng 5-8 câu aliases ngắn gọn tiếng Việt."""
    # Dưới 5 câu
    bad_few = dict(valid_intent_dict, aliases=["câu 1", "câu 2", "câu 3"])
    assert any("5 đến 8 câu" in e for e in validate_intent(bad_few))

    # Trên 8 câu
    bad_many = dict(valid_intent_dict, aliases=[f"câu {i}" for i in range(1, 10)])
    assert any("5 đến 8 câu" in e for e in validate_intent(bad_many))

    # Chứa phần tử rỗng hoặc không phải chuỗi
    bad_elem = dict(valid_intent_dict, aliases=["câu 1", "câu 2", "", "câu 4", "câu 5"])
    assert any("không rỗng" in e for e in validate_intent(bad_elem))

    # Trường hợp đặc biệt OUT_OF_SCOPE: cho phép aliases rỗng
    oos = {
        "intent": "OUT_OF_SCOPE",
        "category": "OUT_OF_SCOPE",
        "describe": "Ngoài phạm vi trợ lý LaTeX.",
        "aliases": [],
        "slots": {},
        "templates": {"latex": "", "python": "", "cpp": ""},
        "kind": "snippet",
        "requires": [],
    }
    assert validate_intent(oos) == []


def test_aliases_cannot_reuse_intent_name_or_contain_digits(valid_intent_dict: dict) -> None:
    copied = dict(valid_intent_dict, aliases=[
        "latex test intent chuan", "cú pháp latex test intent chuan",
        "tạo tài liệu kiểm thử", "mẫu văn bản test", "chèn tài liệu mẫu",
    ])
    assert any("bản sao không dấu" in e for e in validate_intent(copied))
    with_digit = dict(valid_intent_dict, aliases=["mẫu cấp 2", *valid_intent_dict["aliases"][1:]])
    assert any("không được chứa chữ số" in e for e in validate_intent(with_digit))


def test_alias_must_not_be_shared_across_intents(valid_intent_dict: dict) -> None:
    other = dict(valid_intent_dict, intent="LATEX_OTHER_INTENT")
    errors = validate_intents([valid_intent_dict, other])
    assert any("nhiều intent" in message for message in errors["LATEX_TEST_INTENT_CHUAN"])
    assert any("nhiều intent" in message for message in errors["LATEX_OTHER_INTENT"])


def test_future_slug_preserves_vietnamese_d_mapping() -> None:
    assert slugify_intent_name("Chữ đậm") == "CHU_DAM"


def test_slot_specifications_type_and_regex(valid_intent_dict: dict) -> None:
    """Kiểm tra đặc tả slot: kiểu dữ liệu hợp lệ và regex bắt buộc có đúng 1 capturing group."""
    # Kiểu dữ liệu không hợp lệ
    bad_type = dict(
        valid_intent_dict,
        slots={
            "my_slot": {
                "type": "unsupported_type",
                "regex": r"([0-9]+)",
            }
        },
        templates={"latex": "<<my_slot>>"},
    )
    assert any("không hợp lệ" in e for e in validate_intent(bad_type))

    # Regex không có capturing group nào (0 nhóm)
    bad_regex_0_group = dict(
        valid_intent_dict,
        slots={
            "my_slot": {
                "type": "number",
                "regex": r"[0-9]+",
            }
        },
        templates={"latex": "<<my_slot>>"},
    )
    assert any("chính xác đúng 1 capturing group" in e for e in validate_intent(bad_regex_0_group))

    # Regex có 2 capturing groups
    bad_regex_2_groups = dict(
        valid_intent_dict,
        slots={
            "my_slot": {
                "type": "number",
                "regex": r"([0-9]+)-([a-z]+)",
            }
        },
        templates={"latex": "<<my_slot>>"},
    )
    assert any("chính xác đúng 1 capturing group" in e for e in validate_intent(bad_regex_2_groups))

    # Regex bị lỗi cú pháp
    bad_regex_syntax = dict(
        valid_intent_dict,
        slots={
            "my_slot": {
                "type": "number",
                "regex": r"([0-9+",
            }
        },
        templates={"latex": "<<my_slot>>"},
    )
    assert any("không hợp lệ" in e for e in validate_intent(bad_regex_syntax))


def test_template_placeholders_match(valid_intent_dict: dict) -> None:
    """Kiểm tra tính nhất quán giữa placeholder <<slot>> và khai báo slots."""
    # Placeholder trong template nhưng chưa khai báo trong slots
    bad_extra_placeholder = dict(
        valid_intent_dict,
        templates={"latex": "\\title{<<title>>}\n\\setcounter{page}{<<page_num>>}\n<<unknown_slot>>"},
    )
    assert any("chưa được khai báo trong 'slots'" in e for e in validate_intent(bad_extra_placeholder))

    # Slot khai báo nhưng không dùng placeholder trong template
    bad_unused_slot = dict(
        valid_intent_dict,
        templates={"latex": "\\title{<<title>>}"},  # Thiếu <<page_num>>
    )
    assert any("không xuất hiện dưới dạng <<slot>>" in e for e in validate_intent(bad_unused_slot))


def test_kind_requires_post_attributes(valid_intent_dict: dict) -> None:
    """Kiểm tra ràng buộc các thuộc tính kind, requires và post."""
    # Kind không hợp lệ
    bad_kind = dict(valid_intent_dict, kind="unsupported_kind")
    assert any("Thuộc tính 'kind'" in e for e in validate_intent(bad_kind))

    # Requires không phải danh sách string
    bad_req = dict(valid_intent_dict, requires="amsmath")  # String thay vì list
    assert any("Thuộc tính 'requires'" in e for e in validate_intent(bad_req))

    # Post không hợp lệ (ví dụ: chuỗi rỗng)
    bad_post = dict(valid_intent_dict, post="   ")
    assert any("Thuộc tính 'post'" in e for e in validate_intent(bad_post))


def test_validate_real_latex_intents_from_file() -> None:
    """Kiểm tra toàn bộ 43 intent LaTeX trong tệp data/intents.json đều thỏa mãn validator."""
    errors = validate_intents_file(DEFAULT_INTENTS_PATH, domain="latex")
    assert errors == {}, f"Phát hiện lỗi trong các kịch bản LaTeX của intents.json: {errors}"


def test_intent_validator_class_and_raise_on_error(valid_intent_dict: dict) -> None:
    """Kiểm tra lớp đối tượng IntentValidator và tùy chọn raise_on_error."""
    validator = IntentValidator()

    # Kịch bản hợp lệ
    res = validator.validate_intents([valid_intent_dict])
    assert res == {}

    # Kịch bản lỗi kèm raise_on_error=True
    bad_item = dict(valid_intent_dict, intent="bad_name")
    with pytest.raises(IntentValidationError) as exc_info:
        validator.validate_intents([bad_item], raise_on_error=True)

    assert "bad_name" in exc_info.value.errors
