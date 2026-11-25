"""
Kiểm thử tự động cho module core/scenario.py (Scenario Natural String Representation Generator)
và kiểm thực schema kịch bản trong core/validator.py (Schema Validator).

Tuân thủ quy chuẩn kiểm thử AGENTS.md và đặc tả Retrieve-and-Fill (RAF - arXiv:2202.00901):
1. Biểu diễn kịch bản (core/scenario.py):
   - 4 biến thể: 'name', 'slots', 'primed', 'handmade' (Bảng 10, Table 10 trong RAF).
   - Xử lý đa dạng cấu trúc đầu vào: dict, chuỗi tên intent tra cứu ngân hàng kịch bản,
     đối tượng Scenario (dataclass), hoặc đối tượng hỗ trợ to_dict().
   - Trích xuất phần tên từ trường 'describe' (loại bỏ phần tham số sau dấu '|').
   - Định dạng slot và giới hạn tối đa 2 ví dụ phân tách bằng dấu ' / ' (Mục 3.2.2 trong RAF).
   - Kiểm tra các trường hợp biên: không có slot, không có describe, slot dạng danh sách.
2. Kiểm thực schema kịch bản (core/validator.py):
   - Kịch bản hợp lệ: định dạng UPPER_SNAKE_CASE, danh mục trong category whitelist,
     số lượng aliases (5-8 câu tiếng Việt ngắn), regex slot có đúng 1 capturing group,
     tính nhất quán giữa placeholder <<slot>> và khai báo slots, kind/requires/post.
   - Kịch bản không hợp lệ: phát hiện chính xác mọi lỗi vi phạm quy tắc schema.
"""

from __future__ import annotations

import pytest

from core.config import DEFAULT_INTENTS_PATH
from core.scenario import (
    SUPPORTED_VARIANTS,
    extract_name_part,
    extract_slot_examples,
    format_slot_string,
    scenario_string,
)
from core.validator import (
    ALLOWED_KINDS,
    ALLOWED_SLOT_TYPES,
    CATEGORY_WHITELIST,
    IntentValidationError,
    IntentValidator,
    is_valid_intent,
    validate_intent,
    validate_intents,
    validate_intents_file,
)
from src.scenarios import Scenario, SlotDefinition, TemplateMapping

# ==============================================================================
# Fixtures dùng chung cho kiểm thực Validator
# ==============================================================================


@pytest.fixture
def sample_valid_intent() -> dict:
    """Fixture cung cấp một intent chuẩn hoàn toàn hợp lệ theo schema AGENTS.md."""
    return {
        "intent": "MATH_TICH_PHAN_XAC_DINH",
        "category": "tex_math",
        "describe": "tích phân xác định | lower, upper, expr",
        "aliases": [
            "tính tích phân từ a đến b",
            "cú pháp tích phân xác định",
            "chèn tích phân có cận",
            "ký hiệu tích phân từ cận dưới đến cận trên",
            "viết tích phân trong latex",
            "công thức tích phân cận a b",
        ],
        "slots": {
            "lower": {
                "type": "text",
                "regex": r"cận dưới ([a-zA-Z0-9_\-\+]+)",
                "required": True,
                "default": None,
                "desc": "Cận dưới tích phân",
                "examples": ["0", "a"],
            },
            "upper": {
                "type": "text",
                "regex": r"cận trên ([a-zA-Z0-9_\-\+]+)",
                "required": True,
                "default": None,
                "desc": "Cận trên tích phân",
                "examples": ["1", "\\infty"],
            },
            "expr": {
                "type": "text",
                "regex": r"hàm số ([^,\n]+)",
                "required": True,
                "default": None,
                "desc": "Biểu thức dưới dấu tích phân",
                "examples": ["f(x)dx", "x^2 dx"],
            },
        },
        "templates": {
            "latex": "\\int_{<<lower>>}^{<<upper>>} <<expr>>",
            "python": "",
            "cpp": "",
        },
        "kind": "snippet",
        "requires": ["amsmath"],
        "post": "tidy_math",
        "compilable": True,
        "domain": "latex",
    }


# ==============================================================================
# 1. Kiểm thử trích xuất tên và ví dụ slot (core/scenario.py)
# ==============================================================================


def test_extract_name_part_variants() -> None:
    """Kiểm tra hàm extract_name_part trích xuất chính xác phần tên tác vụ."""
    # 1. Chuỗi có dạng '<name> | <slots>'
    assert extract_name_part("phân số | num, den") == "phân số"
    assert extract_name_part("Tạo phương trình bậc hai | hệ số a, b, c") == "Tạo phương trình bậc hai"
    assert extract_name_part("tiêu đề 1 | text") == "tiêu đề 1"

    # 2. Câu mô tả đầy đủ không chứa '|'
    plain_desc = "Tạo tài liệu LaTeX tối thiểu có documentclass, document body và kết thúc."
    assert extract_name_part(plain_desc) == plain_desc

    # 3. Câu mô tả chứa cụm từ 'dấu |' của LaTeX
    pipe_latex_desc = "Tạo đường kẻ ngang bằng hline và đường kẻ dọc bằng dấu | trong khai báo cột."
    assert extract_name_part(pipe_latex_desc) == pipe_latex_desc

    # 4. Chuỗi rỗng hoặc có khoảng trắng
    assert extract_name_part("   ") == ""
    assert extract_name_part("  chèn ảnh | filename  ") == "chèn ảnh"


def test_extract_slot_examples() -> None:
    """Kiểm tra extract_slot_examples giới hạn tối đa 2 ví dụ và chuẩn hóa phân tách."""
    # Danh sách > 2 ví dụ -> chỉ lấy 2 ví dụ đầu theo RAF Mục 3.2.2
    assert extract_slot_examples(["1", "2", "3", "4"]) == ["1", "2"]

    # Danh sách 1 ví dụ
    assert extract_slot_examples(["noi dung mau"]) == ["noi dung mau"]

    # Danh sách rỗng hoặc None
    assert extract_slot_examples([]) == []
    assert extract_slot_examples(None) == []

    # Chuỗi có phân tách ' / '
    assert extract_slot_examples("sec / min / hr") == ["sec", "min"]

    # Chuỗi có phân tách dấu phẩy
    assert extract_slot_examples("F, C, K") == ["F", "C"]

    # Chuỗi đơn
    assert extract_slot_examples("single_value") == ["single_value"]


def test_format_slot_string() -> None:
    """Kiểm tra format_slot_string tạo cú pháp [ slot | ... ] theo đúng chuẩn RAF."""
    # Slot dạng dict có desc và examples
    slot_dict = {
        "desc": "đơn vị đo",
        "examples": ["sec", "min", "hr"],
    }
    # 1. Không kèm ví dụ
    assert format_slot_string("unit", slot_dict, include_examples=False) == "[ slot | đơn vị đo ]"

    # 2. Kèm ví dụ (tối đa 2 ví dụ phân tách bằng ' / ')
    assert format_slot_string("unit", slot_dict, include_examples=True) == "[ slot | đơn vị đo | sec / min ]"

    # 3. Slot dùng khóa 'description' thay vì 'desc'
    slot_desc_key = {"description": "mô tả qua description", "examples": ["ex1"]}
    assert format_slot_string("s", slot_desc_key, include_examples=False) == "[ slot | mô tả qua description ]"
    assert format_slot_string("s", slot_desc_key, include_examples=True) == "[ slot | mô tả qua description | ex1 ]"

    # 4. Slot không có desc -> dùng slot_name
    slot_no_desc = {"examples": ["val1"]}
    assert format_slot_string("my_slot", slot_no_desc, include_examples=False) == "[ slot | my_slot ]"
    assert format_slot_string("my_slot", slot_no_desc, include_examples=True) == "[ slot | my_slot | val1 ]"

    # 5. Slot là chuỗi đơn
    assert format_slot_string("my_slot", "mô tả ngắn", include_examples=False) == "[ slot | mô tả ngắn ]"

    # 6. Slot là đối tượng SlotDefinition của src/scenarios.py
    slot_obj = SlotDefinition(name="target", required=True, default=None)
    assert format_slot_string("target", slot_obj, include_examples=False) == "[ slot | target ]"

    # 7. Slot là đối tượng có thuộc tính desc và examples tùy biến
    class CustomSlot:
        def __init__(self, desc: str, examples: list[str]) -> None:
            self.desc = desc
            self.examples = examples

    duck_slot = CustomSlot(desc="Đối tượng đích", examples=["target_1", "target_2"])
    assert format_slot_string("target", duck_slot, include_examples=False) == "[ slot | Đối tượng đích ]"
    assert format_slot_string("target", duck_slot, include_examples=True) == "[ slot | Đối tượng đích | target_1 / target_2 ]"


# ==============================================================================
# 2. Kiểm thử scenario_string với 4 biến thể RAF (core/scenario.py)
# ==============================================================================


def test_scenario_string_name_variant() -> None:
    """Kiểm tra biến thể 'name' chỉ sinh [ intent | <name part of describe> ]."""
    intent = {
        "intent": "MATH_PHAN_SO",
        "describe": "phân số | num, den",
        "slots": {
            "num": {"desc": "Tử số"},
            "den": {"desc": "Mẫu số"},
        },
    }
    result = scenario_string(intent, variant="name")
    assert result == "[ intent | phân số ]"


def test_scenario_string_slots_variant() -> None:
    """Kiểm tra biến thể 'slots' bổ sung danh sách [ slot | <desc or slot name> ]."""
    intent = {
        "intent": "MATH_PHAN_SO",
        "describe": "phân số | num, den",
        "slots": {
            "num": {"desc": "Tử số"},
            "den": {"desc": "Mẫu số"},
        },
    }
    result = scenario_string(intent, variant="slots")
    assert result == "[ intent | phân số [ slot | Tử số ] [ slot | Mẫu số ] ]"


def test_scenario_string_primed_variant() -> None:
    """Kiểm tra biến thể 'primed' bổ sung tối đa 2 ví dụ cho từng slot."""
    intent = {
        "intent": "MATH_PHAN_SO",
        "describe": "phân số | num, den",
        "slots": {
            "num": {"desc": "Tử số", "examples": ["1", "x", "a"]},
            "den": {"desc": "Mẫu số", "examples": ["2", "y"]},
        },
    }
    result = scenario_string(intent, variant="primed")
    expected = "[ intent | phân số [ slot | Tử số | 1 / x ] [ slot | Mẫu số | 2 / y ] ]"
    assert result == expected


def test_scenario_string_handmade_matches_primed() -> None:
    """Kiểm tra biến thể 'handmade' đồng nhất với biến thể 'primed' dùng khi suy luận."""
    intent = {
        "intent": "IN:GET_WEATHER",
        "describe": "get weather | location, date",
        "slots": {
            "location": {"desc": "location", "examples": ["Seattle", "Boston"]},
            "date": {"desc": "date", "examples": ["today"]},
        },
    }
    primed_res = scenario_string(intent, variant="primed")
    handmade_res = scenario_string(intent, variant="handmade")
    default_res = scenario_string(intent)

    assert handmade_res == primed_res
    assert default_res == handmade_res
    assert handmade_res == "[ intent | get weather [ slot | location | Seattle / Boston ] [ slot | date | today ] ]"


def test_scenario_string_variant_case_insensitivity() -> None:
    """Kiểm tra tên biến thể không phân biệt chữ hoa thường và khoảng trắng."""
    intent = {
        "intent": "MATH_CAN_BAC_HAI",
        "describe": "căn bậc hai | expr",
        "slots": {"expr": {"desc": "Biểu thức dưới căn"}},
    }
    assert scenario_string(intent, variant=" NAME ") == "[ intent | căn bậc hai ]"
    assert scenario_string(intent, variant="Slots") == "[ intent | căn bậc hai [ slot | Biểu thức dưới căn ] ]"


def test_scenario_string_no_slots() -> None:
    """Kiểm tra các kịch bản không có slot thì mọi biến thể đều trả về định dạng intent name."""
    intent = {
        "intent": "LATEX_TAI_LIEU_TOI_GIAN",
        "describe": "Tạo tài liệu LaTeX tối thiểu có documentclass, document body và kết thúc.",
        "slots": {},
    }
    name_res = scenario_string(intent, variant="name")
    slots_res = scenario_string(intent, variant="slots")
    primed_res = scenario_string(intent, variant="primed")
    handmade_res = scenario_string(intent, variant="handmade")

    expected = "[ intent | Tạo tài liệu LaTeX tối thiểu có documentclass, document body và kết thúc. ]"
    assert name_res == expected
    assert slots_res == expected
    assert primed_res == expected
    assert handmade_res == expected


def test_scenario_string_fallback_when_no_describe() -> None:
    """Kiểm tra khi intent không có describe thì sử dụng tên intent làm tên dự phòng."""
    intent_no_desc = {
        "intent": "INTENT_KHONG_CO_DESC",
        "slots": {},
    }
    assert scenario_string(intent_no_desc, variant="name") == "[ intent | INTENT_KHONG_CO_DESC ]"


def test_scenario_string_slots_as_list() -> None:
    """Kiểm tra trường slots được cung cấp dưới dạng danh sách thay vì dict."""
    # Danh sách các chuỗi tên slot
    intent_list_str = {
        "intent": "TEST_INTENT",
        "describe": "thử nghiệm | a, b",
        "slots": ["a", "b"],
    }
    res_str = scenario_string(intent_list_str, variant="slots")
    assert res_str == "[ intent | thử nghiệm [ slot | a ] [ slot | b ] ]"

    # Danh sách các dict slot
    intent_list_dict = {
        "intent": "TEST_INTENT_2",
        "describe": "thử nghiệm 2 | x",
        "slots": [{"name": "x", "desc": "biến x", "examples": ["10"]}],
    }
    res_dict = scenario_string(intent_list_dict, variant="primed")
    assert res_dict == "[ intent | thử nghiệm 2 [ slot | biến x | 10 ] ]"


def test_scenario_string_with_string_lookup() -> None:
    """Kiểm tra tra cứu intent bằng chuỗi tên từ ngân hàng kịch bản data/intents.json."""
    res_name = scenario_string("LATEX_TAI_LIEU_TOI_GIAN", variant="name")
    assert res_name == "[ intent | Tạo tài liệu LaTeX tối thiểu có documentclass, document body và kết thúc. ]"

    res_slots = scenario_string("LATEX_CHEN_HINH_ANH", variant="slots")
    assert "[ intent | Chèn ảnh bằng includegraphics; cần graphicx và file ảnh trong project." in res_slots
    assert "[ slot | Tên tệp hình ảnh ]" in res_slots
    assert "[ slot | Tỉ lệ rộng so với textwidth ]" in res_slots


def test_scenario_string_with_scenario_object() -> None:
    """Kiểm tra truyền đối tượng Scenario của dataclass src/scenarios.py."""
    scenario_obj = Scenario(
        intent="TEST_INTENT",
        category="math",
        describe="tính đạo hàm | x",
        slots={
            "x": SlotDefinition(name="x", required=True, default="x"),
        },
        templates=TemplateMapping(latex="\\frac{d}{dx}"),
    )
    res_slots = scenario_string(scenario_obj, variant="slots")
    assert res_slots == "[ intent | tính đạo hàm [ slot | x ] ]"


def test_scenario_string_with_custom_to_dict_object() -> None:
    """Kiểm tra đối tượng có phương thức to_dict() tùy biến."""
    class CustomScenarioItem:
        def to_dict(self) -> dict:
            return {
                "intent": "CUSTOM_INTENT",
                "describe": "tùy biến | arg",
                "slots": {"arg": {"desc": "tham số"}},
            }

    item = CustomScenarioItem()
    assert scenario_string(item, variant="slots") == "[ intent | tùy biến [ slot | tham số ] ]"


def test_scenario_string_invalid_variant() -> None:
    """Kiểm tra ném lỗi ValueError khi truyền biến thể không được hỗ trợ."""
    intent = {"intent": "TEST", "describe": "test"}
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        scenario_string(intent, variant="unsupported_variant")


def test_scenario_string_nonexistent_intent_string() -> None:
    """Kiểm tra ném lỗi KeyError khi tra cứu tên intent không tồn tại trong ngân hàng kịch bản."""
    with pytest.raises(KeyError, match="Không tìm thấy intent"):
        scenario_string("INTENT_HOAN_TOAN_KHONG_TON_TAI", variant="name")


def test_supported_variants_constant() -> None:
    """Kiểm tra hằng số SUPPORTED_VARIANTS chứa đầy đủ 4 biến thể RAF."""
    assert set(SUPPORTED_VARIANTS) == {"name", "slots", "primed", "handmade"}


# ==============================================================================
# 3. Kiểm thử Validator với Schema hợp lệ (core/validator.py)
# ==============================================================================


def test_validator_valid_intent_passes(sample_valid_intent: dict) -> None:
    """Kiểm tra intent chuẩn thỏa mãn tất cả quy tắc schema và không có bất kỳ lỗi nào."""
    errors = validate_intent(sample_valid_intent)
    assert errors == [], f"Intent hợp lệ không được chứa lỗi: {errors}"
    assert is_valid_intent(sample_valid_intent) is True


def test_validator_all_category_whitelist(sample_valid_intent: dict) -> None:
    """Kiểm tra toàn bộ các danh mục trong CATEGORY_WHITELIST đều được chấp thuận."""
    for cat in CATEGORY_WHITELIST:
        intent_copy = dict(sample_valid_intent, category=cat)
        assert validate_intent(intent_copy) == [], f"Danh mục '{cat}' phải hợp lệ."


def test_validator_all_allowed_slot_types(sample_valid_intent: dict) -> None:
    """Kiểm tra toàn bộ kiểu dữ liệu slot trong ALLOWED_SLOT_TYPES (text, number, label, filename)."""
    for slot_type in ALLOWED_SLOT_TYPES:
        intent_copy = dict(
            sample_valid_intent,
            slots={
                "test_slot": {
                    "type": slot_type,
                    "regex": r"([A-Za-z0-9]+)",
                    "required": True,
                }
            },
            templates={"latex": "<<test_slot>>"},
        )
        assert validate_intent(intent_copy) == [], f"Kiểu slot '{slot_type}' phải hợp lệ."


def test_validator_all_allowed_kinds(sample_valid_intent: dict) -> None:
    """Kiểm tra toàn bộ các kind trong ALLOWED_KINDS (snippet, preamble, document)."""
    for k in ALLOWED_KINDS:
        intent_copy = dict(sample_valid_intent, kind=k)
        assert validate_intent(intent_copy) == [], f"Kind '{k}' phải hợp lệ."


def test_validator_out_of_scope_special_intent() -> None:
    """Kiểm tra trường hợp đặc biệt OUT_OF_SCOPE: cho phép aliases rỗng và slots rỗng."""
    oos_intent = {
        "intent": "OUT_OF_SCOPE",
        "category": "OUT_OF_SCOPE",
        "describe": "Ngoài phạm vi hỗ trợ của trợ lý LaTeX tiếng Việt.",
        "aliases": [],
        "slots": {},
        "templates": {"latex": "", "python": "", "cpp": ""},
        "kind": "snippet",
        "requires": [],
    }
    assert validate_intent(oos_intent) == []
    assert is_valid_intent(oos_intent) is True


def test_validator_real_bank_intents_file() -> None:
    """Kiểm tra toàn bộ các intent LaTeX thực tế trong data/intents.json đều hợp lệ."""
    file_errors = validate_intents_file(DEFAULT_INTENTS_PATH, domain="latex")
    assert file_errors == {}, f"Phát hiện lỗi trong intents.json: {file_errors}"


# ==============================================================================
# 4. Kiểm thử Validator với Schema không hợp lệ (core/validator.py)
# ==============================================================================


def test_validator_invalid_non_dict() -> None:
    """Kiểm tra báo lỗi khi dữ liệu intent không phải là một từ điển dict."""
    errors = validate_intent(["not", "a", "dict"])  # type: ignore[arg-type]
    assert any("Dữ liệu intent phải là dict" in e for e in errors)


def test_validator_missing_required_fields(sample_valid_intent: dict) -> None:
    """Kiểm tra báo lỗi khi thiếu các trường bắt buộc cấp cao."""
    # Thiếu intent
    bad_intent = dict(sample_valid_intent)
    del bad_intent["intent"]
    assert any("Trường 'intent' bắt buộc" in e for e in validate_intent(bad_intent))

    # Thiếu category
    bad_cat = dict(sample_valid_intent)
    del bad_cat["category"]
    assert any("Trường 'category' bắt buộc" in e for e in validate_intent(bad_cat))

    # Thiếu describe
    bad_desc = dict(sample_valid_intent)
    del bad_desc["describe"]
    assert any("Trường 'describe'" in e for e in validate_intent(bad_desc))

    # Thiếu aliases
    bad_aliases = dict(sample_valid_intent)
    del bad_aliases["aliases"]
    assert any("Trường 'aliases' bắt buộc" in e for e in validate_intent(bad_aliases))

    # Thiếu slots
    bad_slots = dict(sample_valid_intent)
    del bad_slots["slots"]
    assert any("Trường 'slots' bắt buộc" in e for e in validate_intent(bad_slots))

    # Thiếu templates
    bad_tpl = dict(sample_valid_intent)
    del bad_tpl["templates"]
    assert any("Trường 'templates' bắt buộc" in e for e in validate_intent(bad_tpl))

    # Thiếu kind
    bad_kind = dict(sample_valid_intent)
    del bad_kind["kind"]
    assert any("Thuộc tính 'kind'" in e for e in validate_intent(bad_kind))

    # Thiếu requires
    bad_req = dict(sample_valid_intent)
    del bad_req["requires"]
    assert any("Thuộc tính 'requires'" in e for e in validate_intent(bad_req))


def test_validator_invalid_intent_naming(sample_valid_intent: dict) -> None:
    """Kiểm tra quy tắc tên intent phải là UPPER_SNAKE_CASE."""
    # Chữ thường
    assert any("UPPER_SNAKE_CASE" in e for e in validate_intent(dict(sample_valid_intent, intent="math_tich_phan")))
    # Ký tự gạch ngang
    assert any("UPPER_SNAKE_CASE" in e for e in validate_intent(dict(sample_valid_intent, intent="MATH-TICH-PHAN")))
    # Chứa khoảng trắng
    assert any("UPPER_SNAKE_CASE" in e for e in validate_intent(dict(sample_valid_intent, intent="MATH TICH PHAN")))
    # Chuỗi rỗng
    assert any("chuỗi ký tự không rỗng" in e for e in validate_intent(dict(sample_valid_intent, intent="")))


def test_validator_duplicate_intent_name(sample_valid_intent: dict) -> None:
    """Kiểm tra phát hiện trùng lặp tên intent thông qua seen_intents."""
    seen: set[str] = set()
    errs1 = validate_intent(sample_valid_intent, seen_intents=seen)
    assert errs1 == []

    # Validate lại intent đó
    errs2 = validate_intent(sample_valid_intent, seen_intents=seen)
    assert any("trùng lặp" in e for e in errs2)


def test_validator_invalid_category(sample_valid_intent: dict) -> None:
    """Kiểm tra danh mục không thuộc category whitelist."""
    bad_cat = dict(sample_valid_intent, category="khong_ton_tai")
    errs = validate_intent(bad_cat)
    assert any("không thuộc whitelist" in e for e in errs)


def test_validator_invalid_describe(sample_valid_intent: dict) -> None:
    """Kiểm tra trường describe rỗng hoặc không phải chuỗi."""
    assert any("không được để rỗng" in e for e in validate_intent(dict(sample_valid_intent, describe="")))
    assert any("không được để rỗng" in e for e in validate_intent(dict(sample_valid_intent, describe="   ")))
    assert any("không được để rỗng" in e for e in validate_intent(dict(sample_valid_intent, describe=None)))


def test_validator_aliases_count_violations(sample_valid_intent: dict) -> None:
    """Kiểm tra ràng buộc 5-8 câu aliases tiếng Việt."""
    # Dưới 5 câu
    bad_few = dict(sample_valid_intent, aliases=["câu 1", "câu 2", "câu 3"])
    assert any("5 đến 8 câu" in e for e in validate_intent(bad_few))

    # Trên 8 câu
    bad_many = dict(sample_valid_intent, aliases=[f"câu {i}" for i in range(1, 10)])
    assert any("5 đến 8 câu" in e for e in validate_intent(bad_many))

    # Phần tử alias rỗng
    bad_empty_item = dict(sample_valid_intent, aliases=["c1", "c2", "  ", "c4", "c5"])
    assert any("không rỗng" in e for e in validate_intent(bad_empty_item))


def test_validator_slot_regex_violations(sample_valid_intent: dict) -> None:
    """Kiểm tra quy tắc regex của slot: đúng 1 capturing group."""
    # 0 capturing group
    bad_0_group = dict(
        sample_valid_intent,
        slots={"s": {"type": "text", "regex": r"[a-z]+"}},
        templates={"latex": "<<s>>"},
    )
    assert any("chính xác đúng 1 capturing group" in e for e in validate_intent(bad_0_group))

    # 2 capturing groups
    bad_2_groups = dict(
        sample_valid_intent,
        slots={"s": {"type": "text", "regex": r"([a-z]+)-([0-9]+)"}},
        templates={"latex": "<<s>>"},
    )
    assert any("chính xác đúng 1 capturing group" in e for e in validate_intent(bad_2_groups))

    # Regex lỗi cú pháp
    bad_syntax = dict(
        sample_valid_intent,
        slots={"s": {"type": "text", "regex": r"([a-z+"}},
        templates={"latex": "<<s>>"},
    )
    assert any("không hợp lệ" in e for e in validate_intent(bad_syntax))


def test_validator_slot_type_violation(sample_valid_intent: dict) -> None:
    """Kiểm tra báo lỗi khi slot type không nằm trong ALLOWED_SLOT_TYPES."""
    bad_type = dict(
        sample_valid_intent,
        slots={"s": {"type": "unsupported_type", "regex": r"([0-9]+)"}},
        templates={"latex": "<<s>>"},
    )
    assert any("không hợp lệ" in e for e in validate_intent(bad_type))


def test_validator_template_placeholder_mismatch(sample_valid_intent: dict) -> None:
    """Kiểm tra phát hiện không khớp giữa placeholder <<slot>> và khai báo slots."""
    # Placeholder xuất hiện trong template nhưng không khai báo trong slots
    bad_extra_placeholder = dict(
        sample_valid_intent,
        templates={"latex": sample_valid_intent["templates"]["latex"] + " <<chua_khai_bao>>"},
    )
    assert any("chưa được khai báo trong 'slots'" in e for e in validate_intent(bad_extra_placeholder))

    # Slot khai báo nhưng không dùng placeholder trong template
    bad_unused_slot = dict(
        sample_valid_intent,
        templates={"latex": "\\int_{<<lower>>}^{<<upper>>}"},  # Thiếu <<expr>>
    )
    assert any("không xuất hiện dưới dạng <<slot>>" in e for e in validate_intent(bad_unused_slot))


def test_validator_kind_and_requires_violations(sample_valid_intent: dict) -> None:
    """Kiểm tra vi phạm thuộc tính kind và requires."""
    # Kind không hợp lệ
    bad_kind = dict(sample_valid_intent, kind="invalid_kind")
    assert any("Thuộc tính 'kind'" in e for e in validate_intent(bad_kind))

    # Requires không phải list
    bad_req = dict(sample_valid_intent, requires="amsmath")
    assert any("Thuộc tính 'requires'" in e for e in validate_intent(bad_req))

    # Post rỗng
    bad_post = dict(sample_valid_intent, post="   ")
    assert any("Thuộc tính 'post'" in e for e in validate_intent(bad_post))


def test_intent_validator_raise_on_error(sample_valid_intent: dict) -> None:
    """Kiểm tra IntentValidator ném ngoại lệ IntentValidationError khi raise_on_error=True."""
    validator = IntentValidator()
    # Danh sách hợp lệ
    assert validator.validate_intents([sample_valid_intent]) == {}

    # Danh sách có lỗi và raise_on_error=True
    bad_intent = dict(sample_valid_intent, intent="invalid_name")
    with pytest.raises(IntentValidationError) as exc_info:
        validator.validate_intents([bad_intent], raise_on_error=True)
    assert "invalid_name" in exc_info.value.errors

    # Hàm validate_intents ngoài module
    all_errs = validate_intents([sample_valid_intent, bad_intent])
    assert "invalid_name" in all_errs
