"""
Module trích xuất tham số dựa trên biểu thức chính quy và chuẩn hóa kiểu dữ liệu (core/slots.py).

Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Kiến trúc luồng xử lý:
  text -> preprocess (NFC, chuẩn hóa ký hiệu toán học, tách từ pyvi cho PhoBERT)
    -> PhoBERT classifier: p = softmax over intents
       - if max p >= tau: intent accepted
       - else retrieval top-3 over aliases + scenario strings -> ask the user to choose
    -> slot extraction (regex per intent) -> if a required slot is missing: ask for exactly the missing slots
    -> render latex template (text slots escaped) -> output
- Quy tắc trung thực (Honesty Rule #1):
  Dự án là một hệ thống kịch bản đơn giản hóa lấy cảm hứng từ Retrieve-and-Fill (RAF),
  bước trích xuất slot dựa trên biểu thức chính quy (simplified regex matching per slot)
  trên văn bản đã chuẩn hóa nhưng chưa tách từ (normalized non-segmented text),
  không sử dụng span-pointer decoder nguyên bản của RAF.
- Chuẩn hóa kiểu dữ liệu chặt chẽ (Type Normalizers):
  + number: Chuẩn hóa dấu (+, -), hệ số đơn vị ngầm định (implicit unit coefficients: '' -> '1', '+' -> '1', '-' -> '-1'),
    dấu thập phân (chuyển ',' tiếng Việt sang '.', '.5' -> '0.5', '5.' -> '5'), khử số 0 âm (-0 -> 0),
    và từ chối các giá trị không phải số hợp lệ (non-numbers, NaN, Inf).
  + text: Thu gọn khoảng trắng thừa (collapsing whitespace), bắt buộc không rỗng (non-empty),
    giới hạn độ dài tối đa 200 ký tự (max 200 chars theo config.MAX_TEXT_SLOT_LENGTH), không thoát ký tự LaTeX (no escaping).
  + label: Chuẩn hóa thành dạng ASCII slug chữ thường nối bởi dấu gạch ngang (ASCII lowercase slug with dashes).
  + filename: Kiểm tra ký tự ASCII thuần túy, cấm duyệt thư mục ('..'), cấm dấu gạch chéo đứng đầu ('/'),
    và kiểm thực phần mở rộng hình ảnh hợp lệ đối chiếu với config.FIGURE_EXTS.
- Ràng buộc slot được khai báo trong dữ liệu và kiểm tra chung, không định tuyến theo tên intent.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Final

from core.config import (
    DEFAULT_INTENTS_PATH,
    FORBIDDEN_LATEX_CONTROL_WORDS,
    FIGURE_EXTS,
    MAX_TEXT_SLOT_LENGTH,
)
from core.preprocess import normalize, normalize_spoken_math

# Biểu thức chính quy cho khoảng trắng liên tiếp
_WHITESPACE_REGEX: Final[re.Pattern[str]] = re.compile(r"\s+")

# Biểu thức chính quy phát hiện ký tự không phải chữ cái và số cho label slug
_LABEL_NON_ALPHANUM_REGEX: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_SPOKEN_MATH_CUE_REGEX: Final[re.Pattern[str]] = re.compile(
    r"\b(?:cộng|trừ|chia|âm|dương|bình\s+phương|mũ|bằng)\b|"
    r"[A-Za-z]\s*\^\s*\d|\d\s*[A-Za-z]", re.IGNORECASE
)
_KEYWORD_TOKEN_REGEX: Final[re.Pattern[str]] = re.compile(r"[^\W\d_]+", re.UNICODE)

# Bộ nhớ đệm ngân hàng kịch bản trong tiến trình để tối ưu hiệu năng
_CACHED_INTENTS_BANK: dict[str, dict[str, Any]] | None = None


class SlotResult(tuple):
    """
    Kết quả trích xuất tham số slot kế thừa 2-tuple (slots, missing) của Python.

    Hỗ trợ hai phương thức truy xuất:
    1. Unpacking dạng tuple: `slots, missing = extract_slots(text, intent)`
    2. Truy cập theo thuộc tính: `result.slots`, `result.missing`, `result.errors`, `result.intent`, `result.is_complete`
    """

    def __new__(
        cls,
        slots: dict[str, str],
        missing: list[str],
        errors: list[str] | None = None,
        intent: str = "",
        error_slots: list[str] | None = None,
    ) -> SlotResult:
        # Khởi tạo tuple 2 phần tử chuẩn (slots, missing)
        instance = super().__new__(cls, (slots, missing))
        instance._slots = slots
        instance._missing = missing
        instance._errors = errors if errors is not None else []
        instance._error_slots = error_slots if error_slots is not None else []
        instance._intent = intent
        return instance

    @property
    def slots(self) -> dict[str, str]:
        """Từ điển các tham số slot đã trích xuất và chuẩn hóa."""
        return self._slots

    @property
    def missing(self) -> list[str]:
        """Danh sách các tham số bắt buộc bị thiếu hoặc vi phạm kiểm thực."""
        return self._missing

    @property
    def errors(self) -> list[str]:
        """Danh sách chi tiết các thông báo lỗi chuẩn hóa hoặc kiểm thực."""
        return self._errors

    @property
    def error_slots(self) -> list[str]:
        """Tên slot tương ứng từng lỗi để pipeline không hỏi nhập lỗi như slot."""
        return self._error_slots

    @property
    def intent(self) -> str:
        """Mã định danh intent mục tiêu."""
        return self._intent

    @property
    def is_complete(self) -> bool:
        """True nếu trích xuất đủ tất cả tham số bắt buộc và không có lỗi."""
        return len(self._missing) == 0 and len(self._errors) == 0


def normalize_number(val: Any) -> str:
    """
    Chuẩn hóa giá trị số (number slot):
    - Chuẩn hóa dấu (+5 -> '5', -5 -> '-5').
    - Chuẩn hóa hệ số đơn vị ngầm định (implicit unit coefficients):
      + '' (chuỗi rỗng) -> '1'
      + '+' -> '1'
      + '-' -> '-1'
    - Chuẩn hóa dấu thập phân: dấu phẩy tiếng Việt ',' sang dấu chấm '.',
      '.5' / ',5' -> '0.5', '-.5' / '-,5' -> '-0.5', '5.' -> '5'.
    - Khử số 0 âm (-0, -0.0 -> '0').
    - Từ chối các giá trị không phải số hợp lệ (non-numbers, NaN, Inf).

    Tham số:
        val: Giá trị số đầu vào (chuỗi ký tự, số nguyên hoặc số thực).

    Trả về:
        Chuỗi biểu diễn số đã chuẩn hóa.

    Ngoại lệ:
        ValueError: Nếu giá trị không thể chuyển đổi thành số hợp lệ.
    """
    if val is None:
        raise ValueError("Giá trị tham số kiểu 'number' không được là None.")

    raw_str = str(val).strip()
    if not raw_str:
        # Hệ số đơn vị ngầm định khi không có ký tự trước biến (ví dụ: x^2 -> hệ số 1)
        return "1"

    # Xử lý các hệ số dấu ngầm định đứng trước biến (ví dụ: +x -> 1, -x -> -1)
    if raw_str == "+":
        return "1"
    if raw_str == "-":
        return "-1"

    # Thu gọn khoảng trắng giữa dấu và chữ số (ví dụ: "+ 5" -> "+5", "- 2" -> "-2")
    s = re.sub(r"^([+-])\s+", r"\1", raw_str)

    # Thay thế dấu phẩy thập phân kiểu tiếng Việt/châu Âu bằng dấu chấm
    s = s.replace(",", ".")

    # Loại bỏ dấu dương không cần thiết ở đầu
    s = s.removeprefix("+")

    # Chuẩn hóa số thập phân khuyết chữ số 0 ở phần nguyên (.5 -> 0.5, -.5 -> -0.5)
    if s.startswith("."):
        s = "0" + s
    elif s.startswith("-."):
        s = "-0." + s[2:]

    # Bỏ dấu chấm lửng ở cuối chuỗi (ví dụ: "5." -> "5")
    s = s.removesuffix(".")

    # Kiểm tra tính hợp lệ bằng chuyển đổi float
    try:
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            raise ValueError(f"Giá trị số không hợp lệ (NaN hoặc vô cực): '{val}'")
    except (ValueError, TypeError) as err:
        raise ValueError(f"Không thể phân tích giá trị thành số hợp lệ: '{val}'") from err

    # Chuẩn hóa trường hợp số 0 âm (-0, -0.0) về "0"
    if f == 0.0:
        return "0"

    return s


def normalize_text(val: Any) -> str:
    """
    Chuẩn hóa giá trị văn bản (text slot):
    - Thu gọn khoảng trắng thừa liên tiếp thành khoảng trắng đơn và cắt tỉa 2 đầu.
    - Kiểm tra không rỗng (non-empty).
    - Giới hạn độ dài tối đa 200 ký tự (max 200 chars theo config.MAX_TEXT_SLOT_LENGTH).
    - Không thoát ký tự đặc biệt LaTeX (no escaping, bước thoát ký tự do renderer thực hiện).

    Tham số:
        val: Giá trị văn bản đầu vào.

    Trả về:
        Chuỗi văn bản đã chuẩn hóa.

    Ngoại lệ:
        ValueError: Nếu văn bản rỗng hoặc vượt quá độ dài tối đa cho phép.
    """
    if val is None:
        raise ValueError("Giá trị tham số kiểu 'text' không được là None.")

    collapsed = _WHITESPACE_REGEX.sub(" ", str(val)).strip()
    if not collapsed:
        raise ValueError("Văn bản tham số kiểu 'text' không được để trống.")

    if len(collapsed) > MAX_TEXT_SLOT_LENGTH:
        raise ValueError(
            f"Văn bản tham số kiểu 'text' vượt quá giới hạn tối đa {MAX_TEXT_SLOT_LENGTH} ký tự "
            f"(nhận được {len(collapsed)} ký tự)."
        )

    # Bảo toàn nguyên vẹn văn bản, không thoát ký tự đặc biệt LaTeX tại bước này
    return collapsed


def normalize_math(val: Any) -> str:
    """Giữ ký hiệu và macro LaTeX toán học sau khi loại khoảng trắng không cần thiết."""
    raw_expression = str(val).strip()
    forbidden = r"\\(?:" + "|".join(map(re.escape, FORBIDDEN_LATEX_CONTROL_WORDS)) + r")(?![A-Za-z])"
    if re.search(forbidden, raw_expression, flags=re.IGNORECASE):
        raise ValueError("Biểu thức toán chứa lệnh LaTeX không an toàn.")
    preserved_text: list[str] = []

    def preserve_text_macro(match: re.Match[str]) -> str:
        preserved_text.append(match.group(0))
        return f"LATEXTEXTTOKEN{len(preserved_text) - 1}"

    expression = re.sub(
        r"\\(?:text|mbox|operatorname)\{[^{}]*\}", preserve_text_macro, raw_expression
    )
    # Whitespace can separate function names from arguments (e.g. ``sin x``).
    # Keep those spaces, while compacting around operators and protected macros.
    expression = _WHITESPACE_REGEX.sub(" ", expression)
    expression = re.sub(r"\s*([+*/=^(),{}])\s*", r"\1", expression)
    expression = re.sub(r"\s*(\\[A-Za-z]+(?:\{[^{}]*\})*)\s*", r"\1", expression)
    expression = re.sub(r"\s*(LATEXTEXTTOKEN\d+)\s*", r"\1", expression)
    expression = re.sub(r"\s*(LATEXPROTECTEDTOKEN\d+)\s*", r"\1", expression)
    if not expression:
        raise ValueError("Biểu thức toán không được để trống.")
    if len(expression) > MAX_TEXT_SLOT_LENGTH or not re.fullmatch(
        r"[A-Za-z0-9+*/=().,{}^\\\- ]+", expression
    ):
        raise ValueError("Biểu thức toán chứa ký tự không được hỗ trợ.")
    for index, text_macro in enumerate(preserved_text):
        expression = expression.replace(f"LATEXTEXTTOKEN{index}", text_macro)
    return expression


def normalize_label(val: Any) -> str:
    """
    Chuẩn hóa giá trị nhãn tham chiếu LaTeX (label slot):
    - Chuyển đổi thành dạng ASCII slug chữ thường nối bằng dấu gạch ngang (ASCII lowercase slug with dashes).
    - Khử dấu tiếng Việt (đ/Đ -> d, các ký tự thanh điệu Unicode về dạng không dấu).
    - Thay thế các ký tự không phải chữ cái và chữ số thành dấu gạch ngang '-'.
    - Thu gọn dấu gạch ngang liên tiếp và cắt bỏ gạch ngang ở hai đầu.
    - Từ chối nếu nhãn rỗng sau khi chuẩn hóa.

    Tham số:
        val: Giá trị nhãn đầu vào.

    Trả về:
        Chuỗi ASCII slug chữ thường hợp lệ.

    Ngoại lệ:
        ValueError: Nếu nhãn không hợp lệ hoặc trở thành chuỗi rỗng sau khi chuẩn hóa.
    """
    if val is None:
        raise ValueError("Giá trị tham số kiểu 'label' không được là None.")

    raw = str(val).strip()
    if not raw:
        raise ValueError("Nhãn tham số kiểu 'label' không được để trống.")

    # Thay thế thủ công ký tự 'đ' và 'Đ' đặc thù tiếng Việt trước khi phân rã NFKD
    cleaned = raw.replace("đ", "d").replace("Đ", "D")

    # Phân rã dạng tổ hợp NFKD và lọc bỏ các dấu thanh điệu ngoài bảng mã ASCII
    ascii_str = (
        unicodedata.normalize("NFKD", cleaned)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )

    # Thay thế mọi ký tự không phải chữ số hoặc chữ cái bằng dấu gạch ngang
    slug = _LABEL_NON_ALPHANUM_REGEX.sub("-", ascii_str).strip("-")

    if not slug:
        raise ValueError(f"Nhãn 'label' không hợp lệ hoặc không thể chuyển thành ASCII slug: '{val}'.")

    return slug


def normalize_filename(val: Any) -> str:
    """
    Chuẩn hóa và kiểm thực tên tệp hình ảnh (filename slot):
    - Kiểm tra ký tự ASCII thuần túy (ASCII check).
    - Ngăn chặn tấn công duyệt thư mục (disallowing directory traversal '..').
    - Ngăn chặn đường dẫn tuyệt đối hoặc dấu gạch chéo đứng đầu (disallowing leading '/').
    - Kiểm thực phần mở rộng tệp nằm trong danh sách config.FIGURE_EXTS (.png, .jpg, .jpeg, .pdf).

    Tham số:
        val: Tên tệp hoặc đường dẫn tương đối của tệp hình ảnh.

    Trả về:
        Tên tệp đã được cắt tỉa khoảng trắng.

    Ngoại lệ:
        ValueError: Nếu tên tệp rỗng, chứa ký tự phi ASCII, chứa '..', bắt đầu bằng '/',
                    hoặc có phần mở rộng không hợp lệ.
    """
    if val is None:
        raise ValueError("Giá trị tham số kiểu 'filename' không được là None.")

    cleaned = str(val).strip()
    if not cleaned:
        raise ValueError("Tên tệp tham số kiểu 'filename' không được để trống.")

    if not cleaned.isascii():
        raise ValueError(f"Tên tệp phải là chuỗi ký tự ASCII thuần túy: '{val}'.")

    if ".." in cleaned:
        raise ValueError(f"Tên tệp không được chứa ký tự duyệt thư mục '..': '{val}'.")

    if cleaned.startswith("/") or cleaned.startswith("\\"):
        raise ValueError(f"Tên tệp không được bắt đầu bằng dấu gạch chéo '/': '{val}'.")

    ext = Path(cleaned).suffix.lower()
    if not ext or ext not in FIGURE_EXTS:
        raise ValueError(
            f"Phần mở rộng tệp '{ext}' không được hỗ trợ. "
            f"Các định dạng hợp lệ bao gồm: {tuple(FIGURE_EXTS)}."
        )

    return cleaned


def normalize_slot_value(val: Any, slot_type: str) -> str:
    """
    Điều phối chuẩn hóa giá trị tham số slot theo kiểu dữ liệu tương ứng.

    Tham số:
        val: Giá trị tham số thô cần chuẩn hóa.
        slot_type: Kiểu dữ liệu tham số ('number', 'text', 'label', 'filename').

    Trả về:
        Chuỗi giá trị đã được chuẩn hóa.
    """
    normalized_type = slot_type.strip().lower() if slot_type else "text"
    if normalized_type == "number":
        return normalize_number(val)
    elif normalized_type == "math":
        return normalize_math(val)
    elif normalized_type == "text":
        return normalize_text(val)
    elif normalized_type == "label":
        return normalize_label(val)
    elif normalized_type == "filename":
        return normalize_filename(val)
    else:
        # Mặc định xử lý theo kiểu text nếu kiểu dữ liệu chưa được định nghĩa riêng
        return normalize_text(val)


def _load_intents_bank(intents_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Nạp và lưu đệm ngân hàng intent từ data/intents.json.

    Tham số:
        intents_path: Đường dẫn tệp JSON intent (mặc định lấy từ core.config.DEFAULT_INTENTS_PATH).

    Trả về:
        Từ điển ánh xạ intent_name -> intent_data_dict.
    """
    global _CACHED_INTENTS_BANK
    if _CACHED_INTENTS_BANK is not None and intents_path is None:
        return _CACHED_INTENTS_BANK

    target_path = intents_path or DEFAULT_INTENTS_PATH
    if not target_path.exists():
        return {}

    with target_path.open(encoding="utf-8") as f:
        data = json.load(f)

    bank: dict[str, dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "intent" in item:
                bank[item["intent"]] = item

    if intents_path is None:
        _CACHED_INTENTS_BANK = bank

    return bank


def _resolve_intent_definition(
    intent: str | dict[str, Any] | Any,
    intents_bank: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Phân giải tham số intent thành (intent_name, slots_dict).

    Tham số:
        intent: Tên intent dạng chuỗi, đối tượng dict kịch bản, hoặc đối tượng có thuộc tính slots.
        intents_bank: Ngân hàng kịch bản dự phòng nếu intent là chuỗi tên.

    Trả về:
        Tuple gồm (tên intent, từ điển các định nghĩa slot).
    """
    if isinstance(intent, str):
        bank = intents_bank if intents_bank is not None else _load_intents_bank()
        intent_data = bank.get(intent)
        if intent_data is None:
            return intent, {}
        return intent, intent_data.get("slots", {})

    if isinstance(intent, dict):
        intent_name = str(intent.get("intent", ""))
        slots_data = intent.get("slots", {})
        return intent_name, slots_data

    # Duck-typing cho đối tượng Scenario hoặc tương tự
    intent_name = getattr(intent, "intent", "")
    slots_data = getattr(intent, "slots", {})
    return str(intent_name), slots_data


def _leading_anchor_keywords(pattern: str | None) -> list[str]:
    """Lấy các từ khóa literal đầu regex để slot khác có thể cắt giá trị tham lam."""
    if not pattern:
        return []
    candidate = pattern.lstrip("^")
    group = re.match(r"\(\?:((?:\\.|[^()])*)\)", candidate)
    if not group:
        return []
    anchors: list[str] = []
    for alternative in group.group(1).split("|"):
        phrase = re.sub(r"\\[wds]", " ", alternative).replace("\\.", ".")
        phrase = re.sub(r"[^\wÀ-ỹ]+", " ", phrase, flags=re.UNICODE).strip()
        if phrase and _KEYWORD_TOKEN_REGEX.search(phrase):
            anchors.append(phrase)
    return anchors


def _math_context(text: str) -> bool:
    """Nhận diện ngữ cảnh toán rộng để chuẩn hóa số nói nhưng không chạm text slot."""
    return bool(
        _SPOKEN_MATH_CUE_REGEX.search(text)
        or re.search(r"\b(?:tổng|tích|căn|phân số|tử số|mẫu số|giới hạn|tiến tới)\b", text, re.I)
        or re.search(r"\\[A-Za-z]+", text)
    )


def extract_slots(
    text: str,
    intent: str | dict[str, Any] | Any,
    intents_bank: dict[str, dict[str, Any]] | None = None,
) -> SlotResult:
    """
    Trích xuất tham số từ văn bản người dùng dựa trên biểu thức chính quy của từng slot trong kịch bản.

    Tuân thủ quy tắc kiến trúc AGENTS.md:
    - Tiền xử lý văn bản: Chuẩn hóa Unicode NFC, chuẩn hóa ký hiệu toán học (² -> ^2), thu gọn khoảng trắng,
      giữ nguyên văn bản không tách từ vựng (normalized non-segmented text).
    - Khớp regex đơn giản hóa theo từng slot (simplified regex matching per slot).
    - Chuẩn hóa kiểu dữ liệu nghiêm ngặt theo từng loại slot (number, text, label, filename).
    - Thực thi các móc kiểm thực (validation hooks) sau khi trích xuất (ví dụ: a != 0 cho phương trình bậc hai).
    - Báo cáo chính xác các slot bắt buộc còn thiếu hoặc vi phạm ràng buộc.

    Tham số:
        text: Chuỗi văn bản truy vấn của người dùng.
        intent: Tên intent (str), từ điển intent (dict), hoặc đối tượng kịch bản (Scenario).
        intents_bank: Ngân hàng intent tùy chọn để tra cứu khi intent là chuỗi.

    Trả về:
        Đối tượng SlotResult (kế thừa 2-tuple (slots, missing)) chứa bảng ánh xạ slot và danh sách slot thiếu/lỗi.
    """
    # Hỗ trợ đổi thứ tự tham số nếu caller truyền (intent, text)
    if not isinstance(text, str) and isinstance(intent, str):
        text, intent = str(intent), text

    # Bước 1: Tiền xử lý văn bản bằng core.preprocess.normalize (NFC, look-alikes, không phân đoạn từ)
    normalized_text = normalize(text)

    # Bước 2: Phân giải kịch bản mục tiêu và định nghĩa các slot
    intent_name, slots_def = _resolve_intent_definition(intent, intents_bank)

    extracted_slots: dict[str, str] = {}
    missing_slots: list[str] = []
    errors: list[str] = []
    error_slots: list[str] = []

    # Duyệt qua từng tham số slot được định nghĩa trong kịch bản
    slots_items = slots_def.items() if isinstance(slots_def, dict) else []
    spoken_math_text = normalize_spoken_math(normalized_text) if _math_context(normalized_text) else normalized_text
    has_zero_equation = bool(re.search(r"=\s*0\b", spoken_math_text))

    for slot_name, s_def in slots_items:
        # Trích xuất thông tin định nghĩa slot (hỗ trợ cả dict và đối tượng SlotDefinition)
        if isinstance(s_def, dict):
            slot_type = str(s_def.get("type", "text"))
            pattern = s_def.get("regex") or s_def.get("pattern")
            required = bool(s_def.get("required", False))
            default = s_def.get("default")
            absent_means = s_def.get("absent_means")
        else:
            slot_type = str(getattr(s_def, "type", "text"))
            pattern = getattr(s_def, "regex", None) or getattr(s_def, "pattern", None)
            required = bool(getattr(s_def, "required", False))
            default = getattr(s_def, "default", None)
            absent_means = getattr(s_def, "absent_means", None)

        slot_text = (
            spoken_math_text if slot_type in {"number", "math"} else normalized_text
        )

        matched_raw_val: str | None = None

        if pattern:
            try:
                # A bare catch-all cannot prove which part of a sentence is the slot value.
                catch_all_value = str(pattern).strip() in {"(.+)", "(.*)", "(.+?)"}
                if catch_all_value:
                    pattern = None
            except (TypeError, AttributeError):
                pattern = None

        if pattern:
            try:
                # 1. Thử khớp trực tiếp mẫu regex gốc
                match = re.search(pattern, slot_text, flags=re.IGNORECASE | re.UNICODE)
                if match:
                    raw_group = match.group(1)
                    if raw_group is not None:
                        capture_start, capture_end = match.span(1)
                        cut_at = capture_end
                        for other_name, other_spec in slots_items:
                            if other_name == slot_name:
                                continue
                            other_pattern = (
                                other_spec.get("regex") or other_spec.get("pattern")
                                if isinstance(other_spec, dict)
                                else getattr(other_spec, "regex", None)
                            )
                            for anchor in _leading_anchor_keywords(other_pattern):
                                anchor_pattern = re.escape(anchor).replace(r"\ ", r"\s+")
                                anchor_match = re.search(
                                    rf"(?<!\w){anchor_pattern}(?!\w)",
                                    slot_text[capture_start:cut_at],
                                    flags=re.IGNORECASE | re.UNICODE,
                                )
                                if anchor_match:
                                    cut_at = min(cut_at, capture_start + anchor_match.start())
                        matched_raw_val = slot_text[capture_start:cut_at].strip()

                # 2. Nếu là kiểu số (number) và chưa khớp do hệ số đơn vị ngầm định (ví dụ: x^2, -x^2, +x):
                # Thử áp dụng mẫu linh hoạt cho phép chuỗi số rỗng hoặc dấu đơn (+, -)
                if (
                    matched_raw_val is None
                    and slot_type == "number"
                    and r"\s*x" in str(pattern).lower()
                ):
                    flex_pattern = re.sub(r"\\d\+", r"\\d*", str(pattern))
                    flex_pattern = flex_pattern.replace(r"\.", r"[.,]")
                    flex_match = re.search(flex_pattern, slot_text, flags=re.IGNORECASE | re.UNICODE)
                    if flex_match:
                        flex_group = flex_match.group(1)
                        if flex_group is not None:
                            matched_raw_val = flex_group.strip()

            except re.error:
                # Bỏ qua lỗi cú pháp regex không mong muốn một cách an toàn
                pass

        # Chỉ suy số 0 cho term vắng mặt khi slot khai báo rõ và câu là phương trình = 0.
        if matched_raw_val is None and absent_means is not None and slot_type == "number" and has_zero_equation:
            matched_raw_val = str(absent_means)

        # Bước 3: Chuẩn hóa giá trị trích xuất được hoặc áp dụng giá trị mặc định
        if matched_raw_val is not None:
            try:
                normalized_val = normalize_slot_value(matched_raw_val, slot_type)
                extracted_slots[slot_name] = normalized_val
                not_equal = (
                    s_def.get("not_equal")
                    if isinstance(s_def, dict)
                    else getattr(s_def, "not_equal", None)
                )
                if slot_type == "number" and not_equal is not None:
                    if float(normalized_val) == float(not_equal):
                        constraint_error = (
                            s_def.get("constraint_error")
                            if isinstance(s_def, dict)
                            else getattr(s_def, "constraint_error", None)
                        )
                        error = str(constraint_error or "Giá trị slot vi phạm ràng buộc.")
                        errors.append(error)
                        error_slots.append(slot_name)
                        if required:
                            missing_slots.append(slot_name)
            except ValueError as val_err:
                errors.append(str(val_err))
                error_slots.append(slot_name)
                if required:
                    missing_slots.append(slot_name)
        elif default is not None:
            try:
                normalized_default = normalize_slot_value(default, slot_type)
                extracted_slots[slot_name] = normalized_default
            except ValueError:
                extracted_slots[slot_name] = str(default)
        elif required:
            missing_slots.append(slot_name)
        else:
            extracted_slots[slot_name] = ""

    return SlotResult(
        slots=extracted_slots,
        missing=missing_slots,
        errors=errors,
        intent=intent_name,
        error_slots=error_slots,
    )
