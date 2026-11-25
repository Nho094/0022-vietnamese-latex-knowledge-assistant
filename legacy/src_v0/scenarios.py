"""
Module quản lý Scenario Bank (ngân hàng kịch bản) và Schema Definition offline.

Tuân thủ kiến trúc AGENTS.md:
- Định nghĩa cấu trúc scenario chuẩn với: intent, utterances tiếng Việt mẫu,
  mô hình slot regex cho từng intent, và ánh xạ template (latex, python, cpp).
- Cung cấp các hàm offline để tải dữ liệu (load), kiểm tra tính hợp lệ (validate),
  trích xuất tham số regex (extract_slots), render template và truy hồi tương đồng (retrieve).
- Lấy cảm hứng từ mô hình đơn giản hóa Retrieve-and-Fill (RAF) theo Honesty Rule #1:
  bước trích xuất slot dựa trên regex thuần túy thay vì span-pointer decoder phức tạp.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Optional, Sequence, Union

from core.config import (
    DEFAULT_INTENTS_PATH,
    DEFAULT_RETRIEVAL_TOP_K,
    REQUIRED_TEMPLATE_KEYS,
    SUPPORTED_TARGETS,
    THETA_RETRIEVAL_THRESHOLD,
)


class ScenarioError(Exception):
    """Lớp ngoại lệ cơ sở cho các lỗi liên quan đến kịch bản."""
    pass


class ScenarioNotFoundError(ScenarioError):
    """Ngoại lệ ném ra khi không tìm thấy intent chỉ định trong ngân hàng kịch bản."""
    pass


class ScenarioValidationError(ScenarioError):
    """Ngoại lệ ném ra khi kịch bản không đáp ứng ràng buộc schema."""
    def __init__(self, errors: dict[str, list[str]]) -> None:
        self.errors = errors
        summary = "; ".join(f"{k}: {', '.join(v)}" for k, v in list(errors.items())[:5])
        super().__init__(f"Phát hiện lỗi kiểm thực schema kịch bản ({len(errors)} intent lỗi): {summary}")


@dataclass
class SlotDefinition:
    """
    Định nghĩa một slot (tham số) trong kịch bản.

    Thuộc tính:
        name: Tên của slot (trùng với placeholder trong template).
        required: Bắt buộc người dùng cung cấp hay không.
        default: Giá trị mặc định nếu người dùng không cung cấp.
        pattern: Biểu thức chính quy regex dùng để bóc tách giá trị từ câu tiếng Việt.
    """
    name: str
    required: bool = False
    default: str = ""
    pattern: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi thành từ điển lưu trữ JSON."""
        return {
            "required": self.required,
            "default": self.default,
            "pattern": self.pattern,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> SlotDefinition:
        """Tạo SlotDefinition từ dữ liệu từ điển."""
        return cls(
            name=name,
            required=bool(data.get("required", False)),
            default=str(data.get("default", "")),
            pattern=str(data.get("pattern", "")),
        )


@dataclass
class TemplateMapping:
    """
    Ánh xạ mã mẫu cho các ngôn ngữ đích khác nhau.

    Thuộc tính:
        latex: Mã LaTeX tương ứng (bắt buộc có nội dung).
        python: Mã Python mẫu (mặc định chuỗi rỗng "").
        cpp: Mã C++ mẫu (mặc định chuỗi rỗng "").
        mathjax: Mã MathJax tương thích (tùy chọn).
    """
    latex: str
    python: str = ""
    cpp: str = ""
    mathjax: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        """Chuyển đổi sang từ điển JSON."""
        res = {
            "latex": self.latex,
            "python": self.python,
            "cpp": self.cpp,
        }
        if self.mathjax is not None:
            res["mathjax"] = self.mathjax
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateMapping:
        """Khởi tạo từ từ điển dữ liệu."""
        return cls(
            latex=str(data.get("latex", "")),
            python=str(data.get("python", "")),
            cpp=str(data.get("cpp", "")),
            mathjax=data.get("mathjax"),
        )


@dataclass
class Scenario:
    """
    Thực thể kịch bản (Scenario / Intent) đầy đủ trong hệ thống trợ lý LaTeX.

    Tuân thủ schema chuẩn:
        - intent: Tên định danh duy nhất (chữ in hoa, gạch dưới).
        - category: Nhóm ngữ nghĩa chính (ví dụ: math_expression, latex_document).
        - subcategory: Nhóm ngữ nghĩa chi tiết.
        - describe: Mô tả tác vụ bằng tiếng Việt.
        - aliases: Các cụm từ đồng nghĩa, từ khóa viết tắt.
        - sample_utterances: Danh sách câu tiếng Việt người dùng mẫu.
        - slots: Ánh xạ tên slot -> SlotDefinition kèm regex trích xuất.
        - templates: Ánh xạ template cho LaTeX, Python, C++.
        - requires: Danh sách package LaTeX cần khai báo trong preamble.
        - render_target: Mục tiêu render chính ('latex' hoặc 'mathjax').
        - source: Xuất xứ tài liệu kiến thức gốc.
        - notes: Ghi chú bổ sung.
    """
    intent: str
    category: str
    subcategory: str = ""
    describe: str = ""
    aliases: list[str] = field(default_factory=list)
    sample_utterances: list[str] = field(default_factory=list)
    slots: dict[str, SlotDefinition] = field(default_factory=dict)
    templates: TemplateMapping = field(default_factory=lambda: TemplateMapping(latex=""))
    requires: list[str] = field(default_factory=list)
    render_target: str = "latex"
    source: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Xuất kịch bản ra định dạng dict chuẩn để lưu file JSON."""
        return {
            "intent": self.intent,
            "category": self.category,
            "subcategory": self.subcategory,
            "describe": self.describe,
            "aliases": self.aliases,
            "sample_utterances": self.sample_utterances,
            "slots": {name: slot.to_dict() for name, slot in self.slots.items()},
            "templates": self.templates.to_dict(),
            "requires": self.requires,
            "render_target": self.render_target,
            "source": self.source,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        """Tạo đối tượng Scenario từ dictionary JSON."""
        slots_data = data.get("slots", {})
        slots = {
            s_name: SlotDefinition.from_dict(s_name, s_meta)
            for s_name, s_meta in slots_data.items()
        }
        templates_data = data.get("templates", {})
        templates = TemplateMapping.from_dict(templates_data)

        return cls(
            intent=str(data.get("intent", "")),
            category=str(data.get("category", "")),
            subcategory=str(data.get("subcategory", "")),
            describe=str(data.get("describe", "")),
            aliases=list(data.get("aliases", [])),
            sample_utterances=list(data.get("sample_utterances", [])),
            slots=slots,
            templates=templates,
            requires=list(data.get("requires", [])),
            render_target=str(data.get("render_target", "latex")),
            source=dict(data.get("source", {})),
            notes=str(data.get("notes", "")),
        )


class ScenarioBank:
    """
    Bộ chứa và quản lý ngân hàng kịch bản offline.
    Hỗ trợ tra cứu theo intent, lọc theo phân loại, và truy xuất danh sách.
    """
    def __init__(self, scenarios: Sequence[Scenario]) -> None:
        self._scenarios: list[Scenario] = list(scenarios)
        self._by_intent: dict[str, Scenario] = {s.intent: s for s in self._scenarios}

    def __len__(self) -> int:
        return len(self._scenarios)

    def __iter__(self) -> Iterable[Scenario]:
        return iter(self._scenarios)

    def __contains__(self, intent_name: object) -> bool:
        if isinstance(intent_name, str):
            return intent_name in self._by_intent
        return False

    def __getitem__(self, intent_name: str) -> Scenario:
        if intent_name not in self._by_intent:
            raise ScenarioNotFoundError(f"Không tìm thấy kịch bản cho intent: {intent_name}")
        return self._by_intent[intent_name]

    def get(self, intent_name: str, default: Optional[Scenario] = None) -> Optional[Scenario]:
        """Lấy kịch bản theo intent name an toàn."""
        return self._by_intent.get(intent_name, default)

    def all(self) -> list[Scenario]:
        """Trả về toàn bộ danh sách kịch bản."""
        return list(self._scenarios)

    def intents(self) -> list[str]:
        """Danh sách tên tất cả các intent."""
        return list(self._by_intent.keys())

    def categories(self) -> list[str]:
        """Danh sách các danh mục độc nhất."""
        return sorted({s.category for s in self._scenarios if s.category})

    def filter(
        self,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
    ) -> list[Scenario]:
        """Lọc kịch bản theo nhóm category hoặc subcategory."""
        results = self._scenarios
        if category is not None:
            results = [s for s in results if s.category == category]
        if subcategory is not None:
            results = [s for s in results if s.subcategory == subcategory]
        return results

    def validate(self) -> dict[str, list[str]]:
        """Kiểm tra tính hợp lệ toàn bộ ngân hàng kịch bản."""
        return validate_scenario_bank(self._scenarios)


def load_scenarios(path: Union[str, Path] = DEFAULT_INTENTS_PATH) -> list[Scenario]:
    """
    Đọc và phân tích file JSON chứa danh sách kịch bản.

    Tham số:
        path: Đường dẫn tới file intents.json (mặc định lấy từ config).

    Trả về:
        Danh sách các đối tượng Scenario.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File ngân hàng kịch bản không tồn tại: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Dữ liệu kịch bản phải là danh sách JSON, nhận được: {type(data).__name__}")

    return [Scenario.from_dict(item) for item in data]


def load_scenario_bank(path: Union[str, Path] = DEFAULT_INTENTS_PATH) -> ScenarioBank:
    """Tải ngân hàng kịch bản vào đối tượng quản lý ScenarioBank."""
    scenarios = load_scenarios(path)
    return ScenarioBank(scenarios)


def save_scenarios(
    scenarios: Union[ScenarioBank, Sequence[Scenario], Sequence[dict[str, Any]]],
    path: Union[str, Path] = DEFAULT_INTENTS_PATH,
) -> None:
    """
    Lưu danh sách kịch bản xuống file JSON với định dạng UTF-8 và thụt lề chuẩn.

    Tham số:
        scenarios: ScenarioBank, danh sách Scenario hoặc danh sách dict.
        path: Đường dẫn file lưu.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data_to_dump: list[dict[str, Any]] = []
    if isinstance(scenarios, ScenarioBank):
        data_to_dump = [s.to_dict() for s in scenarios.all()]
    elif isinstance(scenarios, Sequence):
        for item in scenarios:
            if isinstance(item, Scenario):
                data_to_dump.append(item.to_dict())
            elif isinstance(item, dict):
                data_to_dump.append(item)
            else:
                raise TypeError(f"Kiểu phần tử không được hỗ trợ: {type(item).__name__}")
    else:
        raise TypeError(f"Kiểu dữ liệu scenarios không hợp lệ: {type(scenarios).__name__}")

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data_to_dump, f, ensure_ascii=False, indent=2)


def validate_scenario(scenario: Union[Scenario, dict[str, Any]]) -> list[str]:
    """
    Kiểm tra tính hợp lệ của một kịch bản đơn lẻ theo các ràng buộc schema:
    1. Tên intent không rỗng và chỉ gồm ký tự hợp lệ.
    2. Category không rỗng.
    3. Sample utterances có ít nhất 1 câu mẫu tiếng Việt.
    4. Template mapping chứa đầy đủ các key (latex, python, cpp), trong đó latex không rỗng.
    5. Các slot có định dạng chuẩn và biểu thức regex biên dịch được thành công.

    Trả về:
        Danh sách chuỗi mô tả lỗi (rỗng nếu kịch bản hợp lệ).
    """
    errors: list[str] = []

    if isinstance(scenario, Scenario):
        data = scenario.to_dict()
    elif isinstance(scenario, dict):
        data = scenario
    else:
        return [f"Kiểu dữ liệu không hợp lệ: {type(scenario).__name__}"]

    intent = data.get("intent", "")
    if not intent or not isinstance(intent, str) or not intent.strip():
        errors.append("Trường 'intent' phải là chuỗi không rỗng.")
    elif not re.match(r"^[A-Z0-9_]+$", intent):
        errors.append(f"Tên intent '{intent}' chỉ được chứa chữ hoa, số và dấu gạch dưới.")

    category = data.get("category", "")
    if not category or not isinstance(category, str) or not category.strip():
        errors.append("Trường 'category' phải là chuỗi không rỗng.")

    utterances = data.get("sample_utterances", [])
    if not isinstance(utterances, list) or len(utterances) == 0:
        errors.append("Trường 'sample_utterances' phải là danh sách chứa ít nhất một câu mẫu tiếng Việt.")
    else:
        for idx, u in enumerate(utterances):
            if not isinstance(u, str) or not u.strip():
                errors.append(f"Câu mẫu thứ {idx + 1} trong 'sample_utterances' không được rỗng.")

    templates = data.get("templates", {})
    if not isinstance(templates, dict):
        errors.append("Trường 'templates' phải là từ điển ánh xạ mã mẫu.")
    else:
        for req_key in REQUIRED_TEMPLATE_KEYS:
            if req_key not in templates:
                errors.append(f"Thiếu khóa bắt buộc '{req_key}' trong 'templates'.")
            elif not isinstance(templates[req_key], str):
                errors.append(f"Giá trị của template '{req_key}' phải là chuỗi.")

        latex_val = templates.get("latex", "")
        if not isinstance(latex_val, str) or len(latex_val) == 0:
            errors.append("Khóa 'latex' trong 'templates' phải có nội dung mã template, không được rỗng.")

    slots = data.get("slots", {})
    if not isinstance(slots, dict):
        errors.append("Trường 'slots' phải là từ điển định nghĩa tham số.")
    else:
        for slot_name, slot_meta in slots.items():
            if not isinstance(slot_meta, dict):
                errors.append(f"Định nghĩa slot '{slot_name}' phải là từ điển.")
                continue

            pattern = slot_meta.get("pattern", "")
            if pattern:
                try:
                    compiled = re.compile(pattern)
                    if compiled.groups < 1:
                        errors.append(f"Regex pattern của slot '{slot_name}' phải chứa ít nhất một nhóm capture ().")
                except re.error as e:
                    errors.append(f"Regex pattern của slot '{slot_name}' không hợp lệ: {e}")

    return errors


def validate_scenario_bank(
    scenarios: Union[ScenarioBank, Iterable[Union[Scenario, dict[str, Any]]]],
) -> dict[str, list[str]]:
    """
    Kiểm tra tính hợp lệ toàn diện của ngân hàng kịch bản (toàn bộ intent).
    Đồng thời kiểm tra tính duy nhất của tên intent (không được trùng lặp).

    Trả về:
        Từ điển ánh xạ intent -> danh sách lỗi phát hiện (rỗng nếu toàn bộ hợp lệ).
    """
    items = scenarios.all() if isinstance(scenarios, ScenarioBank) else list(scenarios)
    all_errors: dict[str, list[str]] = {}
    seen_intents: set[str] = set()

    for idx, item in enumerate(items):
        item_intent = item.intent if isinstance(item, Scenario) else item.get("intent", f"UNKNOWN_{idx}")
        item_errors = validate_scenario(item)

        if item_intent in seen_intents:
            item_errors.append(f"Tên intent '{item_intent}' bị trùng lặp trong cơ sở kịch bản.")
        seen_intents.add(item_intent)

        if item_errors:
            all_errors[item_intent] = item_errors

    return all_errors


def _tokenize(text: str) -> list[str]:
    """Tách từ đơn giản và chuẩn hóa chữ thường để so khớp offline."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return [t for t in cleaned.split() if t]


def _compute_match_score(query_tokens: set[str], query_text: str, target_text: str) -> float:
    """
    Tính điểm tương đồng từ khóa giữa câu truy vấn và văn bản mục tiêu.
    Kết hợp Jaccard similarity trên tập token và trọng số khớp chuỗi con liên tiếp.
    """
    if not target_text:
        return 0.0

    target_norm = target_text.lower().strip()
    target_tokens = set(_tokenize(target_norm))

    if not target_tokens or not query_tokens:
        return 0.0

    # 1. Điểm Jaccard overlap
    intersection = len(query_tokens & target_tokens)
    union = len(query_tokens | target_tokens)
    jaccard = intersection / union if union > 0 else 0.0

    # 2. Khớp chuỗi con chính xác (exact substring match boost)
    substr_boost = 0.0
    if query_text in target_norm or target_norm in query_text:
        substr_boost = 0.35

    # Điểm tổng hợp chuẩn hóa về khoảng [0, 1]
    score = min(1.0, jaccard * 0.65 + substr_boost)
    return score


def retrieve_scenarios(
    query: str,
    scenarios: Union[ScenarioBank, Sequence[Scenario]],
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    threshold: float = THETA_RETRIEVAL_THRESHOLD,
) -> list[tuple[Scenario, float]]:
    """
    Thuật toán truy hồi kịch bản tương đồng offline phục vụ bước fallback
    khi bộ phân loại xác suất thấp hơn ngưỡng tau (theo kiến trúc AGENTS.md):
    'else retrieval top-3 over aliases + scenario strings -> ask the user to choose (if best score < theta: say not understood)'.

    Tham số:
        query: Câu truy vấn hoặc yêu cầu nhập vào từ người dùng.
        scenarios: Ngân hàng kịch bản cần tìm kiếm.
        top_k: Số lượng kịch bản tối đa cần lấy ra (mặc định 3).
        threshold: Ngưỡng tương đồng theta tối thiểu để chấp nhận kết quả.

    Trả về:
        Danh sách các cặp (Scenario, điểm số) giảm dần theo độ tương đồng.
    """
    candidate_list = scenarios.all() if isinstance(scenarios, ScenarioBank) else list(scenarios)
    query_norm = query.lower().strip()
    query_tokens = set(_tokenize(query_norm))

    if not query_tokens:
        return []

    scored_results: list[tuple[Scenario, float]] = []

    for sc in candidate_list:
        max_score = 0.0

        # So khớp với từng alias
        for alias in sc.aliases:
            score = _compute_match_score(query_tokens, query_norm, alias)
            if score > max_score:
                max_score = score

        # So khớp với các câu mẫu (sample utterances)
        for utt in sc.sample_utterances:
            score = _compute_match_score(query_tokens, query_norm, utt)
            if score > max_score:
                max_score = score

        # So khớp với mô tả tiếng Việt
        desc_score = _compute_match_score(query_tokens, query_norm, sc.describe) * 0.8
        if desc_score > max_score:
            max_score = desc_score

        # So khớp trực tiếp với intent name
        intent_str = sc.intent.lower().replace("_", " ")
        intent_score = _compute_match_score(query_tokens, query_norm, intent_str) * 0.9
        if intent_score > max_score:
            max_score = intent_score

        if max_score >= threshold:
            scored_results.append((sc, round(max_score, 4)))

    # Sắp xếp giảm dần theo điểm số
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results[:top_k]


def extract_slots(scenario: Scenario, text: str) -> tuple[dict[str, str], list[str]]:
    """
    Trích xuất tham số từ câu đầu vào dựa trên regex của từng slot trong kịch bản.
    Theo kiến trúc AGENTS.md:
    'slot extraction (regex per intent) -> if a required slot is missing: ask for exactly the missing slots'

    Tham số:
        scenario: Kịch bản mục tiêu chứa định nghĩa các slot và regex pattern.
        text: Câu nhập liệu của người dùng.

    Trả về:
        Tuple gồm:
        - dict các slot đã trích xuất được hoặc đã điền giá trị mặc định.
        - list các slot bắt buộc (required=True) bị thiếu để gửi phản hồi yêu cầu người dùng bổ sung.
    """
    extracted_slots: dict[str, str] = {}
    missing_required: list[str] = []

    for slot_name, slot_def in scenario.slots.items():
        val: Optional[str] = None

        if slot_def.pattern:
            try:
                match = re.search(slot_def.pattern, text, flags=re.IGNORECASE | re.UNICODE)
                if match:
                    # Lấy nhóm capture đầu tiên không phải None
                    groups = [g for g in match.groups() if g is not None]
                    if groups:
                        val = groups[0].strip()
            except re.error:
                # shortcut: lỗi pattern regex không mong muốn, nâng cấp khi thêm kiểm tra tiền xử lý
                pass

        if val:
            extracted_slots[slot_name] = val
        elif slot_def.default:
            extracted_slots[slot_name] = slot_def.default
        elif slot_def.required:
            missing_required.append(slot_name)
        else:
            extracted_slots[slot_name] = ""

    return extracted_slots, missing_required


def render_template(
    scenario: Scenario,
    slot_values: Optional[dict[str, str]] = None,
    target: str = "latex",
    escape_latex_special: bool = False,
) -> str:
    """
    Điền các tham số đã trích xuất vào template tương ứng của kịch bản.

    Tham số:
        scenario: Kịch bản mục tiêu.
        slot_values: Giá trị các slot đã được trích xuất (tùy chọn).
        target: Ngôn ngữ đích cần render ('latex', 'python', 'cpp', 'mathjax').
        escape_latex_special: Có escape ký tự đặc biệt cho các slot văn bản hay không.

    Trả về:
        Chuỗi mã template đã được điền đầy đủ tham số.
    """
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"Target '{target}' không được hỗ trợ. Các target hợp lệ: {SUPPORTED_TARGETS}")

    templates_dict = scenario.templates.to_dict()
    template_str = templates_dict.get(target, "")

    if not template_str and target == "latex":
        template_str = scenario.templates.latex or scenario.templates.mathjax or ""

    slots = dict(slot_values or {})

    # Điền giá trị mặc định cho các slot chưa được cung cấp
    for s_name, s_def in scenario.slots.items():
        if s_name not in slots:
            slots[s_name] = s_def.default

    # Thoát các ký tự đặc biệt nếu cần
    def _escape_latex(val: str) -> str:
        special_chars = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        escaped = val
        for char, rep in special_chars.items():
            escaped = escaped.replace(char, rep)
        return escaped

    rendered = template_str
    for key, value in slots.items():
        sub_val = _escape_latex(value) if escape_latex_special else value
        # Hỗ trợ cả placeholder đơn {key} và kép {{key}}
        rendered = rendered.replace(f"{{{key}}}", sub_val)
        rendered = rendered.replace(f"{{{{{key}}}}}", sub_val)

    return rendered
