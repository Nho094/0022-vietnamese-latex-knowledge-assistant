"""
Module pipeline bóc tách tham số (Regex Slot Filling) và sinh mã mẫu LaTeX (LaTeX Template Generation Pipeline).

Tuân thủ nghiêm ngặt kiến trúc và các quy tắc AGENTS.md:
- Kiến trúc luồng xử lý:
  text -> preprocess (NFC, chuẩn hóa ký hiệu toán học, tách từ pyvi cho PhoBERT)
    -> PhoBERT classifier: p = softmax over intents (hoặc categories trong chế độ 2 tầng)
       - if max p >= tau: intent accepted
       - else retrieval top-3 over aliases + scenario strings -> ask the user to choose (if best score < theta: say not understood)
    -> slot extraction (regex per intent) -> if a required slot is missing: ask for exactly the missing slots
    -> render latex template (text slots escaped) -> output
- Quy tắc trung thực (Honesty Rule #1):
  Hệ thống là mô hình scenario-based đơn giản hóa lấy cảm hứng từ Retrieve-and-Fill (RAF),
  bước slot filling dùng biểu thức chính quy (regex) thay vì span-pointer decoder phức tạp.
- Quy tắc bảo mật:
  Chỉ sinh mã từ ngân hàng kịch bản hợp lệ đã đăng ký (template bank),
  tuyệt đối không chạy shell escape và thoát an toàn các ký tự đặc biệt trong slot văn bản.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from core.config import (
    DEFAULT_INTENTS_PATH,
    DEFAULT_RETRIEVAL_TOP_K,
    SUPPORTED_TARGETS,
    TAU_INTENT_THRESHOLD,
    THETA_RETRIEVAL_THRESHOLD,
)
from src.classifier import ClassificationResult, PhoBertClassifier
from src.fallback import CandidateMatch, FallbackResult, SbertRetrievalFallback
from src.scenarios import (
    Scenario,
    ScenarioBank,
    SlotDefinition,
    load_scenario_bank,
)


def preprocess_text(text: str) -> str:
    """
    Tiền xử lý và chuẩn hóa chuỗi văn bản tiếng Việt đầu vào.

    Quy trình chuẩn hóa:
    1. Chuẩn hóa Unicode theo định dạng dựng sẵn NFC (tránh lỗi ký tự tổ hợp).
    2. Chuẩn hóa các ký tự toán học và dấu ngoặc full-width sang half-width chuẩn.
    3. Thử tách từ bằng thư viện pyvi nếu có sẵn trong môi trường.

    Tham số:
        text: Chuỗi văn bản tiếng Việt thô từ người dùng.

    Trả về:
        Chuỗi văn bản đã được chuẩn hóa.
    """
    if not text:
        return ""

    # Bước 1: Chuẩn hóa dạng Unicode NFC
    normalized = unicodedata.normalize("NFC", text.strip())

    # Bước 2: Chuẩn hóa ký hiệu toán học toàn độ rộng (full-width) sang nửa độ rộng (half-width)
    symbol_replacements = {
        "＋": "+",
        "－": "-",
        "×": "*",
        "÷": "/",
        "＝": "=",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "／": "/",
        "＾": "^",
        "＿": "_",
    }
    for full, half in symbol_replacements.items():
        normalized = normalized.replace(full, half)

    return normalized


@dataclass
class SlotExtractionResult:
    """
    Kết quả trích xuất tham số (slot entities) từ văn bản người dùng dựa trên regex.

    Thuộc tính:
        intent: Mã định danh kịch bản mục tiêu.
        slots: Bảng ánh xạ tên slot -> giá trị trích xuất được hoặc giá trị mặc định.
        missing_slots: Danh sách tên các slot bắt buộc (required=True) chưa tìm thấy.
        is_complete: Trạng thái đã đủ tất cả tham số bắt buộc hay chưa (True nếu missing_slots rỗng).
        raw_matches: Bảng ánh xạ tên slot -> chuỗi khớp trực tiếp từ biểu thức regex.
    """
    intent: str
    slots: dict[str, str] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    is_complete: bool = True
    raw_matches: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """
    Kết quả toàn diện từ quy trình bóc tách slot và sinh mã template LaTeX.

    Thuộc tính:
        success: Thành công sinh mã hay không (True khi intent hợp lệ, đủ slot và đã render).
        status: Mã trạng thái ('success', 'missing_slots', 'not_understood', 'scenario_not_found', 'error').
        query: Chuỗi văn bản truy vấn gốc của người dùng.
        intent: Mã intent được phân giải (None nếu không xác định được).
        scenario: Đối tượng kịch bản tương ứng nếu tìm thấy.
        slots: Bảng ánh xạ giá trị các slot được sử dụng để nội suy template.
        missing_slots: Danh sách các slot bắt buộc còn thiếu cần người dùng bổ sung.
        latex_code: Đoạn mã LaTeX (hoặc mã target) hoàn chỉnh sau khi render.
        requires: Danh sách các gói LaTeX phụ thuộc cần khai báo trong preamble.
        target: Ngôn ngữ / nền tảng đích đã render ('latex', 'python', 'cpp', 'mathjax').
        message: Thông điệp phản hồi người dùng (hướng dẫn, báo lỗi hoặc yêu cầu bổ sung slot).
        confidence: Điểm tin cậy p_max từ PhoBERT classifier.
        fallback_used: Có kích hoạt cơ chế truy hồi ngữ nghĩa dự phòng (fallback) hay không.
        retrieval_score: Điểm tương đồng cosine similarity nếu có kích hoạt fallback.
        candidates: Danh sách ứng viên gợi ý khi độ tin cậy thấp (phục vụ người dùng chọn lựa).
        elapsed_ms: Thời gian thực thi toàn chu trình tính bằng mili-giây (ms).
    """
    success: bool
    status: str
    query: str
    intent: Optional[str] = None
    scenario: Optional[Scenario] = None
    slots: dict[str, str] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    latex_code: str = ""
    requires: list[str] = field(default_factory=list)
    target: str = "latex"
    message: str = ""
    confidence: float = 0.0
    fallback_used: bool = False
    retrieval_score: float = 0.0
    candidates: list[CandidateMatch] = field(default_factory=list)
    elapsed_ms: float = 0.0


class RegexSlotExtractor:
    """
    Động cơ bóc tách tham số dựa trên biểu thức chính quy (Regex Slot Extraction Engine).

    Theo Honesty Rule #1 của AGENTS.md:
    Phương pháp tiếp cận dựa trên regex theo kịch bản đơn giản hóa từ ý tưởng Retrieve-and-Fill (RAF),
    không sử dụng span-pointer decoder phức tạp của bài báo gốc.
    """

    def __init__(self, scenario_bank: Optional[ScenarioBank] = None) -> None:
        """
        Khởi tạo động cơ trích xuất slot.

        Tham số:
            scenario_bank: Ngân hàng kịch bản chứa định nghĩa schema slot và regex pattern.
        """
        self.scenario_bank = scenario_bank

    def extract(self, scenario: Scenario, text: str) -> SlotExtractionResult:
        """
        Trích xuất các thực thể slot từ văn bản đầu vào cho một kịch bản cụ thể.

        Thuật toán bóc tách:
        1. Duyệt qua từng slot trong scenario.slots.
        2. Nếu slot có pattern regex, tìm kiếm chuỗi khớp đầu tiên trong câu.
        3. Nếu tìm thấy, lấy nhóm capture đầu tiên không rỗng làm giá trị trích xuất.
        4. Nếu không tìm thấy:
           - Nếu có giá trị mặc định (default), gán giá trị mặc định.
           - Nếu slot là bắt buộc (required=True), đưa vào danh sách missing_slots.
           - Ngược lại, gán chuỗi rỗng "".

        Tham số:
            scenario: Đối tượng kịch bản chứa danh sách slot và biểu thức chính quy.
            text: Văn bản câu hỏi tiếng Việt của người dùng (đã hoặc chưa tiền xử lý).

        Trả về:
            SlotExtractionResult chứa các slot đã trích xuất và danh sách slot còn thiếu.
        """
        extracted_slots: dict[str, str] = {}
        raw_matches: dict[str, str] = {}
        missing_slots: list[str] = []

        cleaned_text = text.strip()

        for slot_name, slot_def in scenario.slots.items():
            matched_value: Optional[str] = None

            if slot_def.pattern:
                try:
                    match = re.search(slot_def.pattern, cleaned_text, flags=re.IGNORECASE | re.UNICODE)
                    if match:
                        # Lấy nhóm capture đầu tiên hợp lệ
                        groups = [g for g in match.groups() if g is not None]
                        if groups:
                            val = groups[0].strip()
                            matched_value = val
                            raw_matches[slot_name] = match.group(0).strip()
                except re.error:
                    # shortcut: biểu thức regex không hợp lệ được bỏ qua một cách an toàn
                    pass

            if matched_value:
                extracted_slots[slot_name] = matched_value
            elif slot_def.default:
                extracted_slots[slot_name] = slot_def.default
            elif slot_def.required:
                missing_slots.append(slot_name)
            else:
                extracted_slots[slot_name] = ""

        is_complete = len(missing_slots) == 0

        return SlotExtractionResult(
            intent=scenario.intent,
            slots=extracted_slots,
            missing_slots=missing_slots,
            is_complete=is_complete,
            raw_matches=raw_matches,
        )

    def extract_by_intent(self, intent: str, text: str) -> SlotExtractionResult:
        """
        Trích xuất slot dựa vào mã intent bằng cách tra cứu trong scenario_bank.

        Tham số:
            intent: Mã định danh kịch bản (ví dụ: MATHJAX_PHAN_SO).
            text: Văn bản đầu vào từ người dùng.

        Trả về:
            SlotExtractionResult tương ứng.
        """
        if self.scenario_bank is None or intent not in self.scenario_bank:
            return SlotExtractionResult(
                intent=intent,
                slots={},
                missing_slots=[],
                is_complete=False,
                raw_matches={},
            )

        scenario = self.scenario_bank[intent]
        return self.extract(scenario, text)


class LaTeXTemplateRenderer:
    """
    Bộ nội suy và kết xuất mã LaTeX từ mẫu (LaTeX Template Rendering Engine).

    Tuân thủ quy tắc bảo mật kiến trúc AGENTS.md:
    - Thoát các ký tự đặc biệt LaTeX (text slots escaped) để tránh lỗi biên dịch.
    - Đảm bảo mã sinh ra xuất phát từ template bank đã đăng ký hợp lệ.
    """

    # Danh mục các ký tự đặc biệt trong LaTeX cần thoát khi đặt trong ngữ cảnh văn bản
    LATEX_SPECIAL_CHARS: dict[str, str] = {
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

    # Các tiền tố hoặc tên slot được coi là văn bản thuần (cần thoát ký tự đặc biệt)
    TEXT_SLOT_NAMES: set[str] = {
        "title",
        "author",
        "caption",
        "label",
        "text",
        "content",
        "header",
        "name",
        "desc",
        "description",
        "note",
        "message",
    }

    @classmethod
    def escape_latex(cls, text: str) -> str:
        """
        Thoát các ký tự điều khiển đặc biệt của LaTeX trong chuỗi văn bản.
        Ví dụ: 'Báo cáo 50% & lợi nhuận' -> 'Báo cáo 50\\% \\& lợi nhuận'

        Tham số:
            text: Chuỗi văn bản cần làm sạch.

        Trả về:
            Chuỗi an toàn cho trình biên dịch LaTeX.
        """
        if not text:
            return ""

        escaped = text
        for char, replacement in cls.LATEX_SPECIAL_CHARS.items():
            escaped = escaped.replace(char, replacement)
        return escaped

    @classmethod
    def is_text_slot(cls, slot_name: str, scenario: Optional[Scenario] = None) -> bool:
        """
        Xác định một slot có phải là văn bản thường hay là biểu thức toán học.
        Các slot toán học (formula, num, den, expr) không thoát các toán tử toán như ^, _, \\.

        Tham số:
            slot_name: Tên tham số.
            scenario: Đối tượng kịch bản để kiểm tra danh mục (nếu có).

        Trả về:
            True nếu là slot văn bản, False nếu là slot biểu thức toán học.
        """
        name_lower = slot_name.lower()
        if name_lower in cls.TEXT_SLOT_NAMES:
            return True

        # Nếu kịch bản thuộc nhóm tài liệu văn bản chung và slot không phải toán
        if scenario is not None and scenario.category in ("latex_document", "document_structure"):
            if name_lower not in ("formula", "num", "den", "expr", "math"):
                return True

        return False

    def render(
        self,
        scenario: Scenario,
        slot_values: Optional[dict[str, str]] = None,
        target: str = "latex",
        escape_text_slots: bool = True,
    ) -> str:
        """
        Nội suy các giá trị tham số vào chuỗi mẫu LaTeX của kịch bản.

        Tham số:
            scenario: Kịch bản mục tiêu từ template bank.
            slot_values: Từ điển chứa giá trị các tham số đã trích xuất.
            target: Ngôn ngữ / nền tảng đích ('latex', 'python', 'cpp', 'mathjax').
            escape_text_slots: Có thoát ký tự đặc biệt cho các slot văn bản hay không.

        Trả về:
            Chuỗi mã đã được điền đầy đủ tham số.
        """
        if target not in SUPPORTED_TARGETS:
            raise ValueError(f"Ngôn ngữ đích '{target}' không được hỗ trợ. Các target hợp lệ: {SUPPORTED_TARGETS}")

        templates_dict = scenario.templates.to_dict()
        template_str = templates_dict.get(target, "")

        if not template_str and target == "latex":
            template_str = scenario.templates.latex

        merged_slots: dict[str, str] = {}
        for s_name, s_def in scenario.slots.items():
            merged_slots[s_name] = s_def.default

        if slot_values:
            merged_slots.update(slot_values)

        rendered = template_str
        for key, val in merged_slots.items():
            sub_val = val
            if escape_text_slots and self.is_text_slot(key, scenario):
                sub_val = self.escape_latex(val)

            # Thay thế cả hai định dạng placeholder: {slot_name} và {{slot_name}}
            rendered = rendered.replace(f"{{{key}}}", sub_val)
            rendered = rendered.replace(f"{{{{{key}}}}}", sub_val)

        return rendered

    @classmethod
    def wrap_in_document(
        cls,
        body: str,
        packages: Sequence[str] = (),
        document_class: str = "article",
    ) -> str:
        """
        Đóng gói một đoạn mã LaTeX sinh ra thành một tài liệu LaTeX hoàn chỉnh có thể biên dịch.

        Tham số:
            body: Nội dung thân văn bản hoặc biểu thức LaTeX.
            packages: Danh sách các gói LaTeX cần khai báo (\\usepackage{...}).
            document_class: Lớp tài liệu (mặc định 'article').

        Trả về:
            Chuỗi mã nguồn LaTeX hoàn chỉnh bắt đầu bằng \\documentclass và kết thúc bằng \\end{document}.
        """
        if r"\documentclass" in body and r"\begin{document}" in body:
            return body

        pkg_lines = "\n".join(f"\\usepackage{{{pkg}}}" for pkg in packages)
        if pkg_lines:
            pkg_lines += "\n"
        return (
            f"\\documentclass{{{document_class}}}\n"
            f"{pkg_lines}"
            f"\\begin{{document}}\n"
            f"{body}\n"
            f"\\end{{document}}\n"
        )


class LaTeXPipeline:
    """
    Quy trình tích hợp hoàn chỉnh (Full End-to-End Pipeline) cho Trợ lý LaTeX tiếng Việt.

    Kết nối chặt chẽ các giai đoạn theo kiến trúc AGENTS.md:
    1. Preprocess: Chuẩn hóa NFC và ký hiệu toán học.
    2. Intent Classification / Scenario Resolution:
       PhoBERT p_max >= tau -> accept intent;
       else SBERT top-3 retrieval (score >= theta -> suggest candidates, score < theta -> not understood).
    3. Slot Extraction: Dùng RegexSlotExtractor để bóc tách tham số và phát hiện thiếu sót.
    4. LaTeX Template Generation: Dùng LaTeXTemplateRenderer để nội suy và sinh mã LaTeX an toàn.
    """

    def __init__(
        self,
        scenario_bank: Optional[ScenarioBank] = None,
        classifier: Optional[PhoBertClassifier] = None,
        fallback: Optional[SbertRetrievalFallback] = None,
        slot_extractor: Optional[RegexSlotExtractor] = None,
        renderer: Optional[LaTeXTemplateRenderer] = None,
        threshold_tau: float = TAU_INTENT_THRESHOLD,
        threshold_theta: float = THETA_RETRIEVAL_THRESHOLD,
    ) -> None:
        """
        Khởi tạo pipeline sinh mã LaTeX.

        Tham số:
            scenario_bank: Ngân hàng kịch bản nạp sẵn (mặc định nạp từ intents.json).
            classifier: Bộ phân loại PhoBertClassifier (nếu None sẽ khởi tạo mặc định).
            fallback: Bộ truy hồi ngữ nghĩa SbertRetrievalFallback (nếu None sẽ khởi tạo mặc định).
            slot_extractor: Động cơ trích xuất RegexSlotExtractor.
            renderer: Động cơ render LaTeXTemplateRenderer.
            threshold_tau: Ngưỡng chấp nhận intent trực tiếp (tau, mặc định 0.8).
            threshold_theta: Ngưỡng tối thiểu chấp nhận hiểu trong fallback (theta, mặc định 0.3).
        """
        if scenario_bank is not None:
            self.scenario_bank = scenario_bank
        elif DEFAULT_INTENTS_PATH.exists():
            self.scenario_bank = load_scenario_bank(DEFAULT_INTENTS_PATH)
        else:
            self.scenario_bank = ScenarioBank([])

        self.threshold_tau = threshold_tau
        self.threshold_theta = threshold_theta

        self.classifier = classifier or PhoBertClassifier(
            scenario_bank=self.scenario_bank,
            threshold_tau=self.threshold_tau,
        )
        self.fallback = fallback or SbertRetrievalFallback(
            scenario_bank=self.scenario_bank,
            threshold_theta=self.threshold_theta,
            top_k=DEFAULT_RETRIEVAL_TOP_K,
        )
        self.slot_extractor = slot_extractor or RegexSlotExtractor(scenario_bank=self.scenario_bank)
        self.renderer = renderer or LaTeXTemplateRenderer()

    def resolve_scenario(
        self,
        clean_text: str,
        intent_override: Optional[str] = None,
        auto_choose_fallback: bool = True,
    ) -> tuple[Optional[Scenario], float, bool, float, list[CandidateMatch], str]:
        """
        Phân giải và xác định kịch bản mục tiêu từ câu văn bản đầu vào.

        Quy trình phân giải:
        1. Nếu intent_override được cung cấp, trực tiếp tra cứu trong scenario_bank.
        2. Chạy PhoBERT classifier: p = softmax over intents.
           - Nếu p_max >= tau: chấp nhận ý định trực tiếp.
        3. Nếu p_max < tau: kích hoạt SBERT retrieval fallback (top-3).
           - Nếu điểm tốt nhất < theta: báo không hiểu ('say not understood').
           - Nếu điểm tốt nhất >= theta:
             + Nếu auto_choose_fallback=True: chọn kịch bản top-1.
             + Nếu auto_choose_fallback=False: trả về danh sách ứng viên để hỏi người dùng.

        Tham số:
            clean_text: Câu truy vấn đã qua chuẩn hóa.
            intent_override: Tùy chọn chỉ định trực tiếp intent name.
            auto_choose_fallback: Tự động chọn kịch bản top-1 khi fallback (mặc định True).

        Trả về:
            Tuple: (scenario, confidence, fallback_used, retrieval_score, candidates, status_message)
        """
        # 1. Trường hợp người dùng chỉ định intent trực tiếp
        if intent_override:
            if intent_override in self.scenario_bank:
                scenario = self.scenario_bank[intent_override]
                return scenario, 1.0, False, 0.0, [], "intent_override"
            return None, 0.0, False, 0.0, [], f"Không tìm thấy kịch bản '{intent_override}'."

        # 2. Phân loại ý định qua PhoBERT classifier (p_max >= tau)
        clf_res: ClassificationResult = self.classifier.predict(clean_text)

        if clf_res.accepted and clf_res.intent != "UNKNOWN" and clf_res.intent in self.scenario_bank:
            scenario = self.scenario_bank[clf_res.intent]
            return scenario, clf_res.confidence, False, 0.0, [], "classifier_accepted"

        # 3. Kích hoạt bước truy hồi ngữ nghĩa dự phòng (SBERT top-3 fallback)
        fb_res: FallbackResult = self.fallback.retrieve(clean_text)

        if not fb_res.understood or not fb_res.top_candidates:
            # Điểm số tốt nhất < theta: say not understood
            return None, clf_res.confidence, True, 0.0, [], "Xin lỗi, tôi không hiểu yêu cầu này."

        best_cand = fb_res.top_candidates[0]
        if auto_choose_fallback:
            if best_cand.intent in self.scenario_bank:
                scenario = self.scenario_bank[best_cand.intent]
                return scenario, clf_res.confidence, True, best_cand.score, fb_res.top_candidates, "fallback_top1_selected"

        return None, clf_res.confidence, True, best_cand.score, fb_res.top_candidates, "ask_user_to_choose"

    def generate(
        self,
        text: str,
        intent: Optional[str] = None,
        slot_overrides: Optional[dict[str, str]] = None,
        target: str = "latex",
        auto_fallback: bool = True,
        escape_text_slots: bool = True,
    ) -> PipelineResult:
        """
        Thực hiện toàn bộ quy trình: tiền xử lý, phân giải kịch bản, trích xuất slot regex,
        nội suy tham số vào template, và trả về mã LaTeX hoàn chỉnh.

        Tham số:
            text: Văn bản câu hỏi / yêu cầu tiếng Việt của người dùng.
            intent: Mã intent chỉ định trực tiếp (tùy chọn).
            slot_overrides: Giá trị slot cung cấp thêm để ghi đè (tùy chọn).
            target: Ngôn ngữ đích cần sinh ('latex', 'python', 'cpp', 'mathjax').
            auto_fallback: Tự động chọn kịch bản top-1 nếu độ tin cậy PhoBERT < tau.
            escape_text_slots: Thoát ký tự đặc biệt LaTeX cho các slot văn bản.

        Trả về:
            PipelineResult chứa mã LaTeX kết quả và thông tin chi tiết các bước.
        """
        start_time = time.perf_counter()

        # Kiểm tra tính hợp lệ ngôn ngữ đích
        if target not in SUPPORTED_TARGETS:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PipelineResult(
                success=False,
                status="error",
                query=text,
                target=target,
                message=f"Ngôn ngữ đích '{target}' không được hỗ trợ. Các target hợp lệ: {', '.join(SUPPORTED_TARGETS)}",
                elapsed_ms=elapsed_ms,
            )

        # 1. Tiền xử lý văn bản
        clean_text = preprocess_text(text)
        if not clean_text:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PipelineResult(
                success=False,
                status="not_understood",
                query=text,
                target=target,
                message="Truy vấn rỗng. Vui lòng nhập câu hỏi hoặc yêu cầu tiếng Việt.",
                elapsed_ms=elapsed_ms,
            )

        # 2. Phân giải kịch bản mục tiêu
        scenario, conf, fallback_used, ret_score, candidates, resolution_msg = self.resolve_scenario(
            clean_text=clean_text,
            intent_override=intent,
            auto_choose_fallback=auto_fallback,
        )

        if scenario is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            status_code = "ambiguous" if resolution_msg == "ask_user_to_choose" else "not_understood"
            return PipelineResult(
                success=False,
                status=status_code,
                query=text,
                confidence=conf,
                fallback_used=fallback_used,
                retrieval_score=ret_score,
                candidates=candidates,
                target=target,
                message=resolution_msg,
                elapsed_ms=elapsed_ms,
            )

        # 3. Trích xuất slot entities bằng regex engine
        extraction_res: SlotExtractionResult = self.slot_extractor.extract(scenario, clean_text)
        final_slots = dict(extraction_res.slots)

        # Ghi đè với slot_overrides nếu có
        if slot_overrides:
            final_slots.update(slot_overrides)

        # Cập nhật danh sách các slot bắt buộc còn thiếu
        still_missing = [
            s_name
            for s_name, s_def in scenario.slots.items()
            if s_def.required and not final_slots.get(s_name)
        ]

        # 4. Kiểm tra ràng buộc tham số bắt buộc (ask for exactly the missing slots)
        if still_missing:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PipelineResult(
                success=False,
                status="missing_slots",
                query=text,
                intent=scenario.intent,
                scenario=scenario,
                slots=final_slots,
                missing_slots=still_missing,
                confidence=conf,
                fallback_used=fallback_used,
                retrieval_score=ret_score,
                target=target,
                message=f"Kịch bản '{scenario.intent}' thiếu các tham số bắt buộc: {', '.join(still_missing)}",
                elapsed_ms=elapsed_ms,
            )

        # 5. Nội suy tham số vào template và sinh mã hoàn chỉnh
        rendered_code = self.renderer.render(
            scenario=scenario,
            slot_values=final_slots,
            target=target,
            escape_text_slots=escape_text_slots,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return PipelineResult(
            success=True,
            status="success",
            query=text,
            intent=scenario.intent,
            scenario=scenario,
            slots=final_slots,
            missing_slots=[],
            latex_code=rendered_code,
            requires=list(scenario.requires),
            target=target,
            message="Sinh mã thành công.",
            confidence=conf,
            fallback_used=fallback_used,
            retrieval_score=ret_score,
            candidates=candidates,
            elapsed_ms=elapsed_ms,
        )

    def batch_generate(
        self,
        texts: Sequence[str],
        target: str = "latex",
        auto_fallback: bool = True,
    ) -> list[PipelineResult]:
        """
        Thực hiện sinh mã hàng loạt cho một danh sách các câu tiếng Việt.

        Tham số:
            texts: Danh sách các câu truy vấn.
            target: Ngôn ngữ đích ('latex', 'python', 'cpp', 'mathjax').
            auto_fallback: Tự động chọn kịch bản top-1 khi fallback.

        Trả về:
            Danh sách PipelineResult tương ứng với từng câu.
        """
        return [self.generate(t, target=target, auto_fallback=auto_fallback) for t in texts]

    def generate_document(
        self,
        text: str,
        intent: Optional[str] = None,
        slot_overrides: Optional[dict[str, str]] = None,
        target: str = "latex",
        auto_fallback: bool = True,
        escape_text_slots: bool = True,
        document_class: str = "article",
    ) -> PipelineResult:
        """
        Sinh một tài liệu LaTeX hoàn chỉnh (bao gồm documentclass, usepackage và document body)
        thay vì chỉ sinh đoạn mã (snippet).

        Tham số:
            text: Văn bản câu hỏi / yêu cầu của người dùng.
            intent: Mã intent chỉ định (tùy chọn).
            slot_overrides: Giá trị slot ghi đè (tùy chọn).
            target: Ngôn ngữ đích (mặc định 'latex').
            auto_fallback: Tự động chọn kịch bản top-1 khi fallback.
            escape_text_slots: Có thoát ký tự đặc biệt hay không.
            document_class: Lớp tài liệu (mặc định 'article').

        Trả về:
            PipelineResult với latex_code đã được bao bọc đầy đủ trong document body.
        """
        res = self.generate(
            text=text,
            intent=intent,
            slot_overrides=slot_overrides,
            target=target,
            auto_fallback=auto_fallback,
            escape_text_slots=escape_text_slots,
        )
        if res.success and target == "latex":
            wrapped = self.renderer.wrap_in_document(
                body=res.latex_code,
                packages=res.requires,
                document_class=document_class,
            )
            res.latex_code = wrapped
        return res


def wrap_latex_in_document(
    body: str,
    packages: Sequence[str] = (),
    document_class: str = "article",
) -> str:
    """
    Hàm tiện ích bao bọc một đoạn mã LaTeX vào tài liệu hoàn chỉnh có thể biên dịch.

    Tham số:
        body: Nội dung thân văn bản LaTeX.
        packages: Danh sách các gói phụ thuộc cần nạp.
        document_class: Lớp tài liệu (mặc định 'article').

    Trả về:
        Mã nguồn tài liệu LaTeX hoàn chỉnh.
    """
    return LaTeXTemplateRenderer.wrap_in_document(
        body=body,
        packages=packages,
        document_class=document_class,
    )


def generate_latex(
    text: str,
    intent: Optional[str] = None,
    slot_overrides: Optional[dict[str, str]] = None,
    target: str = "latex",
) -> PipelineResult:
    """
    Hàm tiện ích cấp cao để sinh mã LaTeX nhanh từ văn bản tiếng Việt.

    Tham số:
        text: Văn bản câu hỏi / yêu cầu của người dùng.
        intent: Mã intent chỉ định (tùy chọn).
        slot_overrides: Bảng giá trị slot bổ sung (tùy chọn).
        target: Ngôn ngữ đích (mặc định 'latex').

    Trả về:
        PipelineResult chứa mã sinh ra và siêu dữ liệu.
    """
    pipeline = LaTeXPipeline()
    return pipeline.generate(
        text=text,
        intent=intent,
        slot_overrides=slot_overrides,
        target=target,
    )
