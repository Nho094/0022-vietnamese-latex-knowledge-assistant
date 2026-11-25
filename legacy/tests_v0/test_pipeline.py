"""
Kiểm thử đơn vị cho pipeline bóc tách tham số regex và sinh mã LaTeX (tests/test_pipeline.py).

Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Kiểm tra toàn diện các thành phần của src/pipeline.py:
  + Tiền xử lý chuỗi văn bản (NFC, ký hiệu toán học).
  + Động cơ bóc tách tham số dựa trên regex (RegexSlotExtractor).
  + Động cơ render template LaTeX và cơ chế thoát ký tự đặc biệt (LaTeXTemplateRenderer).
  + Toàn bộ luồng tích hợp của LaTeXPipeline (resolve scenario, slot filling, render template).
- Thực thi hoàn toàn ngoại tuyến, độc lập, không truy cập mạng.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.config import (
    DEFAULT_INTENTS_PATH,
    TAU_INTENT_THRESHOLD,
    THETA_RETRIEVAL_THRESHOLD,
)
from src.classifier import ClassificationResult, PhoBertClassifier
from src.fallback import CandidateMatch, FallbackResult, SbertRetrievalFallback
from src.pipeline import (
    LaTeXPipeline,
    LaTeXTemplateRenderer,
    PipelineResult,
    RegexSlotExtractor,
    SlotExtractionResult,
    generate_latex,
    preprocess_text,
    wrap_latex_in_document,
)
from src.scenarios import (
    Scenario,
    ScenarioBank,
    SlotDefinition,
    TemplateMapping,
)


@pytest.fixture
def sample_scenario_bank() -> ScenarioBank:
    """Fixture tạo ScenarioBank phục vụ kiểm thử pipeline."""
    scenarios = [
        Scenario(
            intent="MATHJAX_PHAN_SO",
            category="math_expression",
            describe="Chèn phân số toán học",
            aliases=["phân số", "phân thức", "a trên b"],
            sample_utterances=["tạo phân số với tử a và mẫu b", "chèn phân thức"],
            slots={
                "num": SlotDefinition(name="num", pattern=r"(?:tử số|tử)\s+([a-zA-Z0-9\^_\+\-]+)", default="a"),
                "den": SlotDefinition(name="den", pattern=r"(?:mẫu số|mẫu)\s+([a-zA-Z0-9\^_\+\-]+)", default="b"),
            },
            templates=TemplateMapping(
                latex=r"\frac{" + "{num}" + "}{" + "{den}" + "}",
                mathjax=r"\frac{" + "{num}" + "}{" + "{den}" + "}",
            ),
            requires=["amsmath"],
        ),
        Scenario(
            intent="LATEX_SECTION",
            category="document_structure",
            describe="Tạo đề mục section",
            aliases=["section", "tiêu đề", "đề mục"],
            sample_utterances=["tạo section mới với tiêu đề"],
            slots={
                "title": SlotDefinition(name="title", required=True, pattern=r"tiêu đề\s+là\s+([^\n,;]+)"),
            },
            templates=TemplateMapping(
                latex=r"\section{" + "{title}" + "}",
            ),
            requires=[],
        ),
        Scenario(
            intent="LATEX_TABLE_ENV",
            category="latex_document",
            describe="Môi trường bảng table",
            aliases=["bảng", "table", "tạo bảng"],
            sample_utterances=["tạo bảng với chú thích"],
            slots={
                "caption": SlotDefinition(name="caption", required=False, default="Bảng dữ liệu", pattern=r"chú thích\s+([^\n,;]+)"),
                "label": SlotDefinition(name="label", required=False, default="tab:data", pattern=r"nhãn\s+([a-zA-Z0-9_:]+)"),
            },
            templates=TemplateMapping(
                latex=r"\begin{table}" + "\n" + r"\caption{" + "{caption}" + "}\n" + r"\label{" + "{label}" + "}\n" + r"\end{table}",
            ),
            requires=[],
        ),
    ]
    return ScenarioBank(scenarios)


def test_preprocess_text_normalization() -> None:
    """Kiểm tra tiền xử lý chuẩn hóa Unicode NFC và ký hiệu toán học toàn độ rộng."""
    raw = "tính ＋ và － và × và ÷ và ＝"
    cleaned = preprocess_text(raw)
    assert "+" in cleaned
    assert "-" in cleaned
    assert "*" in cleaned
    assert "/" in cleaned
    assert "=" in cleaned

    # Kiểm tra chuỗi rỗng
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""


def test_regex_slot_extractor_success(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra trích xuất slot thành công với biểu thức regex."""
    extractor = RegexSlotExtractor(scenario_bank=sample_scenario_bank)
    scenario = sample_scenario_bank["MATHJAX_PHAN_SO"]

    result: SlotExtractionResult = extractor.extract(scenario, "Tạo phân số với tử số x và mẫu số y")
    assert result.intent == "MATHJAX_PHAN_SO"
    assert result.slots["num"] == "x"
    assert result.slots["den"] == "y"
    assert result.is_complete is True
    assert len(result.missing_slots) == 0


def test_regex_slot_extractor_default_values(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra gán giá trị mặc định khi người dùng không chỉ định slot."""
    extractor = RegexSlotExtractor(scenario_bank=sample_scenario_bank)
    scenario = sample_scenario_bank["MATHJAX_PHAN_SO"]

    result: SlotExtractionResult = extractor.extract(scenario, "Chèn phân số thông thường")
    assert result.slots["num"] == "a"  # default
    assert result.slots["den"] == "b"  # default
    assert result.is_complete is True
    assert len(result.missing_slots) == 0


def test_regex_slot_extractor_missing_required(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra phát hiện thiếu slot bắt buộc (required=True)."""
    extractor = RegexSlotExtractor(scenario_bank=sample_scenario_bank)
    scenario = sample_scenario_bank["LATEX_SECTION"]

    result: SlotExtractionResult = extractor.extract(scenario, "Tạo một section mới trong tài liệu")
    assert result.is_complete is False
    assert "title" in result.missing_slots
    assert result.slots.get("title", "") == ""


def test_latex_template_renderer_basic(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra nội suy cơ bản của LaTeXTemplateRenderer."""
    renderer = LaTeXTemplateRenderer()
    scenario = sample_scenario_bank["MATHJAX_PHAN_SO"]

    rendered = renderer.render(scenario, {"num": "1", "den": "2"})
    assert rendered == r"\frac{1}{2}"


def test_latex_template_renderer_escaping_text_slots(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra cơ chế thoát ký tự đặc biệt LaTeX cho các slot văn bản thuần."""
    renderer = LaTeXTemplateRenderer()
    scenario = sample_scenario_bank["LATEX_SECTION"]

    # Chứa các ký tự đặc biệt: %, &, $, #, _
    raw_title = "Doanh thu 50% & Lợi nhuận #1_Mục_tiêu $1000"
    rendered = renderer.render(scenario, {"title": raw_title}, escape_text_slots=True)

    assert r"50\%" in rendered
    assert r"\&" in rendered
    assert r"\#1" in rendered
    assert r"\_Mục\_tiêu" in rendered
    assert r"\$1000" in rendered


def test_latex_template_renderer_math_slots_not_escaped(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra slot biểu thức toán học không bị thoát dấu mũ, gạch dưới và dấu gạch chéo."""
    renderer = LaTeXTemplateRenderer()
    scenario = sample_scenario_bank["MATHJAX_PHAN_SO"]

    # Biểu thức toán chứa toán tử mũ ^ và biến gạch dưới _
    rendered = renderer.render(scenario, {"num": "x^2", "den": "y_1"}, escape_text_slots=True)
    assert rendered == r"\frac{x^2}{y_1}"


def test_latex_template_renderer_unsupported_target(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra ném lỗi khi chỉ định target không được hỗ trợ."""
    renderer = LaTeXTemplateRenderer()
    scenario = sample_scenario_bank["MATHJAX_PHAN_SO"]

    with pytest.raises(ValueError, match="không được hỗ trợ"):
        renderer.render(scenario, target="unsupported_lang")


def test_pipeline_generate_direct_accepted(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra pipeline sinh mã thành công khi bộ phân loại đạt ngưỡng tau."""
    mock_clf = MagicMock(spec=PhoBertClassifier)
    mock_clf.predict.return_value = ClassificationResult(
        intent="MATHJAX_PHAN_SO",
        confidence=0.92,
        accepted=True,
        category="math_expression",
    )
    mock_fb = MagicMock(spec=SbertRetrievalFallback)

    pipeline = LaTeXPipeline(
        scenario_bank=sample_scenario_bank,
        classifier=mock_clf,
        fallback=mock_fb,
    )

    res: PipelineResult = pipeline.generate("tạo phân số với tử số 3 và mẫu số 4")
    assert res.success is True
    assert res.status == "success"
    assert res.intent == "MATHJAX_PHAN_SO"
    assert res.latex_code == r"\frac{3}{4}"
    assert res.requires == ["amsmath"]
    assert res.confidence == 0.92
    assert res.fallback_used is False


def test_pipeline_generate_missing_slots(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra pipeline khi thiếu tham số bắt buộc."""
    pipeline = LaTeXPipeline(scenario_bank=sample_scenario_bank)

    res: PipelineResult = pipeline.generate("tạo section mới", intent="LATEX_SECTION")
    assert res.success is False
    assert res.status == "missing_slots"
    assert "title" in res.missing_slots
    assert "thiếu các tham số bắt buộc" in res.message


def test_pipeline_generate_with_slot_overrides(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra pipeline cung cấp slot_overrides để bổ sung tham số thiếu."""
    pipeline = LaTeXPipeline(scenario_bank=sample_scenario_bank)

    res: PipelineResult = pipeline.generate(
        text="tạo section mới",
        intent="LATEX_SECTION",
        slot_overrides={"title": "Giới thiệu tổng quan"},
    )
    assert res.success is True
    assert res.status == "success"
    assert res.latex_code == r"\section{Giới thiệu tổng quan}"


def test_pipeline_fallback_accepted(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra pipeline kích hoạt fallback khi confidence < tau và chọn top-1."""
    mock_clf = MagicMock(spec=PhoBertClassifier)
    mock_clf.predict.return_value = ClassificationResult(
        intent="UNKNOWN",
        confidence=0.45,
        accepted=False,
    )
    mock_fb = MagicMock(spec=SbertRetrievalFallback)
    mock_fb.retrieve.return_value = FallbackResult(
        query="phân số",
        understood=True,
        top_candidates=[
            CandidateMatch(
                intent="MATHJAX_PHAN_SO",
                score=0.88,
                describe="Chèn phân số toán học",
                matched_text="phân số",
            )
        ],
    )

    pipeline = LaTeXPipeline(
        scenario_bank=sample_scenario_bank,
        classifier=mock_clf,
        fallback=mock_fb,
    )

    res: PipelineResult = pipeline.generate("phân số với tử 5 và mẫu 6")
    assert res.success is True
    assert res.fallback_used is True
    assert res.retrieval_score == 0.88
    assert res.latex_code == r"\frac{5}{6}"


def test_pipeline_fallback_not_understood(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra pipeline khi câu hỏi ngoài phạm vi (score < theta)."""
    mock_clf = MagicMock(spec=PhoBertClassifier)
    mock_clf.predict.return_value = ClassificationResult(
        intent="UNKNOWN",
        confidence=0.1,
        accepted=False,
    )
    mock_fb = MagicMock(spec=SbertRetrievalFallback)
    mock_fb.retrieve.return_value = FallbackResult(
        query="thời tiết",
        understood=False,
        top_candidates=[],
        prompt_message="Xin lỗi, tôi không hiểu yêu cầu này.",
    )

    pipeline = LaTeXPipeline(
        scenario_bank=sample_scenario_bank,
        classifier=mock_clf,
        fallback=mock_fb,
    )

    res: PipelineResult = pipeline.generate("thời tiết hôm nay thế nào")
    assert res.success is False
    assert res.status == "not_understood"


def test_pipeline_batch_generate(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra phương thức batch_generate xử lý nhiều câu hỏi đồng thời."""
    pipeline = LaTeXPipeline(scenario_bank=sample_scenario_bank)

    texts = [
        "tử số 1 và mẫu số 2",
        "tử số x và mẫu số y",
    ]
    results = pipeline.batch_generate(texts)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, PipelineResult)


def test_resolve_scenario_intent_override_valid(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra resolve_scenario khi người dùng chỉ định intent_override hợp lệ."""
    pipeline = LaTeXPipeline(scenario_bank=sample_scenario_bank)
    scenario, conf, fb_used, ret_score, cands, msg = pipeline.resolve_scenario(
        clean_text="nội dung bất kỳ",
        intent_override="MATHJAX_PHAN_SO",
    )
    assert scenario is not None
    assert scenario.intent == "MATHJAX_PHAN_SO"
    assert conf == 1.0
    assert fb_used is False
    assert msg == "intent_override"


def test_resolve_scenario_intent_override_invalid(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra resolve_scenario khi intent_override không tồn tại thì báo lỗi rõ ràng."""
    pipeline = LaTeXPipeline(scenario_bank=sample_scenario_bank)
    scenario, conf, fb_used, ret_score, cands, msg = pipeline.resolve_scenario(
        clean_text="tạo phân số",
        intent_override="NON_EXISTENT_INTENT",
    )
    assert scenario is None
    assert "Không tìm thấy kịch bản" in msg
    assert conf == 0.0


def test_resolve_scenario_auto_choose_fallback_false(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra chế độ fallback auto_choose_fallback=False trả về ask_user_to_choose."""
    mock_clf = MagicMock(spec=PhoBertClassifier)
    mock_clf.predict.return_value = ClassificationResult(
        intent="UNKNOWN",
        confidence=0.4,
        accepted=False,
    )
    mock_fb = MagicMock(spec=SbertRetrievalFallback)
    mock_fb.retrieve.return_value = FallbackResult(
        query="phân số",
        understood=True,
        top_candidates=[
            CandidateMatch(
                intent="MATHJAX_PHAN_SO",
                score=0.85,
                describe="Chèn phân số toán học",
                matched_text="phân số",
            )
        ],
    )
    pipeline = LaTeXPipeline(
        scenario_bank=sample_scenario_bank,
        classifier=mock_clf,
        fallback=mock_fb,
    )
    scenario, conf, fb_used, ret_score, cands, msg = pipeline.resolve_scenario(
        clean_text="phân số",
        auto_choose_fallback=False,
    )
    assert scenario is None
    assert fb_used is True
    assert ret_score == 0.85
    assert msg == "ask_user_to_choose"
    assert len(cands) == 1


def test_pipeline_generate_empty_or_whitespace_query(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra pipeline trả về not_understood khi câu truy vấn rỗng hoặc chỉ toàn khoảng trắng."""
    pipeline = LaTeXPipeline(scenario_bank=sample_scenario_bank)
    res_empty = pipeline.generate("")
    assert res_empty.success is False
    assert res_empty.status == "not_understood"

    res_spaces = pipeline.generate("   \n\t  ")
    assert res_spaces.success is False
    assert res_spaces.status == "not_understood"


def test_regex_slot_extractor_extract_by_intent(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra phương thức extract_by_intent của RegexSlotExtractor."""
    extractor = RegexSlotExtractor(scenario_bank=sample_scenario_bank)

    # Trường hợp intent tồn tại
    res_valid = extractor.extract_by_intent("MATHJAX_PHAN_SO", "tử số m và mẫu số n")
    assert res_valid.intent == "MATHJAX_PHAN_SO"
    assert res_valid.slots["num"] == "m"
    assert res_valid.slots["den"] == "n"
    assert res_valid.is_complete is True

    # Trường hợp intent không tồn tại
    res_invalid = extractor.extract_by_intent("UNKNOWN_INTENT", "tử số m")
    assert res_invalid.intent == "UNKNOWN_INTENT"
    assert res_invalid.is_complete is False


def test_regex_slot_extractor_multiple_slots(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra trích xuất nhiều slot trong một câu truy vấn phức tạp có dấu phân cách."""
    extractor = RegexSlotExtractor(scenario_bank=sample_scenario_bank)
    scenario = sample_scenario_bank["LATEX_TABLE_ENV"]

    text = "tạo bảng có chú thích Bảng số liệu kinh tế, nhãn tab:kinh_te"
    result = extractor.extract(scenario, text)
    assert result.is_complete is True
    assert result.slots["caption"] == "Bảng số liệu kinh tế"
    assert result.slots["label"] == "tab:kinh_te"


def test_latex_template_renderer_wrap_in_document() -> None:
    """Kiểm tra đóng gói đoạn mã LaTeX vào tài liệu hoàn chỉnh qua wrap_in_document."""
    snippet = r"\frac{a}{b}"
    packages = ["amsmath", "amssymb"]

    doc = LaTeXTemplateRenderer.wrap_in_document(snippet, packages=packages, document_class="article")
    assert r"\documentclass{article}" in doc
    assert r"\usepackage{amsmath}" in doc
    assert r"\usepackage{amssymb}" in doc
    assert r"\begin{document}" in doc
    assert snippet in doc
    assert r"\end{document}" in doc

    # Kiểm tra không bọc lặp nếu chuỗi đã là tài liệu hoàn chỉnh
    already_doc = r"\documentclass{article}" + "\n" + r"\begin{document}" + "\nHello\n" + r"\end{document}"
    assert LaTeXTemplateRenderer.wrap_in_document(already_doc) == already_doc


def test_latex_template_renderer_comprehensive_special_characters(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra cơ chế thoát toàn bộ các ký tự đặc biệt LaTeX của slot văn bản."""
    renderer = LaTeXTemplateRenderer()
    scenario = sample_scenario_bank["LATEX_SECTION"]

    # Chứa toàn bộ các ký tự đặc biệt: \, &, %, $, #, _, {, }, ~, ^
    special_text = r"Chi phí \ 100% & doanh thu $50 #1_hạng {mục} ~ 10^2"
    rendered = renderer.render(scenario, {"title": special_text}, escape_text_slots=True)

    assert r"\%" in rendered
    assert r"\&" in rendered
    assert r"\$" in rendered
    assert r"\#" in rendered
    assert r"\_" in rendered
    assert r"\{" in rendered
    assert r"\}" in rendered
    assert r"\textasciitilde{}" in rendered
    assert r"\textasciicircum{}" in rendered


def test_pipeline_generate_document_method(sample_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra phương thức generate_document sinh tài liệu hoàn chỉnh chứa documentclass và usepackage."""
    pipeline = LaTeXPipeline(scenario_bank=sample_scenario_bank)
    res = pipeline.generate_document(
        text="tạo phân số với tử số 5 và mẫu số 6",
        intent="MATHJAX_PHAN_SO",
    )
    assert res.success is True
    assert r"\documentclass{article}" in res.latex_code
    assert r"\usepackage{amsmath}" in res.latex_code
    assert r"\begin{document}" in res.latex_code
    assert r"\frac{5}{6}" in res.latex_code
    assert r"\end{document}" in res.latex_code


def test_pipeline_integration_end_to_end_classifier_path(sample_scenario_bank: ScenarioBank) -> None:
    """
    Kiểm thử tích hợp đầu-cuối (End-to-End Integration Test) theo luồng bộ phân loại trực tiếp.
    Văn bản tiếng Việt -> Tiền xử lý -> Phân loại trực tiếp -> Trích xuất regex -> Sinh mã LaTeX.
    """
    pipeline = LaTeXPipeline(scenario_bank=sample_scenario_bank)
    query = "tạo phân số với tử số 12 và mẫu số 35"
    result = pipeline.generate(query)

    assert result.success is True
    assert result.status == "success"
    assert result.intent == "MATHJAX_PHAN_SO"
    assert result.slots["num"] == "12"
    assert result.slots["den"] == "35"
    assert result.latex_code == r"\frac{12}{35}"
    assert "amsmath" in result.requires


def test_pipeline_integration_end_to_end_fallback_path(sample_scenario_bank: ScenarioBank) -> None:
    """
    Kiểm thử tích hợp đầu-cuối (End-to-End Integration Test) theo luồng fallback truy hồi ngữ nghĩa.
    Văn bản chứa alias -> PhoBERT p < tau -> Fallback SBERT top-3 -> Chọn top-1 -> Trích xuất slot -> Sinh LaTeX.
    """
    pipeline = LaTeXPipeline(scenario_bank=sample_scenario_bank)
    query = "chèn cấu trúc phân thức với tử số 99 và mẫu số 100"
    result = pipeline.generate(query)

    assert result.success is True
    assert result.intent == "MATHJAX_PHAN_SO"
    assert result.slots["num"] == "99"
    assert result.slots["den"] == "100"
    assert result.latex_code == r"\frac{99}{100}"


def test_pipeline_integration_missing_slots_dialogue_flow(sample_scenario_bank: ScenarioBank) -> None:
    """
    Kiểm thử tích hợp quy trình hội thoại hai bước khi người dùng ban đầu thiếu slot bắt buộc.
    Bước 1: Người dùng yêu cầu tạo section nhưng không cung cấp tiêu đề -> Pipeline yêu cầu bổ sung title.
    Bước 2: Người dùng cung cấp title qua slot_overrides -> Pipeline hoàn tất sinh mã LaTeX.
    """
    pipeline = LaTeXPipeline(scenario_bank=sample_scenario_bank)

    # Bước 1: Thiếu slot bắt buộc 'title'
    step1_res = pipeline.generate("tạo một section mới cho tài liệu", intent="LATEX_SECTION")
    assert step1_res.success is False
    assert step1_res.status == "missing_slots"
    assert step1_res.missing_slots == ["title"]
    assert "thiếu các tham số bắt buộc" in step1_res.message

    # Bước 2: Bổ sung slot bị thiếu qua slot_overrides
    step2_res = pipeline.generate(
        text="tạo một section mới cho tài liệu",
        intent="LATEX_SECTION",
        slot_overrides={"title": "Kết quả thực nghiệm"},
    )
    assert step2_res.success is True
    assert step2_res.status == "success"
    assert step2_res.latex_code == r"\section{Kết quả thực nghiệm}"


def test_pipeline_integration_with_default_intents_file() -> None:
    """
    Kiểm thử tích hợp với toàn bộ ngân hàng kịch bản thực tế từ data/intents.json (nếu có).
    Xác minh pipeline hoạt động trơn tru với dữ liệu thực tế của dự án.
    """
    if not DEFAULT_INTENTS_PATH.exists():
        pytest.skip(f"Tệp kịch bản {DEFAULT_INTENTS_PATH} không tồn tại trên đĩa.")

    pipeline = LaTeXPipeline()
    assert len(pipeline.scenario_bank) > 0

    # Kiểm tra intent cơ bản đã đăng ký trong data/intents.json
    res = pipeline.generate("khung latex tối thiểu", intent="LATEX_TAI_LIEU_TOI_GIAN")
    assert res.success is True
    assert r"\documentclass" in res.latex_code


def test_generate_latex_convenience_helper() -> None:
    """Kiểm tra hàm tiện ích cấp cao generate_latex."""
    res = generate_latex("tài liệu tối giản", intent="LATEX_TAI_LIEU_TOI_GIAN")
    if DEFAULT_INTENTS_PATH.exists():
        assert res.success is True
        assert r"\documentclass" in res.latex_code


def test_wrap_latex_in_document_helper() -> None:
    """Kiểm tra hàm tiện ích cấp cao wrap_latex_in_document."""
    doc = wrap_latex_in_document(r"\frac{1}{2}", packages=["amsmath"])
    assert r"\documentclass{article}" in doc
    assert r"\usepackage{amsmath}" in doc
    assert r"\frac{1}{2}" in doc
