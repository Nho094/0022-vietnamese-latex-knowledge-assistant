"""
Kiểm thử đơn vị cho giao diện dòng lệnh CLI trợ lý LaTeX (src/cli.py).

Tuân thủ quy tắc AGENTS.md:
- Kiểm tra toàn diện luồng: preprocess -> classifier -> fallback -> slot extraction -> render.
- Kiểm tra chế độ tương tác (interactive REPL / prompts) và chế độ đơn lẻ (--query, --raw).
- Chạy hoàn toàn offline, độc lập và có tính tái lập.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.config import (
    DEFAULT_INTENTS_PATH,
    TAU_INTENT_THRESHOLD,
    THETA_RETRIEVAL_THRESHOLD,
)
from src.cli import (
    AssistantEngine,
    CLIProcessResult,
    create_cli_parser,
    format_cli_output,
    main,
    preprocess_text,
)
from src.classifier import ClassificationResult, PhoBertClassifier
from src.fallback import CandidateMatch, FallbackResult, SbertRetrievalFallback
from src.scenarios import (
    Scenario,
    ScenarioBank,
    SlotDefinition,
    TemplateMapping,
)


@pytest.fixture
def mock_scenario_bank() -> ScenarioBank:
    """Fixture tạo ngân hàng kịch bản phục vụ kiểm thử CLI."""
    scenarios = [
        Scenario(
            intent="MATHJAX_PHAN_SO",
            category="math_expression",
            describe="Chèn phân số toán học",
            aliases=["phân số", "phân thức", "a trên b"],
            sample_utterances=["tạo phân số với tử a và mẫu b", "chèn phân thức"],
            slots={
                "num": SlotDefinition(name="num", pattern=r"(?:tử|tử số)\s+([a-zA-Z0-9]+)", default="a"),
                "den": SlotDefinition(name="den", pattern=r"(?:mẫu|mẫu số)\s+([a-zA-Z0-9]+)", default="b"),
            },
            templates=TemplateMapping(
                latex=r"\frac{" + "{num}" + "}{" + "{den}" + "}",
                mathjax=r"\frac{" + "{num}" + "}{" + "{den}" + "}",
            ),
        ),
        Scenario(
            intent="LATEX_SECTION",
            category="document_structure",
            describe="Tạo đề mục section",
            aliases=["section", "tiêu đề", "đề mục"],
            sample_utterances=["tạo section mới"],
            slots={
                "title": SlotDefinition(name="title", required=True, pattern=r"tiêu đề\s+là\s+([a-zA-Z0-9_]+)"),
            },
            templates=TemplateMapping(
                latex=r"\section{" + "{title}" + "}",
            ),
        ),
    ]
    return ScenarioBank(scenarios)


def test_preprocess_text_nfc_and_math_symbols() -> None:
    """Kiểm tra tiền xử lý chuẩn hóa Unicode NFC và ký hiệu toán học."""
    # Ký tự toán học full-width: ＋, －, ×, ÷
    raw = "tính ＋ và － và × và ÷"
    cleaned = preprocess_text(raw)
    assert "+" in cleaned
    assert "-" in cleaned
    assert "*" in cleaned
    assert "/" in cleaned


def test_preprocess_text_empty() -> None:
    """Kiểm tra xử lý văn bản rỗng."""
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""


def test_assistant_engine_direct_intent_accepted(mock_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra động cơ khi phân loại đạt ngưỡng tau (chấp nhận trực tiếp)."""
    mock_clf = MagicMock(spec=PhoBertClassifier)
    mock_clf.predict.return_value = ClassificationResult(
        intent="MATHJAX_PHAN_SO",
        confidence=0.95,
        accepted=True,
        category="math_expression",
    )
    mock_fb = MagicMock(spec=SbertRetrievalFallback)

    engine = AssistantEngine(
        scenario_bank=mock_scenario_bank,
        classifier=mock_clf,
        fallback=mock_fb,
    )

    res = engine.process_query("tạo phân số với tử 5 và mẫu 10")
    assert res.status == "success"
    assert res.intent == "MATHJAX_PHAN_SO"
    assert res.slots["num"] == "5"
    assert res.slots["den"] == "10"
    assert r"\frac{5}{10}" in res.rendered_code
    assert not res.fallback_used


def test_assistant_engine_fallback_user_choice(mock_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra động cơ kích hoạt fallback khi confidence < tau và người dùng chọn gợi ý."""
    mock_clf = MagicMock(spec=PhoBertClassifier)
    mock_clf.predict.return_value = ClassificationResult(
        intent="UNKNOWN",
        confidence=0.45,
        accepted=False,
    )
    mock_fb = MagicMock(spec=SbertRetrievalFallback)
    mock_fb.retrieve.return_value = FallbackResult(
        query="công thức phân số",
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

    engine = AssistantEngine(
        scenario_bank=mock_scenario_bank,
        classifier=mock_clf,
        fallback=mock_fb,
    )

    # Giả lập người dùng nhập '1' để chọn ứng viên thứ nhất
    user_inputs = iter(["1"])
    res = engine.process_query(
        "công thức phân số",
        interactive_input=lambda _: next(user_inputs),
    )

    assert res.status == "success"
    assert res.intent == "MATHJAX_PHAN_SO"
    assert res.fallback_used is True
    assert res.retrieval_score == 0.85


def test_assistant_engine_fallback_user_cancels(mock_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra người dùng chọn '0' để hủy khi fallback đưa ra gợi ý."""
    mock_clf = MagicMock(spec=PhoBertClassifier)
    mock_clf.predict.return_value = ClassificationResult(
        intent="UNKNOWN",
        confidence=0.35,
        accepted=False,
    )
    mock_fb = MagicMock(spec=SbertRetrievalFallback)
    mock_fb.retrieve.return_value = FallbackResult(
        query="cái gì đó không rõ",
        understood=True,
        top_candidates=[
            CandidateMatch(intent="MATHJAX_PHAN_SO", score=0.6, describe="Phân số", matched_text="phân số")
        ],
    )

    engine = AssistantEngine(
        scenario_bank=mock_scenario_bank,
        classifier=mock_clf,
        fallback=mock_fb,
    )

    # Giả lập người dùng nhập '0' để hủy
    res = engine.process_query(
        "cái gì đó không rõ",
        interactive_input=lambda _: "0",
    )
    assert res.status == "cancelled"


def test_assistant_engine_not_understood(mock_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra trường hợp fallback không hiểu (score < theta)."""
    mock_clf = MagicMock(spec=PhoBertClassifier)
    mock_clf.predict.return_value = ClassificationResult(
        intent="UNKNOWN",
        confidence=0.1,
        accepted=False,
    )
    mock_fb = MagicMock(spec=SbertRetrievalFallback)
    mock_fb.retrieve.return_value = FallbackResult(
        query="hôm nay ăn gì",
        understood=False,
        top_candidates=[],
    )

    engine = AssistantEngine(
        scenario_bank=mock_scenario_bank,
        classifier=mock_clf,
        fallback=mock_fb,
    )

    res = engine.process_query("hôm nay ăn gì")
    assert res.status == "not_understood"


def test_assistant_engine_missing_required_slots_interactive(mock_scenario_bank: ScenarioBank) -> None:
    """Kiểm tra hỏi bổ sung tham số bắt buộc còn thiếu trong tương tác."""
    mock_clf = MagicMock(spec=PhoBertClassifier)
    mock_clf.predict.return_value = ClassificationResult(
        intent="LATEX_SECTION",
        confidence=0.92,
        accepted=True,
    )

    engine = AssistantEngine(
        scenario_bank=mock_scenario_bank,
        classifier=mock_clf,
    )

    # Câu nhập không có tiêu đề -> thiếu slot 'title'
    user_inputs = iter(["TongQuan"])
    res = engine.process_query(
        "tạo section mới",
        interactive_input=lambda _: next(user_inputs),
    )

    assert res.status == "success"
    assert res.slots["title"] == "TongQuan"
    assert r"\section{TongQuan}" in res.rendered_code


def test_format_cli_output() -> None:
    """Kiểm tra định dạng kết quả hiển thị CLI."""
    success_res = CLIProcessResult(
        status="success",
        query="tạo phân số",
        intent="MATHJAX_PHAN_SO",
        rendered_code=r"\frac{a}{b}",
        target="latex",
    )
    out = format_cli_output(success_res, verbose=True)
    assert "MATHJAX_PHAN_SO" in out
    assert r"\frac{a}{b}" in out

    fail_res = CLIProcessResult(
        status="not_understood",
        query="alo 123",
        message="Không hiểu yêu cầu",
    )
    out_fail = format_cli_output(fail_res)
    assert "KHÔNG HIỂU YÊU CẦU" in out_fail


def test_cli_main_single_query(capsys: pytest.CaptureFixture[str]) -> None:
    """Kiểm tra thực thi CLI ở chế độ lệnh đơn qua hàm main."""
    exit_code = main(["--query", "tạo phân số", "--raw"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert r"\frac" in captured.out
