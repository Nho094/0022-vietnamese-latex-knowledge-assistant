"""
Giao diện dòng lệnh tương tác (Interactive CLI) cho trợ lý tri thức LaTeX tiếng Việt.

Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Kiến trúc luồng xử lý:
  text -> preprocess (NFC, chuẩn hóa ký hiệu toán học, tách từ pyvi cho PhoBERT)
    -> PhoBERT classifier: p = softmax over intents
       - if max p >= tau: intent accepted
       - else retrieval top-3 over aliases + scenario strings -> ask the user to choose (if best score < theta: say not understood)
    -> slot extraction (regex per intent) -> if a required slot is missing: ask for exactly the missing slots
    -> render latex template (text slots escaped) -> output
- Cảm hứng thiết kế từ hệ thống Retrieve-and-Fill (RAF) đơn giản hóa theo Honesty Rule #1:
  bước trích xuất slot dùng regex thay vì span-pointer decoder phức tạp.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

# Đảm bảo đường dẫn gốc dự án có trong sys.path khi thực thi trực tiếp qua python3 src/cli.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    CLI_BANNER_TITLE,
    CLI_EXIT_COMMANDS,
    CLI_PROMPT,
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
    extract_slots,
    load_scenario_bank,
    render_template,
)


def preprocess_text(text: str) -> str:
    """
    Tiền xử lý chuỗi văn bản đầu vào của người dùng.
    
    Quy trình:
    1. Chuẩn hóa dạng biểu diễn Unicode NFC (tránh lỗi ký tự tổ hợp tiếng Việt).
    2. Chuẩn hóa các ký hiệu toán học phổ biến (dấu cộng/trừ/nhân/chia toàn độ rộng).
    3. Thử phân đoạn từ vựng bằng pyvi nếu có sẵn thư viện; nếu không dùng tách từ mặc định.

    Tham số:
        text: Chuỗi văn bản tiếng Việt thô từ người dùng.

    Trả về:
        Chuỗi văn bản đã qua chuẩn hóa NFC và ký hiệu toán học.
    """
    if not text:
        return ""

    # Bước 1: Chuẩn hóa Unicode NFC
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
class CLIProcessResult:
    """
    Kết quả xử lý một câu truy vấn của trợ lý LaTeX.

    Thuộc tính:
        status: Trạng thái xử lý ('success', 'not_understood', 'cancelled', 'missing_slots', 'error').
        query: Chuỗi truy vấn gốc của người dùng.
        intent: Mã intent được xác định hoặc người dùng lựa chọn (None nếu không hiểu).
        scenario: Đối tượng kịch bản tương ứng nếu tìm thấy.
        confidence: Độ tin cậy p_max từ mô hình phân loại PhoBERT.
        fallback_used: Có kích hoạt bước truy hồi ngữ nghĩa dự phòng hay không.
        retrieval_score: Điểm cosine similarity nếu có kích hoạt fallback.
        slots: Bảng ánh xạ giá trị các slot đã trích xuất hoặc người dùng bổ sung.
        missing_slots: Danh sách slot bắt buộc còn thiếu (nếu có).
        rendered_code: Mã LaTeX (hoặc mã target) kết quả sau khi render template.
        target: Ngôn ngữ đích đã render ('latex', 'python', 'cpp', 'mathjax').
        message: Thông điệp giải thích hoặc phản hồi tương tác người dùng.
        elapsed_ms: Thời gian xử lý tính bằng mili-giây (ms).
    """
    status: str
    query: str
    intent: Optional[str] = None
    scenario: Optional[Scenario] = None
    confidence: float = 0.0
    fallback_used: bool = False
    retrieval_score: float = 0.0
    slots: dict[str, str] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    rendered_code: str = ""
    target: str = "latex"
    message: str = ""
    elapsed_ms: float = 0.0


class AssistantEngine:
    """
    Động cơ xử lý truy vấn trợ lý tri thức LaTeX tiếng Việt.
    Đóng gói toàn bộ chu trình xử lý theo kiến trúc quy định tại AGENTS.md.
    """

    def __init__(
        self,
        scenario_bank: Optional[ScenarioBank] = None,
        classifier: Optional[PhoBertClassifier] = None,
        fallback: Optional[SbertRetrievalFallback] = None,
        intents_path: Optional[Union[str, Path]] = None,
        threshold_tau: float = TAU_INTENT_THRESHOLD,
        threshold_theta: float = THETA_RETRIEVAL_THRESHOLD,
    ) -> None:
        """
        Khởi tạo động cơ trợ lý.

        Tham số:
            scenario_bank: Ngân hàng kịch bản nạp sẵn; nếu None sẽ nạp từ intents_path.
            classifier: Bộ phân loại PhoBERT; nếu None sẽ khởi tạo tự động.
            fallback: Bộ truy hồi SbertRetrievalFallback; nếu None sẽ khởi tạo tự động.
            intents_path: Đường dẫn tệp dữ liệu intents.json.
            threshold_tau: Ngưỡng xác suất softmax chấp nhận intent trực tiếp (tau).
            threshold_theta: Ngưỡng điểm tương đồng tối thiểu truy hồi fallback (theta).
        """
        path = intents_path or DEFAULT_INTENTS_PATH
        if scenario_bank is not None:
            self.scenario_bank = scenario_bank
        else:
            self.scenario_bank = load_scenario_bank(path)

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

    def process_query(
        self,
        query: str,
        target: str = "latex",
        interactive_input: Optional[Callable[[str], str]] = None,
        auto_choose_top_candidate: bool = False,
    ) -> CLIProcessResult:
        """
        Xử lý truy vấn tiếng Việt từ đầu vào đến sinh mã hoàn chỉnh.

        Tham số:
            query: Câu hỏi / yêu cầu tiếng Việt của người dùng.
            target: Ngôn ngữ đích render template ('latex', 'python', 'cpp', 'mathjax').
            interactive_input: Hàm callback nhận đầu vào tương tác từ người dùng (ví dụ: built-in input).
                               Nếu None, hệ thống chạy ở chế độ không tương tác (non-interactive).
            auto_choose_top_candidate: Tự động chọn ứng viên top-1 khi fallback nếu không có interactive_input.

        Trả về:
            CLIProcessResult chứa toàn bộ dữ liệu suy luận, slot và mã render.
        """
        start_time = time.perf_counter()

        # Kiểm tra tính hợp lệ của ngôn ngữ đích
        if target not in SUPPORTED_TARGETS:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return CLIProcessResult(
                status="error",
                query=query,
                target=target,
                message=f"Ngôn ngữ đích '{target}' không được hỗ trợ. Hỗ trợ: {', '.join(SUPPORTED_TARGETS)}",
                elapsed_ms=elapsed_ms,
            )

        # 1. Tiền xử lý (NFC, ký hiệu toán học)
        clean_text = preprocess_text(query)
        if not clean_text:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return CLIProcessResult(
                status="not_understood",
                query=query,
                target=target,
                message="Truy vấn rỗng. Vui lòng nhập câu hỏi hoặc yêu cầu tiếng Việt.",
                elapsed_ms=elapsed_ms,
            )

        # 2. Phân loại ý định qua PhoBERT (softmax over intents, p_max >= tau)
        clf_res: ClassificationResult = self.classifier.predict(clean_text)
        chosen_intent: Optional[str] = None
        fallback_used = False
        retrieval_score = 0.0

        if clf_res.accepted and clf_res.intent != "UNKNOWN":
            # Ý định được chấp nhận trực tiếp do p_max >= tau
            chosen_intent = clf_res.intent
        else:
            # 3. Kích hoạt fallback truy hồi ngữ nghĩa top-3
            fallback_used = True
            fb_res: FallbackResult = self.fallback.retrieve(clean_text)

            if not fb_res.understood or not fb_res.top_candidates:
                # Điểm số tốt nhất < theta (say not understood)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return CLIProcessResult(
                    status="not_understood",
                    query=query,
                    confidence=clf_res.confidence,
                    fallback_used=True,
                    target=target,
                    message="Không hiểu yêu cầu: Hệ thống không tìm thấy mẫu kịch bản LaTeX phù hợp.",
                    elapsed_ms=elapsed_ms,
                )

            # Điểm số >= theta -> Yêu cầu người dùng lựa chọn (ask the user to choose)
            best_cand = fb_res.top_candidates[0]
            retrieval_score = best_cand.score

            if interactive_input is not None:
                # Giao diện tương tác: hiển thị top-3 kịch bản để người dùng chọn
                print(f"\n[Gợi ý kịch bản] Độ tin cậy trực tiếp thấp ({clf_res.confidence:.2f} < {self.threshold_tau}).")
                print("Vui lòng chọn kịch bản phù hợp nhất từ các gợi ý sau:")
                for idx, cand in enumerate(fb_res.top_candidates, start=1):
                    print(f"  [{idx}] {cand.intent}: {cand.describe} (Điểm tương đồng: {cand.score:.2f})")
                print("  [0] Hủy bỏ thao tác / Không kịch bản nào phù hợp")

                try:
                    choice_str = interactive_input("Chọn số tương ứng [1-3, 0 để hủy]: ").strip()
                    if choice_str in ("1", "2", "3"):
                        chosen_idx = int(choice_str) - 1
                        if chosen_idx < len(fb_res.top_candidates):
                            chosen_cand = fb_res.top_candidates[chosen_idx]
                            chosen_intent = chosen_cand.intent
                            retrieval_score = chosen_cand.score
                    else:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                        return CLIProcessResult(
                            status="cancelled",
                            query=query,
                            confidence=clf_res.confidence,
                            fallback_used=True,
                            retrieval_score=retrieval_score,
                            target=target,
                            message="Người dùng đã hủy lựa chọn kịch bản.",
                            elapsed_ms=elapsed_ms,
                        )
                except (EOFError, KeyboardInterrupt):
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return CLIProcessResult(
                        status="cancelled",
                        query=query,
                        confidence=clf_res.confidence,
                        fallback_used=True,
                        target=target,
                        message="Ngắt thao tác bởi người dùng.",
                        elapsed_ms=elapsed_ms,
                    )
            elif auto_choose_top_candidate:
                # Chế độ tự động chọn ứng viên điểm cao nhất
                chosen_intent = best_cand.intent
                retrieval_score = best_cand.score
            else:
                # Chế độ non-interactive không bật tự động chọn: trả về danh sách gợi ý
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                cand_summaries = [f"{c.intent} ({c.score:.2f})" for c in fb_res.top_candidates]
                return CLIProcessResult(
                    status="ambiguous",
                    query=query,
                    confidence=clf_res.confidence,
                    fallback_used=True,
                    retrieval_score=retrieval_score,
                    target=target,
                    message=f"Cần xác nhận lựa chọn kịch bản: {', '.join(cand_summaries)}",
                    elapsed_ms=elapsed_ms,
                )

        if not chosen_intent or chosen_intent not in self.scenario_bank:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return CLIProcessResult(
                status="error",
                query=query,
                confidence=clf_res.confidence,
                fallback_used=fallback_used,
                target=target,
                message=f"Kịch bản '{chosen_intent}' không tồn tại trong ngân hàng kịch bản.",
                elapsed_ms=elapsed_ms,
            )

        scenario = self.scenario_bank[chosen_intent]

        # 4. Trích xuất slot (regex per intent)
        extracted, missing = extract_slots(scenario, clean_text)
        final_slots = dict(extracted)

        # Nếu thiếu slot bắt buộc (required=True) -> hỏi chính xác các slot còn thiếu
        if missing:
            if interactive_input is not None:
                print(f"\n[Bổ sung tham số] Kịch bản '{scenario.intent}' yêu cầu thêm tham số:")
                for slot_name in missing:
                    slot_def = scenario.slots.get(slot_name)
                    hint = f" ({slot_def.pattern})" if slot_def and slot_def.pattern else ""
                    try:
                        val = interactive_input(f"  Nhập giá trị cho '{slot_name}'{hint}: ").strip()
                        if val:
                            final_slots[slot_name] = val
                    except (EOFError, KeyboardInterrupt):
                        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                        return CLIProcessResult(
                            status="cancelled",
                            query=query,
                            intent=chosen_intent,
                            scenario=scenario,
                            confidence=clf_res.confidence,
                            fallback_used=fallback_used,
                            slots=final_slots,
                            missing_slots=missing,
                            target=target,
                            message="Ngắt bổ sung tham số bởi người dùng.",
                            elapsed_ms=elapsed_ms,
                        )
                # Kiểm tra lại các slot còn thiếu sau khi tương tác
                still_missing = [s for s in missing if not final_slots.get(s)]
                if still_missing:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return CLIProcessResult(
                        status="missing_slots",
                        query=query,
                        intent=chosen_intent,
                        scenario=scenario,
                        confidence=clf_res.confidence,
                        fallback_used=fallback_used,
                        slots=final_slots,
                        missing_slots=still_missing,
                        target=target,
                        message=f"Chưa cung cấp đủ tham số bắt buộc: {', '.join(still_missing)}",
                        elapsed_ms=elapsed_ms,
                    )
            else:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return CLIProcessResult(
                    status="missing_slots",
                    query=query,
                    intent=chosen_intent,
                    scenario=scenario,
                    confidence=clf_res.confidence,
                    fallback_used=fallback_used,
                    slots=final_slots,
                    missing_slots=missing,
                    target=target,
                    message=f"Thiếu tham số bắt buộc: {', '.join(missing)}",
                    elapsed_ms=elapsed_ms,
                )

        # 5. Render template (với text slots escaped theo Honesty Rule #7 & Architecture)
        rendered_code = render_template(
            scenario=scenario,
            slot_values=final_slots,
            target=target,
            escape_latex_special=True,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return CLIProcessResult(
            status="success",
            query=query,
            intent=chosen_intent,
            scenario=scenario,
            confidence=clf_res.confidence,
            fallback_used=fallback_used,
            retrieval_score=retrieval_score,
            slots=final_slots,
            missing_slots=[],
            rendered_code=rendered_code,
            target=target,
            message="Xử lý thành công và sinh mã.",
            elapsed_ms=elapsed_ms,
        )


def format_cli_output(result: CLIProcessResult, verbose: bool = False) -> str:
    """
    Định dạng kết quả trả về của trợ lý hiển thị trên terminal CLI.

    Tham số:
        result: Đối tượng kết quả CLIProcessResult.
        verbose: Hiển thị thêm chi tiết kỹ thuật (độ tin cậy, slots, thời gian xử lý).

    Trả về:
        Chuỗi văn bản được căn chỉnh hiển thị đẹp mắt.
    """
    lines: list[str] = []

    if result.status == "success":
        lines.append("=" * 64)
        lines.append(f"  Ý ĐỊNH PHÁT HIỆN: {result.intent}")
        if result.scenario and result.scenario.describe:
            lines.append(f"  MÔ TẢ:            {result.scenario.describe}")
        lines.append(f"  NGÔN NGỮ ĐÍCH:    {result.target}")
        if verbose:
            confidence_str = f"{result.confidence:.4f}"
            if result.fallback_used:
                confidence_str += f" (Fallback SBERT cosine score: {result.retrieval_score:.4f})"
            lines.append(f"  ĐỘ TIN CẬY:       {confidence_str}")
            lines.append(f"  THỜI GIAN:        {result.elapsed_ms:.2f} ms")
            if result.slots:
                slots_str = ", ".join(f"{k}='{v}'" for k, v in result.slots.items())
                lines.append(f"  THAM SỐ (SLOTS):  {slots_str}")
        lines.append("-" * 64)
        lines.append("  MÃ NGUỒN KẾT QUẢ:")
        lines.append("-" * 64)
        for code_line in result.rendered_code.splitlines():
            lines.append(f"    {code_line}")
        lines.append("=" * 64)
    elif result.status == "not_understood":
        lines.append("!" * 64)
        lines.append("  KHÔNG HIỂU YÊU CẦU:")
        lines.append(f"  {result.message}")
        if verbose:
            lines.append(f"  Độ tin cậy cao nhất: {result.confidence:.4f}")
            lines.append(f"  Thời gian: {result.elapsed_ms:.2f} ms")
        lines.append("!" * 64)
    elif result.status == "missing_slots":
        lines.append("-" * 64)
        lines.append(f"  THIẾU THAM SỐ CHO Ý ĐỊNH '{result.intent}':")
        lines.append(f"  {result.message}")
        lines.append("-" * 64)
    elif result.status == "cancelled":
        lines.append(f"[Đã hủy] {result.message}")
    else:
        lines.append(f"[Lỗi] {result.message}")

    return "\n".join(lines)


def print_banner() -> None:
    """In biểu ngữ chào mừng của ứng dụng CLI."""
    banner = f"""
================================================================
  {CLI_BANNER_TITLE}
  Trợ lý chuyển đổi yêu cầu tiếng Việt sang mã LaTeX (Kiểu RAF)
================================================================
  - Gõ yêu cầu bằng tiếng Việt (ví dụ: 'tạo phân số a trên b').
  - Gõ 'help' để xem hướng dẫn, 'exit' hoặc 'quit' để thoát.
================================================================
"""
    print(banner.strip())


def print_help() -> None:
    """In hướng dẫn sử dụng dòng lệnh."""
    help_text = """
HƯỚNG DẪN SỬ DỤNG:
  - Nhập câu lệnh tiếng Việt tự nhiên:
      * 'tạo ma trận 3 hàng 3 cột'
      * 'chèn công thức tích phân từ a đến b'
      * 'tạo bảng tabular với 3 cột'
      * 'viết hệ phương trình 2 ẩn'
  - Các lệnh điều khiển:
      * 'help': In trợ giúp này.
      * 'exit', 'quit', ':q': Thoát chương trình.
"""
    print(help_text.strip())


def run_interactive_repl(
    engine: AssistantEngine,
    target: str = "latex",
    verbose: bool = False,
) -> None:
    """
    Vòng lặp tương tác đọc-đánh giá-in (REPL loop) cho người dùng cuối.

    Tham số:
        engine: Đối tượng động cơ AssistantEngine.
        target: Ngôn ngữ đích render template ('latex', 'python', 'cpp', 'mathjax').
        verbose: Hiển thị thêm thông số kỹ thuật suy luận.
    """
    print_banner()
    print()

    while True:
        try:
            user_input = input(CLI_PROMPT).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nĐang thoát trợ lý. Tạm biệt!")
            break

        if not user_input:
            continue

        cmd_lower = user_input.lower()
        if cmd_lower in CLI_EXIT_COMMANDS:
            print("Đang thoát trợ lý. Tạm biệt!")
            break
        elif cmd_lower in ("help", "--help", "-h"):
            print_help()
            print()
            continue

        result = engine.process_query(
            query=user_input,
            target=target,
            interactive_input=input,
            auto_choose_top_candidate=False,
        )

        formatted = format_cli_output(result, verbose=verbose)
        print(formatted)
        print()


def create_cli_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh (CLI argument parser)."""
    parser = argparse.ArgumentParser(
        prog="latex-assistant",
        description="Trợ lý sinh mã LaTeX từ câu hỏi tiếng Việt dựa trên PhoBERT và Retrieval Fallback.",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Câu truy vấn tiếng Việt cần xử lý trực tiếp (chế độ chạy đơn lẻ).",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="latex",
        choices=list(SUPPORTED_TARGETS),
        help="Ngôn ngữ hoặc định dạng đích render mã (mặc định: 'latex').",
    )
    parser.add_argument(
        "--intents-path",
        "-i",
        type=str,
        default=None,
        help="Đường dẫn tùy chỉnh đến tệp intents.json (mặc định: data/intents.json).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=TAU_INTENT_THRESHOLD,
        help=f"Ngưỡng xác suất softmax chấp nhận intent (mặc định: {TAU_INTENT_THRESHOLD}).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=THETA_RETRIEVAL_THRESHOLD,
        help=f"Ngưỡng cosine similarity truy hồi fallback (mặc định: {THETA_RETRIEVAL_THRESHOLD}).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Chỉ in ra chuỗi mã nguồn kết quả (dễ dàng dùng trong pipe hoặc chuyển hướng).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="In chi tiết độ tin cậy mô hình, thời gian xử lý và các tham số slot.",
    )
    parser.add_argument(
        "--auto-fallback",
        action="store_true",
        help="Tự động chọn ứng viên top-1 khi fallback trong chế độ query đơn lẻ.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Điểm nhập chính khi chạy module từ dòng lệnh."""
    parser = create_cli_parser()
    args = parser.parse_args(argv)

    try:
        engine = AssistantEngine(
            intents_path=args.intents_path,
            threshold_tau=args.tau,
            threshold_theta=args.theta,
        )
    except Exception as exc:
        print(f"[Lỗi khởi tạo động cơ]: {exc}", file=sys.stderr)
        return 1

    if args.query:
        # Chế độ đơn lẻ (single query mode)
        res = engine.process_query(
            query=args.query,
            target=args.target,
            interactive_input=None,
            auto_choose_top_candidate=args.auto_fallback or True,
        )
        if args.raw:
            if res.status == "success":
                print(res.rendered_code)
                return 0
            else:
                print(f"[Error: {res.status}] {res.message}", file=sys.stderr)
                return 1
        else:
            print(format_cli_output(res, verbose=args.verbose))
            return 0 if res.status == "success" else 1

    # Chế độ vòng lặp tương tác (interactive REPL)
    run_interactive_repl(
        engine=engine,
        target=args.target,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
