"""
Kiểm tra biên dịch LaTeX và xác minh tính sẵn sàng của trình biên dịch pdflatex (tests/test_latex.py).

Tuân thủ nghiêm ngặt các quy tắc AGENTS.md:
- Quy tắc bảo mật:
  + Tuyệt đối không chạy shell escape (luôn luôn bắt buộc -no-shell-escape).
  + Chỉ biên dịch mã xuất phát từ ngân hàng kịch bản hợp lệ (template bank).
  + Giới hạn thời gian thực thi (timeout) và làm việc trong thư mục tạm thời cách ly.
- Quy tắc trung thực (Honesty Rule #7):
  + Không bao giờ tuyên bố một template biên dịch được trừ khi bài kiểm tra biên dịch thực sự
    chạy và vượt qua trong phiên làm việc này; nếu chỉ chạy kiểm tra tĩnh thì phải ghi rõ 'static check only'.
- Yêu cầu kiểm thử:
  + Kiểm tra tính sẵn sàng của pdflatex trên hệ thống.
  + Nếu pdflatex có mặt: thực hiện kiểm thử biên dịch tài liệu LaTeX thực tế sang tệp PDF.
  + Nếu pdflatex vắng mặt: bỏ qua các bài kiểm thử biên dịch với thông báo mô tả rõ ràng, tường minh.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Final, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

from core.config import (
    DEFAULT_INTENTS_PATH,
    DEFAULT_LATEX_ENGINE,
    LATEX_COMMAND_FLAGS,
    LATEX_COMPILE_TIMEOUT_SECONDS,
    LATEX_STATIC_CHECK_ONLY,
    SUPPORTED_LATEX_ENGINES,
)
from src.pipeline import LaTeXPipeline, LaTeXTemplateRenderer, wrap_latex_in_document
from src.scenarios import load_scenario_bank

# Thông điệp mô tả tường minh khi bỏ qua kiểm thử biên dịch do thiếu pdflatex trên hệ thống
SKIP_REASON_PDFLATEX_ABSENT: Final[str] = (
    "pdflatex không có sẵn trên hệ thống (TeX Live chưa được cài đặt). "
    "Bỏ qua kiểm thử biên dịch tài liệu LaTeX theo quy tắc AGENTS.md."
)


def is_pdflatex_available(engine: str = DEFAULT_LATEX_ENGINE) -> bool:
    """
    Kiểm tra sự hiện diện của trình biên dịch LaTeX (mặc định pdflatex) trên hệ điều hành.

    Tham số:
        engine: Tên tệp thực thi của trình biên dịch (pdflatex, xelatex, lualatex).

    Trả về:
        True nếu engine có sẵn trong biến môi trường PATH, ngược lại False.
    """
    return shutil.which(engine) is not None


def compile_latex_document(
    latex_code: str,
    engine: str = DEFAULT_LATEX_ENGINE,
    flags: Sequence[str] = LATEX_COMMAND_FLAGS,
    timeout_seconds: int = LATEX_COMPILE_TIMEOUT_SECONDS,
    work_dir: Optional[Path] = None,
) -> tuple[bool, str, Optional[Path]]:
    """
    Biên dịch an toàn tài liệu LaTeX sang PDF với cờ bảo mật bắt buộc -no-shell-escape.

    Tuân thủ quy tắc bảo mật AGENTS.md:
    - Luôn thực thi với -no-shell-escape, cấm tuyệt đối -shell-escape.
    - Chạy trong môi trường thư mục tạm thời cách ly.
    - Giới hạn thời gian thực thi (timeout) để ngăn treo tiến trình.

    Tham số:
        latex_code: Chuỗi mã nguồn tài liệu LaTeX hoàn chỉnh.
        engine: Tên trình biên dịch LaTeX (mặc định 'pdflatex').
        flags: Danh sách cờ dòng lệnh (mặc định LATEX_COMMAND_FLAGS).
        timeout_seconds: Thời gian chạy tối đa tính bằng giây.
        work_dir: Thư mục làm việc (nếu None sẽ tự động tạo thư mục tạm).

    Trả về:
        Tuple gồm (thành_công: bool, log_hoặc_thông_báo: str, đường_dẫn_pdf: Optional[Path]).
    """
    # 1. Kiểm tra an toàn bảo mật (AGENTS.md Security Rules)
    if "-shell-escape" in flags:
        raise ValueError("Vi phạm bảo mật: Tuyệt đối không được sử dụng cờ -shell-escape.")

    if "-no-shell-escape" not in flags:
        raise ValueError("Vi phạm bảo mật: Cờ -no-shell-escape là bắt buộc theo quy định AGENTS.md.")

    if engine not in SUPPORTED_LATEX_ENGINES:
        raise ValueError(f"Trình biên dịch '{engine}' không được hỗ trợ. Các engine hợp lệ: {SUPPORTED_LATEX_ENGINES}")

    # 2. Kiểm tra tính sẵn sàng của trình biên dịch
    if not is_pdflatex_available(engine):
        return False, f"Trình biên dịch '{engine}' không tồn tại trên hệ thống.", None

    # 3. Chuẩn bị thư mục làm việc và tệp nguồn .tex
    def _run_compile(target_dir: Path) -> tuple[bool, str, Optional[Path]]:
        tex_path = target_dir / "document.tex"
        pdf_path = target_dir / "document.pdf"
        tex_path.write_text(latex_code, encoding="utf-8")

        cmd = [engine, *flags, "document.tex"]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(target_dir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            output_log = f"{result.stdout}\n{result.stderr}"
            if result.returncode == 0 and pdf_path.exists():
                return True, output_log, pdf_path
            return False, output_log, None
        except subprocess.TimeoutExpired:
            return False, f"Quá thời gian biên dịch cho phép ({timeout_seconds} giây).", None
        except FileNotFoundError:
            return False, f"Không tìm thấy tệp thực thi '{engine}'.", None

    if work_dir is not None:
        return _run_compile(work_dir)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        return _run_compile(Path(tmp_dir_str))


def static_latex_check(latex_code: str) -> tuple[bool, str]:
    """
    Kiểm tra cú pháp tĩnh (static check) cho mã nguồn LaTeX khi không có trình biên dịch thực tế.

    Tuân thủ Honesty Rule #7:
    - Nếu chỉ chạy kiểm tra tĩnh mà không biên dịch thực tế, báo cáo kết quả là 'static check only'.

    Kiểm tra:
    1. Tính cân bằng của các cặp dấu ngoặc nhọn `{` và `}`.
    2. Tính cân bằng của các khối môi trường \\begin{env} và \\end{env}.

    Tham số:
        latex_code: Chuỗi mã LaTeX cần kiểm tra.

    Trả về:
        Tuple gồm (hợp_lệ: bool, thông_báo: str).
    """
    if not latex_code.strip():
        return False, "Mã LaTeX rỗng."

    # Kiểm tra cân bằng dấu ngoặc nhọn
    brace_balance = 0
    for i, ch in enumerate(latex_code):
        if ch == "{" and (i == 0 or latex_code[i - 1] != "\\"):
            brace_balance += 1
        elif ch == "}" and (i == 0 or latex_code[i - 1] != "\\"):
            brace_balance -= 1
            if brace_balance < 0:
                return False, f"Dấu đóng ngoặc nhọn '}}' thừa tại vị trí {i}."

    if brace_balance != 0:
        return False, f"Mất cân bằng dấu ngoặc nhọn (chênh lệch {brace_balance})."

    # Kiểm tra khớp cặp \begin{env} và \end{env}
    begin_pattern = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
    end_pattern = re.compile(r"\\end\{([a-zA-Z*]+)\}")

    env_stack: list[str] = []
    tokens = re.finditer(r"\\(begin|end)\{([a-zA-Z*]+)\}", latex_code)
    for token in tokens:
        action, env_name = token.group(1), token.group(2)
        if action == "begin":
            env_stack.append(env_name)
        elif action == "end":
            if not env_stack:
                return False, f"Khối \\end{{{env_name}}} không có \\begin tương ứng."
            last_env = env_stack.pop()
            if last_env != env_name:
                return False, f"Môi trường không khớp: \\begin{{{last_env}}} nhưng gặp \\end{{{env_name}}}."

    if env_stack:
        return False, f"Các môi trường chưa được đóng: {', '.join(env_stack)}."

    # Trả về thành công với nhãn trung thực static check only theo Honesty Rule #7
    return True, LATEX_STATIC_CHECK_ONLY


# =========================================================================
# Các bài kiểm thử đơn vị và Mock (Chạy trong mọi môi trường)
# =========================================================================

def test_pdflatex_availability_detector() -> None:
    """Xác minh hàm is_pdflatex_available trả về giá trị boolean tương ứng với hệ thống."""
    available = is_pdflatex_available("pdflatex")
    assert isinstance(available, bool)
    assert available == (shutil.which("pdflatex") is not None)


def test_latex_compilation_security_enforcement() -> None:
    """Kiểm tra cơ chế chặn vi phạm bảo mật khi thiếu -no-shell-escape hoặc chứa -shell-escape."""
    dummy_code = r"\documentclass{article}\begin{document}Test\end{document}"

    # Lỗi nếu cố tình thêm cờ nguy hiểm -shell-escape
    with pytest.raises(ValueError, match="Tuyệt đối không được sử dụng cờ -shell-escape"):
        compile_latex_document(dummy_code, flags=["-shell-escape", "-no-shell-escape"])

    # Lỗi nếu thiếu cờ bắt buộc -no-shell-escape
    with pytest.raises(ValueError, match="Cờ -no-shell-escape là bắt buộc"):
        compile_latex_document(dummy_code, flags=["-interaction=nonstopmode"])


def test_unsupported_latex_engine_rejection() -> None:
    """Kiểm tra ném lỗi khi chỉ định engine không nằm trong SUPPORTED_LATEX_ENGINES."""
    dummy_code = r"\documentclass{article}\begin{document}Test\end{document}"
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        compile_latex_document(dummy_code, engine="insecure_custom_engine")


def test_compile_latex_document_engine_missing() -> None:
    """Kiểm tra xử lý nhã nhặn khi engine không tồn tại trên hệ thống."""
    dummy_code = r"\documentclass{article}\begin{document}Test\end{document}"
    with patch("shutil.which", return_value=None):
        success, msg, pdf_path = compile_latex_document(dummy_code, engine="pdflatex")
        assert success is False
        assert "không tồn tại trên hệ thống" in msg
        assert pdf_path is None


def test_mocked_pdflatex_compilation_success(tmp_path: Path) -> None:
    """Kiểm tra hàm compile_latex_document xử lý biên dịch thành công thông qua mock subprocess."""
    dummy_code = r"\documentclass{article}\begin{document}Hello World\end{document}"
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy pdf content")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Output written on document.pdf (1 page, 1234 bytes)."
    mock_result.stderr = ""

    with patch("tests.test_latex.is_pdflatex_available", return_value=True), \
         patch("subprocess.run", return_value=mock_result):
        success, log, path = compile_latex_document(
            latex_code=dummy_code,
            work_dir=tmp_path,
        )
        assert success is True
        assert path == pdf_file
        assert "Output written on document.pdf" in log


def test_mocked_pdflatex_compilation_failure(tmp_path: Path) -> None:
    """Kiểm tra hàm compile_latex_document khi lệnh biên dịch gặp lỗi cú pháp LaTeX."""
    dummy_code = r"\documentclass{article}\begin{document}\undefinedmacro\end{document}"

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = "! Undefined control sequence.\nl.1 \\undefinedmacro"
    mock_result.stderr = ""

    with patch("tests.test_latex.is_pdflatex_available", return_value=True), \
         patch("subprocess.run", return_value=mock_result):
        success, log, path = compile_latex_document(
            latex_code=dummy_code,
            work_dir=tmp_path,
        )
        assert success is False
        assert path is None
        assert "! Undefined control sequence." in log


def test_mocked_pdflatex_compilation_timeout(tmp_path: Path) -> None:
    """Kiểm tra xử lý khi biên dịch vượt quá giới hạn thời gian quy định."""
    dummy_code = r"\documentclass{article}\begin{document}Test\end{document}"

    with patch("tests.test_latex.is_pdflatex_available", return_value=True), \
         patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["pdflatex"], timeout=10)):
        success, log, path = compile_latex_document(
            latex_code=dummy_code,
            work_dir=tmp_path,
        )
        assert success is False
        assert path is None
        assert "Quá thời gian biên dịch cho phép" in log


def test_static_latex_check_valid_template() -> None:
    """Kiểm tra kiểm tra tĩnh (static check) thành công và trả về nhãn Honesty Rule #7."""
    valid_doc = (
        r"\documentclass{article}" + "\n"
        r"\usepackage{amsmath}" + "\n"
        r"\begin{document}" + "\n"
        r"\frac{a}{b}" + "\n"
        r"\end{document}"
    )
    is_valid, msg = static_latex_check(valid_doc)
    assert is_valid is True
    # Tuân thủ Honesty Rule #7: Phải ghi rõ 'static check only'
    assert msg == LATEX_STATIC_CHECK_ONLY


def test_static_latex_check_unbalanced_braces() -> None:
    """Kiểm tra phát hiện lỗi dấu ngoặc nhọn mất cân bằng."""
    invalid_code = r"\frac{1}{2"
    is_valid, msg = static_latex_check(invalid_code)
    assert is_valid is False
    assert "ngoặc nhọn" in msg


def test_static_latex_check_mismatched_environments() -> None:
    r"""Kiểm tra phát hiện lỗi môi trường \begin và \end không khớp."""
    invalid_code = r"\begin{table} Nội dung \end{figure}"
    is_valid, msg = static_latex_check(invalid_code)
    assert is_valid is False
    assert "Môi trường không khớp" in msg


def test_skip_message_descriptive_when_pdflatex_absent() -> None:
    """Xác minh thông báo skip rõ ràng, tường minh và đề cập chi tiết nguyên nhân vắng mặt pdflatex."""
    assert "pdflatex" in SKIP_REASON_PDFLATEX_ABSENT
    assert "TeX Live" in SKIP_REASON_PDFLATEX_ABSENT
    assert "AGENTS.md" in SKIP_REASON_PDFLATEX_ABSENT


# =========================================================================
# Các bài kiểm thử biên dịch thực tế (Live Compilation Tests)
# Nếu pdflatex vắng mặt -> Bỏ qua với thông báo tường minh mô tả lý do.
# Nếu pdflatex có mặt   -> Tiến hành biên dịch tài liệu sang tệp PDF thật.
# =========================================================================

@pytest.mark.skipif(not is_pdflatex_available(), reason=SKIP_REASON_PDFLATEX_ABSENT)
def test_live_minimal_latex_document_compilation() -> None:
    """Kiểm thử biên dịch thực tế tài liệu LaTeX tối thiểu khi pdflatex có mặt."""
    doc = (
        r"\documentclass{article}" + "\n"
        r"\begin{document}" + "\n"
        r"Xin chao Viet Nam - Vietnamese LaTeX Assistant." + "\n"
        r"\end{document}"
    )
    success, log, pdf_path = compile_latex_document(doc)
    assert success is True, f"Biên dịch thất bại. Log:\n{log}"
    assert pdf_path is not None
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


@pytest.mark.skipif(not is_pdflatex_available(), reason=SKIP_REASON_PDFLATEX_ABSENT)
def test_live_template_compilation_from_scenario_bank() -> None:
    """Kiểm thử biên dịch thực tế mẫu LaTeX sinh ra từ quy trình pipeline khi pdflatex có mặt."""
    pipeline = LaTeXPipeline()
    result = pipeline.generate_document(
        text="tạo phân số với tử số 3 và mẫu số 4",
        intent="MATHJAX_PHAN_SO",
    )
    assert result.success is True
    doc_code = result.latex_code

    success, log, pdf_path = compile_latex_document(doc_code)
    assert success is True, f"Biên dịch kịch bản thất bại. Log:\n{log}"
    assert pdf_path is not None
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


@pytest.mark.skipif(not is_pdflatex_available(), reason=SKIP_REASON_PDFLATEX_ABSENT)
def test_live_compilation_syntax_error_detection() -> None:
    """Kiểm thử trình biên dịch pdflatex phát hiện chính xác lỗi cú pháp khi biên dịch trực tiếp."""
    invalid_doc = (
        r"\documentclass{article}" + "\n"
        r"\begin{document}" + "\n"
        r"\badcommandnonexistent{" + "\n"
        r"\end{document}"
    )
    success, log, pdf_path = compile_latex_document(invalid_doc)
    assert success is False
    assert pdf_path is None
    assert ("Undefined control sequence" in log or "error" in log.lower())
