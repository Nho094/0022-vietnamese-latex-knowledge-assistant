"""
Kiểm thử tự động cho module core/latex_tools.py (LaTeX String Utilities & Static Checker).

Tuân thủ nghiêm ngặt quy tắc kiến trúc và an toàn trong AGENTS.md:
- An toàn biên dịch LaTeX: kiểm tra tĩnh cú pháp và bảo mật trước khi biên dịch.
- escape_text: Thoát ký tự đặc biệt LaTeX theo cơ chế một lượt (single-pass) để tránh thoát lặp.
- strip_comments: Lọc bỏ chú thích LaTeX (% không bị thoát) trong khi bảo toàn ngắt dòng.
- static_check: Kiểm tra độ cân bằng dấu ngoặc nhọn, kiểm tra lồng nhau của các khối môi trường,
  và chặn đứng các từ điều khiển LaTeX nguy hiểm (input, include, write, openout, openin,
  read, immediate, catcode, directlua) với cơ chế khớp nguyên từ.
- wrap_snippet: Đóng gói mã nguồn LaTeX thành tài liệu hoàn chỉnh hỗ trợ các loại 'snippet',
  'preamble', và 'document'.
"""

from __future__ import annotations

import pytest

from core.latex_tools import (
    FORBIDDEN_WORDS,
    LATEX_ESCAPE_MAP,
    VALID_KINDS,
    escape_text,
    static_check,
    strip_comments,
    wrap_snippet,
)

# ==============================================================================
# 1. Kiểm thử escape_text (Single-pass character escaping)
# ==============================================================================


def test_escape_text_all_ten_special_characters() -> None:
    """Kiểm tra escape_text ánh xạ chính xác toàn bộ 10 ký tự đặc biệt của LaTeX."""
    # Toàn bộ 10 ký tự: \, {, }, $, &, #, ^, _, ~, %
    assert escape_text("\\") == r"\textbackslash{}"
    assert escape_text("{") == r"\{"
    assert escape_text("}") == r"\}"
    assert escape_text("$") == r"\$"
    assert escape_text("&") == r"\&"
    assert escape_text("#") == r"\#"
    assert escape_text("^") == r"\textasciicircum{}"
    assert escape_text("_") == r"\_"
    assert escape_text("~") == r"\textasciitilde{}"
    assert escape_text("%") == r"\%"


def test_escape_text_single_pass_no_double_escaping() -> None:
    """
    Kiểm tra cơ chế single-pass tránh lỗi thoát lặp đệ quy (double escaping).

    Khi ký tự '\\' được thay bằng '\\textbackslash{}', các dấu ngoặc nhọn '{' và '}'
    bên trong chuỗi thay thế không được tiếp tục bị biến đổi thành '\\{' và '\\}'.
    """
    input_str = r"\ { } $ & # ^ _ ~ %"
    escaped = escape_text(input_str)
    expected = (
        r"\textbackslash{} \{ \} \$ \& \# \textasciicircum{} \_ \textasciitilde{} \%"
    )
    assert escaped == expected

    # Kiểm tra chuỗi chứa nhiều dấu gạch chéo ngược liên tiếp và ngoặc nhọn
    complex_input = r"\\{test}"
    complex_escaped = escape_text(complex_input)
    assert complex_escaped == r"\textbackslash{}\textbackslash{}\{test\}"


def test_escape_text_edge_cases() -> None:
    """Kiểm tra các trường hợp biên của escape_text: rỗng, không có ký tự đặc biệt, văn bản tiếng Việt."""
    # Chuỗi rỗng
    assert escape_text("") == ""

    # Chuỗi không chứa ký tự đặc biệt
    plain_text = "Hoc may va xu ly ngon ngu tu nhien tieng Viet"
    assert escape_text(plain_text) == plain_text

    # Chuỗi tiếng Việt có dấu và ký hiệu số học/đơn vị
    mixed_text = "Giảm giá 50% cho hóa đơn #123 & thanh toán $99.99 tại C:\\tailieu\\~nguoidung"
    expected = (
        r"Giảm giá 50\% cho hóa đơn \#123 \& thanh toán \$99.99 tại "
        r"C:\textbackslash{}tailieu\textbackslash{}\textasciitilde{}nguoidung"
    )
    assert escape_text(mixed_text) == expected


def test_latex_escape_map_completeness() -> None:
    """Kiểm tra hằng số LATEX_ESCAPE_MAP có đủ 10 ký tự đặc biệt."""
    assert len(LATEX_ESCAPE_MAP) == 10
    expected_keys = {"\\", "{", "}", "$", "&", "#", "^", "_", "~", "%"}
    assert set(LATEX_ESCAPE_MAP.keys()) == expected_keys


# ==============================================================================
# 2. Kiểm thử strip_comments (Comment filtering)
# ==============================================================================


def test_strip_comments_simple_line() -> None:
    """Kiểm tra strip_comments loại bỏ nội dung chú thích sau % nhưng giữ nguyên ngắt dòng."""
    code = "x = 10 % Đây là chú thích\ny = 20"
    cleaned = strip_comments(code)
    assert cleaned == "x = 10 \ny = 20"


def test_strip_comments_full_line_comment() -> None:
    """Kiểm tra dòng chú thích toàn bộ được thay bằng dòng trống hoặc giữ ngắt dòng."""
    code = "% Dòng chú thích đầu tiên\nx = 1\n% Dòng chú thích thứ hai"
    cleaned = strip_comments(code)
    assert cleaned == "\nx = 1\n"


def test_strip_comments_escaped_percent_sign() -> None:
    """Kiểm tra dấu \\% bị thoát thì KHÔNG bị coi là chú thích."""
    code = r"Tỉ lệ thành công là 95\% theo thống kê % chú thích ở cuối dòng"
    cleaned = strip_comments(code)
    assert cleaned == r"Tỉ lệ thành công là 95\% theo thống kê "


def test_strip_comments_even_vs_odd_backslashes() -> None:
    """
    Kiểm tra xử lý số lượng dấu gạch chéo ngược đứng trước %:
    - Số chẵn (\\\\%): \\\\ là lệnh ngắt dòng của LaTeX, dấu % không bị thoát -> là chú thích!
    - Số lẻ (\\\\\\%): \\\\ là ngắt dòng, \\% là dấu % được thoát -> KHÔNG phải chú thích!
    """
    # 2 backslashes: chẵn -> % là chú thích
    code_even = r"Dòng 1\\% chú thích này phải bị xóa" + "\nDòng 2"
    cleaned_even = strip_comments(code_even)
    assert cleaned_even == r"Dòng 1\\" + "\nDòng 2"

    # 3 backslashes: lẻ -> \% là ký tự thoát, không xóa
    code_odd = r"Dòng 1\\\% nội dung 95\% hiển thị bình thường"
    cleaned_odd = strip_comments(code_odd)
    assert cleaned_odd == code_odd


def test_strip_comments_no_comments() -> None:
    """Kiểm tra chuỗi không có chú thích thì giữ nguyên tuyệt đối."""
    code = "\\begin{equation}\n  E = mc^2\n\\end{equation}"
    assert strip_comments(code) == code


# ==============================================================================
# 3. Kiểm thử static_check: Brace balancing edge cases
# ==============================================================================


def test_static_check_balanced_braces() -> None:
    """Kiểm tra các trường hợp dấu ngoặc nhọn cân bằng hoàn hảo không sinh lỗi."""
    valid_codes = [
        "",
        "   ",
        r"\textbf{Xin chào}",
        r"\frac{1}{2} + \sqrt{x^{2} + 1}",
        r"{ { { a } } }",
        r"\begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix}",
    ]
    for code in valid_codes:
        assert static_check(code) == [], f"Mã hợp lệ không được báo lỗi: {code}"


def test_static_check_unbalanced_open_braces() -> None:
    """Kiểm tra phát hiện thiếu dấu đóng ngoặc nhọn (depth > 0)."""
    code = r"\textbf{Chưa đóng ngoặc"
    problems = static_check(code)
    assert len(problems) == 1
    assert "Mất cân bằng dấu ngoặc nhọn: còn 1 dấu mở ngoặc '{' chưa được đóng." in problems[0]

    code_multi = r"\frac{1}{2"
    problems_multi = static_check(code_multi)
    assert len(problems_multi) == 1
    assert "còn 1 dấu mở ngoặc" in problems_multi[0]


def test_static_check_unbalanced_extra_closing_braces() -> None:
    """Kiểm tra phát hiện thừa dấu đóng ngoặc nhọn (depth < 0)."""
    code = r"\textbf{Chuẩn}} thêm ngoặc đóng"
    problems = static_check(code)
    assert any("Dấu đóng ngoặc nhọn '}' thừa" in p for p in problems)


def test_static_check_escaped_braces_ignored() -> None:
    """Kiểm tra các dấu ngoặc nhọn đã được thoát \\{ và \\} không ảnh hưởng đến bộ đếm độ sâu."""
    code = r"Tập hợp \{ x \mid x > 0 \} có giá trị trong {khoảng này}"
    assert static_check(code) == []


def test_static_check_even_backslashes_before_brace() -> None:
    """
    Kiểm tra trường hợp \\\\{ (2 dấu gạch chéo ngược):
    Dấu { không bị thoát (vì \\\\ là ngắt dòng) nên phải tính vào độ sâu ngoặc.
    """
    code_unbalanced = r"Dòng 1\\{chưa đóng"
    problems = static_check(code_unbalanced)
    assert any("còn 1 dấu mở ngoặc" in p for p in problems)


def test_static_check_braces_inside_comments_ignored() -> None:
    """Kiểm tra dấu ngoặc nhọn nằm trong đoạn chú thích % bị bỏ qua hoàn toàn."""
    code = "x = 1 % { dấu mở ngoặc trong comment\ny = 2"
    assert static_check(code) == []


# ==============================================================================
# 4. Kiểm thử static_check: Environment nesting checks
# ==============================================================================


def test_static_check_valid_environment_nesting() -> None:
    """Kiểm tra các khối môi trường hợp lệ lồng nhau đúng chuẩn ngăn xếp (stack)."""
    code = (
        r"\begin{table}" "\n"
        r"  \begin{tabular}{cc}" "\n"
        r"    1 & 2 \\" "\n"
        r"  \end{tabular}" "\n"
        r"\end{table}"
    )
    assert static_check(code) == []


def test_static_check_environments_with_special_characters() -> None:
    """Kiểm tra tên môi trường chứa ký tự đặc biệt hợp lệ như *, -, _ (ví dụ: align*, table*)."""
    code = (
        r"\begin{align*}" "\n"
        r"  a &= b + c \\" "\n"
        r"  d &= e" "\n"
        r"\end{align*}"
    )
    assert static_check(code) == []


def test_static_check_unclosed_environment() -> None:
    """Kiểm tra phát hiện môi trường chưa được đóng \\end."""
    code = r"\begin{figure}[h]\centering"
    problems = static_check(code)
    assert any("Các môi trường chưa được đóng: \\begin{figure}" in p for p in problems)


def test_static_check_unopened_end_environment() -> None:
    """Kiểm tra phát hiện \\end xuất hiện mà không có \\begin tương ứng."""
    code = r"\end{equation}"
    problems = static_check(code)
    assert any(r"Khối \end{equation} xuất hiện mà không có \begin tương ứng" in p for p in problems)


def test_static_check_mismatched_environment_nesting() -> None:
    """Kiểm tra phát hiện đóng sai thứ tự lồng nhau của các khối môi trường."""
    code = (
        r"\begin{enumerate}" "\n"
        r"  \begin{itemize}" "\n"
        r"  \end{enumerate}" "\n"
        r"\end{itemize}"
    )
    problems = static_check(code)
    assert any(r"Môi trường lồng nhau không khớp: \begin{itemize} bị đóng bởi \end{enumerate}" in p for p in problems)


# ==============================================================================
# 5. Kiểm thử static_check: Forbidden control word detection
# ==============================================================================


@pytest.mark.parametrize("word", FORBIDDEN_WORDS)
def test_static_check_detects_forbidden_words(word: str) -> None:
    """Kiểm tra phát hiện tất cả các từ điều khiển bị cấm vì lý do an toàn bảo mật."""
    code = f"\\{word}{{payload}}"
    problems = static_check(code)
    assert any(f"Phát hiện từ điều khiển LaTeX bị cấm vì lý do bảo mật: '\\{word.lower()}'" in p for p in problems)


def test_static_check_forbidden_words_case_insensitive() -> None:
    """Kiểm tra phát hiện từ điều khiển cấm không phân biệt chữ hoa chữ thường."""
    code = r"\INPUT{secret.tex} \Write18{rm -rf /} \DirectLua{os.execute()}"
    problems = static_check(code)
    assert any(r"'\input'" in p for p in problems)
    assert any(r"'\write'" in p for p in problems)
    assert any(r"'\directlua'" in p for p in problems)


def test_static_check_allows_safe_commands_with_forbidden_prefix() -> None:
    """
    Kiểm tra cơ chế khớp nguyên từ: KHÔNG chặn nhầm các lệnh an toàn hợp lệ
    có chứa tiền tố là từ bị cấm (ví dụ: \\includegraphics chứa 'include').
    """
    safe_code = (
        r"\begin{figure}" "\n"
        r"  \includegraphics[width=0.8\textwidth]{anh_minh_hoa.png}" "\n"
        r"  \inputlineno" "\n"
        r"\end{figure}"
    )
    problems = static_check(safe_code)
    assert problems == [], f"Lệnh an toàn không được bị chặn nhầm: {problems}"


def test_static_check_escaped_backslash_before_forbidden_word() -> None:
    """
    Kiểm tra trường hợp \\\\input (2 backslashes: lệnh ngắt dòng và chữ 'input'):
    Không được xem là control word \\input vì dấu gạch chéo đã kết thúc cặp \\\\.
    """
    code = r"Kết thúc dòng 1\\input dữ liệu mới"
    assert static_check(code) == []

    # 3 backslashes: \\\\ là ngắt dòng, theo sau là \\input thật sự -> phải bị chặn!
    code_dangerous = r"Dòng 1\\\input{secret.tex}"
    problems = static_check(code_dangerous)
    assert any(r"'\input'" in p for p in problems)


def test_static_check_forbidden_words_in_comments_ignored() -> None:
    """Kiểm tra các từ cấm nằm trong dòng chú thích % không gây cảnh báo giả."""
    code = r"% Chú ý: không sử dụng \input{file.tex} ở đây" "\n" r"\textbf{Hợp lệ}"
    assert static_check(code) == []


# ==============================================================================
# 6. Kiểm thử wrap_snippet: Snippet wrapping
# ==============================================================================


def test_wrap_snippet_default_kind_snippet() -> None:
    """Kiểm tra wrap_snippet với kind mặc định ('snippet') tạo tài liệu hoàn chỉnh."""
    code = r"\frac{a}{b}"
    wrapped = wrap_snippet(code, requires=["amsmath"])
    assert r"\documentclass{article}" in wrapped
    assert r"\usepackage{amsmath}" in wrapped
    assert r"\begin{document}" in wrapped
    assert r"\frac{a}{b}" in wrapped
    assert r"\end{document}" in wrapped


def test_wrap_snippet_kind_preamble() -> None:
    """Kiểm tra wrap_snippet với kind='preamble' đặt mã nguồn trước \\begin{document}."""
    code = r"\newcommand{\mycmd}{Hello}"
    wrapped = wrap_snippet(code, kind="preamble", requires=["xcolor"])
    assert r"\documentclass{article}" in wrapped
    assert r"\usepackage{xcolor}" in wrapped
    assert r"\newcommand{\mycmd}{Hello}" in wrapped
    # Mã lệnh phải nằm trước \begin{document}
    cmd_idx = wrapped.find(r"\newcommand{\mycmd}{Hello}")
    doc_idx = wrapped.find(r"\begin{document}")
    assert cmd_idx != -1 and doc_idx != -1
    assert cmd_idx < doc_idx


def test_wrap_snippet_kind_document() -> None:
    """Kiểm tra wrap_snippet với kind='document' bổ sung documentclass và các gói nếu thiếu."""
    code = r"\begin{center} Nội dung \end{center}"
    wrapped = wrap_snippet(code, kind="document", requires=["graphicx"])
    assert r"\documentclass{article}" in wrapped
    assert r"\usepackage{graphicx}" in wrapped
    assert r"\begin{document}" in wrapped
    assert r"\end{document}" in wrapped


def test_wrap_snippet_already_full_document() -> None:
    """Kiểm tra khi mã nguồn đã là tài liệu hoàn chỉnh, không bọc lặp lại mà chèn thêm gói thiếu."""
    full_doc = (
        r"\documentclass{report}" "\n"
        r"\begin{document}" "\n"
        r"Nội dung tài liệu" "\n"
        r"\end{document}"
    )
    wrapped = wrap_snippet(full_doc, requires=["amsmath"])
    # Không tạo 2 lần \documentclass hoặc \begin{document}
    assert wrapped.count(r"\documentclass") == 1
    assert wrapped.count(r"\begin{document}") == 1
    assert r"\usepackage{amsmath}" in wrapped


def test_wrap_snippet_avoids_duplicate_packages() -> None:
    """Kiểm tra gói đã được khai báo trong mã nguồn thì không bị chèn trùng lặp."""
    code = (
        r"\documentclass{article}" "\n"
        r"\usepackage{amsmath}" "\n"
        r"\begin{document}" "\n"
        r"E = mc^2" "\n"
        r"\end{document}"
    )
    wrapped = wrap_snippet(code, requires=["amsmath", "amsfonts"])
    assert wrapped.count(r"\usepackage{amsmath}") == 1
    assert r"\usepackage{amsfonts}" in wrapped


def test_wrap_snippet_custom_document_class() -> None:
    """Kiểm tra tham số tùy chỉnh document_class (ví dụ: 'book', 'report')."""
    wrapped = wrap_snippet("Nội dung", document_class="report")
    assert r"\documentclass{report}" in wrapped


def test_wrap_snippet_invalid_kind_raises_error() -> None:
    """Kiểm tra wrap_snippet ném ngoại lệ ValueError khi truyền kind không hợp lệ."""
    with pytest.raises(ValueError, match="Loại kịch bản 'kind' không hợp lệ"):
        wrap_snippet("test", kind="invalid_kind")


def test_valid_kinds_constant() -> None:
    """Kiểm tra hằng số VALID_KINDS chứa 3 loại: snippet, preamble, document."""
    assert VALID_KINDS == frozenset({"snippet", "preamble", "document"})
