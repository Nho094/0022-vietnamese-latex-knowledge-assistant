"""
Module tiện ích chuỗi LaTeX thuần túy và bộ kiểm tra cú pháp, an ninh tĩnh (core/latex_tools.py).

Tuân thủ nghiêm ngặt quy tắc kiến trúc và an toàn trong AGENTS.md:
- An toàn biên dịch LaTeX:
  + Không bao giờ thực thi shell escape (luôn luôn -no-shell-escape).
  + Kiểm tra an toàn tĩnh (static check) trước khi đưa vào biên dịch hoặc trả về kết quả.
  + Chặn đứng các từ điều khiển nguy hiểm có thể gây rò rỉ tập tin hoặc thực thi mã tùy ý:
    input, include, write, openout, openin, read, immediate, catcode, directlua.
  + Cơ chế khớp nguyên từ để không chặn nhầm các lệnh an toàn hợp lệ như \\includegraphics.
- Tiện ích xử lý chuỗi:
  + escape_text(s): Sử dụng re.sub một lượt (single-pass) với ánh xạ từ điển để tránh lỗi
    thoát lặp đệ quy (ví dụ \\ -> \\textbackslash{} không bị các bước sau tiếp tục thoát { và }).
  + wrap_snippet(code, requires, kind): Sinh tài liệu LaTeX hoàn chỉnh có thể biên dịch
    cho các loại kịch bản: 'snippet', 'preamble', hoặc 'document'.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Callable, Final

from core.config import (
    FORBIDDEN_LATEX_CONTROL_WORDS,
    POST_ACTION_TIDY_MATH,
    TEMPLATE_TARGET_FALLBACKS,
)

# Ánh xạ thoát ký tự đặc biệt LaTeX (toàn bộ 10 ký tự đặc biệt: \, {, }, $, &, #, ^, _, ~, %)
LATEX_ESCAPE_MAP: Final[dict[str, str]] = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "^": r"\textasciicircum{}",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "%": r"\%",
}

# Biểu thức chính quy cho single-pass escape_text
ESCAPE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\\{}$&#^_~%]")

# Danh sách từ điều khiển bị cấm vì lý do an toàn bảo mật
FORBIDDEN_WORDS: Final[Sequence[str]] = FORBIDDEN_LATEX_CONTROL_WORDS

# Mẫu regex phát hiện từ điều khiển LaTeX bị cấm khớp nguyên từ (control words)
# Yêu cầu:
# 1. Bắt đầu bằng dấu gạch chéo ngược không bị thoát: (?<!\\)(?:\\\\)*\\
# 2. Khớp chính xác tên từ điều khiển
# 3. Ký tự tiếp theo không phải là chữ cái: (?![a-zA-Z]) để cho phép \includegraphics
FORBIDDEN_WORDS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\\)(?:\\\\)*\\(" + "|".join(re.escape(w) for w in FORBIDDEN_WORDS) + r")(?![a-zA-Z])",
    re.IGNORECASE,
)

# Mẫu regex phát hiện các cặp môi trường \begin{env} và \end{env}
ENV_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\\)(?:\\\\)*\\(begin|end)\{([a-zA-Z0-9*_-]+)\}"
)

# Các loại kind hợp lệ cho wrap_snippet
VALID_KINDS: Final[frozenset[str]] = frozenset({"snippet", "preamble", "document"})


def resolve_template(intent_data: dict[str, object], target: str) -> str | None:
    """Resolve a target template using only explicitly configured fallbacks."""
    templates = intent_data.get("templates", {})
    if not isinstance(templates, dict):
        return None
    direct = templates.get(target)
    if isinstance(direct, str) and direct.strip():
        return direct
    for fallback in TEMPLATE_TARGET_FALLBACKS.get(target, ()):
        candidate = templates.get(fallback)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def tidy_math(code: str) -> str:
    """Chuẩn hóa dấu và hệ số đơn vị theo các dạng đại số phổ biến."""
    value = re.sub(r"\+\s*-\s*(?=\d)", "- ", code)
    value = re.sub(
        r"(?<![\w.])(-?)1(?=[A-Za-z])", lambda match: "-" if match.group(1) else "", value
    )
    variable_term = r"0\s*[A-Za-z](?:\s*\^\s*(?:\{\d+\}|\d+))?"
    value = re.sub(rf"(?<!\w)[+-]\s*{variable_term}(?=\s*[+=-])", "", value)
    value = re.sub(rf"(?<!\w){variable_term}(?=\s*[+=-])", "", value)
    value = re.sub(r"\s*[+-]\s*0(?=\s*=)", "", value)
    value = re.sub(r"(?m)(=\s*)\+\s*", r"\1", value)
    value = re.sub(r"(?m)(^|[\n{])\s*\+\s*", r"\1", value)
    value = re.sub(r"-\s*-\s*(?=\d)", "+ ", value)
    return re.sub(r"[ \t]{2,}", " ", value)


POST_ACTIONS: Final[dict[str, Callable[[str], str]]] = {
    POST_ACTION_TIDY_MATH: tidy_math,
}


def escape_text(s: str) -> str:
    """
    Thoát các ký tự điều khiển đặc biệt của LaTeX trong chuỗi văn bản.

    Sử dụng kỹ thuật single-pass re.sub với từ điển ánh xạ cho 10 ký tự đặc biệt:
    \\, {, }, $, &, #, ^, _, ~, %.
    Tránh hiện tượng thoát lặp (double-escaping) khi các chuỗi thay thế chứa ký tự đặc biệt.

    Tham số:
        s: Chuỗi ký tự đầu vào.

    Trả về:
        Chuỗi đã được thoát ký tự đặc biệt an toàn cho LaTeX.
    """
    if not s:
        return ""
    return ESCAPE_PATTERN.sub(lambda m: LATEX_ESCAPE_MAP[m.group(0)], s)


def strip_comments(code: str) -> str:
    """
    Loại bỏ các đoạn chú thích (comment text) bắt đầu bằng ký tự % không bị thoát.
    Giữ nguyên cấu trúc các dòng để không làm sai lệch vị trí dòng.

    Tham số:
        code: Chuỗi mã nguồn LaTeX gốc.

    Trả về:
        Chuỗi mã nguồn LaTeX đã loại bỏ chú thích.
    """
    result: list[str] = []
    i = 0
    n = len(code)
    while i < n:
        ch = code[i]
        if ch == "%":
            # Đếm số lượng dấu gạch chéo ngược đứng ngay trước nó
            bs_count = 0
            j = i - 1
            while j >= 0 and code[j] == "\\":
                bs_count += 1
                j -= 1
            if bs_count % 2 == 0:
                # Dấu % không bị thoát -> Bắt đầu chú thích đến hết dòng
                while i < n and code[i] != "\n":
                    i += 1
                if i < n and code[i] == "\n":
                    result.append("\n")
                    i += 1
                continue
        result.append(ch)
        i += 1
    return "".join(result)


def static_check(code: str) -> list[str]:
    """
    Kiểm tra tĩnh tính hợp lệ cú pháp và an toàn bảo mật cho mã nguồn LaTeX.

    Các tiêu chí kiểm tra:
    1. Độ sâu dấu ngoặc nhọn { và } cân bằng và không âm tại mọi thời điểm
       (bỏ qua các dấu ngoặc nhọn đã được thoát \\{, \\} và nội dung chú thích).
    2. Các cặp môi trường \\begin{env} và \\end{env} lồng nhau chuẩn xác theo cấu trúc ngăn xếp (stack).
    3. Từ chối tuyệt đối các từ điều khiển bị cấm vì lý do bảo mật:
       input, include, write, openout, openin, read, immediate, catcode, directlua.
       Khớp nguyên từ để cho phép các lệnh an toàn như \\includegraphics.

    Tham số:
        code: Đoạn mã nguồn LaTeX cần kiểm tra.

    Trả về:
        Danh sách các thông báo lỗi phát hiện được (danh sách rỗng [] nếu hoàn toàn hợp lệ).
    """
    problems: list[str] = []

    if not code or not code.strip():
        return problems

    # Loại bỏ nội dung chú thích trước khi phân tích cú pháp
    clean_code = strip_comments(code)

    # 1. Kiểm tra cân bằng độ sâu dấu ngoặc nhọn không âm
    depth = 0
    i = 0
    n = len(clean_code)
    while i < n:
        ch = clean_code[i]
        if ch in ("{", "}"):
            bs_count = 0
            j = i - 1
            while j >= 0 and clean_code[j] == "\\":
                bs_count += 1
                j -= 1
            if bs_count % 2 == 0:
                # Ký tự ngoặc nhọn không bị thoát
                if ch == "{":
                    depth += 1
                else:  # ch == "}"
                    depth -= 1
                    if depth < 0:
                        problems.append(
                            f"Dấu đóng ngoặc nhọn '}}' thừa tại vị trí {i} (độ sâu ngoặc âm)."
                        )
                        depth = 0  # Đặt lại 0 để không gây lỗi lan truyền giả
        i += 1

    if depth > 0:
        problems.append(
            f"Mất cân bằng dấu ngoặc nhọn: còn {depth} dấu mở ngoặc '{{' chưa được đóng."
        )

    # 2. Kiểm tra lồng nhau chuẩn xác của các khối môi trường \begin{env} và \end{env}
    env_stack: list[str] = []
    for match in ENV_PATTERN.finditer(clean_code):
        tag, env_name = match.group(1), match.group(2)
        if tag == "begin":
            env_stack.append(env_name)
        elif tag == "end":
            if not env_stack:
                problems.append(
                    f"Khối \\end{{{env_name}}} xuất hiện mà không có \\begin tương ứng."
                )
            else:
                top = env_stack.pop()
                if top != env_name:
                    problems.append(
                        f"Môi trường lồng nhau không khớp: \\begin{{{top}}} bị đóng bởi \\end{{{env_name}}}."
                    )

    if env_stack:
        unclosed = ", ".join(f"\\begin{{{e}}}" for e in env_stack)
        problems.append(f"Các môi trường chưa được đóng: {unclosed}.")

    # 3. Kiểm tra và từ chối các từ điều khiển nguy hiểm bị cấm (khớp nguyên từ)
    found_forbidden: set[str] = set()
    for match in FORBIDDEN_WORDS_PATTERN.finditer(clean_code):
        word = match.group(1).lower()
        found_forbidden.add(word)

    for word in sorted(found_forbidden):
        problems.append(
            f"Phát hiện từ điều khiển LaTeX bị cấm vì lý do bảo mật: '\\{word}'."
        )

    return problems


def _ensure_packages_in_document(code: str, requires: Sequence[str]) -> str:
    """
    Chèn thêm các gói phụ thuộc cần thiết vào phần preamble của tài liệu nếu chưa có.
    """
    if not requires:
        return code

    # Thu thập danh sách các gói chưa được nạp
    missing_pkgs: list[str] = []
    for pkg in requires:
        clean_pkg = pkg.strip()
        if not clean_pkg:
            continue
        # Kiểm tra xem package đã được gọi trong \usepackage{...} chưa
        pattern = re.compile(r"\\usepackage(\[[^\]]*\])?\{[^}]*\b" + re.escape(clean_pkg) + r"\b[^}]*\}")
        if not pattern.search(code):
            missing_pkgs.append(clean_pkg)

    if not missing_pkgs:
        return code

    pkg_declarations = "\n".join(f"\\usepackage{{{p}}}" for p in missing_pkgs) + "\n"

    # Chèn ngay trước \begin{document}
    idx = code.find(r"\begin{document}")
    if idx != -1:
        return code[:idx] + pkg_declarations + code[idx:]

    # Nếu không tìm thấy \begin{document}, chèn sau \documentclass nếu có
    doc_match = re.search(r"(\\documentclass(\[[^\]]*\])?\{[^}]+\}\n?)", code)
    if doc_match:
        end_idx = doc_match.end()
        return code[:end_idx] + pkg_declarations + code[end_idx:]

    return pkg_declarations + code


def wrap_snippet(
    code: str,
    requires: Sequence[str] | None = None,
    kind: str = "snippet",
    document_class: str = "article",
) -> str:
    """
    Đóng gói đoạn mã LaTeX thành tài liệu hoàn chỉnh có thể biên dịch.

    Hỗ trợ 3 loại kịch bản:
    - 'snippet': Đoạn mã nằm trong thân tài liệu (giữa \\begin{document} và \\end{document}).
    - 'preamble': Đoạn mã là khai báo gói, định nghĩa lệnh, nằm ở phần mở đầu (trước \\begin{document}).
    - 'document': Đoạn mã vốn là một tài liệu đầy đủ hoặc cấu trúc tài liệu chính.

    Tham số:
        code: Chuỗi mã nguồn LaTeX.
        requires: Danh sách các gói LaTeX phụ thuộc cần khai báo.
        kind: Loại kịch bản ('snippet', 'preamble', hoặc 'document').
        document_class: Lớp tài liệu (mặc định 'article').

    Trả về:
        Mã nguồn LaTeX tài liệu hoàn chỉnh từ \\documentclass tới \\end{document}.
    """
    clean_kind = (kind or "snippet").strip().lower()
    if clean_kind not in VALID_KINDS:
        raise ValueError(
            f"Loại kịch bản 'kind' không hợp lệ: '{kind}'. Các giá trị được phép: {sorted(VALID_KINDS)}."
        )

    req_list: list[str] = [p.strip() for p in (requires or []) if p.strip()]

    # Nếu mã nguồn đã là một tài liệu hoàn chỉnh (chứa cả \documentclass và \begin{document})
    is_full_doc = (r"\documentclass" in code) and (r"\begin{document}" in code)
    if is_full_doc:
        return _ensure_packages_in_document(code, req_list)

    if clean_kind == "preamble":
        # Preamble: mã nguồn nằm ở phần khai báo trước \begin{document}
        parts: list[str] = []
        if r"\documentclass" not in code:
            parts.append(f"\\documentclass{{{document_class}}}")

        for pkg in req_list:
            if f"\\usepackage{{{pkg}}}" not in code:
                parts.append(f"\\usepackage{{{pkg}}}")

        if code.strip():
            parts.append(code.strip())

        parts.append("\\begin{document}")
        parts.append("\\end{document}\n")
        return "\n".join(parts)

    elif clean_kind == "document":
        # Document: cấu trúc tài liệu
        if r"\documentclass" in code:
            # Đã có documentclass nhưng thiếu begin{document}
            doc_with_pkgs = _ensure_packages_in_document(code, req_list)
            if r"\begin{document}" not in doc_with_pkgs:
                doc_with_pkgs = f"{doc_with_pkgs}\n\\begin{{document}}\n\\end{{document}}\n"
            return doc_with_pkgs

        pkg_lines = "\n".join(f"\\usepackage{{{pkg}}}" for pkg in req_list)
        if pkg_lines:
            pkg_lines += "\n"

        return (
            f"\\documentclass{{{document_class}}}\n"
            f"{pkg_lines}"
            f"\\begin{{document}}\n"
            f"{code}\n"
            f"\\end{{document}}\n"
        )

    else:
        # Default: 'snippet'
        pkg_lines = "\n".join(f"\\usepackage{{{pkg}}}" for pkg in req_list)
        if pkg_lines:
            pkg_lines += "\n"

        return (
            f"\\documentclass{{{document_class}}}\n"
            f"{pkg_lines}"
            f"\\begin{{document}}\n"
            f"{code}\n"
            f"\\end{{document}}\n"
        )


__all__ = [
    "FORBIDDEN_WORDS",
    "LATEX_ESCAPE_MAP",
    "POST_ACTIONS",
    "VALID_KINDS",
    "escape_text",
    "resolve_template",
    "static_check",
    "strip_comments",
    "tidy_math",
    "wrap_snippet",
]
