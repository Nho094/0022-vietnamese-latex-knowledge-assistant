"""
Module tiền xử lý văn bản tiếng Việt và phân đoạn từ được bảo vệ (Protected Word Segmentation).

Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Kiến trúc xử lý:
  text -> preprocess (NFC, chuẩn hóa ký hiệu toán học, tách từ pyvi cho PhoBERT)
       -> PhoBERT classifier: p = softmax over intents
       - if max p >= tau: intent accepted
       - else retrieval top-3 over aliases + scenario strings -> ask the user to choose
- Cơ sở lý thuyết và công thức tham chiếu:
  1. Chuẩn hóa dạng biểu diễn Unicode NFC (ISO/IEC 10646) theo chuẩn tiếng Việt dựng sẵn.
  2. PhoBERT (Nguyen & Tuan Nguyen, 2020 - "PhoBERT: Pre-trained language models for Vietnamese"):
     PhoBERT sử dụng kiến trúc RoBERTa tiền huấn luyện ở cấp độ từ (word-level). Do tiếng Việt
     là ngôn ngữ đơn lập, ranh giới từ ghép cần được nối bằng dấu gạch dưới '_' (ví dụ: 'đạo_hàm').
     Đồng thời bảo vệ các ký hiệu toán học và câu lệnh LaTeX (regex \\\\[A-Za-z]+) khỏi bị bẻ gãy.
  3. Sentence-BERT (Reimers & Gurevych, 2019 - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"):
     Chuẩn bị biểu diễn văn bản cho truy hồi ngữ nghĩa dự phòng (Retrieval Fallback).
  4. Vân tay tiền xử lý (preprocess_fingerprint) tạo mã băm tất định (deterministic SHA-256) đảm bảo
     khả năng tái lập và phát hiện lệch phiên bản phân đoạn từ (segmenter drift).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import shutil
import unicodedata
from typing import Final

from core.config import (
    DEFAULT_SEGMENTER,
    EMBED_MODEL_NAME,
    EMBED_MODEL_REQUIRES_WORD_SEGMENTATION,
    LATEX_COMMAND_REGEX,
    LOWERCASE_PHOBERT,
    NORMALIZE_MATH_LOOKALIKES,
    SPOKEN_MATH_OPERATOR_MAP,
    SUPPORTED_SEGMENTERS,
    VIETNAMESE_NUMBER_TENS,
    VIETNAMESE_NUMBER_UNITS,
)

# Bảng tra cứu chuẩn hóa ký hiệu toán học tương đồng (math look-alikes) sang ký tự chuẩn ASCII.
# Chuyển đổi các ký tự Unicode mô phỏng toán học, phân số, số mũ, dấu gạch ngang và ký tự full-width.
MATH_LOOKALIKES_MAP: Final[dict[str, str]] = {
    # Số mũ (Superscript 0-9 và ký hiệu mũ)
    "²": "^2",
    "³": "^3",
    "⁰": "^0",
    "¹": "^1",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
    "⁺": "^+",
    "⁻": "^-",
    "⁼": "^=",
    "⁽": "^(",
    "⁾": "^)",
    "ⁿ": "^n",
    # Dấu trừ và gạch ngang Unicode (Minus / Dash look-alikes sang '-')
    "−": "-",  # U+2212 Minus Sign
    "–": "-",  # U+2013 En Dash
    "—": "-",  # U+2014 Em Dash
    "‐": "-",  # U+2010 Hyphen
    "‑": "-",  # U+2011 Non-breaking Hyphen
    "˗": "-",  # U+02D7 Modifier Letter Minus Sign
    "－": "-",  # U+FF0D Fullwidth Hyphen-Minus
    # Ký hiệu phép nhân sang '*'
    "×": "*",  # U+00D7 Multiplication Sign
    "⋅": "*",  # U+22C5 Dot Operator
    "✕": "*",  # U+2715 Multiplication X
    "✖": "*",  # U+2716 Heavy Multiplication X
    "∗": "*",  # U+2217 Asterisk Operator
    "＊": "*",  # U+FF0A Fullwidth Asterisk
    # Ký hiệu phép chia sang '/'
    "÷": "/",  # U+00F7 Division Sign
    "∕": "/",  # U+2215 Division Slash
    "⁄": "/",  # U+2044 Fraction Slash
    "／": "/",  # U+FF0F Fullwidth Solidus
    # Các toán tử và dấu đóng/mở ngoặc toàn độ rộng (Full-width sang Half-width ASCII)
    "＋": "+",
    "＝": "=",
    "＜": "<",
    "＞": ">",
    "（": "(",
    "）": ")",
    "［": "[",
    "］": "]",
    "｛": "{",
    "｝": "}",
    "＾": "^",
    "＿": "_",
    "＼": "\\",
    "％": "%",
    "＆": "&",
    "＄": "$",
}

# Bổ sung 10 chữ số toàn độ rộng (Full-width digits U+FF10 - U+FF19) sang chữ số ASCII '0'-'9'
for _digit_idx in range(10):
    MATH_LOOKALIKES_MAP[chr(0xFF10 + _digit_idx)] = str(_digit_idx)

# Danh sách biểu thức chính quy nhận dạng các vùng biểu thức toán học và lệnh LaTeX cần bảo vệ
# Quy tắc: Bảo vệ lệnh LaTeX (\\\\[A-Za-z]+), công thức trong dấu $, môi trường LaTeX và biểu thức đại số.
PROTECTED_MATH_PATTERNS: Final[tuple[str, ...]] = (
    # Khối toán học hiển thị trong cặp $$...$$
    r"\$\$[\s\S]*?\$\$",
    # Biểu thức toán học nội dòng trong cặp $...$
    r"\$[^\$\n]+?\$",
    # Cặp lệnh toán học chuẩn LaTeX \\[...\\] và \\(...\\)
    r"\\\[[\s\S]*?\\\]",
    r"\\\(.*?\\\)",
    # Khối môi trường LaTeX \\begin{env}...\\end{env}
    r"\\begin\{[a-zA-Z*]+\}[\s\S]*?\\end\{[a-zA-Z*]+\}",
    # Lệnh LaTeX với tham số tùy chọn và nhóm ngoặc nhọn: ví dụ \\frac{a}{b}, \\int_{0}^{1}
    r"\\[A-Za-z]+(?:(?:[_^](?:\{[^{}\n]*\}|[0-9a-zA-Z]))|\{[^{}\n]*\})*",
    # Lệnh LaTeX đơn lẻ theo đặc tả yêu cầu: regex \\\\[A-Za-z]+
    LATEX_COMMAND_REGEX,
    # Biểu thức đại số chứa toán tử (^, _, =, +, -, *, /, <, >): ví dụ x^2 + 2x + 1 = 0
    r"(?:(?:\b[a-zA-Z0-9_]+\b|\([^\)]+\))\s*[\^+\-*/=<>]\s*)+(?:[a-zA-Z0-9_]+|\([^\)]+\))",
    # Tên hàm toán học kèm đối số: ví dụ f(x), g(x, y)
    r"\b[a-zA-Z]\([a-zA-Z0-9,\s+\-*/^]+\)",
    # Biến số có lũy thừa hoặc chỉ số dưới: ví dụ x^2, a_1
    r"\b[a-zA-Z0-9]+[\^_][a-zA-Z0-9]+",
)

# Biên dịch trước biểu thức tổng hợp các vùng cần bảo vệ để tối ưu tốc độ thực thi
_COMBINED_PROTECTION_REGEX = re.compile("|".join(f"(?:{p})" for p in PROTECTED_MATH_PATTERNS))

# Biểu thức chính quy thu gọn khoảng trắng liên tiếp (whitespace collapsing)
_WHITESPACE_REGEX = re.compile(r"\s+")


def get_pyvi_version() -> str:
    """
    Lấy thông tin phiên bản cài đặt của thư viện pyvi một cách an toàn.

    Trả về:
        Chuỗi phiên bản (ví dụ '0.1.1') hoặc 'unknown' nếu không thể xác định.
    """
    try:
        return importlib.metadata.version("pyvi")
    except (importlib.metadata.PackageNotFoundError, OSError):
        try:
            import pyvi  # type: ignore[import-untyped]

            return str(getattr(pyvi, "__version__", "unknown"))
        except (AttributeError, ImportError):
            return "unknown"


def normalize(
    text: str,
    lowercase: bool = LOWERCASE_PHOBERT,
    convert_lookalikes: bool = NORMALIZE_MATH_LOOKALIKES,
) -> str:
    """
    Chuẩn hóa chuỗi văn bản đầu vào của người dùng theo chuẩn Unicode tiếng Việt và ký hiệu toán học.

    Quy trình chuẩn hóa:
    1. Chuẩn hóa định dạng Unicode dựng sẵn NFC (tránh lỗi lệch tổ hợp ký tự tiếng Việt có dấu).
    2. Chuyển đổi các ký hiệu toán học tương đồng (superscript 2/3 sang ^2/^3, unicode minus/en dash sang '-',
       phép nhân/chia sang '*'/ '/', chữ số toàn độ rộng sang ASCII).
    3. Thu gọn khoảng trắng thừa (collapse whitespace) và cắt bỏ khoảng trắng hai đầu chuỗi.
    4. Giữ nguyên dấu tiếng Việt, kiểu chữ hoa/thường (khi lowercase=False), chữ số và các ký tự
       toán/LaTeX hợp lệ (+ - = ^ * / ( ) x \\ { } _ $ % &).

    Tham số:
        text: Chuỗi văn bản tiếng Việt thô từ người dùng.
        lowercase: Cờ chuyển đổi chữ thường (mặc định False theo LOWERCASE_PHOBERT để bảo toàn kiểu chữ PhoBERT).
        convert_lookalikes: Cờ chuyển đổi ký hiệu toán tương đồng (mặc định True).

    Trả về:
        Chuỗi văn bản sau khi chuẩn hóa hoàn tất.
    """
    if not text:
        return ""

    # Bước 1: Chuẩn hóa dạng biểu diễn Unicode NFC
    normalized_text = unicodedata.normalize("NFC", text)

    # Bước 2: Chuyển đổi ký hiệu toán học tương đồng sang chuẩn ASCII
    if convert_lookalikes:
        for original_sym, ascii_sym in MATH_LOOKALIKES_MAP.items():
            normalized_text = normalized_text.replace(original_sym, ascii_sym)

    # Bước 3: Thu gọn khoảng trắng thừa (collapse whitespace)
    normalized_text = _WHITESPACE_REGEX.sub(" ", normalized_text).strip()

    # Bước 4: Chuyển đổi chữ thường nếu được yêu cầu
    if lowercase:
        normalized_text = normalized_text.lower()

    return normalized_text


def segment(text: str, backend: str = DEFAULT_SEGMENTER) -> str:
    """
    Phân đoạn từ vựng tiếng Việt với dấu gạch dưới '_' kết nối từ ghép, đồng thời bảo vệ
    nguyên vẹn các biểu thức toán học và câu lệnh LaTeX (regex \\\\[A-Za-z]+).

    Cơ chế bảo vệ (Math & LaTeX Shielding):
    1. Quét tìm tất cả các vùng biểu thức toán học, môi trường LaTeX và câu lệnh LaTeX (\\\\[A-Za-z]+).
    2. Tạm thời thay thế các vùng này bằng nhãn giữ chỗ tất định (placeholder) dạng '__SHIELD_MATH_{idx}__'.
    3. Thực hiện tách từ bằng backend được chỉ định ('pyvi' sử dụng pyvi.ViTokenizer).
    4. Phục hồi chính xác các biểu thức toán học ban đầu từ danh sách vùng đã lưu trữ.

    Tham số:
        text: Chuỗi văn bản tiếng Việt đã qua chuẩn hóa.
        backend: Trình tách từ ('pyvi' hoặc 'rdr'). Mặc định là 'pyvi'.

    Trả về:
        Chuỗi văn bản đã phân đoạn từ tiếng Việt với các cấu trúc toán học được bảo toàn.

    Ngoại lệ:
        ValueError: Nếu backend không nằm trong danh sách hỗ trợ ('pyvi', 'rdr').
        RuntimeError: Nếu backend 'rdr' được yêu cầu nhưng thiếu Java hoặc py_vncorenlp.
    """
    if not text:
        return ""

    if backend not in SUPPORTED_SEGMENTERS:
        raise ValueError(
            f"Backend phân đoạn từ không được hỗ trợ: '{backend}'. "
            f"Các backend khả dụng: {list(SUPPORTED_SEGMENTERS)}"
        )

    # Xử lý backend RDRsegmenter (py_vncorenlp)
    if backend == "rdr":
        return _segment_rdr(text)

    # Backend mặc định: pyvi
    return _segment_pyvi(text)


def _segment_pyvi(text: str) -> str:
    """
    Tách từ vựng bằng pyvi.ViTokenizer kết hợp cơ chế che chắn biểu thức toán học.
    """
    try:
        from pyvi import ViTokenizer  # type: ignore[import-untyped]
    except ImportError as err:
        # shortcut: khi pyvi không có sẵn trong môi trường, trả về văn bản gốc đã chuẩn hóa.
        # Nâng cấp khi cài đặt gói pyvi trong môi trường ảo venv.
        raise RuntimeError("Thư viện pyvi chưa được cài đặt trong môi trường.") from err

    # Bước 1: Trích xuất và che chắn các biểu thức toán học và lệnh LaTeX
    protected_spans: list[str] = []

    def _shield_match(match: re.Match[str]) -> str:
        idx = len(protected_spans)
        protected_spans.append(match.group(0))
        return f"__SHIELD_MATH_{idx}__"

    shielded_text = _COMBINED_PROTECTION_REGEX.sub(_shield_match, text)

    # Bước 2: Phân đoạn từ tiếng Việt với ViTokenizer
    tokenized_text = ViTokenizer.tokenize(shielded_text)

    # Bước 3: Phục hồi lại các biểu thức toán học nguyên bản
    for idx, original_span in enumerate(protected_spans):
        placeholder = f"__SHIELD_MATH_{idx}__"
        tokenized_text = tokenized_text.replace(placeholder, original_span)

    return tokenized_text


def _segment_rdr(text: str) -> str:
    """
    Tách từ vựng bằng backend RDRsegmenter qua thư viện py_vncorenlp.
    Kiểm tra điều kiện tiên quyết: Java runtime và gói py_vncorenlp.
    """
    # Kiểm tra sự tồn tại của môi trường thực thi Java
    if shutil.which("java") is None:
        raise RuntimeError(
            "Backend 'rdr' yêu cầu Java Runtime Environment (JRE/JDK 1.8+) được cài đặt và "
            "khả dụng trong biến môi trường PATH, nhưng không tìm thấy lệnh 'java'."
        )

    # Kiểm tra gói py_vncorenlp
    try:
        import py_vncorenlp  # type: ignore[import-untyped] # noqa: F401
    except ImportError as err:
        raise RuntimeError(
            "Backend 'rdr' yêu cầu thư viện 'py_vncorenlp' nhưng chưa được cài đặt. "
            "Vui lòng cài đặt py_vncorenlp và mô hình word segmenter tương ứng."
        ) from err

    # shortcut: RDRsegmenter yêu cầu tải mô hình VnCoreNLP cục bộ; hiện tại chuyển giao ngoại lệ nếu chưa nạp.
    # Nâng cấp khi bổ sung đường dẫn mô hình VnCoreNLP vào thư mục models/vncornlp.
    raise RuntimeError(
        "Backend 'rdr' được kích hoạt nhưng mô hình VnCoreNLP chưa được nạp vào đường dẫn cục bộ."
    )


def prepare_for_phobert(
    text: str,
    lowercase: bool = LOWERCASE_PHOBERT,
    backend: str = DEFAULT_SEGMENTER,
) -> str:
    """
    Quy trình tiền xử lý hoàn chỉnh chuẩn bị chuỗi đầu vào cho mô hình PhoBERT.

    Theo bài báo PhoBERT (Nguyen & Tuan Nguyen, 2020), chuỗi đầu vào cần được:
    1. Chuẩn hóa Unicode NFC và chuẩn hóa ký hiệu toán học tương đồng.
    2. Phân đoạn từ vựng ghép bằng dấu gạch dưới '_' với các lệnh LaTeX và toán được bảo vệ.
    3. Giữ nguyên chữ hoa/thường (khi lowercase=False) nhằm bảo toàn thực thể và nhãn toán.

    Tham số:
        text: Chuỗi văn bản tiếng Việt thô từ người dùng.
        lowercase: Cờ chuyển chữ thường (mặc định False theo LOWERCASE_PHOBERT).
        backend: Trình tách từ (mặc định 'pyvi').

    Trả về:
        Chuỗi văn bản đã qua chuẩn hóa và phân đoạn từ sẵn sàng đưa vào PhoBERT tokenizer.
    """
    normalized = normalize(text, lowercase=lowercase)
    return segment(normalized, backend=backend)


def is_word_segmentation_required(model_name: str | None = None) -> bool:
    """
    Kiểm tra xem mô hình embedding có yêu cầu văn bản đầu vào được phân đoạn từ (word-segmented)
    với dấu gạch dưới '_' hay không dựa trên model card và kiến trúc mô hình nền (backbone).

    Kết quả audit model card:
    - Mô hình cấu hình 'bkai-foundation-models/vietnamese-bi-encoder' (EMBED_MODEL_NAME) sử dụng
      backbone PhoBERT (vinai/phobert-base-v2) với tokenizer BPE mức từ (word-level BPE).
    - Model card trên Hugging Face quy định rõ: 'Because the model is built on top of PhoBERT,
      input text must be word-segmented beforehand when using sentence-transformers directly'.
    - Do đó, đối với các mô hình họ PhoBERT/BKAI bi-encoder, phân đoạn từ là bắt buộc (True).
    - Đối với các mô hình đa ngữ hoặc subword/syllable (như XLM-RoBERTa, paraphrase-multilingual),
      phân đoạn từ ghép bằng gạch dưới là không mong muốn (False) vì tokenizer đa ngữ hoạt động
      trực tiếp trên âm tiết tự nhiên.

    Tham số:
        model_name: Tên định danh mô hình trên HuggingFace Hub hoặc đường dẫn cục bộ.
                    Nếu None, sử dụng cấu hình mặc định EMBED_MODEL_NAME.

    Trả về:
        True nếu mô hình yêu cầu phân đoạn từ ghép, False nếu chỉ cần chuẩn hóa văn bản tự nhiên.
    """
    if model_name is None or model_name == EMBED_MODEL_NAME:
        return EMBED_MODEL_REQUIRES_WORD_SEGMENTATION

    name_lower = str(model_name).strip().lower()
    return bool(any(keyword in name_lower for keyword in ("phobert", "vietnamese-bi-encoder", "bkai")))


def validate_for_embedding(
    text: str,
    model_name: str | None = None,
) -> bool:
    """
    Kiểm tra xác thực nghiêm ngặt chuỗi văn bản đã được tiền xử lý đúng chuẩn hay chưa
    trước khi đưa vào bộ mã hóa embedding nhằm ngăn ngừa chuỗi chưa tiền xử lý bị mã hóa
    (prevents un-preprocessed strings from being encoded).

    Ngoại lệ phát sinh nếu chuỗi chưa tiền xử lý:
    - TypeError nếu không phải chuỗi ký tự (str).
    - ValueError nếu chuỗi rỗng hoặc chỉ chứa khoảng trắng.
    - ValueError nếu chuỗi chưa chuẩn hóa Unicode NFC.
    - ValueError nếu chuỗi còn chứa ký hiệu toán học tương đồng (math look-alikes).
    - ValueError nếu chuỗi chứa khoảng trắng thừa chưa được thu gọn.

    Tham số:
        text: Chuỗi văn bản cần kiểm tra.
        model_name: Tên định danh mô hình embedding (mặc định lấy từ EMBED_MODEL_NAME).

    Trả về:
        True nếu chuỗi hoàn toàn hợp lệ và đã qua tiền xử lý chuẩn.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"Văn bản kiểm tra tiền xử lý embedding phải là chuỗi ký tự (str), nhận: {type(text).__name__}"
        )

    if not text.strip():
        raise ValueError(
            "Chuỗi văn bản không được để trống hoặc chỉ chứa khoảng trắng khi đưa vào embedding."
        )

    # 1. Kiểm tra chuẩn hóa Unicode NFC
    if unicodedata.normalize("NFC", text) != text:
        raise ValueError("Văn bản chưa được chuẩn hóa dạng Unicode dựng sẵn NFC.")

    # 2. Kiểm tra ký hiệu toán học tương đồng chưa chuyển đổi
    for original_sym in MATH_LOOKALIKES_MAP:
        if original_sym in text:
            raise ValueError(
                f"Văn bản còn chứa ký hiệu toán học tương đồng chưa chuẩn hóa: '{original_sym}'"
            )

    # 3. Kiểm tra khoảng trắng thừa chưa thu gọn
    if "  " in text or text != text.strip():
        raise ValueError("Văn bản còn chứa khoảng trắng thừa hoặc chưa được cắt tỉa hai đầu.")

    return True


def prepare_for_embedding(
    text: str,
    lowercase: bool = False,
    segment_words: bool | None = None,
    backend: str = DEFAULT_SEGMENTER,
    model_name: str | None = None,
) -> str:
    """
    Quy trình tiền xử lý chuẩn bị chuỗi đầu vào cho mô hình Sentence-BERT / Text Embedding.

    Tuân thủ kết quả audit model card và kiến trúc dự án:
    1. Ngăn chặn triệt để chuỗi chưa tiền xử lý, chuỗi rỗng hoặc kiểu dữ liệu không hợp lệ
       bị đưa vào bộ mã hóa embedding (prevent un-preprocessed strings from being encoded).
    2. Chuẩn hóa định dạng Unicode dựng sẵn NFC và quy đổi các ký hiệu toán học tương đồng (math look-alikes).
    3. Xác định cơ chế phân đoạn từ (word segmentation) phù hợp với đặc tả mô hình:
       - Nếu segment_words được chỉ định tường minh (True/False): Tuân theo cờ chỉ định của người gọi.
       - Nếu segment_words là None: Tự động quyết định dựa trên model card thông qua
         is_word_segmentation_required(model_name). Đối với EMBED_MODEL_NAME
         ('bkai-foundation-models/vietnamese-bi-encoder' dựa trên PhoBERT), tự động phân đoạn từ
         ghép bằng pyvi kết hợp che chắn công thức toán học/lệnh LaTeX.
         Đối với mô hình multilingual/syllable (như xlm-roberta), chỉ chuẩn hóa không nối từ.

    Tham số:
        text: Chuỗi văn bản tiếng Việt thô.
        lowercase: Cờ chuyển chữ thường (mặc định False).
        segment_words: Cờ phân đoạn từ vựng ghép. Nếu None, tự động suy luận theo đặc tả model card.
        backend: Trình tách từ sử dụng nếu cần phân đoạn từ (mặc định 'pyvi').
        model_name: Tên hoặc định danh mô hình embedding (mặc định lấy từ EMBED_MODEL_NAME).

    Trả về:
        Chuỗi văn bản đã tiền xử lý hoàn chỉnh sẵn sàng cho bước tạo vector nhúng (embedding).

    Ngoại lệ:
        TypeError: Nếu đầu vào không phải là chuỗi ký tự (str).
        ValueError: Nếu chuỗi đầu vào rỗng, chỉ chứa khoảng trắng, hoặc backend không hợp lệ.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"Văn bản đầu vào cho embedding phải là chuỗi ký tự (str), nhận: {type(text).__name__}"
        )

    if not text.strip():
        raise ValueError(
            "Chuỗi văn bản đầu vào không được để trống hoặc chỉ chứa khoảng trắng khi tạo vector nhúng."
        )

    # Bước 1: Chuẩn hóa NFC và ký hiệu toán học tương đồng
    normalized = normalize(text, lowercase=lowercase)

    # Đảm bảo chuỗi sau chuẩn hóa không bị rỗng
    if not normalized.strip():
        raise ValueError(
            "Văn bản sau khi chuẩn hóa rỗng, không thể đưa vào bộ mã hóa embedding."
        )

    # Bước 2: Xác định cờ phân đoạn từ theo đặc tả mô hình
    should_segment = (
        segment_words
        if segment_words is not None
        else is_word_segmentation_required(model_name)
    )

    if should_segment:
        processed = segment(normalized, backend=backend)
    else:
        processed = normalized
    # Ensure final string passes all embedding validation checks
    validate_for_embedding(processed, model_name=model_name)
    return processed


def preprocess_fingerprint(
    backend: str = DEFAULT_SEGMENTER,
    lowercase: bool = LOWERCASE_PHOBERT,
) -> str:
    """
    Tạo mã băm tất định (deterministic SHA-256 fingerprint) đại diện cho toàn bộ
    cấu hình quy tắc chuẩn hóa và phiên bản bộ phân đoạn từ vựng đang được áp dụng.

    Mã băm này được sử dụng để:
    1. Kiểm tra tính nhất quán giữa tập dữ liệu tiền xử lý offline và môi trường suy luận online.
    2. Phát hiện sớm sự trôi dạt phiên bản phân đoạn từ (segmenter drift).

    Tham số:
        backend: Tên trình tách từ (mặc định 'pyvi').
        lowercase: Cờ chữ thường.

    Trả về:
        Chuỗi băm thập lục phân SHA-256 độ dài 64 ký tự.
    """
    fingerprint_payload = {
        "segmenter": backend,
        "pyvi_version": get_pyvi_version(),
        "nfc": True,
        "collapse_whitespace": True,
        "lowercase": lowercase,
        "math_lookalikes": NORMALIZE_MATH_LOOKALIKES,
        "latex_command_regex": LATEX_COMMAND_REGEX,
        "protected_patterns_count": len(PROTECTED_MATH_PATTERNS),
    }

    serialized = json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_spoken_math(text: str) -> str:
    """Chuẩn hóa cách đọc toán tiếng Việt thành ký hiệu trước khi khớp slot.

    Bảng từ vựng nằm trong config để luật dùng chung giữa các intent. Caller chỉ
    dùng kết quả cho slot number/math khi câu có dấu hiệu toán, tránh sửa text slot.
    """
    value = normalize(text)
    protected_commands: list[str] = []

    def protect_latex(match: re.Match[str]) -> str:
        protected_commands.append(match.group(0))
        return f"LATEXPROTECTEDTOKEN{len(protected_commands) - 1}"

    value = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})*", protect_latex, value)
    value = re.sub(r"\bbằng\s+không\b", "= 0", value, flags=re.IGNORECASE)
    tens_options = sorted(VIETNAMESE_NUMBER_TENS, key=len, reverse=True)
    units_options = sorted(VIETNAMESE_NUMBER_UNITS, key=len, reverse=True)
    alternatives = "|".join(re.escape(word) for word in tens_options)
    units = "|".join(re.escape(word) for word in units_options)
    number_pattern = re.compile(
        rf"(?<!\w)(?:(?P<tens>{alternatives})(?:\s+(?P<unit>{units}))?|"
        rf"(?P<ten>mười)(?:\s+(?P<teen>{units}))?|(?P<single>{units}))(?!\w)",
        re.IGNORECASE,
    )

    def number_value(match: re.Match[str]) -> str:
        if re.search(r"\bbậc\s+$", value[: match.start()], flags=re.IGNORECASE):
            return match.group(0)
        if match.group("tens"):
            base = VIETNAMESE_NUMBER_TENS[match.group("tens").casefold()]
            unit = VIETNAMESE_NUMBER_UNITS.get((match.group("unit") or "").casefold(), 0)
            return str(base + unit)
        if match.group("ten"):
            teen = VIETNAMESE_NUMBER_UNITS.get((match.group("teen") or "").casefold(), 0)
            return str(VIETNAMESE_NUMBER_TENS["mười"] + teen)
        return str(VIETNAMESE_NUMBER_UNITS[match.group("single").casefold()])

    value = number_pattern.sub(number_value, value)
    for phrase, symbol in sorted(
        SPOKEN_MATH_OPERATOR_MAP.items(), key=lambda item: len(item[0]), reverse=True
    ):
        value = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", symbol, value, flags=re.IGNORECASE)
    value = re.sub(r"\bmũ\s*(\d+)\b", r"^\1", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<!\w)([+-])\s+(?=[A-Za-z0-9])", r"\1", value)
    value = re.sub(r"(?<!\w)([+-]?\d+(?:\.\d+)?)\s+([A-Za-z])(?=\s*(?:\^|[+=-]|$))", r"\1\2", value)
    value = re.sub(r"\s+\^", "^", value)
    value = re.sub(r"(?<=\d),(?=\d)", ".", value)
    value = _WHITESPACE_REGEX.sub(" ", value).strip()
    for index, command in enumerate(protected_commands):
        value = value.replace(f"LATEXPROTECTEDTOKEN{index}", command)
    return value


def canonicalize_retrieval_numbers(text: str) -> str:
    """Đổi chữ số đứng riêng sang chữ Việt để khớp alias cùng nghĩa về số."""
    value = normalize(text, lowercase=True)
    unit_names: dict[int, str] = {}
    for word, number in VIETNAMESE_NUMBER_UNITS.items():
        unit_names.setdefault(number, word)
    tens_names = {number: word for word, number in VIETNAMESE_NUMBER_TENS.items()}

    def spell_number(match: re.Match[str]) -> str:
        number = int(match.group(0))
        if number < 10:
            return unit_names[number]
        if number == 10:
            return tens_names[10]
        tens, units = divmod(number, 10)
        if number < 20:
            return f"mười {unit_names[units]}"
        prefix = tens_names[tens * 10]
        return f"{prefix} {unit_names[units]}" if units else prefix

    return re.sub(r"(?<![\w^/.-])\d{1,2}(?![\w/.])", spell_number, value)
