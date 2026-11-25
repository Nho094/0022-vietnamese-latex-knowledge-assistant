"""
Kiểm thử đơn vị và bộ xác minh tính đúng đắn cho module tiền xử lý văn bản core/preprocess.py (KA-02).

Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Cơ sở lý thuyết và công thức tham chiếu:
  1. Chuẩn hóa dạng biểu diễn Unicode NFC (ISO/IEC 10646) theo chuẩn tiếng Việt dựng sẵn.
  2. PhoBERT (Nguyen & Tuan Nguyen, 2020 - "PhoBERT: Pre-trained language models for Vietnamese"):
     Phân đoạn từ vựng ghép bằng dấu gạch dưới '_' thông qua pyvi, đồng thời che chắn bảo vệ
     các ký hiệu toán học và câu lệnh LaTeX (regex \\\\[A-Za-z]+) không bị bẻ gãy.
  3. Sentence-BERT (Reimers & Gurevych, 2019): Chuẩn hóa câu tự nhiên cho tầng truy hồi ngữ nghĩa.
  4. Vân tay tiền xử lý (preprocess_fingerprint) tạo mã băm SHA-256 tất định đảm bảo tính tái lập.

Nhiệm vụ kiểm thử (Pre-processing and Span Protection Unit Test Suite):
- Xác minh normalize() có tính lũy đẳng (idempotent), bảo toàn đầy đủ các thanh điệu/dấu tiếng Việt,
  và ánh xạ chính xác các ký hiệu toán tương đồng (math look-alikes).
- Xác minh văn bản tiếng Việt đa từ tạo ra các token ghép nối bằng dấu gạch dưới '_', trong khi
  các vùng được bảo vệ như '2x² + 3x - 5 = 0' (chuyển đổi thành '2x^2 + 3x - 5 = 0') và lệnh LaTeX
  như \\includegraphics được giữ nguyên vẹn không bị bộ phân đoạn từ biến đổi.
- Xác minh việc thay đổi cấu hình bộ phân đoạn từ (segmenter configurations) làm thay đổi
  mã băm preprocess_fingerprint().
- Xác minh việc kích hoạt backend 'rdr' chưa cài đặt ném ra ngoại lệ rõ ràng mà không cố gắng
  thay đổi môi trường (không chạy cài đặt hệ thống hay pip ngầm).
"""

from __future__ import annotations

import unicodedata
from unittest.mock import MagicMock, patch

import pytest

from core.preprocess import (
    MATH_LOOKALIKES_MAP,
    get_pyvi_version,
    is_word_segmentation_required,
    normalize,
    normalize_spoken_math,
    prepare_for_embedding,
    prepare_for_phobert,
    preprocess_fingerprint,
    segment,
    validate_for_embedding,
)


def test_normalize_empty_and_whitespace() -> None:
    """Kiểm tra normalize() xử lý chính xác chuỗi rỗng hoặc chỉ chứa khoảng trắng/ký tự điều khiển."""
    assert normalize("") == ""
    assert normalize("   ") == ""
    assert normalize("\t\n  \r\n") == ""


def test_normalize_spoken_division_operator() -> None:
    """Phép chia đọc thành lời phải được chuẩn hóa như ký hiệu chia."""
    assert normalize_spoken_math("sin x chia x") == "sin x / x"


def test_normalize_idempotent() -> None:
    """
    Xác minh tính lũy đẳng (idempotency) của hàm normalize():
    normalize(normalize(text)) == normalize(text)
    Đảm bảo việc chuẩn hóa nhiều lần liên tiếp không làm biến đổi thêm nội dung văn bản.
    """
    test_cases = [
        "   Tính đạo hàm của hàm số f(x) = 2x² + 3x − 5 = 0   ",
        r"Giải phương trình: \frac{a + b}{c - d} × 4 ÷ 2",
        "Tiếng Việt có dấu: sắc, huyền, hỏi, ngã, nặng và số mũ x³",
        "（ １ ＋ ２ ＝ ３ ） và [ a ＿ b ]",
        r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}",
        r"Chèn đồ thị bằng lệnh \includegraphics và \alpha",
    ]

    for raw_text in test_cases:
        first_pass = normalize(raw_text)
        second_pass = normalize(first_pass)
        third_pass = normalize(second_pass)

        # Lũy đẳng cấp 2 và cấp 3
        assert first_pass == second_pass, f"normalize() không đạt tính lũy đẳng với chuỗi: '{raw_text}'"
        assert second_pass == third_pass


def test_normalize_preserves_vietnamese_diacritics() -> None:
    """
    Xác minh normalize() bảo toàn trọn vẹn toàn bộ các nguyên âm tiếng Việt và 5 thanh điệu
    (sắc, huyền, hỏi, ngã, nặng) cùng các ký tự mũ (â, ê, ô), móc (ơ, ư), trăng (ă) và chữ đ/Đ.
    Đảm bảo không bị mất dấu hoặc suy thoái thành chữ cái không dấu.
    """
    # Mẫu kiểm tra toàn diện các nguyên âm có dấu trong tiếng Việt
    vietnamese_pangram = (
        "Việt Nam đất nước ta ơi! Mênh mông biển lúa đâu trời đẹp hơn. "
        "Học sinh giỏi cần cù, chăm chỉ, ngoan ngoãn, lễ phép, tự giác. "
        "Đạo hàm, tích phân, lượng giác, phương trình vi phân và giải tích."
    )

    normalized = normalize(vietnamese_pangram, lowercase=False)

    # Đảm bảo giữ nguyên vẹn nội dung và dấu tiếng Việt
    assert normalized == vietnamese_pangram

    # Kiểm tra danh sách các ký tự tiếng Việt đặc thù có dấu
    special_chars = [
        "á", "à", "ả", "ã", "ạ",
        "ă", "ắ", "ằ", "ẳ", "ẵ", "ặ",
        "â", "ấ", "ầ", "ẩ", "ẫ", "ậ",
        "é", "è", "ẻ", "ẽ", "ẹ",
        "ê", "ế", "ề", "ể", "ễ", "ệ",
        "í", "ì", "ỉ", "ĩ", "ị",
        "ó", "ò", "ỏ", "õ", "ọ",
        "ô", "ố", "ồ", "ổ", "ỗ", "ộ",
        "ơ", "ớ", "ờ", "ở", "ỡ", "ợ",
        "ú", "ù", "ủ", "ũ", "ụ",
        "ư", "ứ", "ừ", "ử", "ữ", "ự",
        "ý", "ỳ", "ỷ", "ỹ", "ỵ",
        "đ", "Đ",
    ]
    for ch in special_chars:
        text_with_char = f"Ký tự {ch} trong tiếng Việt"
        norm_result = normalize(text_with_char, lowercase=False)
        assert ch in norm_result, f"Ký tự có dấu '{ch}' bị mất hoặc biến dạng sau normalize()!"


def test_normalize_nfc_enforcement() -> None:
    """
    Kiểm tra normalize() bắt buộc chuẩn hóa dạng biểu diễn Unicode NFC (chuẩn dựng sẵn).
    Chuỗi tiếng Việt dạng phân rã (NFD - decomposed) phải được chuyển thành NFC chuẩn.
    """
    raw_nfd = unicodedata.normalize("NFD", "Tiếng Việt có dấu và đạo hàm cấp hai")
    # Xác nhận chuỗi ban đầu thực sự là NFD (khác NFC)
    assert raw_nfd != unicodedata.normalize("NFC", raw_nfd)

    normalized = normalize(raw_nfd)
    assert normalized == "Tiếng Việt có dấu và đạo hàm cấp hai"
    # Kết quả trả về phải đạt chuẩn NFC hoàn toàn
    assert unicodedata.is_normalized("NFC", normalized)


def test_normalize_collapse_whitespace() -> None:
    """Kiểm tra normalize() thu gọn khoảng trắng liên tiếp và cắt lề hai đầu chuỗi."""
    raw = "   Tính    đạo   hàm   \t  bậc   hai \n  của   f(x)   "
    expected = "Tính đạo hàm bậc hai của f(x)"
    assert normalize(raw) == expected


def test_normalize_all_math_lookalikes_mapped() -> None:
    """
    Xác minh tất cả các ký tự trong MATH_LOOKALIKES_MAP đều được chuẩn hóa chính xác
    sang ký tự ASCII tương ứng, không còn ký tự gốc nào sót lại trong kết quả.
    """
    for lookalike, ascii_expected in MATH_LOOKALIKES_MAP.items():
        sample = f"toán {lookalike} mẫu"
        normalized = normalize(sample)
        assert lookalike not in normalized, f"Ký tự look-alike '{lookalike}' chưa được chuẩn hóa!"
        assert ascii_expected in normalized, f"Ký tự ASCII '{ascii_expected}' không xuất hiện sau khi thay '{lookalike}'!"


def test_normalize_superscripts() -> None:
    """
    Kiểm tra normalize() chuyển đổi các ký tự số mũ Unicode (look-alikes) sang ASCII ^digit.
    Đặc biệt là số mũ 2 (²) và 3 (³).
    """
    raw = "2x² + 3x³ - 5⁰ + 4¹ = y⁴ + z⁵ + a⁶ + b⁷ + c⁸ + d⁹"
    normalized = normalize(raw)
    assert normalized == "2x^2 + 3x^3 - 5^0 + 4^1 = y^4 + z^5 + a^6 + b^7 + c^8 + d^9"
    for sup in ["²", "³", "⁰", "¹", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"]:
        assert sup not in normalized


def test_normalize_unicode_minus_and_dashes() -> None:
    """Kiểm tra normalize() chuyển đổi unicode minus (−) và các ký tự gạch ngang tương đồng sang '-'."""
    # U+2212 Minus, U+2013 En Dash, U+2014 Em Dash, U+FF0D Fullwidth Hyphen-Minus
    raw = "x − 5 – 3 — 1 － 2 ‐ 4 ‑ 6"
    normalized = normalize(raw)
    assert normalized == "x - 5 - 3 - 1 - 2 - 4 - 6"
    for dash in ["−", "–", "—", "－", "‐", "‑"]:
        assert dash not in normalized


def test_normalize_multiplication_and_division_symbols() -> None:
    """Kiểm tra normalize() chuyển đổi các ký hiệu phép nhân sang '*' và phép chia sang '/'."""
    raw = "a × b ⋅ c ✕ d ✖ e ∗ f ＊ g ÷ 2 ∕ 3 ⁄ 4 ／ 5"
    normalized = normalize(raw)
    assert normalized == "a * b * c * d * e * f * g / 2 / 3 / 4 / 5"
    for char in ["×", "⋅", "✕", "✖", "∗", "＊", "÷", "∕", "⁄", "／"]:
        assert char not in normalized


def test_normalize_fullwidth_digits_and_operators() -> None:
    """Kiểm tra normalize() chuyển đổi toàn bộ 10 chữ số toàn độ rộng và các toán tử full-width sang ASCII."""
    # Chữ số ０-９ và toán tử ＋, ＝, ＜, ＞, （, ）
    raw = "（ １ ＋ ２ ＝ ３ ＜ ５ ＞ ０ ）"
    normalized = normalize(raw)
    assert normalized == "( 1 + 2 = 3 < 5 > 0 )"
    for i in range(10):
        full_digit = chr(0xFF10 + i)
        assert full_digit not in normalized


def test_normalize_preserves_vietnamese_casing_and_latex_math_characters() -> None:
    """
    Kiểm tra normalize() bảo toàn nguyên vẹn dấu tiếng Việt, chữ hoa/thường (khi lowercase=False),
    chữ số và tập hợp ký tự toán học/LaTeX: + - = ^ * / ( ) x \\ { } _ $ % &.
    """
    sample = r"Giải phương trình: \frac{a + b}{c - d} = x^2 * (y / z) + 100% & $alpha_1$"
    normalized = normalize(sample, lowercase=False)

    # Tiếng Việt có dấu giữ nguyên
    assert "Giải phương trình" in normalized
    # Kiểu chữ giữ nguyên (chữ 'G' hoa không bị chuyển thành thường)
    assert normalized.startswith("G")
    # Lệnh LaTeX và dấu gạch chéo ngược '\' nguyên vẹn
    assert r"\frac" in normalized
    # Dấu ngoặc nhọn '{', '}' nguyên vẹn
    assert "{" in normalized and "}" in normalized
    # Ký tự toán học và ký hiệu đặc biệt (+ - = ^ * / ( ) x _ $ % &)
    for special_char in ["+", "-", "=", "^", "*", "/", "(", ")", "x", "\\", "{", "}", "_", "$", "%", "&"]:
        assert special_char in normalized, f"Ký tự '{special_char}' bị mất sau chuẩn hóa!"


def test_normalize_lowercase_flag() -> None:
    """Kiểm tra cờ lowercase trong normalize(): giữ nguyên kiểu chữ hoặc chuyển thành chữ thường."""
    sample = "Đạo Hàm Của Hàm Số F(X)"
    assert normalize(sample, lowercase=False) == "Đạo Hàm Của Hàm Số F(X)"
    assert normalize(sample, lowercase=True) == "đạo hàm của hàm số f(x)"


def test_segment_vietnamese_words_with_underscores() -> None:
    """
    Xác minh segment() nối từ ghép tiếng Việt đa từ (multi-word text) bằng dấu gạch dưới '_'
    theo quy chuẩn PhoBERT (Nguyen & Tuan Nguyen, 2020) thông qua pyvi.
    """
    raw = "tính đạo hàm và tích phân của hàm số bậc hai"
    segmented = segment(raw, backend="pyvi")
    # Các từ ghép toán học tiếng Việt phổ biến phải được gạch dưới nối liền
    assert "đạo_hàm" in segmented
    assert "tích_phân" in segmented
    assert "hàm_số" in segmented


def test_segment_protected_spans_math_expression() -> None:
    """
    Xác minh các vùng toán học được bảo vệ (protected spans) như '2x² + 3x - 5 = 0'
    (sau khi normalize() chuyển thành '2x^2 + 3x - 5 = 0') được giữ nguyên vẹn 100%
    bởi bộ phân đoạn từ segment(), không bị chèn dấu gạch dưới vào giữa các toán hạng hay toán tử.
    """
    # 1. Kiểm tra qua luồng normalize() kết hợp segment()
    raw = "giải phương trình bậc hai 2x² + 3x − 5 = 0"
    normalized = normalize(raw)
    assert "2x^2 + 3x - 5 = 0" in normalized

    segmented = segment(normalized, backend="pyvi")
    # Vùng biểu thức toán học phải được giữ nguyên hoàn toàn
    assert "2x^2 + 3x - 5 = 0" in segmented
    # Từ ghép tiếng Việt xung quanh vẫn được phân đoạn
    assert "phương_trình" in segmented

    # 2. Đảm bảo không có dấu gạch dưới bất thường bên trong biểu thức toán học
    assert "2x^2_+_3x" not in segmented
    assert "3x_-_5" not in segmented
    assert "_=_0" not in segmented


def test_segment_shields_latex_commands() -> None:
    """
    Xác minh segment() che chắn và giữ nguyên vẹn các câu lệnh LaTeX (regex \\\\[A-Za-z]+),
    đặc biệt là lệnh \\includegraphics và các lệnh phổ biến như \\frac, \\alpha, \\begin.
    Đảm bảo bộ tách từ không nuốt dấu gạch chéo ngược '\\' hay chèn dấu gạch dưới vào lệnh.
    """
    raw = r"chèn hình ảnh minh họa bằng lệnh \includegraphics và tạo phân số \frac"
    segmented = segment(raw, backend="pyvi")

    # Lệnh \includegraphics và \frac phải được bảo toàn nguyên vẹn
    assert r"\includegraphics" in segmented
    assert r"\frac" in segmented

    # Kiểm tra lệnh \includegraphics kèm tham số và đối số
    raw_with_args = r"sử dụng \includegraphics[width=0.8\textwidth]{plot.png} để hiển thị"
    segmented_with_args = segment(raw_with_args, backend="pyvi")
    assert r"\includegraphics" in segmented_with_args

    # Từ ghép tiếng Việt xung quanh vẫn được phân đoạn chính xác
    assert "hình_ảnh" in segmented
    assert "phân_số" in segmented


def test_segment_shields_math_expressions() -> None:
    """Kiểm tra segment() bảo vệ toàn vẹn các biểu thức toán học nội dòng ($...$) và tên hàm f(x)."""
    raw = "giải phương trình $x^2 + 2x + 1 = 0$ và f(x) = y"
    segmented = segment(raw, backend="pyvi")

    assert "$x^2 + 2x + 1 = 0$" in segmented
    assert "f(x) = y" in segmented
    assert "phương_trình" in segmented


def test_segment_shields_latex_environments() -> None:
    """Kiểm tra segment() bảo vệ các khối môi trường LaTeX phức tạp (\\begin{env}...\\end{env})."""
    raw = r"tạo ma trận \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} cấp hai"
    segmented = segment(raw, backend="pyvi")

    assert r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}" in segmented
    assert "ma_trận" in segmented


def test_segment_unsupported_backend_raises_value_error() -> None:
    """Kiểm tra segment() ném ngoại lệ ValueError nếu backend không nằm trong SUPPORTED_SEGMENTERS."""
    with pytest.raises(ValueError, match="Backend phân đoạn từ không được hỗ trợ"):
        segment("kiểm tra", backend="unsupported_segmenter")


def test_segment_rdr_raises_clear_exception_without_env_modifications() -> None:
    """
    Xác minh việc gọi backend 'rdr' chưa sẵn sàng ném ra ngoại lệ tường minh (RuntimeError)
    mà TUYỆT ĐỐI KHÔNG cố gắng thực hiện các thay đổi môi trường ngầm định:
    - Không gọi subprocess (không gọi apt, dnf, pacman, java installation script).
    - Không gọi pip install ngầm.
    - Báo lỗi rõ ràng hướng dẫn người dùng về yêu cầu JRE hoặc py_vncorenlp.
    """
    # Tạo mock giám sát các phương thức có thể làm biến đổi môi trường hệ thống
    mock_run = MagicMock()
    mock_popen = MagicMock()

    with patch("subprocess.run", mock_run), patch("subprocess.Popen", mock_popen):
        # Tình huống 1: Môi trường thiếu Java runtime
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                segment("câu tiếng Việt kiểm tra", backend="rdr")
            # Thông báo lỗi phải rõ ràng chỉ ra thiếu Java Runtime Environment
            assert "yêu cầu Java Runtime Environment" in str(exc_info.value)
            assert "PATH" in str(exc_info.value)

        # Tình huống 2: Java có sẵn nhưng thiếu thư viện py_vncorenlp
        with patch("shutil.which", return_value="/usr/bin/java"), patch.dict(
            "sys.modules", {"py_vncorenlp": None}
        ):
            with pytest.raises(RuntimeError) as exc_info:
                segment("câu tiếng Việt kiểm tra", backend="rdr")
            # Thông báo lỗi phải rõ ràng chỉ ra thiếu gói py_vncorenlp
            assert "py_vncorenlp" in str(exc_info.value)

        # Tình huống 3: Gọi trực tiếp trong môi trường thực tế hiện tại
        with pytest.raises(RuntimeError) as exc_info:
            segment("câu tiếng Việt kiểm tra", backend="rdr")
        error_msg = str(exc_info.value)
        assert ("Java Runtime Environment" in error_msg) or ("py_vncorenlp" in error_msg) or ("VnCoreNLP" in error_msg)

        # Xác nhận KHÔNG CÓ bất kỳ lệnh subprocess nào được thực thi nhằm cố ý sửa đổi môi trường
        assert mock_run.call_count == 0, "segment('rdr') không được phép gọi subprocess.run để sửa môi trường!"
        assert mock_popen.call_count == 0, "segment('rdr') không được phép gọi subprocess.Popen để sửa môi trường!"


def test_prepare_for_phobert_end_to_end() -> None:
    """
    Xác minh quy trình trọn gói prepare_for_phobert():
    Chuẩn hóa NFC + Ánh xạ ký hiệu toán học tương đồng + Phân đoạn từ tiếng Việt
    kết hợp bảo vệ toàn vẹn biểu thức đại số ('2x² + 3x - 5 = 0') và lệnh LaTeX (\\includegraphics).
    """
    raw = "   Tính đạo hàm f(x) = 2x² ＋ 3x − 5 = 0 và chèn hình ảnh bằng \\includegraphics{chart.png}   "
    prepared = prepare_for_phobert(raw)

    # 1. Khoảng trắng thừa được thu gọn
    assert "   " not in prepared
    # 2. Chuẩn hóa look-alikes: ² thành ^2, ＋ thành +, − thành -
    assert "2x^2 + 3x - 5 = 0" in prepared
    # 3. Lệnh LaTeX \includegraphics được bảo toàn nguyên vẹn
    assert r"\includegraphics{chart.png}" in prepared
    # 4. Từ ghép tiếng Việt được phân đoạn bằng dấu gạch dưới
    assert "đạo_hàm" in prepared
    assert "hình_ảnh" in prepared


def test_is_word_segmentation_required() -> None:
    """
    Kiểm tra is_word_segmentation_required() nhận diện chính xác yêu cầu phân đoạn từ
    theo đặc tả model card:
    - EMBED_MODEL_NAME ('bkai-foundation-models/vietnamese-bi-encoder') dựa trên PhoBERT: Yêu cầu True.
    - Các mô hình dựa trên PhoBERT: Yêu cầu True.
    - Các mô hình đa ngữ (XLM-RoBERTa, MiniLM): Trả về False.
    """
    # Mặc định theo cấu hình EMBED_MODEL_NAME (bkai-foundation-models/vietnamese-bi-encoder)
    assert is_word_segmentation_required() is True
    assert is_word_segmentation_required(None) is True
    assert is_word_segmentation_required("bkai-foundation-models/vietnamese-bi-encoder") is True
    assert is_word_segmentation_required("vinai/phobert-base-v2") is True
    assert is_word_segmentation_required("vinai/phobert-large") is True

    # Mô hình đa ngữ không yêu cầu từ ghép nối gạch dưới
    assert is_word_segmentation_required("xlm-roberta-base") is False
    assert is_word_segmentation_required("sentence-transformers/all-MiniLM-L6-v2") is False
    assert is_word_segmentation_required("sentence-transformers/paraphrase-multilingual-mpnet-base-v2") is False


def test_prepare_for_embedding() -> None:
    """
    Kiểm tra prepare_for_embedding() cho mô hình embedding:
    - Khi segment_words=None và dùng cấu hình mặc định (bkai-foundation-models/vietnamese-bi-encoder):
      Tự động phân đoạn từ ghép theo model card của PhoBERT backbone.
    - Khi chỉ định tường minh segment_words=False: Chỉ chuẩn hóa, không nối gạch dưới.
    - Khi chỉ định tường minh segment_words=True: Luôn phân đoạn từ ghép.
    - Khi chỉ định model_name đa ngữ (ví dụ xlm-roberta-base): Tự động không nối gạch dưới.
    """
    raw = "Tính đạo hàm cấp hai với x² ＝ ４"

    # 1. Mặc định theo EMBED_MODEL_NAME (bkai-foundation-models/vietnamese-bi-encoder): Phân đoạn từ
    embed_text_default = prepare_for_embedding(raw)
    assert "đạo_hàm" in embed_text_default
    assert "x^2 = 4" in embed_text_default

    # 2. Tùy chọn tường minh: không tách từ gạch dưới
    embed_text_no_segment = prepare_for_embedding(raw, segment_words=False)
    assert embed_text_no_segment == "Tính đạo hàm cấp hai với x^2 = 4"
    assert "đạo_hàm" not in embed_text_no_segment

    # 3. Tùy chọn tường minh: có tách từ gạch dưới
    embed_text_segmented = prepare_for_embedding(raw, segment_words=True)
    assert "đạo_hàm" in embed_text_segmented
    assert "x^2 = 4" in embed_text_segmented

    # 4. Khi model_name là mô hình đa ngữ không cần nối từ
    embed_text_xlm = prepare_for_embedding(raw, model_name="xlm-roberta-base")
    assert embed_text_xlm == "Tính đạo hàm cấp hai với x^2 = 4"
    assert "đạo_hàm" not in embed_text_xlm


def test_prepare_for_embedding_prevents_un_preprocessed_strings() -> None:
    """
    Xác minh prepare_for_embedding() ngăn chặn các chuỗi không hợp lệ hoặc chưa qua tiền xử lý
    bị đưa vào bộ mã hóa embedding (prevent un-preprocessed strings from being encoded).
    """
    # 1. Ngăn chặn kiểu dữ liệu không phải chuỗi (None, int, list)
    with pytest.raises(TypeError) as exc_info:
        prepare_for_embedding(None)  # type: ignore[arg-type]
    assert "chuỗi ký tự (str)" in str(exc_info.value)

    with pytest.raises(TypeError):
        prepare_for_embedding(12345)  # type: ignore[arg-type]

    # 2. Ngăn chặn chuỗi rỗng hoặc chỉ chứa khoảng trắng
    with pytest.raises(ValueError) as exc_info:
        prepare_for_embedding("")
    assert "không được để trống" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        prepare_for_embedding("   \t  \n  ")
    assert "không được để trống" in str(exc_info.value)


def test_validate_for_embedding() -> None:
    """
    Kiểm tra validate_for_embedding() xác thực tính hợp lệ của chuỗi đã qua tiền xử lý,
    phát hiện và ném lỗi rõ ràng đối với chuỗi thô/chưa tiền xử lý (un-preprocessed strings).
    """
    # Chuỗi hợp lệ đã qua tiền xử lý
    valid_text = "tính đạo_hàm cấp hai với x^2 = 4"
    assert validate_for_embedding(valid_text) is True

    # 1. Phát hiện chuỗi không phải str
    with pytest.raises(TypeError):
        validate_for_embedding(None)  # type: ignore[arg-type]

    # 2. Phát hiện chuỗi rỗng
    with pytest.raises(ValueError) as exc_info:
        validate_for_embedding("   ")
    assert "không được để trống" in str(exc_info.value)

    # 3. Phát hiện chuỗi chưa chuẩn hóa NFC (chuỗi NFD tổ hợp)
    nfd_text = unicodedata.normalize("NFD", "Tiếng Việt chuẩn hóa")
    with pytest.raises(ValueError) as exc_info:
        validate_for_embedding(nfd_text)
    assert "chưa được chuẩn hóa dạng Unicode dựng sẵn NFC" in str(exc_info.value)

    # 4. Phát hiện chuỗi còn chứa ký hiệu toán học tương đồng (math look-alikes)
    with pytest.raises(ValueError) as exc_info:
        validate_for_embedding("Tính x² + y = 0")
    assert "ký hiệu toán học tương đồng" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        validate_for_embedding("3 × 4 = 12")
    assert "ký hiệu toán học tương đồng" in str(exc_info.value)

    # 5. Phát hiện khoảng trắng thừa chưa thu gọn
    with pytest.raises(ValueError) as exc_info:
        validate_for_embedding("tính  đạo hàm")
    assert "khoảng trắng thừa" in str(exc_info.value)


def test_preprocess_fingerprint_deterministic_and_unique() -> None:
    """
    Kiểm tra mã vân tay preprocess_fingerprint() đảm bảo tính tất định (deterministic)
    và thay đổi khi các cấu hình bộ phân đoạn từ (segmenter configurations) bị thay đổi.
    """
    fp_default = preprocess_fingerprint(backend="pyvi", lowercase=False)
    fp_repeat = preprocess_fingerprint(backend="pyvi", lowercase=False)

    # Tính tất định: cùng cấu hình sinh ra cùng một mã băm SHA-256
    assert fp_default == fp_repeat
    assert isinstance(fp_default, str)
    assert len(fp_default) == 64  # SHA-256 hex digest dài đúng 64 ký tự

    # Thay đổi cấu hình backend (segmenter backend alteration: 'pyvi' -> 'rdr')
    fp_rdr = preprocess_fingerprint(backend="rdr", lowercase=False)
    assert fp_default != fp_rdr, "Mã vân tay phải thay đổi khi đổi backend tách từ ('pyvi' -> 'rdr')!"

    # Thay đổi cấu hình chữ thường (lowercase configuration: False -> True)
    fp_lower = preprocess_fingerprint(backend="pyvi", lowercase=True)
    assert fp_default != fp_lower, "Mã vân tay phải thay đổi khi đổi cờ lowercase!"


def test_get_pyvi_version() -> None:
    """Kiểm tra get_pyvi_version() trả về chuỗi phiên bản pyvi hợp lệ."""
    ver = get_pyvi_version()
    assert isinstance(ver, str)
    assert ver != ""
    assert ver == "0.1.1" or ver != "unknown"
