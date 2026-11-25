"""
Module phân tích phân phối subword và kiểm tra độ dài chuỗi token PhoBERT (BPE Tokenization Inspection).

Tuân thủ nghiêm ngặt các quy tắc kiến trúc và tính trung thực trong AGENTS.md:
- Kiến trúc tiền xử lý:
  text -> preprocess (NFC, chuẩn hóa ký hiệu toán học, tách từ pyvi bảo vệ toán/LaTeX)
       -> PhoBERT BPE Tokenizer -> kiểm tra giới hạn độ dài chuỗi (sequence length <= 64 tokens).
- Cơ sở lý thuyết và tham chiếu học thuật:
  1. PhoBERT (Nguyen & Tuan Nguyen, 2020 - "PhoBERT: Pre-trained language models for Vietnamese"):
     PhoBERT sử dụng kiến trúc RoBERTa tiền huấn luyện ở cấp độ từ (word-level). Văn bản đầu vào
     phải qua bước tách từ (nối từ ghép bằng dấu gạch dưới '_') trước khi đi qua thuật toán
     Byte-Pair Encoding (BPE) của fastBPE/RoBERTa để chia tách thành các subwords với ký tự hậu tố '@@'.
  2. Giới hạn độ dài chuỗi (Maximum Sequence Length limit = 64 tokens) được cấu hình tập trung
     tại core.config.PHOBERT_MAX_SEQUENCE_LIMIT nhằm đảm bảo thời gian suy luận thấp và phù hợp
     với phân phối độ dài của các câu lệnh truy vấn ngắn trong trợ lý LaTeX.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# Bổ sung thư mục gốc của dự án vào sys.path để hỗ trợ thực thi trực tiếp dạng script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import (
    PHOBERT_DIR,
    PHOBERT_MAX_SEQUENCE_LIMIT,
    PHOBERT_MODEL_NAME,
)
from core.preprocess import prepare_for_phobert

# 5 mẫu truy vấn đại diện bao phủ: biểu thức toán học, câu lệnh LaTeX và tiếng Việt đàm thoại
SAMPLE_QUERIES: Final[list[dict[str, str]]] = [
    {
        "category": "math_expression",
        "description": "Biểu thức toán học đại số chứa số mũ tương đồng và dấu trừ Unicode",
        "query": "giải phương trình bậc hai 2x² + 3x - 5 = 0 giúp tôi",
    },
    {
        "category": "latex_command",
        "description": "Câu lệnh LaTeX chèn hình ảnh và tham số độ rộng tài liệu",
        "query": r"chèn hình ảnh bằng lệnh \includegraphics với độ rộng 0.8\textwidth",
    },
    {
        "category": "colloquial_vietnamese",
        "description": "Tiếng Việt đàm thoại đời thường kèm xưng hô thân mật",
        "query": "ê bạn ơi làm sao để tạo bảng 3 cột đẹp đẹp xíu trong bài báo cáo vậy",
    },
    {
        "category": "math_and_latex",
        "description": "Kết hợp ký hiệu giải tích toán học và cú pháp lệnh tích phân LaTeX",
        "query": r"hướng dẫn tạo công thức tích phân \int_{0}^{1} x^2 dx trong LaTeX",
    },
    {
        "category": "latex_environment",
        "description": "Môi trường cấu trúc khối ma trận toán học LaTeX begin/end",
        "query": r"viết ma trận kích thước 2x2 trong môi trường \begin{matrix} và \end{matrix}",
    },
]


@dataclass
class TokenInspectionResult:
    """
    Kết quả phân tích tokenization cho từng câu truy vấn.

    Thuộc tính:
        raw_text: Chuỗi văn bản tiếng Việt thô ban đầu.
        category: Nhóm phân loại mẫu (toán học, lệnh LaTeX, tiếng Việt đàm thoại).
        segmented_text: Chuỗi văn bản sau khi qua module chuẩn hóa và phân đoạn từ (prepare_for_phobert).
        subwords: Danh sách các subword token được tạo bởi PhoBERT BPE.
        subword_count: Tổng số lượng subword token.
        is_compliant: Trạng thái tuân thủ giới hạn độ dài chuỗi (subword_count <= limit).
        limit: Ngưỡng giới hạn số lượng token tối đa cho phép (mặc định 64).
    """

    raw_text: str
    category: str
    segmented_text: str
    subwords: list[str]
    subword_count: int
    is_compliant: bool
    limit: int = PHOBERT_MAX_SEQUENCE_LIMIT


def load_phobert_tokenizer(
    model_path: str | Path | None = None,
) -> Any:
    """
    Nạp PhoBERT tokenizer an toàn, ưu tiên nạp ngoại tuyến từ bộ nhớ đệm cục bộ models/hf_cache.

    Tham số:
        model_path: Đường dẫn thư mục mô hình cục bộ hoặc tên định danh HuggingFace Hub.
                    Mặc định sử dụng core.config.PHOBERT_DIR nếu tồn tại, ngược lại PHOBERT_MODEL_NAME.

    Trả về:
        Đối tượng PhobertTokenizer hoặc PreTrainedTokenizerFast.

    Ngoại lệ:
        RuntimeError: Khi không thể nạp tokenizer từ môi trường cục bộ hay trực tuyến.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
    except ImportError as err:
        raise RuntimeError(
            "Thư viện 'transformers' chưa được cài đặt trong môi trường. "
            "Vui lòng kiểm tra lại môi trường ảo python."
        ) from err

    cache_path = Path(model_path) if model_path is not None else PHOBERT_DIR

    # Ưu tiên 1: Tải ngoại tuyến từ thư mục cục bộ models/hf_cache
    if cache_path.exists() and (cache_path / "vocab.txt").exists():
        try:
            return AutoTokenizer.from_pretrained(str(cache_path), local_files_only=True)
        except (OSError, ValueError):
            # shortcut: file cache hỏng cục bộ, chuyển tiếp sang tải từ Hub
            pass

    # Ưu tiên 2: Tải từ HuggingFace Hub với định danh chuẩn
    target_name = str(model_path) if model_path is not None else PHOBERT_MODEL_NAME
    try:
        return AutoTokenizer.from_pretrained(target_name)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"Không thể khởi tạo PhoBERT tokenizer từ '{target_name}'. "
            f"Chi tiết lỗi: {exc}"
        ) from exc


def inspect_query_tokens(
    raw_text: str,
    category: str = "general",
    tokenizer: Any | None = None,
    limit: int = PHOBERT_MAX_SEQUENCE_LIMIT,
) -> TokenInspectionResult:
    """
    Tiền xử lý và phân tích chi tiết các subword tokens cho một câu truy vấn tiếng Việt.

    Quy trình:
    1. Chuẩn hóa và phân đoạn từ được bảo vệ qua prepare_for_phobert (NFC, look-alikes, math/LaTeX shielding).
    2. Chuyển đổi chuỗi phân đoạn thành danh sách BPE subwords bằng tokenizer.tokenize.
    3. Đếm số lượng subwords và so sánh với ngưỡng giới hạn limit (64 tokens).

    Tham số:
        raw_text: Chuỗi văn bản tiếng Việt ban đầu.
        category: Phân loại câu truy vấn.
        tokenizer: Bộ tách từ PhoBERT (nếu None sẽ tự nạp qua load_phobert_tokenizer).
        limit: Giới hạn độ dài chuỗi token tối đa cho phép.

    Trả về:
        Đối tượng TokenInspectionResult chứa thông tin phân tích đầy đủ.
    """
    if tokenizer is None:
        tokenizer = load_phobert_tokenizer()

    # Bước 1: Tiền xử lý chuẩn hóa và tách từ bảo vệ toán/LaTeX
    segmented_text = prepare_for_phobert(raw_text)

    # Bước 2: Tách subword BPE bằng PhoBERT tokenizer
    subwords: list[str] = tokenizer.tokenize(segmented_text)
    subword_count = len(subwords)

    # Bước 3: Xác minh tuân thủ giới hạn 64 tokens
    is_compliant = subword_count <= limit

    return TokenInspectionResult(
        raw_text=raw_text,
        category=category,
        segmented_text=segmented_text,
        subwords=subwords,
        subword_count=subword_count,
        is_compliant=is_compliant,
        limit=limit,
    )


def inspect_sample_queries(
    samples: Sequence[dict[str, str]] | None = None,
    tokenizer: Any | None = None,
    limit: int = PHOBERT_MAX_SEQUENCE_LIMIT,
) -> list[TokenInspectionResult]:
    """
    Thực hiện kiểm tra đồng loạt danh sách các mẫu truy vấn đại diện.

    Tham số:
        samples: Danh sách các mẫu truy vấn (mặc định sử dụng SAMPLE_QUERIES gồm 5 mẫu).
        tokenizer: Đối tượng tokenizer PhoBERT.
        limit: Giới hạn token tối đa.

    Trả về:
        Danh sách kết quả TokenInspectionResult tương ứng với từng mẫu.
    """
    if samples is None:
        samples = SAMPLE_QUERIES

    if tokenizer is None:
        tokenizer = load_phobert_tokenizer()

    results: list[TokenInspectionResult] = []
    for item in samples:
        res = inspect_query_tokens(
            raw_text=item["query"],
            category=item.get("category", "general"),
            tokenizer=tokenizer,
            limit=limit,
        )
        results.append(res)
    return results


def format_inspection_table(
    results: Sequence[TokenInspectionResult],
    truncate_subwords_at: int = 8,
) -> str:
    """
    Tạo bảng định dạng báo cáo tổng quan phân phối subwords và trạng thái tuân thủ.

    Tham số:
        results: Danh sách kết quả phân tích TokenInspectionResult.
        truncate_subwords_at: Số lượng token hiển thị tối đa trong cột bảng trước khi rút gọn (tránh vỡ layout).

    Trả về:
        Chuỗi biểu diễn bảng ASCII/grid đã căn lề đẹp mắt.
    """
    headers = [
        "STT",
        "Phân loại",
        "Đầu vào thô (Raw input string)",
        "Chuỗi phân đoạn (Segmented text)",
        "PhoBERT BPE Subwords",
        "Số token",
        "Giới hạn",
        "Tuân thủ (<= 64)",
    ]

    rows = []
    for idx, r in enumerate(results, start=1):
        if len(r.subwords) <= truncate_subwords_at:
            subwords_str = " ".join(r.subwords)
        else:
            head = " ".join(r.subwords[: truncate_subwords_at // 2])
            tail = " ".join(r.subwords[-(truncate_subwords_at // 2) :])
            subwords_str = f"{head} ... {tail}"

        status_str = "ĐẠT (PASS)" if r.is_compliant else "VƯỢT (FAIL)"
        rows.append([
            idx,
            r.category,
            r.raw_text,
            r.segmented_text,
            subwords_str,
            r.subword_count,
            f"<= {r.limit}",
            status_str,
        ])

    try:
        from tabulate import tabulate  # type: ignore[import-untyped]

        return tabulate(rows, headers=headers, tablefmt="grid")
    except ImportError:
        # Tự tạo bảng định dạng Markdown nếu tabulate không có sẵn
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)


def format_detailed_cards(results: Sequence[TokenInspectionResult]) -> str:
    """
    Tạo báo cáo chi tiết theo từng mẫu truy vấn dạng danh thiếp (card view),
    hiển thị trọn vẹn 100% các subword token không bị rút gọn.

    Tham số:
        results: Danh sách kết quả phân tích TokenInspectionResult.

    Trả về:
        Chuỗi văn bản định dạng chi tiết từng mẫu.
    """
    cards: list[str] = []
    separator = "=" * 80

    for idx, r in enumerate(results, start=1):
        compliance_label = (
            f"ĐẠT YÊU CẦU ({r.subword_count}/{r.limit} tokens <= {r.limit})"
            if r.is_compliant
            else f"KHÔNG ĐẠT ({r.subword_count}/{r.limit} tokens > {r.limit})"
        )
        subwords_formatted = ", ".join(f"'{tok}'" for tok in r.subwords)

        card_content = (
            f"{separator}\n"
            f"MẪU THỬ NGHIỆM #{idx} [{r.category.upper()}]\n"
            f"- Chuỗi gốc thô (Raw input):   {r.raw_text}\n"
            f"- Chuỗi đã tách từ (Segmented): {r.segmented_text}\n"
            f"- Tổng số subword tokens:      {r.subword_count} (Giới hạn tối đa: {r.limit})\n"
            f"- Đánh giá tuân thủ:           {compliance_label}\n"
            f"- Danh sách PhoBERT BPE tokens:\n"
            f"  [{subwords_formatted}]\n"
        )
        cards.append(card_content)

    cards.append(separator)
    return "\n".join(cards)


def main() -> int:
    """
    Hàm thực thi chính của script show_tokens.py.
    """
    parser = argparse.ArgumentParser(
        description="Script phân tích phân phối token PhoBERT và kiểm tra giới hạn 64 tokens."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Đường dẫn thư mục hoặc checkpoint PhoBERT (mặc định nạp từ models/hf_cache).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=PHOBERT_MAX_SEQUENCE_LIMIT,
        help=f"Giới hạn độ dài token tối đa (mặc định {PHOBERT_MAX_SEQUENCE_LIMIT}).",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Chỉ in bảng tóm tắt, ẩn phần chi tiết các token.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("PHOBERT BPE TOKENIZATION & SUBWORD LENGTH INSPECTION")
    print("Trợ lý tiếng Việt LaTeX (Vietnamese LaTeX Knowledge Assistant)")
    print("=" * 80)

    try:
        tokenizer = load_phobert_tokenizer(args.model_path)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"\n[LỖI] Không thể tải PhoBERT tokenizer: {exc}", file=sys.stderr)
        return 1

    results = inspect_sample_queries(SAMPLE_QUERIES, tokenizer=tokenizer, limit=args.limit)

    # In báo cáo chi tiết các mẫu
    if not args.table_only:
        print("\n--- CHI TIẾT CÁC MẪU TRUY VẤN ĐẠI DIỆN ---")
        print(format_detailed_cards(results))

    # In bảng tổng hợp
    print("\n--- BẢNG TỔNG HỢP KIỂM TRA PHÂN PHỐI SUBWORD & TUÂN THỦ GIỚI HẠN ---")
    print(format_inspection_table(results))

    # Kiểm tra toàn bộ có đạt yêu cầu <= 64 tokens
    all_passed = all(r.is_compliant for r in results)
    max_tokens = max(r.subword_count for r in results)
    avg_tokens = sum(r.subword_count for r in results) / len(results)

    print("\n--- THỐNG KÊ PHÂN PHỐI SUBWORD ---")
    print(f"Tổng số mẫu kiểm tra: {len(results)}")
    print(f"Số lượng token lớn nhất: {max_tokens} / {args.limit}")
    print(f"Số lượng token trung bình: {avg_tokens:.2f}")
    print(
        f"Trạng thái tuân thủ toàn bộ mẫu (<= {args.limit} tokens): "
        f"{'ĐẠT (PASS)' if all_passed else 'KHÔNG ĐẠT (FAIL)'}\n"
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
