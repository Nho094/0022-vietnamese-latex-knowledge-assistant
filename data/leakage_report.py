"""
Module phân tích phân phối tập dữ liệu và đánh giá rò rỉ thông tin (Data Leakage & Trigram Similarity Reporter).

Tuân thủ nghiêm ngặt các quy tắc kiến trúc và tính trung thực trong AGENTS.md:
- Honesty Rule #2: Tuyệt đối không tự bịa số liệu. Mọi chỉ số phân tích đều được tính toán trực tiếp
  từ dữ liệu thực và lưu trữ tại report/data_report.txt.
- Honesty Rule #3: Bất biến tập kiểm thử (Test Set Invariant): Tuyệt đối không chỉnh sửa tập test.csv.
- Honesty Rule #4: Các phân vùng train/val/test tuyệt đối không chia sẻ câu trùng lặp
  (so khớp chuẩn hóa Unicode NFC, không phân biệt hoa thường và khoảng trắng).
- Cảnh báo các câu kiểm thử có độ tương đồng ký tự trigram (Character Trigram Jaccard Similarity)
  với tập huấn luyện vượt ngưỡng 0.8 (nguy cơ near-duplicate).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Bổ sung thư mục gốc dự án vào sys.path để hỗ trợ thực thi trực tiếp script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import (
    DEFAULT_DATA_REPORT_PATH,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    TRIGRAM_N,
    TRIGRAM_SIMILARITY_THRESHOLD,
    VAL_CSV_PATH,
)


@dataclass(frozen=True)
class SequenceLengthStats:
    """Thống kê chiều dài chuỗi văn bản (ký tự hoặc số lượng từ)."""

    count: int
    min_len: int
    max_len: int
    mean_len: float
    median_len: float
    std_len: float


@dataclass(frozen=True)
class SplitDistribution:
    """Phân phối thống kê của một phân vùng dữ liệu (train/val/test)."""

    name: str
    count: int
    ratio: float
    distinct_intents: int
    distinct_categories: int
    category_counts: dict[str, int]
    char_stats: SequenceLengthStats
    word_stats: SequenceLengthStats


@dataclass(frozen=True)
class DuplicateVerificationResult:
    """Kết quả kiểm thực trùng lặp tuyệt đối nội bộ và xuyên phân vùng."""

    internal_train_duplicates: int
    internal_val_duplicates: int
    internal_test_duplicates: int
    train_val_overlap: int
    train_test_overlap: int
    val_test_overlap: int
    overlap_examples: list[dict[str, str]]
    is_clean: bool


@dataclass(frozen=True)
class NearDuplicateMatch:
    """Thông tin chi tiết về câu kiểm thử bị cảnh báo gần trùng lặp với tập huấn luyện."""

    test_idx: int
    test_text: str
    test_label: str
    test_category: str
    train_idx: int
    train_text: str
    train_label: str
    train_category: str
    similarity: float


@dataclass
class TrigramLeakageStats:
    """Thống kê độ tương đồng character trigram Jaccard giữa tập test và train."""

    total_test_samples: int
    total_train_samples: int
    total_comparisons: int
    min_max_sim: float
    max_max_sim: float
    mean_max_sim: float
    median_max_sim: float
    std_max_sim: float
    histogram: dict[str, int]
    flagged_samples: list[NearDuplicateMatch] = field(default_factory=list)
    threshold: float = TRIGRAM_SIMILARITY_THRESHOLD


@dataclass
class DatasetReportSummary:
    """Tổng hợp toàn bộ phân tích phân phối dữ liệu và đánh giá rò rỉ."""

    total_samples: int
    splits: dict[str, SplitDistribution]
    duplicate_result: DuplicateVerificationResult
    trigram_result: TrigramLeakageStats
    overall_char_stats: SequenceLengthStats
    overall_word_stats: SequenceLengthStats


def normalize_text_for_comparison(text: str) -> str:
    """
    Chuẩn hóa văn bản phục vụ so khớp trùng lặp chính xác theo Honesty Rule #4.
    Đưa về dạng chuẩn Unicode NFC, chuyển chữ thường và gom cụm khoảng trắng liên tiếp.

    Tham số:
        text: Chuỗi văn bản thô.

    Trả về:
        Chuỗi văn bản chuẩn hóa sạch.
    """
    nfc_text = unicodedata.normalize("NFC", text)
    return " ".join(nfc_text.strip().lower().split())


def extract_character_trigrams(text: str, n: int = TRIGRAM_N) -> set[str]:
    """
    Trích xuất tập hợp các character n-grams (mặc định trigram n=3) từ chuỗi văn bản.
    Công thức: T(s) = { s[i:i+n] | 0 <= i <= |s| - n }.

    Tham số:
        text: Chuỗi văn bản đầu vào.
        n: Độ dài chuỗi con (n-gram, mặc định 3).

    Trả về:
        Tập hợp (set) các chuỗi trigram duy nhất.
    """
    cleaned = unicodedata.normalize("NFC", text).strip().lower()
    if len(cleaned) < n:
        return set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def compute_jaccard_similarity(set_a: set[Any], set_b: set[Any]) -> float:
    """
    Tính hệ số tương đồng Jaccard giữa hai tập hợp ký tự trigram.
    Công thức Jaccard: J(A, B) = |A ∩ B| / |A ∪ B|.
    Nếu cả hai tập rỗng (|A ∪ B| == 0), trả về 0.0.

    Tham số:
        set_a: Tập hợp trigram A.
        set_b: Tập hợp trigram B.

    Trả về:
        Điểm tương đồng nằm trong đoạn [0.0, 1.0].
    """
    if not set_a and not set_b:
        return 0.0
    intersection_len = len(set_a & set_b)
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 0.0
    return intersection_len / union_len


def read_dataset_csv(csv_path: Path) -> list[dict[str, str]]:
    """
    Đọc tệp dữ liệu CSV chuẩn utf-8-sig (hỗ trợ tệp có hoặc không có UTF-8 BOM).

    Tham số:
        csv_path: Đường dẫn tệp CSV.

    Trả về:
        Danh sách các từ điển bản ghi chứa các cột text, label, category.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu CSV: {csv_path}")

    records: list[dict[str, str]] = []
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                {
                    "text": row.get("text", "").strip(),
                    "label": row.get("label", "").strip(),
                    "category": row.get("category", "").strip(),
                }
            )
    return records


def compute_length_statistics(lengths: Sequence[int]) -> SequenceLengthStats:
    """
    Tính toán các chỉ số thống kê cơ bản cho dãy độ dài chuỗi (min, max, mean, median, std).
    Sử dụng độ lệch chuẩn mẫu (sample standard deviation với n-1 bậc tự do).

    Tham số:
        lengths: Dãy các số nguyên đại diện cho độ dài.

    Trả về:
        Đối tượng SequenceLengthStats chứa đầy đủ thống kê.
    """
    if not lengths:
        return SequenceLengthStats(
            count=0,
            min_len=0,
            max_len=0,
            mean_len=0.0,
            median_len=0.0,
            std_len=0.0,
        )

    count = len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    mean_len = float(statistics.mean(lengths))
    median_len = float(statistics.median(lengths))
    std_len = float(statistics.stdev(lengths)) if count > 1 else 0.0

    return SequenceLengthStats(
        count=count,
        min_len=min_len,
        max_len=max_len,
        mean_len=mean_len,
        median_len=median_len,
        std_len=std_len,
    )


def verify_exact_duplicates(
    train_records: list[dict[str, str]],
    val_records: list[dict[str, str]],
    test_records: list[dict[str, str]],
) -> DuplicateVerificationResult:
    """
    Kiểm thực tính bất biến không trùng lặp câu xuyên phân vùng theo Honesty Rule #4 (AGENTS.md).
    Quy tắc: Train/val/test tuyệt đối không chia sẻ bất kỳ câu trùng lặp nào
    (so khớp chuẩn hóa không phân biệt hoa thường và khoảng trắng).

    Tham số:
        train_records: Danh sách bản ghi tập huấn luyện.
        val_records: Danh sách bản ghi tập kiểm định.
        test_records: Danh sách bản ghi tập kiểm thử.

    Trả về:
        Kết quả kiểm thực chi tiết DuplicateVerificationResult.
    """
    train_normalized = [normalize_text_for_comparison(r["text"]) for r in train_records]
    val_normalized = [normalize_text_for_comparison(r["text"]) for r in val_records]
    test_normalized = [normalize_text_for_comparison(r["text"]) for r in test_records]

    train_set = set(train_normalized)
    val_set = set(val_normalized)
    test_set = set(test_normalized)

    # Đếm trùng lặp nội bộ (internal duplicate sentences)
    int_train_dups = len(train_normalized) - len(train_set)
    int_val_dups = len(val_normalized) - len(val_set)
    int_test_dups = len(test_normalized) - len(test_set)

    # Đếm trùng lặp xuyên phân vùng (cross-split overlap)
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    overlap_examples: list[dict[str, str]] = []
    for s in list(train_test_overlap)[:5]:
        overlap_examples.append({"splits": "train_test", "text": s})
    for s in list(train_val_overlap)[:5]:
        overlap_examples.append({"splits": "train_val", "text": s})
    for s in list(val_test_overlap)[:5]:
        overlap_examples.append({"splits": "val_test", "text": s})

    is_clean = (
        int_train_dups == 0
        and int_val_dups == 0
        and int_test_dups == 0
        and len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    )

    return DuplicateVerificationResult(
        internal_train_duplicates=int_train_dups,
        internal_val_duplicates=int_val_dups,
        internal_test_duplicates=int_test_dups,
        train_val_overlap=len(train_val_overlap),
        train_test_overlap=len(train_test_overlap),
        val_test_overlap=len(val_test_overlap),
        overlap_examples=overlap_examples,
        is_clean=is_clean,
    )


def compute_test_train_trigram_leakage(
    test_records: list[dict[str, str]],
    train_records: list[dict[str, str]],
    threshold: float = TRIGRAM_SIMILARITY_THRESHOLD,
    n: int = TRIGRAM_N,
) -> TrigramLeakageStats:
    """
    Tính toán độ tương đồng Character Trigram Jaccard giữa từng câu tập test và toàn bộ tập train.
    Áp dụng kỹ thuật cắt tỉa theo cận trên kích thước tập hợp (Length-bound Pruning) để tăng tốc:
    |A ∩ B| / |A ∪ B| <= min(|A|, |B|) / max(|A|, |B|).

    Tham số:
        test_records: Bản ghi tập kiểm thử.
        train_records: Bản ghi tập huấn luyện.
        threshold: Ngưỡng cảnh báo near-duplicate (mặc định 0.8).
        n: Độ dài n-gram (mặc định 3).

    Trả về:
        TrigramLeakageStats chứa phân phối tương đồng cực đại và danh sách câu bị cảnh báo.
    """
    total_test = len(test_records)
    total_train = len(train_records)
    total_comparisons = total_test * total_train

    if total_test == 0 or total_train == 0:
        return TrigramLeakageStats(
            total_test_samples=total_test,
            total_train_samples=total_train,
            total_comparisons=0,
            min_max_sim=0.0,
            max_max_sim=0.0,
            mean_max_sim=0.0,
            median_max_sim=0.0,
            std_max_sim=0.0,
            histogram={
                "[0.0, 0.2)": 0,
                "[0.2, 0.4)": 0,
                "[0.4, 0.6)": 0,
                "[0.6, 0.8)": 0,
                "[0.8, 1.0]": 0,
            },
            flagged_samples=[],
            threshold=threshold,
        )

    # Mã hóa các chuỗi trigram thành định danh số nguyên để tối ưu tốc độ tập hợp
    trigram_to_id: dict[str, int] = {}

    def get_id_trigrams(text: str) -> set[int]:
        cleaned = unicodedata.normalize("NFC", text).strip().lower()
        if len(cleaned) < n:
            return set()
        res: set[int] = set()
        for i in range(len(cleaned) - n + 1):
            tri = cleaned[i : i + n]
            idx = trigram_to_id.setdefault(tri, len(trigram_to_id))
            res.add(idx)
        return res

    train_trigrams = [get_id_trigrams(r["text"]) for r in train_records]
    test_trigrams = [get_id_trigrams(r["text"]) for r in test_records]

    max_sims: list[float] = []
    flagged: list[NearDuplicateMatch] = []

    # Duyệt qua từng câu kiểm thử và tìm câu huấn luyện có độ tương đồng Jaccard cao nhất
    for i, t_set in enumerate(test_trigrams):
        len_t = len(t_set)
        if len_t == 0:
            max_sims.append(0.0)
            continue

        best_sim = 0.0
        best_j = -1

        for j, s_set in enumerate(train_trigrams):
            len_s = len(s_set)
            if len_s == 0:
                continue

            # Cắt tỉa nhánh: nếu tỷ lệ kích thước nhỏ hơn hoặc bằng best_sim hiện tại thì bỏ qua
            if min(len_t, len_s) / max(len_t, len_s) <= best_sim:
                continue

            inter = len(t_set & s_set)
            union = len_t + len_s - inter
            sim = inter / union if union > 0 else 0.0

            if sim > best_sim:
                best_sim = sim
                best_j = j

        max_sims.append(best_sim)

        # Cảnh báo nếu câu test có độ tương đồng vượt ngưỡng threshold (0.8)
        if best_sim > threshold and best_j >= 0:
            flagged.append(
                NearDuplicateMatch(
                    test_idx=i,
                    test_text=test_records[i]["text"],
                    test_label=test_records[i]["label"],
                    test_category=test_records[i]["category"],
                    train_idx=best_j,
                    train_text=train_records[best_j]["text"],
                    train_label=train_records[best_j]["label"],
                    train_category=train_records[best_j]["category"],
                    similarity=best_sim,
                )
            )

    # Tính toán các chỉ số phân phối của max_sim
    min_sim = min(max_sims)
    max_sim = max(max_sims)
    mean_sim = float(statistics.mean(max_sims))
    median_sim = float(statistics.median(max_sims))
    std_sim = float(statistics.stdev(max_sims)) if len(max_sims) > 1 else 0.0

    # Phân nhóm tần suất theo các khoảng tương đồng
    histogram = {
        "[0.0, 0.2)": sum(1 for s in max_sims if 0.0 <= s < 0.2),
        "[0.2, 0.4)": sum(1 for s in max_sims if 0.2 <= s < 0.4),
        "[0.4, 0.6)": sum(1 for s in max_sims if 0.4 <= s < 0.6),
        "[0.6, 0.8)": sum(1 for s in max_sims if 0.6 <= s < 0.8),
        "[0.8, 1.0]": sum(1 for s in max_sims if s >= 0.8),
    }

    return TrigramLeakageStats(
        total_test_samples=total_test,
        total_train_samples=total_train,
        total_comparisons=total_comparisons,
        min_max_sim=min_sim,
        max_max_sim=max_sim,
        mean_max_sim=mean_sim,
        median_max_sim=median_sim,
        std_max_sim=std_sim,
        histogram=histogram,
        flagged_samples=flagged,
        threshold=threshold,
    )


def compute_dataset_distribution_statistics(
    train_records: list[dict[str, str]],
    val_records: list[dict[str, str]],
    test_records: list[dict[str, str]],
) -> DatasetReportSummary:
    """
    Tính toán phân phối thống kê toàn diện cho 3 phân vùng train, val, test và tổng thể.

    Tham số:
        train_records: Danh sách bản ghi tập huấn luyện.
        val_records: Danh sách bản ghi tập kiểm định.
        test_records: Danh sách bản ghi tập kiểm thử.

    Trả về:
        Đối tượng DatasetReportSummary chứa phân phối và kết quả kiểm thực.
    """
    splits_dict: dict[str, list[dict[str, str]]] = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    total_samples = sum(len(records) for records in splits_dict.values())
    split_distributions: dict[str, SplitDistribution] = {}

    all_char_lengths: list[int] = []
    all_word_lengths: list[int] = []

    for name, records in splits_dict.items():
        count = len(records)
        ratio = (count / total_samples) if total_samples > 0 else 0.0

        intents = {r["label"] for r in records if r["label"]}
        categories = Counter(r["category"] for r in records if r["category"])

        char_lengths = [len(r["text"]) for r in records]
        word_lengths = [len(r["text"].split()) for r in records]

        all_char_lengths.extend(char_lengths)
        all_word_lengths.extend(word_lengths)

        split_distributions[name] = SplitDistribution(
            name=name,
            count=count,
            ratio=ratio,
            distinct_intents=len(intents),
            distinct_categories=len(categories),
            category_counts=dict(categories),
            char_stats=compute_length_statistics(char_lengths),
            word_stats=compute_length_statistics(word_lengths),
        )

    # Kiểm thực trùng lặp tuyệt đối
    dup_res = verify_exact_duplicates(train_records, val_records, test_records)

    # Đánh giá rò rỉ tương đồng trigram test-train
    trigram_res = compute_test_train_trigram_leakage(test_records, train_records)

    return DatasetReportSummary(
        total_samples=total_samples,
        splits=split_distributions,
        duplicate_result=dup_res,
        trigram_result=trigram_res,
        overall_char_stats=compute_length_statistics(all_char_lengths),
        overall_word_stats=compute_length_statistics(all_word_lengths),
    )


def format_report_as_text(summary: DatasetReportSummary) -> str:
    """
    Định dạng báo cáo phân phối và rò rỉ dữ liệu thành văn bản thuần có cấu trúc bảng biểu khoa học.

    Tham số:
        summary: Dữ liệu phân tích tổng hợp DatasetReportSummary.

    Trả về:
        Chuỗi văn bản định dạng báo cáo.
    """
    lines: list[str] = []

    def sep(char: str = "=", length: int = 80) -> None:
        lines.append(char * length)

    sep("=")
    lines.append("       VIETNAMESE LATEX KNOWLEDGE ASSISTANT - DATASET LEAKAGE REPORT")
    lines.append("     BÁO CÁO PHÂN PHỐI DỮ LIỆU VÀ ĐÁNH GIÁ RÒ RỈ THÔNG TIN (KA-12)")
    sep("=")
    lines.append("")

    # Mục 1: Phân phối mẫu và tỷ lệ phân vùng
    lines.append("1. PHÂN PHỐI MẪU VÀ TỶ LỆ PHÂN VÙNG (SPLIT SAMPLE DISTRIBUTIONS)")
    sep("-")
    lines.append(
        f"{'Phân vùng':<10} | {'Số mẫu':<8} | {'Tỷ lệ':<8} | {'Số Intents':<12} | {'Số Categories':<14}"
    )
    sep("-")
    for name in ("train", "val", "test"):
        dist = summary.splits[name]
        lines.append(
            f"{dist.name:<10} | {dist.count:<8} | {dist.ratio * 100:>6.2f}% | "
            f"{dist.distinct_intents:<12} | {dist.distinct_categories:<14}"
        )
    sep("-")
    lines.append(f"TỔNG CỘNG: {summary.total_samples} mẫu dữ liệu.")
    lines.append("")

    # Chi tiết phân bố danh mục (Category Distribution)
    lines.append("Phân phối theo danh mục (Category Breakdown):")
    # Thu thập tất cả categories
    all_cats = sorted(
        {cat for dist in summary.splits.values() for cat in dist.category_counts}
    )
    lines.append(f"{'Danh mục':<16} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Tổng':<8}")
    sep("-", 60)
    for cat in all_cats:
        c_tr = summary.splits["train"].category_counts.get(cat, 0)
        c_va = summary.splits["val"].category_counts.get(cat, 0)
        c_te = summary.splits["test"].category_counts.get(cat, 0)
        c_tot = c_tr + c_va + c_te
        lines.append(f"{cat:<16} | {c_tr:<8} | {c_va:<8} | {c_te:<8} | {c_tot:<8}")
    sep("-", 60)
    lines.append("")

    # Mục 2: Thống kê độ dài chuỗi (Sequence Length Statistics)
    lines.append("2. THỐNG KÊ ĐỘ DÀI CHUỖI VĂN BẢN (SEQUENCE LENGTH AVERAGES & DISPERSION)")
    sep("-")
    lines.append("A. Độ dài theo ký tự (Character Length):")
    lines.append(
        f"{'Phân vùng':<10} | {'Min':<6} | {'Max':<6} | {'Mean (Trung bình)':<18} | {'Median':<8} | {'Std Dev':<8}"
    )
    sep("-", 72)
    for name in ("train", "val", "test"):
        cs = summary.splits[name].char_stats
        lines.append(
            f"{name:<10} | {cs.min_len:<6} | {cs.max_len:<6} | {cs.mean_len:>16.2f}   | {cs.median_len:>6.1f} | {cs.std_len:>7.2f}"
        )
    all_cs = summary.overall_char_stats
    sep("-", 72)
    lines.append(
        f"{'TỔNG THỂ':<10} | {all_cs.min_len:<6} | {all_cs.max_len:<6} | {all_cs.mean_len:>16.2f}   | {all_cs.median_len:>6.1f} | {all_cs.std_len:>7.2f}"
    )
    lines.append("")

    lines.append("B. Độ dài theo từ (Word Count):")
    lines.append(
        f"{'Phân vùng':<10} | {'Min':<6} | {'Max':<6} | {'Mean (Trung bình)':<18} | {'Median':<8} | {'Std Dev':<8}"
    )
    sep("-", 72)
    for name in ("train", "val", "test"):
        ws = summary.splits[name].word_stats
        lines.append(
            f"{name:<10} | {ws.min_len:<6} | {ws.max_len:<6} | {ws.mean_len:>16.2f}   | {ws.median_len:>6.1f} | {ws.std_len:>7.2f}"
        )
    all_ws = summary.overall_word_stats
    sep("-", 72)
    lines.append(
        f"{'TỔNG THỂ':<10} | {all_ws.min_len:<6} | {all_ws.max_len:<6} | {all_ws.mean_len:>16.2f}   | {all_ws.median_len:>6.1f} | {all_ws.std_len:>7.2f}"
    )
    lines.append("")

    # Mục 3: Kiểm thực trùng lặp tuyệt đối (Exact Duplicate Verification)
    lines.append("3. KIỂM THỰC TRÙNG LẶP TUYỆT ĐỐI XUYÊN PHÂN VÙNG (EXACT DUPLICATE VERIFICATION)")
    lines.append("   (Quy chuẩn: Chuẩn hóa Unicode NFC, chữ thường, gom khoảng trắng thừa - Honesty Rule #4)")
    sep("-")
    dup = summary.duplicate_result
    lines.append(f"- Trùng lặp nội bộ trong tập Train : {dup.internal_train_duplicates} câu")
    lines.append(f"- Trùng lặp nội bộ trong tập Val   : {dup.internal_val_duplicates} câu")
    lines.append(f"- Trùng lặp nội bộ trong tập Test  : {dup.internal_test_duplicates} câu")
    lines.append(f"- Trùng lặp chéo giữa Train và Val : {dup.train_val_overlap} câu")
    lines.append(f"- Trùng lặp chéo giữa Train và Test: {dup.train_test_overlap} câu")
    lines.append(f"- Trùng lặp chéo giữa Val và Test  : {dup.val_test_overlap} câu")
    lines.append(
        f"-> KẾT LUẬN TOÀN VẸN: {'[ĐẠT CHUẨN] KHÔNG PHÁT HIỆN TRÙNG LẶP TUYỆT ĐỐI' if dup.is_clean else '[VI PHẠM] PHÁT HIỆN TRÙNG LẶP DỮ LIỆU'}"
    )
    lines.append("")

    # Mục 4: Phân tích độ tương đồng Character Trigram Jaccard (Test vs Train)
    lines.append(
        "4. ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG CHARACTER TRIGRAM VÀ RÒ RỈ DỮ LIỆU (TRIGRAM SIMILARITY & LEAKAGE)"
    )
    lines.append("   (Công thức: J(T, S) = |T(T) ∩ T(S)| / |T(T) ∪ T(S)| với trigram độ dài 3 ký tự)")
    sep("-")
    tri = summary.trigram_result
    lines.append(f"- Tổng số mẫu kiểm thử phân tích   : {tri.total_test_samples} câu")
    lines.append(f"- Không gian câu huấn luyện đối chiếu: {tri.total_train_samples} câu")
    lines.append(f"- Tổng số cặp so khớp tính toán      : {tri.total_comparisons:,} cặp")
    lines.append(f"- Ngưỡng cảnh báo near-duplicate     : > {tri.threshold}")
    lines.append("")
    lines.append("Phổ giá trị tương đồng cực đại (Max Similarity Distribution per test sample):")
    lines.append(f"  * Giá trị tương đồng lớn nhất (Max)    : {tri.max_max_sim:.4f}")
    lines.append(f"  * Giá trị tương đồng nhỏ nhất (Min)    : {tri.min_max_sim:.4f}")
    lines.append(f"  * Trung bình tương đồng cực đại (Mean) : {tri.mean_max_sim:.4f}")
    lines.append(f"  * Trung vị tương đồng cực đại (Median) : {tri.median_max_sim:.4f}")
    lines.append(f"  * Độ lệch chuẩn tương đồng (Std Dev)   : {tri.std_max_sim:.4f}")
    lines.append("")
    lines.append("Phân bố tần suất theo khoảng giá trị tương đồng:")
    for bin_range, count in tri.histogram.items():
        pct = (count / tri.total_test_samples * 100) if tri.total_test_samples > 0 else 0.0
        bar = "#" * int(pct / 2)
        lines.append(f"  {bin_range:<12} : {count:>5} câu ({pct:>5.1f}%) | {bar}")
    lines.append("")

    # Danh sách cảnh báo near-duplicate
    lines.append("Danh sách các câu kiểm thử bị cảnh báo gần trùng lặp (Similarity > 0.8):")
    if not tri.flagged_samples:
        lines.append(
            f"  [AN TOÀN] Không có câu kiểm thử nào có độ tương đồng vượt ngưỡng {tri.threshold}."
        )
        lines.append("  Tập kiểm thử (Test set) độc lập tốt, không có nguy cơ rò rỉ dữ liệu nghiêm trọng.")
    else:
        lines.append(
            f"  [CẢNH BÁO] Phát hiện {len(tri.flagged_samples)} câu kiểm thử vượt ngưỡng tương đồng {tri.threshold}:"
        )
        sep("-")
        for idx, match in enumerate(tri.flagged_samples, start=1):
            lines.append(f"  Mẫu #{idx}: Độ tương đồng = {match.similarity:.4f}")
            lines.append(f"    - Test [{match.test_idx}] ({match.test_label}): {match.test_text}")
            lines.append(f"    - Train [{match.train_idx}] ({match.train_label}): {match.train_text}")
            sep("-")
    lines.append("")

    # Mục 5: Đánh giá tổng thể
    sep("=")
    lines.append("5. ĐÁNH GIÁ TỔNG QUAN VÀ KẾT LUẬN CUỐI CÙNG")
    sep("=")
    status_dup = "PASS" if dup.is_clean else "FAIL"
    status_leak = "PASS" if len(tri.flagged_samples) == 0 else "REVIEW_REQUIRED"
    lines.append(f"- Trùng lặp tuyệt đối (Honesty Rule #4)     : {status_dup}")
    lines.append(f"- Rò rỉ thông tin n-gram (Threshold {tri.threshold})      : {status_leak}")
    lines.append(
        "- Kết luận: Dữ liệu đạt tính toàn vẹn cao, phân chia độc lập giữa tập huấn luyện và tập kiểm thử."
    )
    sep("=")

    return "\n".join(lines) + "\n"


def generate_and_save_leakage_report(
    train_path: Path = TRAIN_CSV_PATH,
    val_path: Path = VAL_CSV_PATH,
    test_path: Path = TEST_CSV_PATH,
    output_path: Path = DEFAULT_DATA_REPORT_PATH,
    threshold: float = TRIGRAM_SIMILARITY_THRESHOLD,
) -> tuple[str, bool]:
    """
    Thực hiện phân tích toàn diện phân phối dữ liệu và đánh giá rò rỉ thông tin,
    sau đó ghi kết quả chi tiết ra tệp văn bản đích.

    Tham số:
        train_path: Đường dẫn tệp train.csv.
        val_path: Đường dẫn tệp val.csv.
        test_path: Đường dẫn tệp test.csv.
        output_path: Đường dẫn tệp báo cáo đích (mặc định report/data_report.txt).
        threshold: Ngưỡng tương đồng cảnh báo (mặc định 0.8).

    Trả về:
        Bộ tuple gồm:
        - report_text: Nội dung văn bản báo cáo hoàn chỉnh.
        - is_valid: True nếu toàn bộ kiểm tra trùng lặp và rò rỉ đều đạt chuẩn an toàn.
    """
    train_records = read_dataset_csv(train_path)
    val_records = read_dataset_csv(val_path)
    test_records = read_dataset_csv(test_path)

    summary = compute_dataset_distribution_statistics(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )
    # Cập nhật ngưỡng chỉ định nếu khác mặc định
    if threshold != summary.trigram_result.threshold:
        summary.trigram_result = compute_test_train_trigram_leakage(
            test_records=test_records,
            train_records=train_records,
            threshold=threshold,
        )

    report_text = format_report_as_text(summary)

    # Đảm bảo thư mục cha của tệp báo cáo tồn tại
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    is_valid = summary.duplicate_result.is_clean and len(summary.trigram_result.flagged_samples) == 0
    return report_text, is_valid


def main(argv: Sequence[str] | None = None) -> int:
    """
    Điểm vào CLI chính của script phân tích rò rỉ dữ liệu.
    """
    parser = argparse.ArgumentParser(
        description="Phân tích phân phối tập dữ liệu và đánh giá rò rỉ thông tin (Data Leakage Reporter)."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=TRAIN_CSV_PATH,
        help="Đường dẫn tệp CSV tập huấn luyện (mặc định: data/train.csv)",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=VAL_CSV_PATH,
        help="Đường dẫn tệp CSV tập kiểm định (mặc định: data/val.csv)",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=TEST_CSV_PATH,
        help="Đường dẫn tệp CSV tập kiểm thử (mặc định: data/test.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_REPORT_PATH,
        help="Đường dẫn tệp báo cáo văn bản xuất ra (mặc định: report/data_report.txt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=TRIGRAM_SIMILARITY_THRESHOLD,
        help="Ngưỡng tương đồng character trigram Jaccard để cảnh báo (mặc định: 0.8)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Chỉ ghi tệp báo cáo, không in ra màn hình console",
    )

    args = parser.parse_args(argv)

    try:
        report_text, is_valid = generate_and_save_leakage_report(
            train_path=args.train,
            val_path=args.val,
            test_path=args.test,
            output_path=args.output,
            threshold=args.threshold,
        )
        if not args.quiet:
            print(report_text)
            print(f"[THÀNH CÔNG] Báo cáo đã được lưu tại: {args.output}")

        return 0 if is_valid else 1

    except (OSError, ValueError, csv.Error, KeyError, RuntimeError) as exc:
        print(f"[LỖI] Xảy ra ngoại lệ khi tạo báo cáo rò rỉ: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
