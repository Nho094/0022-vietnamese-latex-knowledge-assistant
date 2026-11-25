"""
Kiểm thử tự động cho module phân tích rò rỉ dữ liệu và tương đồng n-gram (data/leakage_report.py).

Tuân thủ nghiêm ngặt các quy tắc kiến trúc và tính trung thực trong AGENTS.md:
- Honesty Rule #2: Kiểm chứng các chỉ số phân tích thực tế không bịa đặt số liệu.
- Honesty Rule #3: Bất biến tập kiểm thử: Chỉ đọc, không chỉnh sửa test.csv.
- Honesty Rule #4: Kiểm thực không trùng lặp câu xuyên phân vùng (train/val/test).
- Kiểm tra tính đúng đắn toán học của hàm trích xuất trigram và chỉ số Jaccard Similarity.
- Kiểm tra phát hiện và cảnh báo các câu near-duplicate vượt ngưỡng threshold (0.8).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from core.config import (
    DEFAULT_DATA_REPORT_PATH,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    TRIGRAM_SIMILARITY_THRESHOLD,
    VAL_CSV_PATH,
)
from data.leakage_report import (
    DuplicateVerificationResult,
    NearDuplicateMatch,
    TrigramLeakageStats,
    compute_jaccard_similarity,
    compute_length_statistics,
    compute_test_train_trigram_leakage,
    extract_character_trigrams,
    generate_and_save_leakage_report,
    main,
    normalize_text_for_comparison,
    read_dataset_csv,
    verify_exact_duplicates,
)


def test_normalize_text_for_comparison() -> None:
    """Kiểm tra chuẩn hóa Unicode NFC, chữ thường và gom khoảng trắng thừa."""
    raw = "   Tạo   phân   SỐ   toán  học \n \t trong   LaTeX   "
    expected = "tạo phân số toán học trong latex"
    assert normalize_text_for_comparison(raw) == expected

    # Chuỗi rỗng
    assert normalize_text_for_comparison("   \t  \n  ") == ""


def test_extract_character_trigrams() -> None:
    """Kiểm tra hàm trích xuất character trigrams (n=3)."""
    # Chuỗi ngắn hơn 3 ký tự
    assert extract_character_trigrams("") == set()
    assert extract_character_trigrams("a") == set()
    assert extract_character_trigrams("ab") == set()

    # Chuỗi đúng 3 ký tự
    assert extract_character_trigrams("abc") == {"abc"}

    # Chuỗi 5 ký tự: "hello" -> {"hel", "ell", "llo"}
    assert extract_character_trigrams("hello") == {"hel", "ell", "llo"}

    # Tiếng Việt có dấu Unicode NFC
    res = extract_character_trigrams("toán")
    assert res == {"toá", "oán"}


def test_compute_jaccard_similarity() -> None:
    """Kiểm tra tính toán độ tương đồng Jaccard giữa hai tập hợp."""
    # Hai tập giống hệt nhau
    set_a = {"abc", "bcd", "cde"}
    assert compute_jaccard_similarity(set_a, set_a) == 1.0

    # Hai tập rời nhau
    set_b = {"xyz", "yza"}
    assert compute_jaccard_similarity(set_a, set_b) == 0.0

    # Giao 2, hợp 4: Jaccard = 2/4 = 0.5
    set_c = {"bcd", "cde", "def", "efg"}
    assert compute_jaccard_similarity(set_a, set_c) == pytest.approx(2 / 5)

    # Tập rỗng
    assert compute_jaccard_similarity(set(), set()) == 0.0
    assert compute_jaccard_similarity(set_a, set()) == 0.0


def test_compute_length_statistics() -> None:
    """Kiểm tra tính toán các chỉ số thống kê độ dài (min, max, mean, median, std)."""
    # Dãy rỗng
    empty_stats = compute_length_statistics([])
    assert empty_stats.count == 0
    assert empty_stats.mean_len == 0.0

    # Dãy gồm 3 phần tử [10, 20, 30]
    stats = compute_length_statistics([10, 20, 30])
    assert stats.count == 3
    assert stats.min_len == 10
    assert stats.max_len == 30
    assert stats.mean_len == 20.0
    assert stats.median_len == 20.0
    assert stats.std_len == pytest.approx(10.0)

    # Dãy đơn lẻ
    single_stats = compute_length_statistics([42])
    assert single_stats.count == 1
    assert single_stats.std_len == 0.0


def test_verify_exact_duplicates_clean() -> None:
    """Kiểm tra xác minh khi không có bất kỳ câu trùng lặp nào."""
    train = [{"text": "Câu huấn luyện 1", "label": "L1", "category": "C1"}]
    val = [{"text": "Câu kiểm định 1", "label": "L1", "category": "C1"}]
    test = [{"text": "Câu kiểm thử 1", "label": "L1", "category": "C1"}]

    res = verify_exact_duplicates(train, val, test)
    assert res.is_clean is True
    assert res.internal_train_duplicates == 0
    assert res.train_val_overlap == 0
    assert res.train_test_overlap == 0
    assert res.val_test_overlap == 0


def test_verify_exact_duplicates_detected() -> None:
    """Kiểm tra phát hiện trùng lặp nội bộ và trùng lặp xuyên phân vùng."""
    train = [
        {"text": "Tạo phân số trong LaTeX", "label": "L1", "category": "C1"},
        {"text": "tạo   phân SỐ trong   latex", "label": "L1", "category": "C1"},  # Trùng nội bộ
    ]
    val = [{"text": "Khác hoàn toàn", "label": "L1", "category": "C1"}]
    test = [{"text": "Tạo phân số trong LaTeX", "label": "L1", "category": "C1"}]  # Trùng xuyên tập train-test

    res = verify_exact_duplicates(train, val, test)
    assert res.is_clean is False
    assert res.internal_train_duplicates == 1
    assert res.train_test_overlap == 1
    assert len(res.overlap_examples) > 0


def test_compute_test_train_trigram_leakage_flagging() -> None:
    """Kiểm tra tính năng phát hiện và gắn cờ near-duplicate khi độ tương đồng > 0.8."""
    # Test câu gần giống nhau (độ tương đồng cao > 0.8)
    train_records = [
        {"text": "hướng dẫn cách tạo phân số toán học trong tài liệu LaTeX", "label": "PHAN_SO", "category": "tex_math"},
        {"text": "chèn hình ảnh png vào báo cáo", "label": "HINH_ANH", "category": "tex_figures"},
    ]
    test_records = [
        # Câu kiểm thử rất giống câu 1 trong train
        {"text": "hướng dẫn cách tạo phân số toán học trong tài liệu LaTeX.", "label": "PHAN_SO", "category": "tex_math"},
        # Câu kiểm thử hoàn toàn khác
        {"text": "cấu hình lề trang và kích thước giấy A4", "label": "LE_TRANG", "category": "tex_layout"},
    ]

    leakage_stats: TrigramLeakageStats = compute_test_train_trigram_leakage(
        test_records=test_records,
        train_records=train_records,
        threshold=0.8,
    )

    assert leakage_stats.total_test_samples == 2
    assert leakage_stats.total_train_samples == 2
    assert leakage_stats.total_comparisons == 4
    assert len(leakage_stats.flagged_samples) == 1

    flagged: NearDuplicateMatch = leakage_stats.flagged_samples[0]
    assert flagged.test_idx == 0
    assert flagged.train_idx == 0
    assert flagged.similarity > 0.8
    assert flagged.test_label == "PHAN_SO"


def test_compute_test_train_trigram_leakage_empty() -> None:
    """Kiểm tra xử lý biên khi tập kiểm thử hoặc tập huấn luyện rỗng."""
    stats = compute_test_train_trigram_leakage([], [])
    assert stats.total_test_samples == 0
    assert stats.total_comparisons == 0
    assert stats.max_max_sim == 0.0
    assert len(stats.flagged_samples) == 0


def test_read_dataset_csv_missing_file(tmp_path: Path) -> None:
    """Kiểm tra bắt lỗi FileNotFoundError khi tệp không tồn tại."""
    missing_file = tmp_path / "not_found.csv"
    with pytest.raises(FileNotFoundError):
        read_dataset_csv(missing_file)


def test_format_report_as_text_and_generation(tmp_path: Path) -> None:
    """Kiểm tra định dạng báo cáo văn bản đầy đủ và lưu ra tệp thành công."""
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    output_txt = tmp_path / "report" / "data_report.txt"

    def write_csv(p: Path, rows: list[list[str]]) -> None:
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label", "category"])
            writer.writerows(rows)

    write_csv(
        train_csv,
        [
            ["viết giúp tôi tài liệu tối thiểu", "LATEX_TAI_LIEU_TOI_GIAN", "tex_struct"],
            ["tạo tài liệu bài báo", "LATEX_ARTICLE", "tex_doc"],
        ],
    )
    write_csv(
        val_csv,
        [
            ["hướng dẫn làm tài liệu ngắn gọn", "LATEX_TAI_LIEU_TOI_GIAN", "tex_struct"],
        ],
    )
    write_csv(
        test_csv,
        [
            ["cần khung tài liệu khởi tạo cho dự án", "LATEX_TAI_LIEU_TOI_GIAN", "tex_struct"],
        ],
    )

    report_text, is_valid = generate_and_save_leakage_report(
        train_path=train_csv,
        val_path=val_csv,
        test_path=test_csv,
        output_path=output_txt,
        threshold=0.8,
    )

    assert is_valid is True
    assert output_txt.exists()
    assert "SPLIT SAMPLE DISTRIBUTIONS" in report_text
    assert "SEQUENCE LENGTH AVERAGES" in report_text
    assert "EXACT DUPLICATE VERIFICATION" in report_text
    assert "TRIGRAM SIMILARITY & LEAKAGE" in report_text
    assert "[ĐẠT CHUẨN] KHÔNG PHÁT HIỆN TRÙNG LẶP TUYỆT ĐỐI" in report_text


def test_cli_main_invocation(tmp_path: Path) -> None:
    """Kiểm tra thực thi CLI qua hàm main với các cờ dòng lệnh."""
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    output_txt = tmp_path / "data_report.txt"

    with open(train_csv, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([["text", "label", "category"], ["mẫu train một", "L1", "C1"]])
    with open(val_csv, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([["text", "label", "category"], ["mẫu val một", "L1", "C1"]])
    with open(test_csv, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([["text", "label", "category"], ["mẫu test một", "L1", "C1"]])

    exit_code = main(
        [
            "--train",
            str(train_csv),
            "--val",
            str(val_csv),
            "--test",
            str(test_csv),
            "--output",
            str(output_txt),
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert output_txt.exists()


def test_real_project_dataset_leakage_integrity() -> None:
    """
    Kiểm thực tính toàn vẹn dữ liệu thực tế của toàn bộ dự án:
    1. Tệp train.csv, val.csv, test.csv tồn tại.
    2. Không có bất kỳ câu trùng lặp tuyệt đối nào giữa các phân vùng (Honesty Rule #4).
    3. Không có câu kiểm thử nào bị gắn cờ rò rỉ near-duplicate (trigram similarity > 0.8).
    4. Tệp báo cáo mặc định report/data_report.txt tồn tại và có nội dung hợp lệ.
    """
    assert TRAIN_CSV_PATH.exists(), f"Không tìm thấy {TRAIN_CSV_PATH}"
    assert VAL_CSV_PATH.exists(), f"Không tìm thấy {VAL_CSV_PATH}"
    assert TEST_CSV_PATH.exists(), f"Không tìm thấy {TEST_CSV_PATH}"

    train_records = read_dataset_csv(TRAIN_CSV_PATH)
    val_records = read_dataset_csv(VAL_CSV_PATH)
    test_records = read_dataset_csv(TEST_CSV_PATH)

    # Kiểm tra trùng lặp
    dup_res: DuplicateVerificationResult = verify_exact_duplicates(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )
    assert dup_res.is_clean is True, (
        f"Vi phạm Honesty Rule #4: Phát hiện trùng lặp câu! "
        f"train_test: {dup_res.train_test_overlap}, "
        f"train_val: {dup_res.train_val_overlap}, "
        f"val_test: {dup_res.val_test_overlap}"
    )

    # Kiểm tra rò rỉ n-gram trên tập test
    leakage_stats = compute_test_train_trigram_leakage(
        test_records=test_records,
        train_records=train_records,
        threshold=TRIGRAM_SIMILARITY_THRESHOLD,
    )
    assert len(leakage_stats.flagged_samples) == 0, (
        f"Phát hiện {len(leakage_stats.flagged_samples)} câu kiểm thử có nguy cơ near-duplicate > 0.8"
    )
    assert leakage_stats.max_max_sim < 0.8, (
        f"Độ tương đồng cực đại đạt {leakage_stats.max_max_sim:.4f}, vượt ngưỡng an toàn 0.8"
    )

    # Kiểm tra tệp báo cáo tồn tại
    assert DEFAULT_DATA_REPORT_PATH.exists(), f"Chưa sinh tệp {DEFAULT_DATA_REPORT_PATH}"
    content = DEFAULT_DATA_REPORT_PATH.read_text(encoding="utf-8")
    assert "VIETNAMESE LATEX KNOWLEDGE ASSISTANT - DATASET LEAKAGE REPORT" in content
    assert "KẾT LUẬN TOÀN VẸN: [ĐẠT CHUẨN] KHÔNG PHÁT HIỆN TRÙNG LẶP TUYỆT ĐỐI" in content
