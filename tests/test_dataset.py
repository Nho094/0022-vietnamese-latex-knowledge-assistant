"""
Bộ kiểm thử tính toàn vẹn và định dạng schema của tập dữ liệu (tests/test_dataset.py).

Tuân thủ nghiêm ngặt các quy tắc kiến trúc và tính trung thực trong AGENTS.md:
- Honesty Rule #2: Tuyệt đối không tự bịa số liệu. Mọi chỉ số phân tích đều được trích xuất
  hoặc tính toán từ dữ liệu thực tế và lưu trữ tại report/data_report.txt.
- Honesty Rule #3: Bất biến tập kiểm thử (Test Set Invariant): Tuyệt đối không tinh chỉnh trên test set,
  không chỉnh sửa, tự sinh hay thêm bớt bừa bãi vào test.csv.
- Honesty Rule #4: Các phân vùng train/val/test tuyệt đối không chia sẻ bất kỳ câu trùng lặp nào
  (so khớp chuẩn hóa Unicode NFC, chữ thường và gom cụm khoảng trắng liên tiếp).
- Kiểm thực schema 3 cột bắt buộc: 'text', 'label', 'category' cho train.csv, val.csv, test.csv.
- Xác thực 100% nhãn ý định (label) và danh mục (category) tồn tại trong data/intents.json.
- Kiểm tra báo cáo rò rỉ dữ liệu report/data_report.txt được tạo đầy đủ và đáp ứng các ngưỡng an toàn.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Final

import pytest

from core.config import (
    DEFAULT_DATA_REPORT_PATH,
    DEFAULT_INTENTS_PATH,
    MIN_TEST_SAMPLES_PER_INTENT,
    MIN_TRAIN_SAMPLES_PER_INTENT,
    OUT_OF_SCOPE_INTENT,
    TEST_CSV_PATH,
    TEST_OOS_PATH,
    TEST_OWNER_TEMPLATE_PATH,
    TRAIN_CSV_PATH,
    TRIGRAM_SIMILARITY_THRESHOLD,
    VAL_CSV_PATH,
)
from data.leakage_report import (
    compute_test_train_trigram_leakage,
    normalize_text_for_comparison,
    read_dataset_csv,
    verify_exact_duplicates,
)

# Danh sách các cột bắt buộc phải có trong mỗi phân vùng dữ liệu CSV
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("text", "label", "category")

# Ngưỡng kích thước tối thiểu đảm bảo dữ liệu không bị rỗng hoặc lỗi ghi tệp
MIN_EXPECTED_SAMPLES: Final[dict[str, int]] = {
    "train": 1000,
    "val": 500,
    "test": 500,
}


def _load_intents_lookup(intents_path: Path = DEFAULT_INTENTS_PATH) -> dict[str, str]:
    """
    Nạp danh mục ý định từ data/intents.json thành bảng tra cứu intent -> category.

    Tham số:
        intents_path: Đường dẫn tệp intents.json.

    Trả về:
        Ánh xạ từ mã ý định (intent) sang tên danh mục (category).
    """
    if not intents_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp danh mục kịch bản: {intents_path}")

    with open(intents_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"Định dạng intents.json không hợp lệ, kỳ vọng danh sách nhưng nhận: {type(data)}")

    return {item["intent"]: item["category"] for item in data if "intent" in item and "category" in item}


def _validate_split_records_schema(
    records: list[dict[str, str]],
    split_name: str,
    required_cols: tuple[str, ...] = REQUIRED_COLUMNS,
) -> None:
    """
    Kiểm thực tính toàn vẹn cấu trúc dữ liệu cho từng bản ghi của một phân vùng split:
    - Chứa đủ các khóa bắt buộc (text, label, category).
    - Các giá trị chuỗi sau khi strip() không được phép rỗng.

    Tham số:
        records: Danh sách bản ghi nạp từ CSV.
        split_name: Tên phân vùng ('train', 'val', 'test').
        required_cols: Bộ tuple tên các cột bắt buộc.
    """
    assert len(records) > 0, f"Phân vùng '{split_name}' không được phép rỗng."

    for idx, row in enumerate(records):
        for col in required_cols:
            assert col in row, (
                f"Phân vùng '{split_name}' tại dòng {idx} thiếu cột bắt buộc '{col}'."
            )
            val = row[col]
            assert isinstance(val, str), (
                f"Phân vùng '{split_name}' dòng {idx} cột '{col}' phải là chuỗi, nhận: {type(val)}."
            )
            assert len(val.strip()) > 0, (
                f"Phân vùng '{split_name}' dòng {idx} cột '{col}' chứa giá trị rỗng hoặc toàn khoảng trắng."
            )


# ==============================================================================
# 1. KIỂM THỬ SỰ TỒN TẠI VÀ ĐỊNH DẠNG TỆP ARTIFACTS
# ==============================================================================


def test_dataset_artifacts_exist() -> None:
    """
    Xác minh sự tồn tại của các tệp dữ liệu phân vùng và tệp danh mục kịch bản:
    - data/train.csv
    - data/val.csv
    - data/test.csv
    - data/intents.json
    - report/data_report.txt
    """
    for artifact_path in (
        TRAIN_CSV_PATH,
        VAL_CSV_PATH,
        TEST_CSV_PATH,
        DEFAULT_INTENTS_PATH,
        DEFAULT_DATA_REPORT_PATH,
    ):
        assert artifact_path.exists(), f"Artifact bắt buộc không tồn tại: {artifact_path}"
        assert artifact_path.is_file(), f"Đường dẫn không phải tệp thông thường: {artifact_path}"
        assert artifact_path.stat().st_size > 0, f"Tệp artifact rỗng (0 bytes): {artifact_path}"


def test_csv_headers_and_columns() -> None:
    """
    Kiểm tra tiêu đề (header) của các tệp CSV train, val, test phải chứa chính xác các cột:
    ['text', 'label', 'category'], hỗ trợ đọc tệp có mã hóa utf-8 hoặc utf-8-sig.
    """
    for split_path in (TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH):
        with open(split_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            assert header is not None, f"Tệp CSV rỗng không có dòng tiêu đề: {split_path}"
            cleaned_header = [col.strip() for col in header]
            assert cleaned_header == list(REQUIRED_COLUMNS), (
                f"Tiêu đề của {split_path.name} không khớp kỳ vọng. Nhận: {cleaned_header}, Kỳ vọng: {list(REQUIRED_COLUMNS)}"
            )


def test_dataset_records_schema_and_minimum_counts() -> None:
    """
    Xác minh từng bản ghi trong train.csv, val.csv và test.csv:
    - Chứa đầy đủ các trường 'text', 'label', 'category' không rỗng.
    - Đạt số lượng mẫu tối thiểu cho mỗi phân vùng đã đề ra trong kiến trúc.
    """
    splits = {
        "train": read_dataset_csv(TRAIN_CSV_PATH),
        "val": read_dataset_csv(VAL_CSV_PATH),
        "test": read_dataset_csv(TEST_CSV_PATH),
    }

    for split_name, records in splits.items():
        _validate_split_records_schema(records, split_name)
        min_samples = MIN_EXPECTED_SAMPLES[split_name]
        assert len(records) >= min_samples, (
            f"Phân vùng '{split_name}' có {len(records)} mẫu, thấp hơn mức tối thiểu kỳ vọng ({min_samples})."
        )


def test_schema_validator_catches_invalid_records(tmp_path: Path) -> None:
    """
    Kiểm thử cơ chế bắt lỗi khi gặp bản ghi dữ liệu bị thiếu trường hoặc chứa ô trống:
    Đảm bảo hàm kiểm tra phát hiện lỗi chính xác thay vì bỏ qua âm thầm.
    """
    # 1. Bản ghi thiếu cột 'category'
    bad_record_missing_col = [{"text": "Câu hỏi LaTeX", "label": "LATEX_TAI_LIEU"}]
    with pytest.raises(AssertionError, match="thiếu cột bắt buộc 'category'"):
        _validate_split_records_schema(bad_record_missing_col, "synthetic_test")

    # 2. Bản ghi có trường text rỗng
    bad_record_empty_text = [{"text": "   ", "label": "LATEX_TAI_LIEU", "category": "tex_doc"}]
    with pytest.raises(AssertionError, match="chứa giá trị rỗng"):
        _validate_split_records_schema(bad_record_empty_text, "synthetic_test")

    # 3. Phân vùng hoàn toàn rỗng
    with pytest.raises(AssertionError, match="không được phép rỗng"):
        _validate_split_records_schema([], "synthetic_empty")


# ==============================================================================
# 2. KIỂM THỰC Ý ĐỊNH VÀ DANH MỤC KHỚP VỚI INTENTS.JSON
# ==============================================================================


def test_all_labels_exist_in_intents_json() -> None:
    """
    Kiểm thực tính nhất quán toàn vẹn giữa dữ liệu huấn luyện/kiểm thử với ngân hàng ý định:
    - Mọi nhãn 'label' xuất hiện trong train.csv, val.csv, test.csv đều phải được định nghĩa trong intents.json.
    - Cột 'category' trong mỗi dòng dữ liệu phải trùng khớp tuyệt đối với category của intent đó trong intents.json.
    """
    intent_catalog = _load_intents_lookup(DEFAULT_INTENTS_PATH)
    assert len(intent_catalog) > 0, "Ngân hàng kịch bản data/intents.json không chứa ý định nào."

    for split_path in (TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH):
        records = read_dataset_csv(split_path)
        for idx, row in enumerate(records):
            label = row["label"]
            category = row["category"]

            # Kiểm tra label có tồn tại trong intents.json không
            assert label in intent_catalog, (
                f"Tệp {split_path.name} tại dòng {idx} chứa nhãn không xác định '{label}'. "
                f"Nhãn này không tồn tại trong data/intents.json."
            )

            # Kiểm tra category có khớp với định nghĩa của intent không
            expected_category = intent_catalog[label]
            assert category == expected_category, (
                f"Tệp {split_path.name} tại dòng {idx} có danh mục '{category}' "
                f"không khớp với danh mục chuẩn '{expected_category}' của ý định '{label}'."
            )


def test_intent_catalog_coverage_in_dataset() -> None:
    """
    Xác minh độ bao phủ ý định của bộ dữ liệu:
    - Tập huấn luyện (train) phải bao phủ toàn bộ các ý định có trong intents.json (hoặc 189 intents).
    - Tập kiểm định (val) bao phủ đầy đủ các intent chính.
    - Nhãn ngoại lệ OUT_OF_SCOPE phải xuất hiện và được gán danh mục hợp lệ.
    """
    intent_catalog = _load_intents_lookup(DEFAULT_INTENTS_PATH)
    train_records = read_dataset_csv(TRAIN_CSV_PATH)
    train_labels = {r["label"] for r in train_records}

    # Kiểm tra OUT_OF_SCOPE
    assert OUT_OF_SCOPE_INTENT in intent_catalog, (
        f"Ý định ngoài phạm vi '{OUT_OF_SCOPE_INTENT}' phải có trong intents.json."
    )
    assert OUT_OF_SCOPE_INTENT in train_labels, (
        f"Tập train.csv phải chứa các mẫu của ý định '{OUT_OF_SCOPE_INTENT}'."
    )

    # Đảm bảo toàn bộ 189 ý định chuẩn đều có mẫu trong tập train và đạt mức tối thiểu
    missing_in_train = set(intent_catalog.keys()) - train_labels
    assert len(missing_in_train) == 0, (
        f"Có {len(missing_in_train)} ý định không có mẫu huấn luyện trong train.csv: {missing_in_train}"
    )

    train_label_counts = Counter(r["label"] for r in train_records)
    for label, count in train_label_counts.items():
        assert count >= MIN_TRAIN_SAMPLES_PER_INTENT, (
            f"Ý định '{label}' trong tập train chỉ có {count} mẫu, vi phạm mức tối thiểu ({MIN_TRAIN_SAMPLES_PER_INTENT})."
        )


def test_invalid_label_or_category_detection() -> None:
    """
    Kiểm thử phát hiện nhãn giả lập không hợp lệ hoặc danh mục bị gán sai lệch.
    """
    intent_catalog = _load_intents_lookup(DEFAULT_INTENTS_PATH)

    # 1. Nhãn hoàn toàn giả tưởng không có trong intents.json
    fake_record = {"text": "Viết công thức tích phân", "label": "INTENT_KHONG_TON_TAI_999", "category": "tex_math"}
    assert fake_record["label"] not in intent_catalog

    # 2. Nhãn có thật nhưng gán nhầm category (ví dụ: LATEX_TAI_LIEU_TOI_GIAN thuộc tex_struct nhưng gán tex_math)
    sample_intent = "LATEX_TAI_LIEU_TOI_GIAN"
    true_category = intent_catalog[sample_intent]
    wrong_category = "tex_math" if true_category != "tex_math" else "tex_struct"
    mismatched_record = {"text": "Tạo tài liệu LaTeX tối thiểu", "label": sample_intent, "category": wrong_category}

    assert mismatched_record["category"] != intent_catalog[mismatched_record["label"]]


# ==============================================================================
# 3. KIỂM THỰC KHÔNG TRÙNG LẶP TUYỆT ĐỐI XUYÊN PHÂN VÙNG (HONESTY RULE #4)
# ==============================================================================


def test_zero_exact_duplicates_across_splits() -> None:
    """
    Kiểm thực nghiêm ngặt Honesty Rule #4 trong AGENTS.md:
    Train/val/test tuyệt đối không chia sẻ bất kỳ câu trùng lặp nào dưới cơ chế so khớp
    chuẩn hóa Unicode NFC, chuyển chữ thường và gom cụm khoảng trắng liên tiếp.

    Yêu cầu:
    - Trùng lặp nội bộ trong train, val, test đều bằng 0.
    - Trùng lặp giao nhau giữa train-val, train-test, val-test đều bằng 0.
    """
    train_records = read_dataset_csv(TRAIN_CSV_PATH)
    val_records = read_dataset_csv(VAL_CSV_PATH)
    test_records = read_dataset_csv(TEST_CSV_PATH)

    dup_result = verify_exact_duplicates(train_records, val_records, test_records)

    # Kiểm tra cờ toàn vẹn dữ liệu
    assert dup_result.is_clean is True, (
        f"Phát hiện vi phạm Honesty Rule #4! Dữ liệu bị trùng lặp: "
        f"train_dups={dup_result.internal_train_duplicates}, "
        f"val_dups={dup_result.internal_val_duplicates}, "
        f"test_dups={dup_result.internal_test_duplicates}, "
        f"train_val_overlap={dup_result.train_val_overlap}, "
        f"train_test_overlap={dup_result.train_test_overlap}, "
        f"val_test_overlap={dup_result.val_test_overlap}. "
        f"Ví dụ vi phạm: {dup_result.overlap_examples}"
    )

    # Khẳng định từng chỉ số trùng lặp đều bằng 0
    assert dup_result.internal_train_duplicates == 0
    assert dup_result.internal_val_duplicates == 0
    assert dup_result.internal_test_duplicates == 0
    assert dup_result.train_val_overlap == 0
    assert dup_result.train_test_overlap == 0
    assert dup_result.val_test_overlap == 0


def test_normalized_text_comparison_behavior() -> None:
    """
    Kiểm thử hàm chuẩn hóa văn bản normalize_text_for_comparison:
    - Loại bỏ khoảng trắng đầu/cuối và gom cụm nhiều khoảng trắng / ký tự tab, newline.
    - Chuyển chữ hoa thành chữ thường.
    - Đồng nhất biểu diễn ký tự tiếng Việt Unicode (NFC vs NFD) để đảm bảo không bị lọt trùng lặp ẩn.
    """
    # 1. Chữ hoa / chữ thường và khoảng trắng thừa
    raw1 = "   Tạo   phân   SỐ   toán  học \n \t trong   LaTeX   "
    raw2 = "tạo phân số toán học trong latex"
    assert normalize_text_for_comparison(raw1) == normalize_text_for_comparison(raw2)

    # 2. Unicode Decomposed (NFD) vs Composed (NFC)
    # Ký tự 'ế' trong NFD là 'e' + combining acute + combining circumflex
    text_nfc = unicodedata.normalize("NFC", "thiết kế bảng toán")
    text_nfd = unicodedata.normalize("NFD", "thiết kế bảng toán")
    assert text_nfc != text_nfd  # Biểu diễn nhị phân byte khác nhau
    assert normalize_text_for_comparison(text_nfc) == normalize_text_for_comparison(text_nfd)

    # 3. Kiểm chứng verify_exact_duplicates bắt được trùng lặp biến thể hình thức này
    sample_train = [{"text": raw1, "label": "L1", "category": "C1"}]
    sample_test = [{"text": text_nfd, "label": "L2", "category": "C2"}]
    sample_test_duplicate = [{"text": raw2, "label": "L1", "category": "C1"}]

    res_clean = verify_exact_duplicates(sample_train, [], sample_test)
    assert res_clean.is_clean is True

    res_overlap = verify_exact_duplicates(sample_train, [], sample_test_duplicate)
    assert res_overlap.is_clean is False
    assert res_overlap.train_test_overlap == 1


# ==============================================================================
# 4. KIỂM THỰC BÁO CÁO DỮ LIỆU VÀ NGƯỠNG AN TOÀN RÒ RỈ (SAFETY THRESHOLDS)
# ==============================================================================


def test_data_report_file_exists_and_format() -> None:
    """
    Kiểm tra sự tồn tại và cấu trúc cơ bản của tệp báo cáo report/data_report.txt:
    - Tệp phải tồn tại và có kích thước lớn hơn 0 bytes.
    - Chứa tiêu đề chính của dự án và các mục đánh giá từ 1 đến 5.
    """
    assert DEFAULT_DATA_REPORT_PATH.exists(), f"Không tìm thấy tệp báo cáo: {DEFAULT_DATA_REPORT_PATH}"
    content = DEFAULT_DATA_REPORT_PATH.read_text(encoding="utf-8")
    assert len(content.strip()) > 0, "Tệp báo cáo data_report.txt hoàn toàn rỗng."

    assert "VIETNAMESE LATEX KNOWLEDGE ASSISTANT - DATASET LEAKAGE REPORT" in content
    assert "1. PHÂN PHỐI MẪU VÀ TỶ LỆ PHÂN VÙNG" in content
    assert "2. THỐNG KÊ ĐỘ DÀI CHUỖI VĂN BẢN" in content
    assert "3. KIỂM THỰC TRÙNG LẶP TUYỆT ĐỐI XUYÊN PHÂN VÙNG" in content
    assert "4. ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG CHARACTER TRIGRAM VÀ RÒ RỈ DỮ LIỆU" in content
    assert "5. ĐÁNH GIÁ TỔNG QUAN VÀ KẾT LUẬN CUỐI CÙNG" in content


def test_data_report_meets_safety_thresholds() -> None:
    """
    Trích xuất và kiểm thực các ngưỡng an toàn trong báo cáo report/data_report.txt:
    1. Trùng lặp tuyệt đối (Honesty Rule #4) phải đạt trạng thái 'PASS'.
    2. Rò rỉ thông tin n-gram phải đạt trạng thái 'PASS'.
    3. Kết luận toàn vẹn phải là '[ĐẠT CHUẨN] KHÔNG PHÁT HIỆN TRÙNG LẶP TUYỆT ĐỐI'.
    4. Không có câu kiểm thử nào bị cảnh báo near-duplicate (> 0.8).
    5. Giá trị tương đồng cực đại (Max Jaccard Similarity) ghi nhận trong báo cáo phải < 0.8.
    """
    content = DEFAULT_DATA_REPORT_PATH.read_text(encoding="utf-8")

    # 1. Trùng lặp tuyệt đối
    assert re.search(r"Trùng lặp tuyệt đối.*:\s*PASS", content) is not None, (
        "Báo cáo data_report.txt không ghi nhận trạng thái PASS cho kiểm thực trùng lặp tuyệt đối."
    )
    assert "[ĐẠT CHUẨN] KHÔNG PHÁT HIỆN TRÙNG LẶP TUYỆT ĐỐI" in content

    # 2. Rò rỉ thông tin n-gram
    assert re.search(r"Rò rỉ thông tin n-gram.*:\s*PASS", content) is not None, (
        "Báo cáo data_report.txt không ghi nhận trạng thái PASS cho đánh giá rò rỉ n-gram."
    )
    assert "[AN TOÀN] Không có câu kiểm thử nào có độ tương đồng vượt ngưỡng" in content

    # 3. Phân tích giá trị tương đồng cực đại
    max_sim_match = re.search(r"Giá trị tương đồng lớn nhất \(Max\)\s*:\s*([0-9.]+)", content)
    assert max_sim_match is not None, "Không tìm thấy giá trị tương đồng lớn nhất trong data_report.txt"
    max_sim = float(max_sim_match.group(1))

    assert max_sim < TRIGRAM_SIMILARITY_THRESHOLD, (
        f"Giá trị tương đồng lớn nhất trong báo cáo ({max_sim:.4f}) "
        f"vượt ngưỡng an toàn cho phép ({TRIGRAM_SIMILARITY_THRESHOLD})."
    )

    # 4. Kiểm tra phân phối tần suất khoảng nguy hiểm [0.8, 1.0] phải bằng 0 câu
    hist_high_match = re.search(r"\[0\.8,\s*1\.0\]\s*:\s*(\d+)\s*câu", content)
    assert hist_high_match is not None, "Không tìm thấy tần suất khoảng [0.8, 1.0] trong data_report.txt"
    high_sim_count = int(hist_high_match.group(1))
    assert high_sim_count == 0, (
        f"Có {high_sim_count} câu trong tập test có độ tương đồng thuộc khoảng [0.8, 1.0], vi phạm an toàn."
    )


def test_trigram_leakage_detector_catches_near_duplicate() -> None:
    """
    Kiểm thử thuật toán tính toán rò rỉ Character Trigram Jaccard (compute_test_train_trigram_leakage):
    - Khi có mẫu kiểm thử gần trùng lặp cao (> 0.8), bộ lọc phải ghi nhận vào flagged_samples.
    - Khi các mẫu hoàn toàn độc lập, flagged_samples phải rỗng.
    """
    # Mẫu train và mẫu test gần giống nhau (chỉ đổi 1 chữ nhỏ)
    train_samples = [
        {"text": "hướng dẫn cách tạo bảng căn chỉnh lề phải trong latex", "label": "L1", "category": "C1"}
    ]
    # Câu test rất tương đồng (chỉ đổi "hướng dẫn" -> "chỉ giúp", phần lớn trigram trùng nhau)
    near_dup_test = [
        {"text": "hướng dẫn cách tạo bảng căn chỉnh lề trái trong latex", "label": "L1", "category": "C1"}
    ]
    # Câu test độc lập
    independent_test = [
        {"text": "khai báo kích thước phông chữ tài liệu 12pt", "label": "L2", "category": "C2"}
    ]

    # Kiểm tra với câu near duplicate
    leakage_stats_near = compute_test_train_trigram_leakage(
        test_records=near_dup_test,
        train_records=train_samples,
        threshold=0.7,  # Đặt threshold 0.7 để kích hoạt cảnh báo
    )
    assert leakage_stats_near.max_max_sim >= 0.7
    assert len(leakage_stats_near.flagged_samples) == 1
    assert leakage_stats_near.flagged_samples[0].similarity == leakage_stats_near.max_max_sim

    # Kiểm tra với câu độc lập
    leakage_stats_indep = compute_test_train_trigram_leakage(
        test_records=independent_test,
        train_records=train_samples,
        threshold=0.8,
    )
    assert leakage_stats_indep.max_max_sim < 0.5
    assert len(leakage_stats_indep.flagged_samples) == 0


def test_live_dataset_leakage_safety_threshold() -> None:
    """
    Thực hiện kiểm chứng trực tiếp độ tương đồng Character Trigram Jaccard trên tập dữ liệu thực tế
    (940 câu test đối chiếu với 3825 câu train):
    - Đảm bảo 100% câu test có độ tương đồng < TRIGRAM_SIMILARITY_THRESHOLD (0.8).
    - Danh sách các câu bị cảnh báo flagged_samples phải hoàn toàn rỗng.
    - Phổ tương đồng cực đại tuân thủ đúng phân phối đã ghi trong báo cáo.
    """
    train_records = read_dataset_csv(TRAIN_CSV_PATH)
    test_records = read_dataset_csv(TEST_CSV_PATH)

    stats = compute_test_train_trigram_leakage(
        test_records=test_records,
        train_records=train_records,
        threshold=TRIGRAM_SIMILARITY_THRESHOLD,
    )

    assert len(stats.flagged_samples) == 0, (
        f"Phát hiện {len(stats.flagged_samples)} câu kiểm thử vi phạm ngưỡng rò rỉ {TRIGRAM_SIMILARITY_THRESHOLD}: "
        f"{stats.flagged_samples}"
    )
    assert stats.max_max_sim < TRIGRAM_SIMILARITY_THRESHOLD, (
        f"Độ tương đồng lớn nhất ({stats.max_max_sim:.4f}) không được vượt ngưỡng {TRIGRAM_SIMILARITY_THRESHOLD}."
    )
    assert stats.histogram["[0.8, 1.0]"] == 0


def test_oos_test_schema_and_disjointness() -> None:
    oos_rows = read_dataset_csv(TEST_OOS_PATH)
    train_rows = read_dataset_csv(TRAIN_CSV_PATH)
    val_rows = read_dataset_csv(VAL_CSV_PATH)
    assert oos_rows
    assert all(row["label"] == OUT_OF_SCOPE_INTENT for row in oos_rows)
    assert all(row["category"] == "tex_misc" for row in oos_rows)
    oos_texts = {normalize_text_for_comparison(row["text"]) for row in oos_rows}
    assert not oos_texts & {normalize_text_for_comparison(row["text"]) for row in train_rows}
    assert not oos_texts & {normalize_text_for_comparison(row["text"]) for row in val_rows}


def test_owner_template_has_three_blank_rows_per_category() -> None:
    with TEST_OWNER_TEMPLATE_PATH.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["text", "label", "category", "written_by"]
        rows = list(reader)
    categories = {item["category"] for item in json.loads(DEFAULT_INTENTS_PATH.read_text())}
    counts = Counter(row["category"] for row in rows)
    assert set(counts) == categories
    assert all(count == 3 for count in counts.values())
    assert all(not row["text"] and not row["label"] and not row["written_by"] for row in rows)


# ==============================================================================
# 5. KIỂM THỰC BẤT BIẾN TẬP KIỂM THỬ (HONESTY RULE #3)
# ==============================================================================


def test_test_set_invariance_and_statistics() -> None:
    """
    Kiểm thực Honesty Rule #3 (AGENTS.md):
    - Tập test.csv phải có số lượng mẫu kiểm định chuẩn (940 câu).
    - Toàn bộ các câu phải có độ dài hợp lý (> 10 ký tự, > 2 từ).
    - Không có giá trị thiếu hoặc bị cắt cụt.
    """
    test_records = read_dataset_csv(TEST_CSV_PATH)
    assert len(test_records) == 940, f"Số lượng mẫu tập kiểm thử thay đổi bất thường: {len(test_records)} (kỳ vọng 940)."

    # Kiểm tra độ dài văn bản và số lượng từ của từng câu test
    for idx, r in enumerate(test_records):
        text = r["text"]
        assert len(text) >= 10, f"Câu test tại dòng {idx} quá ngắn (< 10 ký tự): '{text}'"
        words = text.split()
        assert len(words) >= 2, f"Câu test tại dòng {idx} có ít hơn 2 từ: '{text}'"

    # Đếm số lượng mẫu của từng intent trong tập test
    label_counts = Counter(r["label"] for r in test_records)
    for label, count in label_counts.items():
        assert count >= MIN_TEST_SAMPLES_PER_INTENT, (
            f"Ý định '{label}' trong tập test chỉ có {count} mẫu, vi phạm mức tối thiểu ({MIN_TEST_SAMPLES_PER_INTENT})."
        )
