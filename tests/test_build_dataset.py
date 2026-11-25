"""
Kiểm thử tự động cho module tổng hợp và kiểm định tập dữ liệu (data/build_dataset.py).

Tuân thủ nghiêm ngặt các quy tắc kiến trúc và tính trung thực trong AGENTS.md:
- Honesty Rule #1: Hệ thống kịch bản đơn giản hóa lấy cảm hứng từ Retrieve-and-Fill (RAF).
- Honesty Rule #3: Bất biến tập kiểm thử: Tập test không được phép chứa câu tổng hợp tự động,
  chỉ bao gồm các câu do chủ sở hữu viết (owner-written).
- Honesty Rule #4: Kiểm tra không trùng lặp câu xuyên tập (train/val/test) với chuẩn hóa lowercasing
  và collapsing whitespace.
- Kiểm thử các tình huống dừng khẩn cấp (abort) khi phát hiện trùng lặp câu test với train/val.
- Kiểm thử cơ chế phát cảnh báo khi số mẫu của intent < 10 (train) hoặc < 5 (test).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from core.config import (
    DEFAULT_INTENTS_PATH,
    OUT_OF_SCOPE_INTENT,
    RAW_SENTENCES_PATH,
)
from data.build_dataset import (
    DatasetRecord,
    compile_and_validate_dataset,
    format_summary_table,
    load_intents_catalog,
    load_out_of_scope_samples,
    load_owner_sentences,
    load_synthetic_sentences,
    main,
    normalize_for_dedup,
    validate_split_integrity,
)


def test_normalize_for_dedup() -> None:
    """Kiểm tra hàm chuẩn hóa chuỗi: chuyển chữ thường, gom cụm khoảng trắng và bảo toàn NFC."""
    # 1. Chuyển chữ thường và gom khoảng trắng thừa
    raw1 = "   Tạo   phân   SỐ   toán  học   "
    expected1 = "tạo phân số toán học"
    assert normalize_for_dedup(raw1) == expected1

    # 2. Xử lý ký tự ngắt dòng và tab
    raw2 = "Tạo\tphân số\ntrong\r\nLaTeX"
    expected2 = "tạo phân số trong latex"
    assert normalize_for_dedup(raw2) == expected2

    # 3. Chuỗi rỗng
    assert normalize_for_dedup("   \t  \n  ") == ""


def test_load_intents_catalog() -> None:
    """Kiểm tra nạp danh mục ý định từ data/intents.json chuẩn 189 intents."""
    intent_map = load_intents_catalog(DEFAULT_INTENTS_PATH)
    assert len(intent_map) == 189
    assert "LATEX_TAI_LIEU_TOI_GIAN" in intent_map
    assert intent_map["LATEX_TAI_LIEU_TOI_GIAN"] == "tex_struct"
    assert OUT_OF_SCOPE_INTENT in intent_map
    assert intent_map[OUT_OF_SCOPE_INTENT] == "tex_misc"


def test_load_intents_catalog_invalid(tmp_path: Path) -> None:
    """Kiểm tra xử lý lỗi khi tệp danh mục không tồn tại hoặc định dạng sai."""
    non_existent = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_intents_catalog(non_existent)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"intent": "single_dict"}', encoding="utf-8")
    with pytest.raises(ValueError, match="kỳ vọng danh sách"):
        load_intents_catalog(bad_json)


def test_load_owner_sentences_valid() -> None:
    """Kiểm tra nạp các câu mẫu tác giả viết từ data/raw_sentences.csv."""
    intent_map = load_intents_catalog(DEFAULT_INTENTS_PATH)
    records = load_owner_sentences(RAW_SENTENCES_PATH, valid_intents=intent_map)

    assert len(records) == 3760
    splits = {r.split for r in records}
    assert splits == {"train", "val", "test"}

    # Kiểm tra toàn bộ mẫu đều có origin là owner_written
    for r in records:
        assert r.origin == "owner_written"
        assert r.label in intent_map
        assert r.category == intent_map[r.label]


def test_load_owner_sentences_invalid(tmp_path: Path) -> None:
    """Kiểm tra kiểm thực phân vùng split và nhãn intent bất hợp lệ trong CSV."""
    intent_map = {"INTENT_A": "cat_a"}

    # Tệp chứa nhãn lạ
    bad_label_csv = tmp_path / "bad_label.csv"
    bad_label_csv.write_text("text,label,split\nXin chào,UNKNOWN_INTENT,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="không tồn tại trong danh mục"):
        load_owner_sentences(bad_label_csv, valid_intents=intent_map)

    # Tệp chứa split lạ
    bad_split_csv = tmp_path / "bad_split.csv"
    bad_split_csv.write_text("text,label,split\nXin chào,INTENT_A,unsupported_split\n", encoding="utf-8")
    with pytest.raises(ValueError, match="không hợp lệ"):
        load_owner_sentences(bad_split_csv, valid_intents=intent_map)


def test_load_synthetic_sentences() -> None:
    """Kiểm tra nạp câu huấn luyện tổng hợp cho phân vùng train."""
    intent_map = load_intents_catalog(DEFAULT_INTENTS_PATH)
    # Lấy mẫu nhỏ 2 mẫu/intent để kiểm tra nhanh
    records = load_synthetic_sentences(
        intents_path=DEFAULT_INTENTS_PATH,
        seed=42,
        samples_per_intent=2,
        valid_intents=intent_map,
    )
    assert len(records) > 0
    for r in records:
        assert r.origin == "synthetic"
        assert r.split == "train"
        assert r.label in intent_map


def test_load_out_of_scope_samples() -> None:
    """Kiểm tra nạp dữ liệu ngoại phạm vi OUT_OF_SCOPE."""
    intent_map = load_intents_catalog(DEFAULT_INTENTS_PATH)
    records = load_out_of_scope_samples(include_test=False, valid_intents=intent_map)

    assert len(records) > 0
    splits = {r.split for r in records}
    # Mặc định test không chứa OOS để đảm bảo độc quyền owner-written
    assert "test" not in splits
    assert "train" in splits
    assert "val" in splits

    for r in records:
        assert r.label == OUT_OF_SCOPE_INTENT
        assert r.category == "tex_misc"


def test_validate_split_integrity_success() -> None:
    """Kiểm tra kiểm định phân vùng thành công khi không có trùng lặp."""
    valid_intents = {"INTENT_A": "cat_a", "INTENT_B": "cat_b"}

    train_recs = [
        DatasetRecord(text=f"câu train A {i}", label="INTENT_A", category="cat_a", split="train", origin="owner_written")
        for i in range(10)
    ] + [
        DatasetRecord(text=f"câu train B {i}", label="INTENT_B", category="cat_b", split="train", origin="owner_written")
        for i in range(10)
    ]

    val_recs = [
        DatasetRecord(text=f"câu val A {i}", label="INTENT_A", category="cat_a", split="val", origin="owner_written")
        for i in range(5)
    ]

    test_recs = [
        DatasetRecord(text=f"câu test A {i}", label="INTENT_A", category="cat_a", split="test", origin="owner_written")
        for i in range(5)
    ]

    counts, warnings_list = validate_split_integrity(
        train_records=train_recs,
        val_records=val_recs,
        test_records=test_recs,
        valid_intents=valid_intents,
    )

    assert counts["INTENT_A"]["train"] == 10
    assert counts["INTENT_A"]["val"] == 5
    assert counts["INTENT_A"]["test"] == 5

    # INTENT_B thiếu val và test nên sẽ có cảnh báo test < 5
    assert any("INTENT_B" in w and "test" in w for w in warnings_list)


def test_abort_on_test_duplicate_train() -> None:
    """Kiểm tra dừng thực thi (abort) khi phát hiện câu test trùng lặp với train."""
    valid_intents = {"INTENT_A": "cat_a"}

    train_recs = [
        DatasetRecord(text="Tạo tài liệu LaTeX tối thiểu.", label="INTENT_A", category="cat_a", split="train", origin="owner_written")
    ]
    val_recs = [
        DatasetRecord(text="Một câu val hoàn toàn khác.", label="INTENT_A", category="cat_a", split="val", origin="owner_written")
    ]
    # Câu test trùng với train (khác biệt chữ hoa và khoảng trắng)
    test_recs = [
        DatasetRecord(text="  tạo tài liệu  latex tối thiểu.  ", label="INTENT_A", category="cat_a", split="test", origin="owner_written")
    ]

    with pytest.raises(ValueError, match="DỪNG THỰC THI \\(ABORT\\): Phát hiện .* câu kiểm thử .* trùng lặp với tập train"):
        validate_split_integrity(train_recs, val_recs, test_recs, valid_intents)


def test_abort_on_test_duplicate_val() -> None:
    """Kiểm tra dừng thực thi (abort) khi phát hiện câu test trùng lặp với val."""
    valid_intents = {"INTENT_A": "cat_a"}

    train_recs = [
        DatasetRecord(text="Câu train riêng biệt.", label="INTENT_A", category="cat_a", split="train", origin="owner_written")
    ]
    val_recs = [
        DatasetRecord(text="Câu mẫu trong val.", label="INTENT_A", category="cat_a", split="val", origin="owner_written")
    ]
    test_recs = [
        DatasetRecord(text="câu mẫu trong val.", label="INTENT_A", category="cat_a", split="test", origin="owner_written")
    ]

    with pytest.raises(ValueError, match="DỪNG THỰC THI \\(ABORT\\): Phát hiện .* câu kiểm thử .* trùng lặp với tập val"):
        validate_split_integrity(train_recs, val_recs, test_recs, valid_intents)


def test_abort_on_val_duplicate_train() -> None:
    """Kiểm tra dừng thực thi (abort) khi phát hiện câu val trùng lặp với train."""
    valid_intents = {"INTENT_A": "cat_a"}

    train_recs = [
        DatasetRecord(text="Cú pháp tạo bảng trong LaTeX", label="INTENT_A", category="cat_a", split="train", origin="owner_written")
    ]
    val_recs = [
        DatasetRecord(text="cú pháp tạo bảng trong latex", label="INTENT_A", category="cat_a", split="val", origin="owner_written")
    ]
    test_recs = [
        DatasetRecord(text="Câu test hoàn toàn độc lập", label="INTENT_A", category="cat_a", split="test", origin="owner_written")
    ]

    with pytest.raises(ValueError, match="DỪNG THỰC THI \\(ABORT\\): Phát hiện .* câu kiểm định .* trùng lặp với tập train"):
        validate_split_integrity(train_recs, val_recs, test_recs, valid_intents)


def test_abort_on_non_owner_test_sentence() -> None:
    """Kiểm tra ràng buộc câu test phải độc quyền do chủ sở hữu viết (owner-written)."""
    valid_intents = {"INTENT_A": "cat_a"}

    train_recs = [
        DatasetRecord(text="Câu train.", label="INTENT_A", category="cat_a", split="train", origin="owner_written")
    ]
    val_recs = [
        DatasetRecord(text="Câu val.", label="INTENT_A", category="cat_a", split="val", origin="owner_written")
    ]
    # Câu test có nguồn gốc từ sinh tổng hợp (synthetic)
    test_recs = [
        DatasetRecord(text="Câu test sinh máy.", label="INTENT_A", category="cat_a", split="test", origin="synthetic")
    ]

    with pytest.raises(ValueError, match="Vi phạm quy tắc phân vùng: Tập test phải độc quyền do chủ sở hữu viết"):
        validate_split_integrity(train_recs, val_recs, test_recs, valid_intents, allow_non_owner_test=False)


def test_warning_on_low_sample_count() -> None:
    """Kiểm tra phát cảnh báo khi một intent có ít hơn 10 train hoặc 5 test."""
    valid_intents = {"INTENT_FEW": "cat_a"}

    train_recs = [
        DatasetRecord(text=f"train {i}", label="INTENT_FEW", category="cat_a", split="train", origin="owner_written")
        for i in range(8)  # 8 < 10
    ]
    val_recs = [
        DatasetRecord(text=f"val {i}", label="INTENT_FEW", category="cat_a", split="val", origin="owner_written")
        for i in range(5)
    ]
    test_recs = [
        DatasetRecord(text=f"test {i}", label="INTENT_FEW", category="cat_a", split="test", origin="owner_written")
        for i in range(3)  # 3 < 5
    ]

    with pytest.warns(UserWarning) as records:
        _, warnings_list = validate_split_integrity(train_recs, val_recs, test_recs, valid_intents)

    assert len(warnings_list) == 2
    assert any("8 mẫu train" in w for w in warnings_list)
    assert any("3 mẫu test" in w for w in warnings_list)
    assert len(records) >= 2


def test_format_summary_table() -> None:
    """Kiểm tra định dạng bảng tổng hợp kết quả phân bổ dữ liệu."""
    intent_counts = {
        "INTENT_A": {"train": 20, "val": 5, "test": 5, "total": 30},
        "INTENT_B": {"train": 10, "val": 5, "test": 5, "total": 20},
    }
    valid_intents = {"INTENT_A": "tex_doc", "INTENT_B": "tex_math"}

    table = format_summary_table(
        intent_counts=intent_counts,
        valid_intents=valid_intents,
        total_train=30,
        total_val=10,
        total_test=10,
    )

    assert "BÁO CÁO TỔNG HỢP VÀ PHÂN BỔ DỮ LIỆU" in table
    assert "INTENT_A" in table
    assert "INTENT_B" in table
    assert "tex_doc" in table
    assert "tex_math" in table


def test_compile_and_validate_dataset_e2e(tmp_path: Path) -> None:
    """Kiểm tra luồng thực thi tích hợp end-to-end với tệp xuất ra đĩa."""
    result = compile_and_validate_dataset(
        raw_sentences_path=RAW_SENTENCES_PATH,
        intents_path=DEFAULT_INTENTS_PATH,
        output_dir=tmp_path,
        include_synthetic=False,  # Tắt synthetic để kiểm tra tốc độ
        include_oos=False,
        dry_run=False,
    )

    assert result.train_count == 1880
    assert result.val_count == 940
    assert result.test_count == 940

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"

    assert train_csv.is_file()
    assert val_csv.is_file()
    assert test_csv.is_file()

    # Kiểm tra cấu trúc cột của CSV
    with open(train_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["text", "label", "category"]
        row = next(reader)
        assert len(row["text"]) > 0
        assert len(row["label"]) > 0
        assert len(row["category"]) > 0


def test_cli_main_dry_run() -> None:
    """Kiểm tra thực thi giao diện dòng lệnh CLI ở chế độ dry-run."""
    exit_code = main(["--dry-run", "--no-synthetic", "--no-oos"])
    assert exit_code == 0
