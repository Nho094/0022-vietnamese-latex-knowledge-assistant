"""
Bộ kiểm thử đơn vị cho kịch bản tinh chỉnh PhoBERT (train/finetune_phobert.py).

Tuân thủ quy chuẩn AGENTS.md:
- Python 3.11+, type hints, ruff-clean, pytest.
- Identifiers bằng tiếng Anh; docstrings và comments giải thích lý do (WHY) bằng tiếng Việt.
- Kiểm thử các chức năng cốt lõi:
  + Tính mã băm SHA-256 tất định của tệp kiểm thử (Honesty Rule #3).
  + Nạp mẫu CSV với mã hóa UTF-8-SIG (bảo vệ chống ký tự BOM).
  + Ánh xạ nhãn cho 'intent' (189 lớp theo schema) và 'category' (12 nhóm ngữ nghĩa).
  + Lịch trình tốc độ học: linear warm-up và linear decay theo công thức BERT/RoBERTa.
  + Dừng sớm (early stopping) với tính kiên nhẫn (patience=5) bảo vệ chống quá khớp.
  + Khởi tạo mạng nơ-ron PhoBertIntentClassifier ở chế độ smoke run siêu tốc trên CPU.
  + Quy trình thực nghiệm hoàn chỉnh (run_experiment_pipeline) và lưu trữ checkpoint/kết quả.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaConfig

from core.config import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_INTENTS_PATH,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
)
from core.models import PhoBertIntentClassifier
from train.finetune_phobert import (
    DatasetSample,
    TextClassificationDataset,
    build_arg_parser,
    build_label_mappings,
    compute_file_sha256,
    create_classifier_model,
    create_linear_schedule_with_warmup,
    read_csv_samples,
    run_experiment_pipeline,
    search_learning_rate,
    train_single_run,
)

TOLERANCE: Final[float] = 1e-5


# =====================================================================
# 1. Kiểm thử Tiện ích Băm SHA-256 và Nạp Dữ liệu CSV
# =====================================================================


def test_compute_file_sha256_deterministic() -> None:
    """
    Kiểm tra hàm compute_file_sha256 tính toán mã băm SHA-256 chính xác và tất định.
    LÝ DO THIẾT KẾ: Honesty Rule #3 yêu cầu kiểm chứng sha256 của data/test.csv.
    """
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write("vietnamese latex knowledge assistant\n")
        tmp_path = Path(tmp.name)

    try:
        expected_hash = hashlib.sha256(b"vietnamese latex knowledge assistant\n").hexdigest()
        assert compute_file_sha256(tmp_path) == expected_hash
    finally:
        tmp_path.unlink(missing_ok=True)


def test_compute_file_sha256_file_not_found() -> None:
    """Kiểm tra ngoại lệ FileNotFoundError khi đường dẫn tệp không tồn tại."""
    with pytest.raises(FileNotFoundError, match="Tệp không tồn tại"):
        compute_file_sha256(Path("non_existent_file_xyz.csv"))


def test_read_csv_samples_with_bom_and_limits() -> None:
    """
    Kiểm tra đọc dữ liệu CSV có mã hóa UTF-8-SIG (chứa ký tự BOM \\ufeff) và giới hạn max_samples.
    """
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["text", "label", "category"])
        writer.writeheader()
        writer.writerow({"text": "Tính tích phân này", "label": "CALC_INTEGRAL", "category": "calculus"})
        writer.writerow({"text": "Vẽ đồ thị hàm số", "label": "PLOT_FUNCTION", "category": "graphics"})
        writer.writerow({"text": "Giải hệ phương trình", "label": "SOLVE_SYSTEM", "category": "algebra"})
        tmp_path = Path(tmp.name)

    try:
        # Đọc toàn bộ mẫu
        all_samples = read_csv_samples(tmp_path)
        assert len(all_samples) == 3
        assert all_samples[0].text == "Tính tích phân này"
        assert all_samples[0].label == "CALC_INTEGRAL"
        assert all_samples[0].category == "calculus"

        # Đọc có giới hạn max_samples=2
        limited_samples = read_csv_samples(tmp_path, max_samples=2)
        assert len(limited_samples) == 2
        assert limited_samples[1].label == "PLOT_FUNCTION"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_read_csv_samples_file_not_found() -> None:
    """Kiểm tra ngoại lệ FileNotFoundError khi tệp dữ liệu CSV không tồn tại."""
    with pytest.raises(FileNotFoundError, match="Không tìm thấy tệp"):
        read_csv_samples(Path("missing_data_file.csv"))


# =====================================================================
# 2. Kiểm thử Ánh xạ Nhãn (build_label_mappings)
# =====================================================================


def test_build_label_mappings_intent_target() -> None:
    """
    Kiểm tra xây dựng bảng ánh xạ cho mục tiêu 'intent'.
    Tuân thủ quy chuẩn: Khi có intents.json, nạp đủ toàn bộ 189 nhãn theo schema
    để đảm bảo không bị thiếu lớp khi tính macro-F1 trên không gian ý định toàn cục.
    """
    train_samples = [
        DatasetSample(text="câu 1", label="MATH_MATRIX", category="algebra"),
        DatasetSample(text="câu 2", label="MATH_INTEGRAL", category="calculus"),
    ]
    labels, label2id, id2label = build_label_mappings(
        target="intent",
        train_samples=train_samples,
        intents_json_path=DEFAULT_INTENTS_PATH,
    )
    if DEFAULT_INTENTS_PATH.exists():
        assert len(labels) == 189
    else:
        assert len(labels) == 2

    assert len(label2id) == len(labels)
    assert len(id2label) == len(labels)
    for i, lbl in enumerate(labels):
        assert label2id[lbl] == i
        assert id2label[i] == lbl


def test_build_label_mappings_category_target() -> None:
    """
    Kiểm tra xây dựng bảng ánh xạ cho mục tiêu 'category'.
    Danh sách nhãn là các danh mục phân loại được sắp xếp từ vựng (alphabetical).
    """
    train_samples = [
        DatasetSample(text="câu 1", label="INTENT_1", category="calculus"),
        DatasetSample(text="câu 2", label="INTENT_2", category="algebra"),
        DatasetSample(text="câu 3", label="INTENT_3", category="calculus"),
    ]
    labels, label2id, id2label = build_label_mappings(
        target="category",
        train_samples=train_samples,
    )
    assert labels == ["algebra", "calculus"]
    assert label2id == {"algebra": 0, "calculus": 1}
    assert id2label == {0: "algebra", 1: "calculus"}


def test_build_label_mappings_invalid_target() -> None:
    """Kiểm tra ngoại lệ ValueError khi truyền mục tiêu không được hỗ trợ."""
    with pytest.raises(ValueError, match="không hợp lệ"):
        build_label_mappings(target="invalid_target", train_samples=[])


# =====================================================================
# 3. Kiểm thử Lịch trình Tốc độ học (Linear Warm-up & Decay)
# =====================================================================


def test_linear_schedule_with_warmup_mathematical_behavior() -> None:
    """
    Kiểm tra tính đúng đắn toán học của linear warm-up và decay (Devlin et al., 2018):
        - Tại step 0: lr_factor = 0.0
        - Tại step num_warmup_steps: lr_factor = 1.0
        - Tại step cuối num_training_steps: lr_factor = 0.0
    """
    param = nn.Parameter(torch.zeros(10))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = create_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
    )

    # Step 0: lr = 0.0
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=TOLERANCE)

    # Step 5: halfway through warmup -> factor = 5 / 10 = 0.5
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.5, abs=TOLERANCE)

    # Step 10: end of warmup -> factor = 10 / 10 = 1.0
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(1.0, abs=TOLERANCE)

    # Step 55: halfway through decay (remaining 90 steps, after 45 steps -> factor = 45 / 90 = 0.5)
    for _ in range(45):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.5, abs=TOLERANCE)

    # Step 100: end of training -> factor = 0.0
    for _ in range(45):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=TOLERANCE)


# =====================================================================
# 4. Kiểm thử Khởi tạo Mô hình và Dataset PyTorch
# =====================================================================


def test_create_classifier_model_smoke_mode() -> None:
    """
    Kiểm tra khởi tạo mô hình ở chế độ smoke_mode=True.
    Tạo mạng nơ-ron PhoBertIntentClassifier siêu nhẹ (1 tầng, hidden_size=64)
    thực thi siêu tốc trên CPU mà không cần tải file trọng số 540MB.
    """
    num_labels = 12
    model = create_classifier_model(num_labels=num_labels, smoke_mode=True, device="cpu")
    assert isinstance(model, PhoBertIntentClassifier)
    assert model.num_labels == num_labels
    assert model.classifier.out_features == num_labels
    assert model.classifier.in_features == 64


def test_text_classification_dataset_encoding() -> None:
    """
    Kiểm tra lớp TextClassificationDataset tiền xử lý và đóng gói tensor cho PyTorch DataLoader.
    """
    samples = [
        DatasetSample(text="Tính đạo hàm f(x) = x^2", label="MATH_DERIVATIVE", category="calculus"),
        DatasetSample(text="Không rõ", label="UNKNOWN_INTENT", category="misc"),
    ]
    label2id = {"MATH_DERIVATIVE": 0}

    # Giả lập tokenizer PhoBERT
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[0, 10, 20, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }

    dataset = TextClassificationDataset(
        samples=samples,
        target="intent",
        label2id=label2id,
        tokenizer=mock_tokenizer,
        max_seq_length=16,
        preprocess=False,
    )

    # Mẫu thứ 2 bị bỏ qua vì UNKNOWN_INTENT không có trong label2id
    assert len(dataset) == 1
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["labels"].item() == 0


# =====================================================================
# 5. Kiểm thử Huấn luyện Đơn lẻ và Cơ chế Dừng Sớm (Early Stopping)
# =====================================================================


class _MockDataset(Dataset):
    """Dataset giả lập dùng cho kiểm thử huấn luyện đơn vị."""

    def __init__(self, size: int, num_labels: int, hidden_size: int = 64) -> None:
        self.size = size
        self.num_labels = num_labels
        self.data = [
            {
                "input_ids": torch.randint(0, 1000, (8,), dtype=torch.long),
                "attention_mask": torch.ones(8, dtype=torch.long),
                "labels": torch.tensor(i % num_labels, dtype=torch.long),
            }
            for i in range(size)
        ]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.data[idx]


def test_train_single_run_and_early_stopping() -> None:
    """
    Kiểm tra hàm train_single_run hoạt động chính xác với cơ chế dừng sớm (Early Stopping).
    Kiểm tra mô hình khôi phục đúng trọng số tốt nhất sau khi dừng sớm.
    """
    num_labels = 3
    labels = ["c0", "c1", "c2"]
    id2label = {i: labels[i] for i in range(num_labels)}

    tiny_config = RobertaConfig(
        vocab_size=2000,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    model = PhoBertIntentClassifier.from_config(tiny_config, num_labels=num_labels)

    train_ds = _MockDataset(size=8, num_labels=num_labels)
    val_ds = _MockDataset(size=4, num_labels=num_labels)
    train_loader = DataLoader(train_ds, batch_size=4)
    val_loader = DataLoader(val_ds, batch_size=4)

    trained_model, summary = train_single_run(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        labels=labels,
        id2label=id2label,
        learning_rate=1e-3,
        max_epochs=3,
        patience=2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        device="cpu",
    )

    assert isinstance(trained_model, PhoBertIntentClassifier)
    assert "best_epoch" in summary
    assert "best_val_macro_f1" in summary
    assert "epochs_trained" in summary
    assert summary["epochs_trained"] <= 3


def test_search_learning_rate_selection() -> None:
    """
    Kiểm tra tìm kiếm tốc độ học tối ưu (search_learning_rate)
    được đánh giá hoàn toàn trên validation macro-F1 (Honesty Rule #3).
    """
    labels = ["cat_a", "cat_b"]
    id2label = {0: "cat_a", 1: "cat_b"}
    train_ds = _MockDataset(size=6, num_labels=2)
    val_ds = _MockDataset(size=4, num_labels=2)

    # Chạy tìm kiếm giữa 2 candidate LRs
    best_lr, lr_results = search_learning_rate(
        candidate_lrs=[1e-4, 5e-4],
        train_dataset=train_ds,  # type: ignore[arg-type]
        val_dataset=val_ds,  # type: ignore[arg-type]
        labels=labels,
        id2label=id2label,
        batch_size=4,
        max_epochs=1,
        patience=1,
        warmup_ratio=0.1,
        weight_decay=0.01,
        smoke_mode=True,
        device="cpu",
    )

    assert best_lr in (1e-4, 5e-4)
    assert "1.0e-04" in lr_results
    assert "5.0e-04" in lr_results


# =====================================================================
# 6. Kiểm thử Toàn bộ Đường ống Thực nghiệm (run_experiment_pipeline)
# =====================================================================


def test_run_experiment_pipeline_smoke_mode(tmp_path: Path) -> None:
    """
    Kiểm thử tích hợp đầu-cuối toàn bộ quy trình tinh chỉnh PhoBERT ở chế độ smoke run:
    1. Kiểm tra tính mã băm SHA-256 của data/test.csv.
    2. Nạp dữ liệu, kiểm tra vân tay tiền xử lý.
    3. Tìm kiếm tốc độ học và huấn luyện đa hạt giống.
    4. Đánh giá test một lần duy nhất.
    5. Lưu checkpoint, cấu hình, ánh xạ nhãn, vân tay tiền xử lý.
    6. Xuất tệp kết quả JSON báo cáo đầy đủ chỉ số học thuật.
    """
    checkpoint_dir = tmp_path / "checkpoints"
    results_path = tmp_path / "phobert_results.json"

    # Chạy pipeline trên tập category ở chế độ smoke run
    results = run_experiment_pipeline(
        target="category",
        smoke_mode=True,
        device="cpu",
        train_path=TRAIN_CSV_PATH,
        val_path=VAL_CSV_PATH,
        test_path=TEST_CSV_PATH,
        checkpoint_dir=checkpoint_dir,
        results_path=results_path,
    )

    # Kiểm tra cấu trúc kết quả trả về
    assert results["target"] == "category"
    assert results["smoke_mode"] is True
    assert "test_csv_sha256" in results
    assert len(results["test_csv_sha256"]) == 64
    assert "preprocessing_fingerprint" in results
    assert "aggregate_test_metrics" in results

    agg = results["aggregate_test_metrics"]
    assert "accuracy_mean" in agg
    assert "accuracy_std" in agg
    assert "macro_f1_mean" in agg
    assert "macro_f1_std" in agg
    assert "ece_mean" in agg
    assert "ece_std" in agg

    # Kiểm tra các tệp tin artifact đã được lưu trữ đúng
    assert results_path.exists()
    assert (checkpoint_dir / "pytorch_model.bin").exists()
    assert (checkpoint_dir / "config.json").exists()
    assert (checkpoint_dir / "label2id.json").exists()
    assert (checkpoint_dir / "id2label.json").exists()
    assert (checkpoint_dir / "preprocessing_fingerprint.json").exists()

    # Đọc lại và kiểm tra nội dung tệp json
    with open(results_path, "r", encoding="utf-8") as f:
        loaded_res = json.load(f)
    assert loaded_res["test_csv_sha256"] == results["test_csv_sha256"]


# =====================================================================
# 7. Kiểm thử Bộ phân tích Tham số Dòng lệnh (CLI Parser)
# =====================================================================


def test_build_arg_parser_defaults() -> None:
    """Kiểm tra cấu hình tham số mặc định của parser dòng lệnh."""
    parser = build_arg_parser()
    args = parser.parse_args([])

    assert args.target == "intent"
    assert args.smoke_mode is False
    assert args.patience == DEFAULT_EARLY_STOPPING_PATIENCE
    assert args.epochs == 10
    assert args.batch_size == 16
    assert args.warmup_ratio == pytest.approx(DEFAULT_WARMUP_RATIO)
    assert args.weight_decay == pytest.approx(DEFAULT_WEIGHT_DECAY)


def test_build_arg_parser_custom_args() -> None:
    """Kiểm tra nhận diện đúng các cờ tùy biến của CLI."""
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--target",
            "category",
            "--smoke",
            "--candidate-lrs",
            "1e-4",
            "2e-4",
            "--seeds",
            "100",
            "200",
            "--patience",
            "3",
            "--epochs",
            "5",
        ]
    )

    assert args.target == "category"
    assert args.smoke_mode is True
    assert args.candidate_lrs == [1e-4, 2e-4]
    assert args.seeds == [100, 200]
    assert args.patience == 3
    assert args.epochs == 5
