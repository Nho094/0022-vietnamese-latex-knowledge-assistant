"""
Bộ kiểm thử đơn vị cho module tiện ích đánh giá học thuật và thống kê (train/evaluate.py).

Tuân thủ quy chuẩn AGENTS.md:
- Python 3.11, type hints, ruff-clean, pytest.
- Identifiers in English; docstrings và comments giải thích lý do (WHY) bằng tiếng Việt.
- Kiểm tra tính đúng đắn của các công thức toán học và trường hợp biên (edge cases).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Final

import pytest

from core.config import (
    DEFAULT_INTENTS_PATH,
    DEFAULT_SEED,
)
from train.evaluate import (
    ClassificationMetricsResult,
    ECEResult,
    compute_accuracy,
    compute_bootstrap_ci,
    compute_classification_metrics,
    compute_ece,
    compute_ece_details,
    compute_macro_f1,
    compute_mcnemar_test,
    compute_mean,
    compute_mrr,
    compute_paired_bootstrap_ci,
    compute_per_class_metrics,
    compute_sample_std,
    compute_schema_macro_f1,
    compute_top_k_accuracy,
    compute_weighted_f1,
    generate_confusion_matrix_figure,
    load_intent_labels,
    main,
    mcnemar_test_from_counts,
)

TOLERANCE: Final[float] = 1e-5


# =====================================================================
# 1. Kiểm thử nạp Intent Labels
# =====================================================================


def test_load_intent_labels_default() -> None:
    """Kiểm tra nạp nhãn từ tệp intents.json chuẩn của dự án."""
    assert DEFAULT_INTENTS_PATH.exists()
    labels = load_intent_labels(DEFAULT_INTENTS_PATH)
    assert len(labels) == 189
    assert "LATEX_TAI_LIEU_TOI_GIAN" in labels
    assert "MATH_MATH_BASIC_PHAN_SO" in labels
    # Đảm bảo danh sách được sắp xếp bảng chữ cái và không có nhãn trùng
    assert labels == sorted(set(labels))


def test_load_intent_labels_nonexistent() -> None:
    """Kiểm tra nạp tệp không tồn tại phát sinh FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_intent_labels("non_existent_intents_file_12345.json")


# =====================================================================
# 2. Kiểm thử Thống kê tóm tắt (Summary Statistics)
# =====================================================================


def test_compute_mean_basic() -> None:
    """Kiểm tra tính giá trị trung bình cộng."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert compute_mean(data) == pytest.approx(3.0, abs=TOLERANCE)


def test_compute_mean_negative_and_zero() -> None:
    """Kiểm tra giá trị trung bình với số âm và 0."""
    data = [-10.0, 0.0, 10.0]
    assert compute_mean(data) == pytest.approx(0.0, abs=TOLERANCE)


def test_compute_mean_empty_raises() -> None:
    """Kiểm tra chuỗi rỗng phát sinh ValueError."""
    with pytest.raises(ValueError, match="values must not be empty"):
        compute_mean([])


def test_compute_sample_std_bessel_correction() -> None:
    """Kiểm tra độ lệch chuẩn mẫu với n-1 bậc tự do (Hiệu chỉnh Bessel)."""
    # Mẫu [2, 4, 4, 4, 5, 5, 7, 9]: n=8, mean=5.0, sum((x-5)^2)=32, s^2 = 32 / 7 = 4.5714
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    expected_std = math.sqrt(32.0 / 7.0)
    assert compute_sample_std(data, ddof=1) == pytest.approx(expected_std, abs=TOLERANCE)


def test_compute_sample_std_single_element() -> None:
    """Kiểm tra an toàn: khi n <= ddof (n=1, ddof=1) trả về 0.0."""
    assert compute_sample_std([42.0], ddof=1) == 0.0
    assert compute_sample_std([], ddof=1) == 0.0


# =====================================================================
# 3. Kiểm thử Chỉ số phân loại chuẩn (Classification Metrics)
# =====================================================================


def test_compute_accuracy_exact() -> None:
    """Kiểm tra độ chính xác phân loại tổng thể."""
    y_true = ["A", "B", "C", "D"]
    y_pred = ["A", "B", "C", "A"]  # 3/4 đúng
    assert compute_accuracy(y_true, y_pred) == pytest.approx(0.75, abs=TOLERANCE)


def test_compute_accuracy_empty_and_mismatched() -> None:
    """Kiểm tra trường hợp rỗng và kích thước không đồng nhất."""
    assert compute_accuracy([], []) == 0.0
    with pytest.raises(ValueError, match="không đồng nhất"):
        compute_accuracy(["A"], ["A", "B"])


def test_compute_per_class_metrics_with_zero_division() -> None:
    """
    Kiểm tra Precision, Recall, F1 từng lớp và cơ chế xử lý chia cho 0.
    Lớp A: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
    Lớp B: TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=2/3
    Lớp C: TP=0, FP=0, FN=1 -> P=0.0, R=0.0, F1=0.0 (zero_division)
    Lớp D (không có trong mẫu): TP=0, FP=0, FN=0 -> P=0.0, R=0.0, F1=0.0, support=0
    """
    y_true = ["A", "B", "C"]
    y_pred = ["A", "B", "B"]
    labels = ["A", "B", "C", "D"]

    metrics = compute_per_class_metrics(y_true, y_pred, labels=labels, zero_division=0.0)

    # Lớp A
    assert metrics["A"]["precision"] == pytest.approx(1.0, abs=TOLERANCE)
    assert metrics["A"]["recall"] == pytest.approx(1.0, abs=TOLERANCE)
    assert metrics["A"]["f1"] == pytest.approx(1.0, abs=TOLERANCE)
    assert metrics["A"]["support"] == 1.0

    # Lớp B
    assert metrics["B"]["precision"] == pytest.approx(0.5, abs=TOLERANCE)
    assert metrics["B"]["recall"] == pytest.approx(1.0, abs=TOLERANCE)
    assert metrics["B"]["f1"] == pytest.approx(2.0 / 3.0, abs=TOLERANCE)
    assert metrics["B"]["support"] == 1.0

    # Lớp C
    assert metrics["C"]["precision"] == 0.0
    assert metrics["C"]["recall"] == 0.0
    assert metrics["C"]["f1"] == 0.0
    assert metrics["C"]["support"] == 1.0

    # Lớp D (chưa quan sát trong ground-truth hoặc prediction)
    assert metrics["D"]["precision"] == 0.0
    assert metrics["D"]["recall"] == 0.0
    assert metrics["D"]["f1"] == 0.0
    assert metrics["D"]["support"] == 0.0


def test_compute_macro_f1_across_classes() -> None:
    """
    Kiểm tra tính Macro-F1 trung bình không trọng số.
    F1(A) = 1.0, F1(B) = 2/3, F1(C) = 0.0.
    Nếu đánh giá trên 3 lớp: Macro-F1 = (1.0 + 2/3 + 0.0) / 3 = 5/9 = 0.5555...
    Nếu đánh giá thêm lớp D (F1=0.0): Macro-F1 = (1.0 + 2/3 + 0.0 + 0.0) / 4 = 5/12 = 0.4166...
    """
    y_true = ["A", "B", "C"]
    y_pred = ["A", "B", "B"]

    macro_3 = compute_macro_f1(y_true, y_pred, labels=["A", "B", "C"])
    assert macro_3 == pytest.approx(5.0 / 9.0, abs=TOLERANCE)

    macro_4 = compute_macro_f1(y_true, y_pred, labels=["A", "B", "C", "D"])
    assert macro_4 == pytest.approx(5.0 / 12.0, abs=TOLERANCE)


def test_compute_schema_macro_f1() -> None:
    """Kiểm tra tiện ích Macro-F1 trên toàn bộ 189 ý định trong schema."""
    y_true = ["MATH_MATH_BASIC_PHAN_SO"]
    y_pred = ["MATH_MATH_BASIC_PHAN_SO"]
    # 1 lớp F1=1.0, 188 lớp còn lại F1=0.0 -> Macro-F1 = 1 / 189
    macro = compute_schema_macro_f1(y_true, y_pred, intents_path=DEFAULT_INTENTS_PATH)
    assert macro == pytest.approx(1.0 / 189.0, abs=TOLERANCE)


def test_compute_weighted_f1() -> None:
    """
    Kiểm tra tính Weighted-F1 theo support.
    Mẫu: A (sup=1, F1=1.0), B (sup=2, F1=0.8).
    Weighted-F1 = (1 * 1.0 + 2 * 0.8) / 3 = 2.6 / 3 = 0.8666...
    """
    y_true = ["A", "B", "B"]
    y_pred = ["A", "B", "A"]
    # Per-class:
    # A: TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=2/3, sup=1
    # B: TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1=2/3, sup=2
    # Weighted-F1 = (1 * 2/3 + 2 * 2/3) / 3 = 2/3
    weighted = compute_weighted_f1(y_true, y_pred)
    assert weighted == pytest.approx(2.0 / 3.0, abs=TOLERANCE)


def test_compute_classification_metrics_summary() -> None:
    """Kiểm tra đối tượng tổng hợp ClassificationMetricsResult."""
    y_true = ["INTENT_1", "INTENT_2"]
    y_pred = ["INTENT_1", "INTENT_2"]
    res = compute_classification_metrics(y_true, y_pred)

    assert isinstance(res, ClassificationMetricsResult)
    assert res.accuracy == 1.0
    assert res.macro_f1 == 1.0
    assert res.weighted_f1 == 1.0
    assert res.total_samples == 2
    assert res.total_classes == 2
    dict_repr = res.to_dict()
    assert "accuracy" in dict_repr
    assert "per_class" in dict_repr


# =====================================================================
# 4. Kiểm thử Chỉ số xếp hạng (Ranking Metrics)
# =====================================================================


def test_compute_top_k_accuracy() -> None:
    """Kiểm tra Top-k Accuracy với k=1 và k=2."""
    y_true = ["A", "B", "C"]
    y_pred_ranked = [
        ["A", "B", "C"],  # Đúng ở top-1
        ["C", "B", "A"],  # Đúng ở top-2
        ["A", "D", "B"],  # Sai cả top-2, chỉ đúng ở top-3 nếu có C
    ]

    top1 = compute_top_k_accuracy(y_true, y_pred_ranked, k=1)
    top2 = compute_top_k_accuracy(y_true, y_pred_ranked, k=2)
    top3 = compute_top_k_accuracy(y_true, y_pred_ranked, k=3)

    assert top1 == pytest.approx(1.0 / 3.0, abs=TOLERANCE)
    assert top2 == pytest.approx(2.0 / 3.0, abs=TOLERANCE)
    assert top3 == pytest.approx(2.0 / 3.0, abs=TOLERANCE)


def test_compute_top_k_accuracy_invalid_k() -> None:
    """Kiểm tra k < 1 phát sinh ngoại lệ ValueError."""
    with pytest.raises(ValueError, match="k phải >= 1"):
        compute_top_k_accuracy(["A"], [["A"]], k=0)


def test_compute_mrr() -> None:
    """
    Kiểm tra Mean Reciprocal Rank (MRR).
    Mẫu 1: A ở rank 1 -> RR = 1/1 = 1.0
    Mẫu 2: B ở rank 2 -> RR = 1/2 = 0.5
    Mẫu 3: C không có trong dự đoán -> RR = 0.0
    Mẫu 4: D ở rank 4 -> RR = 1/4 = 0.25
    MRR = (1.0 + 0.5 + 0.0 + 0.25) / 4 = 1.75 / 4 = 0.4375
    """
    y_true = ["A", "B", "C", "D"]
    y_pred_ranked = [
        ["A", "X", "Y"],
        ["X", "B", "Y"],
        ["X", "Y", "Z"],
        ["X", "Y", "Z", "D"],
    ]

    mrr = compute_mrr(y_true, y_pred_ranked)
    assert mrr == pytest.approx(0.4375, abs=TOLERANCE)


def test_compute_mrr_empty() -> None:
    """Kiểm tra MRR trên chuỗi rỗng trả về 0.0."""
    assert compute_mrr([], []) == 0.0


# =====================================================================
# 5. Kiểm thử Khoảng tin cậy Bootstrap phi tham số
# =====================================================================


def test_compute_bootstrap_ci_reproducibility() -> None:
    """Kiểm tra tính tái lập tất định khi thiết lập seed."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ci1 = compute_bootstrap_ci(data, num_bootstrap=200, confidence_level=0.95, seed=DEFAULT_SEED)
    ci2 = compute_bootstrap_ci(data, num_bootstrap=200, confidence_level=0.95, seed=DEFAULT_SEED)

    assert ci1[0] == pytest.approx(ci2[0], abs=TOLERANCE)
    assert ci1[1] == pytest.approx(ci2[1], abs=TOLERANCE)
    assert ci1[0] < ci1[1]
    # Cận dưới và cận trên phải chứa giá trị trung bình mẫu (5.5)
    assert ci1[0] <= 5.5 <= ci1[1]


def test_compute_bootstrap_ci_single_element() -> None:
    """Kiểm tra dữ liệu 1 phần tử trả về khoảng suy biến (val, val)."""
    assert compute_bootstrap_ci([42.0]) == (42.0, 42.0)


def test_compute_bootstrap_ci_invalid_args() -> None:
    """Kiểm tra các tham số không hợp lệ."""
    with pytest.raises(ValueError, match="data rỗng"):
        compute_bootstrap_ci([])

    with pytest.raises(ValueError, match="num_bootstrap phải >= 1"):
        compute_bootstrap_ci([1.0, 2.0], num_bootstrap=0)

    with pytest.raises(ValueError, match="confidence_level"):
        compute_bootstrap_ci([1.0, 2.0], confidence_level=1.5)


def test_compute_paired_bootstrap_ci() -> None:
    """Kiểm tra Bootstrap theo cặp cho độ chính xác Accuracy."""
    y_true = ["A"] * 20 + ["B"] * 20
    y_pred = ["A"] * 18 + ["B"] * 2 + ["B"] * 18 + ["A"] * 2  # 36/40 = 90%
    ci = compute_paired_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric_fn=compute_accuracy,
        num_bootstrap=200,
        confidence_level=0.95,
        seed=DEFAULT_SEED,
    )
    assert ci[0] <= 0.90 <= ci[1]
    assert 0.70 < ci[0] < 0.95


# =====================================================================
# 6. Kiểm định McNemar (McNemar's Test)
# =====================================================================


def test_mcnemar_zero_discordant_pairs() -> None:
    """
    Kiểm tra xử lý đặc biệt theo yêu cầu: khi b + c = 0 (hai mô hình hoàn toàn đồng thuận).
    Yêu cầu: statistic = 0.0, p-value = 1.0.
    """
    stat, p_val = mcnemar_test_from_counts(b=0, c=0, continuity_correction=True)
    assert stat == 0.0
    assert p_val == 1.0

    stat_no_corr, p_val_no_corr = mcnemar_test_from_counts(b=0, c=0, continuity_correction=False)
    assert stat_no_corr == 0.0
    assert p_val_no_corr == 1.0


def test_mcnemar_identical_models() -> None:
    """Kiểm tra hai mô hình dự đoán hoàn toàn giống nhau cho b+c=0."""
    y_true = ["A", "B", "C", "D"]
    y_pred_a = ["A", "B", "X", "D"]
    y_pred_b = ["A", "B", "X", "D"]

    stat, p_val = compute_mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert stat == 0.0
    assert p_val == 1.0


def test_mcnemar_symmetric_discordance() -> None:
    """
    Khi b = c > 0 (ví dụ b=10, c=10), với hiệu chỉnh Yates: (|10 - 10| - 1) -> max(0, -1) = 0.
    Statistic = 0.0, p-value = 1.0.
    """
    stat, p_val = mcnemar_test_from_counts(b=10, c=10, continuity_correction=True)
    assert stat == 0.0
    assert p_val == 1.0


def test_mcnemar_significant_difference() -> None:
    """
    Kiểm tra trường hợp có sự khác biệt rõ rệt (b=25, c=1).
    Yates chi2 = (|25 - 1| - 1)^2 / 26 = 23^2 / 26 = 529 / 26 = 20.346...
    p-value phải cực nhỏ (< 0.001).
    """
    stat, p_val = mcnemar_test_from_counts(b=25, c=1, continuity_correction=True)
    expected_stat = (23.0**2) / 26.0
    assert stat == pytest.approx(expected_stat, abs=TOLERANCE)
    assert p_val < 0.001


def test_mcnemar_without_continuity_correction() -> None:
    """Kiểm tra công thức chuẩn không hiệu chỉnh Yates: (b - c)^2 / (b + c)."""
    stat, _ = mcnemar_test_from_counts(b=20, c=5, continuity_correction=False)
    expected_stat = (15.0**2) / 25.0  # 225 / 25 = 9.0
    assert stat == pytest.approx(expected_stat, abs=TOLERANCE)


def test_mcnemar_negative_counts_raises() -> None:
    """Kiểm tra số đếm âm phát sinh ValueError."""
    with pytest.raises(ValueError, match="phải là số nguyên không âm"):
        mcnemar_test_from_counts(b=-1, c=5)


# =====================================================================
# 7. Kiểm thử Sai số hiệu chuẩn kỳ vọng (ECE)
# =====================================================================


def test_compute_ece_perfect_calibration() -> None:
    """
    Kiểm tra mô hình hiệu chuẩn hoàn hảo:
    Các mẫu có confidence = 1.0 và dự đoán đúng 100% (acc=1.0) -> ECE = 0.0.
    """
    y_true = ["A", "B", "C"]
    y_pred = ["A", "B", "C"]
    confidences = [1.0, 1.0, 1.0]

    ece = compute_ece(y_true, y_pred, confidences, num_bins=10)
    assert ece == pytest.approx(0.0, abs=TOLERANCE)


def test_compute_ece_complete_miscalibration() -> None:
    """
    Kiểm tra mô hình hoàn toàn mất hiệu chuẩn:
    Confidence = 1.0 nhưng dự đoán sai 100% (acc = 0.0) -> Gap = |0.0 - 1.0| = 1.0 -> ECE = 1.0.
    """
    y_true = ["A", "B", "C"]
    y_pred = ["X", "Y", "Z"]
    confidences = [1.0, 1.0, 1.0]

    ece = compute_ece(y_true, y_pred, confidences, num_bins=10)
    assert ece == pytest.approx(1.0, abs=TOLERANCE)


def test_compute_ece_details_structure() -> None:
    """Kiểm tra đối tượng kết quả chi tiết ECEResult và tính đúng của từng bin."""
    # 2 mẫu rơi vào bin [0.8, 0.9]:
    # Mẫu 1: conf=0.85, đúng
    # Mẫu 2: conf=0.85, sai
    # Bin này: acc=0.5, avg_conf=0.85, gap = |0.5 - 0.85| = 0.35, ECE = 0.35
    y_true = ["A", "A"]
    y_pred = ["A", "B"]
    confidences = [0.85, 0.85]

    res = compute_ece_details(y_true, y_pred, confidences, num_bins=10)
    assert isinstance(res, ECEResult)
    assert res.num_bins == 10
    assert res.total_samples == 2
    assert res.ece == pytest.approx(0.35, abs=TOLERANCE)
    assert len(res.bin_accuracies) == 10
    assert len(res.bin_counts) == 10
    assert res.bin_counts[8] == 2  # 0.85 rơi vào bin index 8 [0.8, 0.9)


def test_compute_ece_empty() -> None:
    """Kiểm tra ECE với danh sách rỗng."""
    assert compute_ece([], [], []) == 0.0


# =====================================================================
# 8. Kiểm thử Sinh biểu đồ ma trận nhầm lẫn (Confusion Matrix Figure)
# =====================================================================


def test_generate_confusion_matrix_figure_saves_file() -> None:
    """Kiểm tra sinh và lưu tệp PNG biểu đồ ma trận nhầm lẫn."""
    y_true = ["LATEX_TAO_BANG", "MATH_MATH_BASIC_PHAN_SO", "LATEX_TAO_BANG", "LATEX_DANH_SACH"]
    y_pred = ["LATEX_TAO_BANG", "MATH_MATH_BASIC_PHAN_SO", "MATH_MATH_BASIC_PHAN_SO", "LATEX_DANH_SACH"]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test_cm.png"
        result_path = generate_confusion_matrix_figure(
            y_true=y_true,
            y_pred=y_pred,
            output_path=out_file,
            title="Kiểm thử Confusion Matrix",
        )

        assert result_path.exists()
        assert result_path == out_file
        # Kiểm tra tệp PNG không rỗng và có kích thước thực tế
        assert result_path.stat().st_size > 1000


def test_generate_confusion_matrix_figure_normalized() -> None:
    """Kiểm tra lưu ma trận nhầm lẫn có chuẩn hóa tỷ lệ."""
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "A", "B", "B"]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "cm_norm.png"
        result_path = generate_confusion_matrix_figure(
            y_true=y_true,
            y_pred=y_pred,
            output_path=out_file,
            normalize="true",
        )
        assert result_path.exists()
        assert result_path.stat().st_size > 1000


def test_generate_confusion_matrix_figure_mismatched_raises() -> None:
    """Kiểm tra độ dài không khớp phát sinh ValueError."""
    with pytest.raises(ValueError, match="không đồng nhất"):
        generate_confusion_matrix_figure(["A"], ["A", "B"])


# =====================================================================
# 9. Kiểm thử Hàm main của CLI
# =====================================================================


def test_evaluate_cli_main() -> None:
    """Kiểm tra thực thi hàm main của module train/evaluate.py."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_fig = Path(tmpdir) / "cli_cm.png"
        exit_code = main(["--output-fig", str(test_fig), "--seed", "42"])
        assert exit_code == 0
        assert test_fig.exists()
        assert test_fig.stat().st_size > 1000
