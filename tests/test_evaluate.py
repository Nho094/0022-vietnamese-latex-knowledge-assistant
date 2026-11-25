"""
Bộ kiểm thử đơn vị với các giá trị đối chuẩn tính tay (Hand-Computed Unit Tests)
cho các chỉ số đo lường đánh giá học thuật và công thức thống kê (train/evaluate.py).

Tuân thủ quy chuẩn dự án và AGENTS.md:
- Python 3.11+, type hints đầy đủ, ruff-clean, pytest.
- Identifiers bằng tiếng Anh; docstrings và giải thích bằng tiếng Việt (Half Vibe Code).
- Kiểm chứng đối chuẩn số học tính tay chính xác (hand-computed ground truth)
  cho mọi công thức thống kê và chỉ số đánh giá.
- Không sử dụng số ma thuật tùy tiện, liên kết trực tiếp cấu hình từ core/config.py.

Các công thức và chỉ số được kiểm thử:
1. Độ chính xác tổng thể (Overall Accuracy).
2. Độ chuẩn xác (Precision), độ thu hồi (Recall), điểm F1 theo từng lớp (Per-class)
   và cơ chế xử lý chia cho 0 (Van Rijsbergen, 1979).
3. Macro-F1 trung bình không trọng số và kiểm thử độ bao phủ toàn bộ intent schema
   với các nhãn có kích thước hỗ trợ bằng 0 (Zero-support schema coverage).
4. Weighted-F1 tính trọng số theo support thực tế.
5. Thống kê tóm tắt: Trung bình mẫu (Mean) và độ lệch chuẩn mẫu hiệu chỉnh Bessel (n-1).
6. Chỉ số xếp hạng: Top-k Accuracy và nghịch đảo thứ hạng trung bình (MRR - Voorhees, 1999).
7. Ước lượng khoảng tin cậy Bootstrap phân vị (Percentile Bootstrap CI - Efron & Tibshirani, 1993).
8. Kiểm định McNemar hiệu chỉnh liên tục Yates và trường hợp biên p = 1.0 khi b + c = 0 (McNemar, 1947; Dietterich, 1998).
9. Sai số hiệu chuẩn kỳ vọng phân bin đều (Equal-width binning ECE - Guo et al., 2017).
"""

from __future__ import annotations

import math
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
    load_intent_labels,
    mcnemar_test_from_counts,
)

# Dung sai số thực cho các phép toán dấu phẩy động
TOLERANCE: Final[float] = 1e-7


# =====================================================================
# 1. Kiểm thử Độ chính xác tổng thể (Overall Accuracy)
# =====================================================================


def test_hand_computed_accuracy_exact() -> None:
    """
    Kiểm tra độ chính xác tổng thể (Accuracy) với tập dữ liệu đối chuẩn tính tay:
    y_true = ["A", "B", "C", "D", "A", "B"] (N = 6)
    y_pred = ["A", "B", "D", "D", "B", "B"]
    Số mẫu đúng = 4 (các chỉ số 0, 1, 3, 5).
    Giá trị tính tay lý thuyết: 4 / 6 = 2 / 3 = 0.6666666667.
    """
    y_true = ["A", "B", "C", "D", "A", "B"]
    y_pred = ["A", "B", "D", "D", "B", "B"]

    acc = compute_accuracy(y_true, y_pred)
    assert acc == pytest.approx(2.0 / 3.0, abs=TOLERANCE)


def test_accuracy_edge_cases() -> None:
    """
    Kiểm tra các trường hợp biên của Accuracy:
    - Tập rỗng: trả về zero_division (mặc định 0.0, hoặc giá trị chỉ định).
    - Toàn bộ đúng: 1.0.
    - Toàn bộ sai: 0.0.
    - Kích thước không khớp: phát sinh ValueError.
    """
    assert compute_accuracy([], [], zero_division=0.0) == 0.0
    assert compute_accuracy([], [], zero_division=1.0) == 1.0
    assert compute_accuracy(["A", "B"], ["A", "B"]) == 1.0
    assert compute_accuracy(["A", "B"], ["B", "A"]) == 0.0

    with pytest.raises(ValueError, match="không đồng nhất"):
        compute_accuracy(["A"], ["A", "B"])


# =====================================================================
# 2. Kiểm thử Per-class Precision, Recall, F1 và Xử lý Chia cho 0
# =====================================================================


def test_hand_computed_per_class_metrics() -> None:
    """
    Kiểm tra đối chuẩn tính tay chi tiết cho từng lớp theo công thức Van Rijsbergen (1979):
    Tập mẫu kích thước N = 10:
        y_true = [A, A, A, B, B, B, B, C, C, D]
        y_pred = [A, A, B, B, B, C, D, C, A, D]
    Danh sách nhãn đánh giá: ["A", "B", "C", "D", "E"] (lớp E có support = 0).

    Chi tiết tính tay từng lớp:
    - Lớp A:
        TP = 2 (mẫu 0, 1)
        FP = 1 (mẫu 8: true=C, pred=A)
        FN = 1 (mẫu 2: true=A, pred=B)
        Support = 3
        Precision = 2 / (2 + 1) = 2/3
        Recall = 2 / (2 + 1) = 2/3
        F1 = 2/3
    - Lớp B:
        TP = 2 (mẫu 3, 4)
        FP = 1 (mẫu 2: true=A, pred=B)
        FN = 2 (mẫu 5: pred=C; mẫu 6: pred=D)
        Support = 4
        Precision = 2 / (2 + 1) = 2/3
        Recall = 2 / (2 + 2) = 2/4 = 1/2
        F1 = 2 * (2/3) * (1/2) / (2/3 + 1/2) = (2/3) / (7/6) = 4/7
    - Lớp C:
        TP = 1 (mẫu 7)
        FP = 1 (mẫu 5: true=B, pred=C)
        FN = 1 (mẫu 8: true=C, pred=A)
        Support = 2
        Precision = 1 / (1 + 1) = 1/2
        Recall = 1 / (1 + 1) = 1/2
        F1 = 1/2 = 0.5
    - Lớp D:
        TP = 1 (mẫu 9)
        FP = 1 (mẫu 6: true=B, pred=D)
        FN = 0
        Support = 1
        Precision = 1 / (1 + 1) = 1/2
        Recall = 1 / (1 + 0) = 1.0
        F1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 1.0 / 1.5 = 2/3
    - Lớp E (Zero-support label):
        TP = 0, FP = 0, FN = 0, Support = 0
        Precision = 0.0 (xử lý zero_division)
        Recall = 0.0 (xử lý zero_division)
        F1 = 0.0 (xử lý zero_division)
    """
    y_true = ["A", "A", "A", "B", "B", "B", "B", "C", "C", "D"]
    y_pred = ["A", "A", "B", "B", "B", "C", "D", "C", "A", "D"]
    labels = ["A", "B", "C", "D", "E"]

    metrics = compute_per_class_metrics(
        y_true, y_pred, labels=labels, zero_division=0.0
    )

    # Kiểm tra lớp A
    assert metrics["A"]["precision"] == pytest.approx(2.0 / 3.0, abs=TOLERANCE)
    assert metrics["A"]["recall"] == pytest.approx(2.0 / 3.0, abs=TOLERANCE)
    assert metrics["A"]["f1"] == pytest.approx(2.0 / 3.0, abs=TOLERANCE)
    assert metrics["A"]["support"] == 3.0

    # Kiểm tra lớp B
    assert metrics["B"]["precision"] == pytest.approx(2.0 / 3.0, abs=TOLERANCE)
    assert metrics["B"]["recall"] == pytest.approx(1.0 / 2.0, abs=TOLERANCE)
    assert metrics["B"]["f1"] == pytest.approx(4.0 / 7.0, abs=TOLERANCE)
    assert metrics["B"]["support"] == 4.0

    # Kiểm tra lớp C
    assert metrics["C"]["precision"] == pytest.approx(0.5, abs=TOLERANCE)
    assert metrics["C"]["recall"] == pytest.approx(0.5, abs=TOLERANCE)
    assert metrics["C"]["f1"] == pytest.approx(0.5, abs=TOLERANCE)
    assert metrics["C"]["support"] == 2.0

    # Kiểm tra lớp D
    assert metrics["D"]["precision"] == pytest.approx(0.5, abs=TOLERANCE)
    assert metrics["D"]["recall"] == pytest.approx(1.0, abs=TOLERANCE)
    assert metrics["D"]["f1"] == pytest.approx(2.0 / 3.0, abs=TOLERANCE)
    assert metrics["D"]["support"] == 1.0

    # Kiểm tra lớp E (Zero-support)
    assert metrics["E"]["precision"] == 0.0
    assert metrics["E"]["recall"] == 0.0
    assert metrics["E"]["f1"] == 0.0
    assert metrics["E"]["support"] == 0.0


# =====================================================================
# 3. Kiểm thử Macro-F1, Weighted-F1 và All-label Schema Coverage
# =====================================================================


def test_hand_computed_macro_and_weighted_f1() -> None:
    """
    Kiểm tra tính tay Macro-F1 và Weighted-F1 trên tập 10 mẫu đã tính toán ở test trên:
    - F1 từng lớp: F1(A) = 2/3, F1(B) = 4/7, F1(C) = 1/2, F1(D) = 2/3, F1(E) = 0.
    - Tổng F1 của 4 lớp quan sát {A, B, C, D}:
        2/3 + 4/7 + 1/2 + 2/3 = 4/3 + 4/7 + 1/2
        Quy đồng mẫu số 42: (56 + 24 + 21) / 42 = 101 / 42.

    1. Macro-F1 trên 4 lớp {A, B, C, D}:
        (101 / 42) / 4 = 101 / 168 ≈ 0.6011904762.
    2. Macro-F1 bao gồm lớp zero-support {A, B, C, D, E} (5 lớp):
        (101 / 42 + 0) / 5 = 101 / 210 ≈ 0.4809523810.
    3. Weighted-F1 trên {A, B, C, D, E}:
        Tổng trọng số support = 3 + 4 + 2 + 1 + 0 = 10.
        Tổng có trọng số = 3 * (2/3) + 4 * (4/7) + 2 * (1/2) + 1 * (2/3) + 0 * 0
                         = 2 + 16/7 + 1 + 2/3 = 3 + 16/7 + 2/3
        Quy đồng mẫu 21: (63 + 48 + 14) / 21 = 125 / 21.
        Weighted-F1 = (125 / 21) / 10 = 125 / 210 = 25 / 42 ≈ 0.5952380952.
    """
    y_true = ["A", "A", "A", "B", "B", "B", "B", "C", "C", "D"]
    y_pred = ["A", "A", "B", "B", "B", "C", "D", "C", "A", "D"]

    # Đánh giá trên 4 lớp xuất hiện
    macro_4 = compute_macro_f1(y_true, y_pred, labels=["A", "B", "C", "D"])
    assert macro_4 == pytest.approx(101.0 / 168.0, abs=TOLERANCE)

    # Đánh giá trên 5 lớp (bao gồm nhãn zero-support E)
    macro_5 = compute_macro_f1(y_true, y_pred, labels=["A", "B", "C", "D", "E"])
    assert macro_5 == pytest.approx(101.0 / 210.0, abs=TOLERANCE)

    # Weighted-F1 trên 5 lớp (lớp E support=0 không làm thay đổi tử số và mẫu số)
    weighted_5 = compute_weighted_f1(y_true, y_pred, labels=["A", "B", "C", "D", "E"])
    assert weighted_5 == pytest.approx(25.0 / 42.0, abs=TOLERANCE)

    # Weighted-F1 trên 4 lớp cũng cho cùng kết quả 25 / 42
    weighted_4 = compute_weighted_f1(y_true, y_pred, labels=["A", "B", "C", "D"])
    assert weighted_4 == pytest.approx(25.0 / 42.0, abs=TOLERANCE)


def test_all_label_schema_coverage_with_zero_support() -> None:
    """
    Kiểm tra độ bao phủ toàn bộ intent schema (189 lớp) khi tập kiểm thử chỉ xuất hiện một số ít lớp.
    Giả sử tập kiểm thử chỉ có 2 intent thuộc schema:
        LATEX_TAI_LIEU_TOI_GIAN: 1 mẫu, dự đoán đúng -> F1 = 1.0
        MATH_MATH_BASIC_PHAN_SO: 1 mẫu, dự đoán đúng -> F1 = 1.0
    Toàn bộ 187 intent còn lại trong schema không có mẫu (zero-support) -> F1 = 0.0.

    Giá trị tính tay lý thuyết:
    - Observed Macro-F1 (trên 2 lớp): (1.0 + 1.0) / 2 = 1.0.
    - Schema Macro-F1 (trên 189 lớp): (1.0 + 1.0 + 187 * 0.0) / 189 = 2 / 189 ≈ 0.0105820106.
    - Weighted-F1: (1 * 1.0 + 1 * 1.0) / 2 = 1.0.
    """
    schema_labels = load_intent_labels(DEFAULT_INTENTS_PATH)
    total_schema_classes = len(schema_labels)
    assert total_schema_classes == 189
    assert "LATEX_TAI_LIEU_TOI_GIAN" in schema_labels
    assert "MATH_MATH_BASIC_PHAN_SO" in schema_labels

    y_true = ["LATEX_TAI_LIEU_TOI_GIAN", "MATH_MATH_BASIC_PHAN_SO"]
    y_pred = ["LATEX_TAI_LIEU_TOI_GIAN", "MATH_MATH_BASIC_PHAN_SO"]

    # 1. Macro-F1 cục bộ chỉ trên các lớp quan sát
    observed_macro = compute_macro_f1(y_true, y_pred, labels=None)
    assert observed_macro == pytest.approx(1.0, abs=TOLERANCE)

    # 2. Schema Macro-F1 trên toàn bộ 189 lớp
    schema_macro = compute_schema_macro_f1(
        y_true, y_pred, intents_path=DEFAULT_INTENTS_PATH
    )
    expected_schema_macro = 2.0 / float(total_schema_classes)
    assert schema_macro == pytest.approx(expected_schema_macro, abs=TOLERANCE)

    # 3. Kiểm tra thông qua hàm tích hợp compute_classification_metrics
    result = compute_classification_metrics(
        y_true,
        y_pred,
        evaluate_against_schema=True,
        intents_path=DEFAULT_INTENTS_PATH,
    )
    assert isinstance(result, ClassificationMetricsResult)
    assert result.accuracy == 1.0
    assert result.weighted_f1 == 1.0
    assert result.macro_f1 == pytest.approx(expected_schema_macro, abs=TOLERANCE)
    assert result.total_classes == 189
    assert result.total_samples == 2
    # Đảm bảo các lớp zero-support có trường support = 0.0 và f1 = 0.0
    assert result.per_class["LATEX_BANG_TABULAR_CO_BAN"]["support"] == 0.0
    assert result.per_class["LATEX_BANG_TABULAR_CO_BAN"]["f1"] == 0.0


# =====================================================================
# 4. Kiểm thử Mean và Sample Standard Deviation (Bessel's Correction n-1)
# =====================================================================


def test_hand_computed_sample_std_bessel_correction() -> None:
    """
    Kiểm tra giá trị đối chuẩn tính tay cho Trung bình mẫu và Độ lệch chuẩn mẫu (n-1):
    Dữ liệu N = 7: [10.0, 12.0, 23.0, 23.0, 16.0, 24.0, 18.0]
    Tổng: 10 + 12 + 23 + 23 + 16 + 24 + 18 = 126
    Trung bình mẫu: x_bar = 126 / 7 = 18.0

    Độ lệch (x_i - x_bar):
        [-8, -6, 5, 5, -2, 6, 0]
    Bình phương độ lệch (x_i - x_bar)^2:
        [64, 36, 25, 25, 4, 36, 0]
    Tổng bình phương độ lệch: 64 + 36 + 25 + 25 + 4 + 36 + 0 = 190.

    1. Độ lệch chuẩn mẫu (ddof=1, hiệu chỉnh Bessel n-1 = 6):
        s^2 = 190 / 6 = 95 / 3 ≈ 31.6666666667
        s = sqrt(95 / 3) ≈ 5.6273143389
    2. Độ lệch chuẩn tổng thể (ddof=0, chia cho n = 7):
        sigma = sqrt(190 / 7) ≈ 5.2098807
    """
    data = [10.0, 12.0, 23.0, 23.0, 16.0, 24.0, 18.0]

    # Kiểm tra Mean
    mean_val = compute_mean(data)
    assert mean_val == pytest.approx(18.0, abs=TOLERANCE)

    # Kiểm tra Sample Standard Deviation (ddof=1)
    sample_std = compute_sample_std(data, ddof=1)
    expected_sample_std = math.sqrt(95.0 / 3.0)
    assert sample_std == pytest.approx(expected_sample_std, abs=TOLERANCE)

    # Kiểm tra khi ddof=0 (phân biệt rõ ràng giữa n và n-1)
    pop_std = compute_sample_std(data, ddof=0)
    expected_pop_std = math.sqrt(190.0 / 7.0)
    assert pop_std == pytest.approx(expected_pop_std, abs=TOLERANCE)
    assert sample_std > pop_std  # Hiệu chỉnh Bessel luôn lớn hơn khi n > 1


def test_sample_std_edge_cases() -> None:
    """
    Kiểm tra các trường hợp biên của compute_sample_std:
    - Khi n <= ddof (ví dụ n = 1 với ddof = 1, hoặc n = 0): trả về 0.0 an toàn.
    - Dãy hằng số [5.0, 5.0, 5.0]: độ lệch chuẩn bằng 0.0 tuyệt đối.
    """
    assert compute_sample_std([], ddof=1) == 0.0
    assert compute_sample_std([42.0], ddof=1) == 0.0
    assert compute_sample_std([5.0, 5.0, 5.0], ddof=1) == 0.0


# =====================================================================
# 5. Kiểm thử Ranking Metrics: Top-k Accuracy & MRR
# =====================================================================


def test_hand_computed_ranking_metrics() -> None:
    """
    Kiểm tra đối chuẩn tính tay cho Top-k Accuracy và MRR (Voorhees, 1999):
    Tập gồm N = 4 truy vấn:
    - Mẫu 0: y_true = "A", preds = ["A", "B", "C", "D"] -> rank = 1, RR = 1 / 1 = 1.0
    - Mẫu 1: y_true = "B", preds = ["C", "B", "A", "D"] -> rank = 2, RR = 1 / 2 = 0.5
    - Mẫu 2: y_true = "C", preds = ["D", "A", "B", "C"] -> rank = 4, RR = 1 / 4 = 0.25
    - Mẫu 3: y_true = "D", preds = ["A", "B", "C", "E"] -> không nằm trong preds -> RR = 0.0

    Giá trị tính tay lý thuyết:
    - Top-1: Chỉ mẫu 0 trúng -> 1 / 4 = 0.25.
    - Top-2: Mẫu 0 và mẫu 1 trúng -> 2 / 4 = 0.50.
    - Top-3: Mẫu 0 và mẫu 1 trúng (mẫu 2 ở hạng 4) -> 2 / 4 = 0.50.
    - Top-4: Mẫu 0, mẫu 1, mẫu 2 trúng -> 3 / 4 = 0.75.
    - MRR = (1.0 + 0.5 + 0.25 + 0.0) / 4 = 1.75 / 4 = 7 / 16 = 0.4375.
    """
    y_true = ["A", "B", "C", "D"]
    y_pred_ranked = [
        ["A", "B", "C", "D"],
        ["C", "B", "A", "D"],
        ["D", "A", "B", "C"],
        ["A", "B", "C", "E"],
    ]

    assert compute_top_k_accuracy(y_true, y_pred_ranked, k=1) == pytest.approx(
        0.25, abs=TOLERANCE
    )
    assert compute_top_k_accuracy(y_true, y_pred_ranked, k=2) == pytest.approx(
        0.50, abs=TOLERANCE
    )
    assert compute_top_k_accuracy(y_true, y_pred_ranked, k=3) == pytest.approx(
        0.50, abs=TOLERANCE
    )
    assert compute_top_k_accuracy(y_true, y_pred_ranked, k=4) == pytest.approx(
        0.75, abs=TOLERANCE
    )

    mrr = compute_mrr(y_true, y_pred_ranked)
    assert mrr == pytest.approx(7.0 / 16.0, abs=TOLERANCE)


def test_ranking_metrics_edge_cases() -> None:
    """Kiểm tra các trường hợp biên của Top-k và MRR."""
    assert compute_top_k_accuracy([], [], k=3) == 0.0
    assert compute_mrr([], []) == 0.0

    with pytest.raises(ValueError, match="k phải >= 1"):
        compute_top_k_accuracy(["A"], [["A"]], k=0)

    with pytest.raises(ValueError, match="không đồng nhất"):
        compute_mrr(["A"], [["A"], ["B"]])


# =====================================================================
# 6. Kiểm thử Bootstrap CI Boundaries (Percentile Method)
# =====================================================================


def test_hand_computed_bootstrap_ci_boundaries() -> None:
    """
    Kiểm tra ước lượng khoảng tin cậy Bootstrap phân vị (Efron & Tibshirani, 1993):
    1. Dãy hằng số [7.5, 7.5, 7.5, 7.5]:
       Mọi mẫu lấy có hoàn lại đều có trung bình là 7.5.
       Do đó cận dưới và cận trên phải hội tụ chính xác về 7.5.
    2. Danh sách 1 phần tử [42.0]:
       Trả về (42.0, 42.0).
    3. Tính chất bao hàm (Bounding property):
       Cận dưới <= Cận trên, và giá trị trung bình mẫu phải nằm trong [ci_lower, ci_upper].
    4. Tính tất định theo hạt giống ngẫu nhiên (Determinism with seed).
    """
    constant_data = [7.5, 7.5, 7.5, 7.5]
    ci_lower, ci_upper = compute_bootstrap_ci(
        constant_data,
        num_bootstrap=100,
        confidence_level=0.95,
        seed=DEFAULT_SEED,
    )
    assert ci_lower == pytest.approx(7.5, abs=TOLERANCE)
    assert ci_upper == pytest.approx(7.5, abs=TOLERANCE)

    single_elem = [42.0]
    assert compute_bootstrap_ci(single_elem) == (42.0, 42.0)

    # Kiểm tra tính tất định và tính chất bao hàm trên dãy số thực
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    mean_val = compute_mean(data)
    ci1_low, ci1_high = compute_bootstrap_ci(
        data,
        num_bootstrap=500,
        confidence_level=0.90,
        seed=123,
    )
    ci2_low, ci2_high = compute_bootstrap_ci(
        data,
        num_bootstrap=500,
        confidence_level=0.90,
        seed=123,
    )
    # Lặp lại cùng seed phải trả về kết quả số thực giống hệt nhau
    assert ci1_low == ci2_low
    assert ci1_high == ci2_high
    assert min(data) <= ci1_low <= mean_val <= ci1_high <= max(data)


def test_paired_bootstrap_ci_edge_cases() -> None:
    """Kiểm tra Paired Bootstrap CI khi hai chuỗi giống hệt nhau hoặc lệch hoàn toàn."""
    y_true = ["A", "B", "C"]
    y_pred_perfect = ["A", "B", "C"]

    # Với accuracy, mô hình hoàn hảo luôn có điểm 1.0 trên mọi mẫu con
    low, high = compute_paired_bootstrap_ci(
        y_true,
        y_pred_perfect,
        metric_fn=compute_accuracy,
        num_bootstrap=100,
        seed=DEFAULT_SEED,
    )
    assert low == 1.0
    assert high == 1.0


# =====================================================================
# 7. Kiểm định McNemar và Trường Hợp Biên b + c = 0
# =====================================================================


def test_mcnemar_edge_case_b_plus_c_zero() -> None:
    """
    Kiểm tra trường hợp biên quan trọng: b + c = 0.
    Khi hai mô hình hoàn toàn đồng thuận về tính đúng/sai trên mọi mẫu dữ liệu (b=0, c=0),
    không có bất kỳ sai lệch nào.
    Yêu cầu đặc tả: statistic = 0.0 và p_value = 1.0.
    """
    stat, p_val = mcnemar_test_from_counts(b=0, c=0, continuity_correction=True)
    assert stat == 0.0
    assert p_val == 1.0

    stat_no_corr, p_val_no_corr = mcnemar_test_from_counts(
        b=0, c=0, continuity_correction=False
    )
    assert stat_no_corr == 0.0
    assert p_val_no_corr == 1.0

    # Kiểm thử thông qua chuỗi dự đoán thực tế giống hệt nhau
    y_true = ["A", "B", "C", "D"]
    y_pred_a = ["A", "B", "C", "A"]  # Mẫu 3 sai
    y_pred_b = ["A", "B", "C", "A"]  # Mẫu 3 cũng sai hệt Model A
    stat_seq, p_val_seq = compute_mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert stat_seq == 0.0
    assert p_val_seq == 1.0


def test_hand_computed_mcnemar_chi_square() -> None:
    """
    Kiểm tra đối chuẩn tính tay công thức kiểm định McNemar (Dietterich, 1998):
    Trường hợp 1: b = 25, c = 5 (tổng bất đồng b + c = 30, |b - c| = 20)
    1. Có hiệu chỉnh Yates:
        corrected_diff = max(0, 20 - 1) = 19
        chi_square = (19^2) / 30 = 361 / 30 ≈ 12.0333333333
    2. Không có hiệu chỉnh Yates:
        chi_square = (20^2) / 30 = 400 / 30 = 40 / 3 ≈ 13.3333333333

    Trường hợp 2: b = 4, c = 1 (tổng bất đồng b + c = 5, |b - c| = 3)
    1. Có hiệu chỉnh Yates:
        chi_square = (3 - 1)^2 / 5 = 4 / 5 = 0.80
    2. Không có hiệu chỉnh Yates:
        chi_square = 3^2 / 5 = 9 / 5 = 1.80

    Trường hợp 3: |b - c| <= 1 có hiệu chỉnh Yates:
        b = 5, c = 5 -> |5 - 5| = 0 -> stat = 0.0, p_val = 1.0
        b = 6, c = 5 -> |6 - 5| = 1 -> max(0, 1 - 1) = 0 -> stat = 0.0, p_val = 1.0
    """
    # Trường hợp 1: b = 25, c = 5
    stat_yates, p_yates = mcnemar_test_from_counts(
        b=25, c=5, continuity_correction=True
    )
    assert stat_yates == pytest.approx(361.0 / 30.0, abs=TOLERANCE)
    assert p_yates < 0.001

    stat_raw, p_raw = mcnemar_test_from_counts(b=25, c=5, continuity_correction=False)
    assert stat_raw == pytest.approx(40.0 / 3.0, abs=TOLERANCE)
    assert p_raw < 0.001

    # Trường hợp 2: b = 4, c = 1
    stat_2, _ = mcnemar_test_from_counts(b=4, c=1, continuity_correction=True)
    assert stat_2 == pytest.approx(0.8, abs=TOLERANCE)

    stat_2_raw, _ = mcnemar_test_from_counts(b=4, c=1, continuity_correction=False)
    assert stat_2_raw == pytest.approx(1.8, abs=TOLERANCE)

    # Trường hợp 3: |b - c| <= 1 với Yates
    stat_equal, p_equal = mcnemar_test_from_counts(b=5, c=5, continuity_correction=True)
    assert stat_equal == 0.0
    assert p_equal == 1.0

    stat_diff1, p_diff1 = mcnemar_test_from_counts(b=6, c=5, continuity_correction=True)
    assert stat_diff1 == 0.0
    assert p_diff1 == 1.0


# =====================================================================
# 8. Kiểm thử ECE Binning (Expected Calibration Error)
# =====================================================================


def test_hand_computed_ece_binning() -> None:
    """
    Kiểm tra đối chuẩn tính tay công thức ECE (Guo et al., ICML 2017) với phân bin đều:
    Tập gồm N = 8 mẫu, chia thành M = 4 bins (độ rộng bin = 0.25):
        bin_edges = [0.0, 0.25, 0.50, 0.75, 1.00]
        bin_idx = min(int(conf * 4), 3)

    Chi tiết các mẫu:
    - Mẫu 1: yt=1, yp=1 (đúng), conf=0.10 -> Bin 0: [0.0, 0.25)
    - Mẫu 2: yt=1, yp=0 (sai),  conf=0.20 -> Bin 0: [0.0, 0.25)
    - Mẫu 3: yt=0, yp=0 (đúng), conf=0.40 -> Bin 1: [0.25, 0.50)
    (Bin 2: [0.50, 0.75) không có mẫu nào rơi vào -> Empty bin!)
    - Mẫu 4: yt=1, yp=1 (đúng), conf=0.80 -> Bin 3: [0.75, 1.00]
    - Mẫu 5: yt=1, yp=1 (đúng), conf=0.85 -> Bin 3: [0.75, 1.00]
    - Mẫu 6: yt=0, yp=1 (sai),  conf=0.90 -> Bin 3: [0.75, 1.00]
    - Mẫu 7: yt=1, yp=1 (đúng), conf=0.95 -> Bin 3: [0.75, 1.00]
    - Mẫu 8: yt=1, yp=1 (đúng), conf=1.00 -> Bin 3: [0.75, 1.00]

    Tính toán theo từng bin:
    - Bin 0:
        |B_0| = 2. Số mẫu đúng = 1 -> acc(B_0) = 1 / 2 = 0.50
        conf(B_0) = (0.10 + 0.20) / 2 = 0.15
        Lệch hiệu chuẩn: |0.50 - 0.15| = 0.35
        Đóng góp vào ECE: (2 / 8) * 0.35 = 0.0875
    - Bin 1:
        |B_1| = 1. Số mẫu đúng = 1 -> acc(B_1) = 1.00
        conf(B_1) = 0.40
        Lệch hiệu chuẩn: |1.00 - 0.40| = 0.60
        Đóng góp vào ECE: (1 / 8) * 0.60 = 0.0750
    - Bin 2:
        |B_2| = 0. Thùng rỗng -> acc = 0.0, conf = 0.0
        Đóng góp vào ECE: 0.0
    - Bin 3:
        |B_3| = 5. Số mẫu đúng = 4 (mẫu 4, 5, 7, 8) -> acc(B_3) = 4 / 5 = 0.80
        conf(B_3) = (0.80 + 0.85 + 0.90 + 0.95 + 1.00) / 5 = 4.50 / 5 = 0.90
        Lệch hiệu chuẩn: |0.80 - 0.90| = 0.10
        Đóng góp vào ECE: (5 / 8) * 0.10 = 0.0625

    Tổng ECE tính tay:
        ECE = 0.0875 + 0.0750 + 0.0 + 0.0625 = 0.2250.
    """
    y_true = [1, 1, 0, 1, 1, 0, 1, 1]
    y_pred = [1, 0, 0, 1, 1, 1, 1, 1]
    confidences = [0.10, 0.20, 0.40, 0.80, 0.85, 0.90, 0.95, 1.00]
    num_bins = 4

    result: ECEResult = compute_ece_details(
        y_true, y_pred, confidences, num_bins=num_bins
    )

    # 1. Kiểm tra cấu trúc phân bin
    assert result.num_bins == 4
    assert result.total_samples == 8
    assert result.bin_edges == [0.0, 0.25, 0.50, 0.75, 1.00]
    assert result.bin_counts == [2, 1, 0, 5]

    # 2. Kiểm tra độ chính xác và độ tin cậy từng bin
    assert result.bin_accuracies == [
        pytest.approx(0.50, abs=TOLERANCE),
        pytest.approx(1.00, abs=TOLERANCE),
        0.0,
        pytest.approx(0.80, abs=TOLERANCE),
    ]
    assert result.bin_confidences == [
        pytest.approx(0.15, abs=TOLERANCE),
        pytest.approx(0.40, abs=TOLERANCE),
        0.0,
        pytest.approx(0.90, abs=TOLERANCE),
    ]

    # 3. Kiểm tra tổng ECE chính xác
    assert result.ece == pytest.approx(0.2250, abs=TOLERANCE)
    assert compute_ece(y_true, y_pred, confidences, num_bins=num_bins) == pytest.approx(
        0.2250, abs=TOLERANCE
    )


def test_ece_edge_cases() -> None:
    """
    Kiểm tra các trường hợp biên của ECE:
    - Tập dữ liệu rỗng: ECE = 0.0.
    - Hiệu chuẩn hoàn hảo (Perfect calibration): acc == conf trên mọi bin -> ECE = 0.0.
    - Hoàn toàn mất hiệu chuẩn (Uncalibrated): conf = 1.0 nhưng đoán sai toàn bộ -> ECE = 1.0.
    """
    assert compute_ece([], [], [], num_bins=5) == 0.0

    # Hoàn hảo: 2 mẫu tin cậy 1.0 đoán đúng cả 2
    assert compute_ece([1, 1], [1, 1], [1.0, 1.0], num_bins=5) == 0.0

    # Sai hoàn toàn nhưng tự tin tuyệt đối: ECE = |0.0 - 1.0| = 1.0
    assert compute_ece([1, 1], [0, 0], [1.0, 1.0], num_bins=5) == 1.0
