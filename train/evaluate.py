"""
Module tiện ích đo lường đánh giá học thuật và phân tích thống kê (Evaluation Metrics and Statistical Utilities).

Tệp tin: train/evaluate.py
Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Identifiers in English, docstrings và comments bằng tiếng Việt giải thích lý do (WHY) và công thức bài báo khoa học.
- Toàn bộ hằng số cấu hình được nạp từ core/config.py (không dùng số ma thuật).
- Thiết lập seed tất định thông qua core.config.set_seed.
- Trung thực số liệu (Honesty Rules): Không bịa đặt chỉ số, mọi chỉ số đều được tính toán thực tế.

Các thành phần chính:
1. Phân loại chuẩn (Classification Metrics):
   - Accuracy
   - Per-class Precision, Recall, F1 với xử lý chia cho 0 (zero-division handling) (Van Rijsbergen, 1979)
   - Macro-F1 đánh giá trên toàn bộ các lớp định nghĩa trong intent schema
   - Weighted-F1 trọng số theo kích thước hỗ trợ (support-weighted)
2. Chỉ số xếp hạng (Ranking Metrics):
   - Top-k Accuracy (kịch bản k ứng viên hàng đầu)
   - Mean Reciprocal Rank (MRR - Voorhees, 1999)
3. Thống kê tóm tắt (Summary Statistics):
   - Mean (Giá trị trung bình mẫu)
   - Sample Standard Deviation với n-1 bậc tự do (Hiệu chỉnh Bessel - Bessel's correction)
4. Ước lượng khoảng tin cậy Bootstrap phi tham số (Non-parametric Bootstrap Confidence Interval):
   - Phương pháp phân vị (Percentile method - Efron & Tibshirani, 1993)
5. Kiểm định McNemar (McNemar's Test):
   - Hiệu chỉnh liên tục Yates (Yates' continuity correction - McNemar, 1947; Dietterich, 1998)
   - Xử lý trường hợp b + c = 0 khi hai mô hình có dự đoán hoàn toàn đồng thuận
6. Sai số hiệu chuẩn kỳ vọng (Expected Calibration Error - ECE):
   - Các khoảng tin cậy có độ rộng bằng nhau (equal-width confidence bins - Guo et al., 2017)
7. Trực quan hóa ma trận nhầm lẫn (Confusion Matrix Figure):
   - Sử dụng matplotlib lưu tự động vào report/figures/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

# Đảm bảo đường dẫn gốc dự án có trong sys.path khi thực thi trực tiếp script
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

# Thiết lập backend Agg không tương tác để an toàn trên môi trường máy chủ headless không có màn hình X11
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats

    HAS_SCIPY: bool = True
except ImportError:
    HAS_SCIPY = False

from core.config import (
    DEFAULT_BOOTSTRAP_ROUNDS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_ECE_BINS,
    DEFAULT_INTENTS_PATH,
    DEFAULT_RANKING_TOP_K,
    DEFAULT_SEED,
    DEFAULT_ZERO_DIVISION,
    set_seed,
)

# =====================================================================
# Cấu trúc dữ liệu kết quả đo lường (Data Classes)
# =====================================================================


@dataclass
class ClassificationMetricsResult:
    """
    Kết quả tính toán toàn diện các chỉ số phân loại ý định (Classification Metrics).
    """

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    total_samples: int
    total_classes: int

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang định dạng dictionary phục vụ lưu trữ JSON."""
        return asdict(self)


@dataclass
class ECEResult:
    """
    Kết quả tính sai số hiệu chuẩn kỳ vọng (Expected Calibration Error - ECE).
    Tham chiếu: Guo et al., ICML 2017 'On Calibration of Modern Neural Networks'.
    """

    ece: float
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]
    bin_edges: list[float]
    num_bins: int
    total_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary."""
        return asdict(self)


@dataclass
class McNemarResult:
    """
    Kết quả kiểm định McNemar so sánh hai mô hình phân loại trên cùng tập dữ liệu.
    Tham chiếu: McNemar (1947), Dietterich (1998).
    """

    statistic: float
    p_value: float
    b: int  # Model A đúng, Model B sai
    c: int  # Model A sai, Model B đúng
    contingency_table: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary."""
        return asdict(self)


# =====================================================================
# 1. Nạp nhãn kịch bản từ Intent Schema
# =====================================================================


def load_intent_labels(intents_path: str | Path = DEFAULT_INTENTS_PATH) -> list[str]:
    """
    Nạp danh sách định danh intent chuẩn hóa được định nghĩa trong schema kịch bản (data/intents.json).

    Lý do:
        Macro-F1 cần được tính toán trên toàn bộ không gian lớp của schema (kể cả các intent không xuất
        hiện trong tập kiểm thử cục bộ), nhằm phản ánh chính xác độ bao phủ của bộ phân loại.

    Tham số:
        intents_path: Đường dẫn tới tệp JSON kịch bản.

    Trả về:
        Danh sách tên intent đã được sắp xếp bảng chữ cái.
    """
    path = Path(intents_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp intent schema tại: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "intent" in item:
                labels.append(str(item["intent"]))
    elif isinstance(data, dict) and "intents" in data:
        for item in data["intents"]:
            if isinstance(item, dict) and "intent" in item:
                labels.append(str(item["intent"]))

    return sorted(set(labels))


# =====================================================================
# 2. Thống kê tóm tắt (Summary Statistics)
# =====================================================================


def compute_mean(values: Sequence[float]) -> float:
    """
    Tính giá trị trung bình cộng mẫu (Sample Mean).

    Công thức toán học:
        \\bar{x} = \\frac{1}{n} \\sum_{i=1}^n x_i

    Tham số:
        values: Chuỗi các giá trị số thực.

    Trả về:
        Giá trị trung bình cộng.

    Ngoại lệ:
        ValueError nếu chuỗi rỗng.
    """
    if not values:
        raise ValueError("Không thể tính giá trị trung bình trên chuỗi rỗng (values must not be empty).")
    return float(sum(values) / len(values))


def compute_sample_std(values: Sequence[float], ddof: int = 1) -> float:
    """
    Tính độ lệch chuẩn mẫu với n - ddof bậc tự do (mặc định ddof=1, hiệu chỉnh Bessel).

    Công thức toán học (Bessel's correction):
        s = \\sqrt{ \\frac{1}{n - 1} \\sum_{i=1}^n (x_i - \\bar{x})^2 }

    Lý do:
        Ước lượng phương sai mẫu với mẫu số n - 1 là ước lượng không chệch (unbiased estimator)
        của phương sai tổng thể. Khi n <= ddof (ví dụ n = 1 với ddof = 1), hàm trả về 0.0 an toàn.

    Tham số:
        values: Chuỗi các giá trị số thực.
        ddof: Bậc tự do hiệu chỉnh (Delta Degrees of Freedom, mặc định 1 cho mẫu thống kê).

    Trả về:
        Độ lệch chuẩn mẫu số thực.
    """
    n = len(values)
    if n <= ddof:
        return 0.0

    mean_val = compute_mean(values)
    sum_squared_diff = sum((x - mean_val) ** 2 for x in values)
    variance = sum_squared_diff / (n - ddof)
    return float(math.sqrt(max(0.0, variance)))


# =====================================================================
# 3. Phân loại chuẩn (Classification Metrics)
# =====================================================================


def compute_accuracy(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    zero_division: float = DEFAULT_ZERO_DIVISION,
) -> float:
    """
    Tính độ chính xác tổng thể (Overall Accuracy).

    Công thức toán học:
        \\text{Accuracy} = \\frac{1}{N} \\sum_{i=1}^N \\mathbf{1}(y_{true, i} = y_{pred, i})

    Tham số:
        y_true: Danh sách nhãn thực tế ground-truth.
        y_pred: Danh sách nhãn dự đoán từ mô hình.
        zero_division: Giá trị trả về khi số lượng mẫu N = 0 (mặc định 0.0).

    Trả về:
        Tỷ lệ dự đoán chính xác trong khoảng [0.0, 1.0].
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Kích thước y_true ({len(y_true)}) và y_pred ({len(y_pred)}) không đồng nhất."
        )
    if not y_true:
        return float(zero_division)

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return float(correct / len(y_true))


def compute_per_class_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    zero_division: float = DEFAULT_ZERO_DIVISION,
) -> dict[str, dict[str, float]]:
    """
    Tính độ chuẩn xác (Precision), độ thu hồi (Recall), điểm F1 và số lượng hỗ trợ (Support) cho từng lớp.

    Công thức toán học (Van Rijsbergen, 1979 - Information Retrieval):
        P_c = \\frac{TP_c}{TP_c + FP_c}, \\quad R_c = \\frac{TP_c}{TP_c + FN_c}, \\quad F1_c = \\frac{2 P_c R_c}{P_c + R_c}

    Xử lý chia cho 0 (zero-division handling):
        Khi mẫu số bằng 0 (ví dụ không có dự đoán nào hoặc không có mẫu thực tế), gán giá trị bằng zero_division (0.0).

    Tham số:
        y_true: Chuỗi nhãn thực tế.
        y_pred: Chuỗi nhãn dự đoán.
        labels: Danh sách các lớp cần đánh giá. Nếu None, tự động trích xuất tập hợp các nhãn duy nhất.
        zero_division: Giá trị thay thế khi mẫu số bằng 0.

    Trả về:
        Dictionary dạng: {label_name: {"precision": float, "recall": float, "f1": float, "support": float}}
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Kích thước y_true ({len(y_true)}) và y_pred ({len(y_pred)}) không đồng nhất."
        )

    if labels is None:
        target_labels = sorted({str(y) for y in y_true} | {str(y) for y in y_pred})
    else:
        target_labels = [str(lbl) for lbl in labels]

    # Đếm các đại lượng TP, FP, FN, Support cho từng lớp
    tp_counts: dict[str, int] = {lbl: 0 for lbl in target_labels}
    fp_counts: dict[str, int] = {lbl: 0 for lbl in target_labels}
    fn_counts: dict[str, int] = {lbl: 0 for lbl in target_labels}
    support_counts: dict[str, int] = {lbl: 0 for lbl in target_labels}

    label_set = set(target_labels)

    for yt_raw, yp_raw in zip(y_true, y_pred):
        yt = str(yt_raw)
        yp = str(yp_raw)

        if yt in label_set:
            support_counts[yt] += 1

        if yt == yp:
            if yt in label_set:
                tp_counts[yt] += 1
        else:
            if yp in label_set:
                fp_counts[yp] += 1
            if yt in label_set:
                fn_counts[yt] += 1

    results: dict[str, dict[str, float]] = {}
    for lbl in target_labels:
        tp = tp_counts[lbl]
        fp = fp_counts[lbl]
        fn = fn_counts[lbl]
        sup = support_counts[lbl]

        # Tính Precision
        if tp + fp > 0:
            precision = float(tp / (tp + fp))
        else:
            precision = float(zero_division)

        # Tính Recall
        if tp + fn > 0:
            recall = float(tp / (tp + fn))
        else:
            recall = float(zero_division)

        # Tính F1-score
        if precision + recall > 0:
            f1 = float((2 * precision * recall) / (precision + recall))
        else:
            f1 = float(zero_division)

        results[lbl] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sup),
        }

    return results


def compute_macro_f1(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    zero_division: float = DEFAULT_ZERO_DIVISION,
) -> float:
    """
    Tính Macro-F1 trung bình không trọng số trên tập hợp các lớp được chỉ định.

    Công thức toán học:
        \\text{Macro-F1} = \\frac{1}{|C|} \\sum_{c \\in C} F1_c

    Đặc tả yêu cầu:
        Khi truyền vào danh sách nhãn từ intent schema (189 intents), Macro-F1 sẽ được tính
        trên toàn bộ 189 lớp. Các lớp không xuất hiện trong test set có F1=0 và làm giảm điểm
        tổng thể phản ánh mức độ thiếu hụt dữ liệu kiểm thử.

    Tham số:
        y_true: Danh sách nhãn thực tế.
        y_pred: Danh sách nhãn dự đoán.
        labels: Danh sách toàn bộ các lớp cần tính trung bình (ví dụ schema intents).
        zero_division: Giá trị xử lý chia 0.

    Trả về:
        Điểm Macro-F1 số thực trong đoạn [0.0, 1.0].
    """
    per_class = compute_per_class_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        zero_division=zero_division,
    )
    if not per_class:
        return float(zero_division)

    f1_list = [metrics["f1"] for metrics in per_class.values()]
    return float(sum(f1_list) / len(f1_list))


def compute_schema_macro_f1(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    intents_path: str | Path = DEFAULT_INTENTS_PATH,
    zero_division: float = DEFAULT_ZERO_DIVISION,
) -> float:
    """
    Hàm tiện ích tính Macro-F1 trên toàn bộ các lớp định nghĩa trong intent schema.
    """
    schema_labels = load_intent_labels(intents_path=intents_path)
    return compute_macro_f1(
        y_true=y_true,
        y_pred=y_pred,
        labels=schema_labels,
        zero_division=zero_division,
    )


def compute_weighted_f1(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    zero_division: float = DEFAULT_ZERO_DIVISION,
) -> float:
    """
    Tính Weighted-F1 tính trọng số theo kích thước hỗ trợ (support) thực tế của từng lớp.

    Công thức toán học:
        \\text{Weighted-F1} = \\frac{ \\sum_{c \\in C} \\text{support}(c) \\cdot F1_c }{ \\sum_{c \\in C} \\text{support}(c) }

    Tham số:
        y_true: Danh sách nhãn thực tế.
        y_pred: Danh sách nhãn dự đoán.
        labels: Danh sách lớp tùy chọn.
        zero_division: Giá trị xử lý khi tổng support bằng 0.

    Trả về:
        Điểm Weighted-F1 số thực.
    """
    per_class = compute_per_class_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        zero_division=zero_division,
    )
    total_support = sum(metrics["support"] for metrics in per_class.values())
    if total_support == 0:
        return float(zero_division)

    weighted_f1_sum = sum(metrics["f1"] * metrics["support"] for metrics in per_class.values())
    return float(weighted_f1_sum / total_support)


def compute_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    evaluate_against_schema: bool = False,
    intents_path: str | Path = DEFAULT_INTENTS_PATH,
    zero_division: float = DEFAULT_ZERO_DIVISION,
) -> ClassificationMetricsResult:
    """
    Tính toán tập hợp đầy đủ các chỉ số phân loại: Accuracy, Macro-F1, Weighted-F1 và Per-class F1.

    Tham số:
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        labels: Danh sách lớp đánh giá tường minh.
        evaluate_against_schema: Nếu True và labels là None, tự động nạp toàn bộ nhãn từ intents_path.
        intents_path: Đường dẫn tệp intent schema.
        zero_division: Xử lý chia 0.

    Trả về:
        Đối tượng ClassificationMetricsResult.
    """
    if evaluate_against_schema and labels is None:
        target_labels: Sequence[Any] | None = load_intent_labels(intents_path)
    else:
        target_labels = labels

    accuracy = compute_accuracy(y_true, y_pred, zero_division=zero_division)
    per_class = compute_per_class_metrics(
        y_true, y_pred, labels=target_labels, zero_division=zero_division
    )
    macro_f1 = compute_macro_f1(
        y_true, y_pred, labels=target_labels, zero_division=zero_division
    )
    weighted_f1 = compute_weighted_f1(
        y_true, y_pred, labels=target_labels, zero_division=zero_division
    )

    return ClassificationMetricsResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class=per_class,
        total_samples=len(y_true),
        total_classes=len(per_class),
    )


# =====================================================================
# 4. Chỉ số xếp hạng (Ranking Metrics)
# =====================================================================


def compute_top_k_accuracy(
    y_true: Sequence[Any],
    y_pred_ranked: Sequence[Sequence[Any]],
    k: int = DEFAULT_RANKING_TOP_K,
) -> float:
    """
    Tính độ chính xác Top-k (Top-k Accuracy) trong bài toán xếp hạng intent hoặc retrieval.

    Công thức toán học:
        \\text{Top-}k = \\frac{1}{N} \\sum_{i=1}^N \\mathbf{1}(y_{true, i} \\in \\hat{Y}_{i, 1..k})

    Tham số:
        y_true: Chuỗi nhãn ground-truth độ dài N.
        y_pred_ranked: Chuỗi các danh sách nhãn dự đoán đã được sắp xếp giảm dần theo điểm tin cậy/tương đồng.
        k: Số lượng ứng viên hàng đầu được xem xét (k >= 1).

    Trả về:
        Tỷ lệ nhãn thực tế nằm trong top-k dự đoán.
    """
    if len(y_true) != len(y_pred_ranked):
        raise ValueError(
            f"Kích thước y_true ({len(y_true)}) và y_pred_ranked ({len(y_pred_ranked)}) không đồng nhất."
        )
    if k < 1:
        raise ValueError(f"Giá trị k phải >= 1, nhận được k={k}.")
    if not y_true:
        return 0.0

    hits = 0
    for yt, preds in zip(y_true, y_pred_ranked):
        # Kiểm tra xem nhãn thực tế có nằm trong k phần tử đầu tiên của danh sách dự đoán
        if yt in preds[:k]:
            hits += 1

    return float(hits / len(y_true))


def compute_mrr(
    y_true: Sequence[Any],
    y_pred_ranked: Sequence[Sequence[Any]],
) -> float:
    """
    Tính chỉ số nghịch đảo thứ hạng trung bình (Mean Reciprocal Rank - MRR).

    Công thức toán học (Voorhees, 1999 - TREC-8; Cranfield paradigm):
        \\text{MRR} = \\frac{1}{N} \\sum_{i=1}^N \\frac{1}{\\text{rank}_i}
        trong đó \\text{rank}_i là vị trí (1-based) đầu tiên xuất hiện nhãn đúng; nếu không xuất hiện thì bằng 0.

    Tham số:
        y_true: Chuỗi nhãn ground-truth.
        y_pred_ranked: Chuỗi danh sách các ứng viên dự đoán có xếp hạng.

    Trả về:
        Điểm MRR trong đoạn [0.0, 1.0].
    """
    if len(y_true) != len(y_pred_ranked):
        raise ValueError(
            f"Kích thước y_true ({len(y_true)}) và y_pred_ranked ({len(y_pred_ranked)}) không đồng nhất."
        )
    if not y_true:
        return 0.0

    rr_sum = 0.0
    for yt, preds in zip(y_true, y_pred_ranked):
        try:
            # Vị trí xuất hiện đầu tiên (1-based rank)
            rank = preds.index(yt) + 1
            rr_sum += 1.0 / rank
        except ValueError:
            # Nhãn thực tế không nằm trong danh sách gợi ý
            pass

    return float(rr_sum / len(y_true))


# =====================================================================
# 5. Khoảng tin cậy Bootstrap phi tham số (Bootstrap Confidence Interval)
# =====================================================================


def compute_bootstrap_ci(
    data: Sequence[Any],
    metric_fn: Callable[[Sequence[Any]], float] | None = None,
    num_bootstrap: int = DEFAULT_BOOTSTRAP_ROUNDS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """
    Ước lượng khoảng tin cậy phi tham số bằng phương pháp Bootstrap phân vị (Percentile Bootstrap CI).

    Tham chiếu học thuật:
        Efron & Tibshirani (1993), 'An Introduction to the Bootstrap', Chapman & Hall.

    Phương pháp:
        Lấy mẫu có hoàn lại B lần từ tập dữ liệu ban đầu, tính toán chỉ số thống kê trên từng mẫu bootstrap,
        sau đó lấy các phân vị tương ứng với mức ý nghĩa alpha = 1 - confidence_level:
            CI = [ \\text{Percentile}(100 \\cdot \\alpha / 2), \\text{Percentile}(100 \\cdot (1 - \\alpha / 2)) ]

    Tham số:
        data: Danh sách mẫu dữ liệu quan sát.
        metric_fn: Hàm tính chỉ số trên tập con mẫu (mặc định tính compute_mean).
        num_bootstrap: Số lượt lấy mẫu lặp lại (mặc định 1000).
        confidence_level: Mức độ tin cậy (ví dụ 0.95 tương ứng 95% CI).
        seed: Hạt giống ngẫu nhiên đảm bảo tính tái lập kết quả.

    Trả về:
        Tuple (ci_lower, ci_upper) đại diện cho cận dưới và cận trên.
    """
    if not data:
        raise ValueError("Dữ liệu data rỗng, không thể ước lượng Bootstrap.")
    if num_bootstrap < 1:
        raise ValueError(f"Số vòng lặp num_bootstrap phải >= 1, nhận được {num_bootstrap}.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"Mức tin cậy confidence_level phải trong khoảng (0, 1), nhận được {confidence_level}."
        )

    fn = metric_fn if metric_fn is not None else compute_mean

    n = len(data)
    if n == 1:
        val = float(fn(data))
        return (val, val)

    rng = np.random.default_rng(seed)
    bootstrap_metrics: list[float] = []

    # Chuyển đổi dữ liệu sang numpy array hoặc list hỗ trợ đánh chỉ mục nhanh
    data_list = list(data)

    for _ in range(num_bootstrap):
        # Lấy mẫu có hoàn lại (sampling with replacement) kích thước n
        sampled_indices = rng.choice(n, size=n, replace=True)
        sampled_subset = [data_list[idx] for idx in sampled_indices]
        metric_value = fn(sampled_subset)
        bootstrap_metrics.append(metric_value)

    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * (alpha / 2.0)
    upper_percentile = 100.0 * (1.0 - (alpha / 2.0))

    ci_lower = float(np.percentile(bootstrap_metrics, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_metrics, upper_percentile))

    return (ci_lower, ci_upper)


def compute_paired_bootstrap_ci(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    metric_fn: Callable[[Sequence[Any], Sequence[Any]], float],
    num_bootstrap: int = DEFAULT_BOOTSTRAP_ROUNDS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """
    Ước lượng khoảng tin cậy Bootstrap cho các chỉ số đánh giá cặp (y_true, y_pred) như F1 hoặc Accuracy.
    """
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError(
            f"Kích thước y_true ({n}) và y_pred ({len(y_pred)}) không đồng nhất."
        )
    if n == 0:
        raise ValueError("Dữ liệu y_true và y_pred không được rỗng.")

    paired_data = list(zip(y_true, y_pred))

    def wrapped_metric(subset: Sequence[tuple[Any, Any]]) -> float:
        sub_true = [item[0] for item in subset]
        sub_pred = [item[1] for item in subset]
        return metric_fn(sub_true, sub_pred)

    return compute_bootstrap_ci(
        data=paired_data,
        metric_fn=wrapped_metric,
        num_bootstrap=num_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )


# =====================================================================
# 6. Kiểm định McNemar (McNemar's Test)
# =====================================================================


def mcnemar_test_from_counts(
    b: int,
    c: int,
    continuity_correction: bool = True,
) -> tuple[float, float]:
    """
    Thực hiện kiểm định McNemar dựa trên số đếm bất đồng (discordant pairs) giữa 2 mô hình.

    Bảng liên kết 2x2:
        - b: Số mẫu Model A đoán ĐÚNG nhưng Model B đoán SAI.
        - c: Số mẫu Model A đoán SAI nhưng Model B đoán ĐÚNG.

    Công thức hiệu chỉnh liên tục Yates (Yates' continuity correction - Dietterich, 1998):
        \\chi^2 = \\frac{ (\\max(0, |b - c| - 1))^2 }{ b + c }

    Xử lý đặc biệt khi b + c = 0:
        Khi cả hai mô hình hoàn toàn giống hệt nhau về dự đoán đúng/sai trên mọi mẫu,
        không có bất kỳ sai lệch nào. Do đó statistic = 0.0 và p_value = 1.0.

    Tham số:
        b: Số lượng bất đồng ủng hộ Model A.
        c: Số lượng bất đồng ủng hộ Model B.
        continuity_correction: Áp dụng hiệu chỉnh liên tục Yates (mặc định True).

    Trả về:
        Tuple (statistic, p_value) với bậc tự do df=1.
    """
    if b < 0 or c < 0:
        raise ValueError(f"Số đếm b ({b}) và c ({c}) phải là số nguyên không âm.")

    # Xử lý trường hợp b + c = 0 theo yêu cầu bài toán
    if b + c == 0:
        return (0.0, 1.0)

    diff = abs(b - c)
    if continuity_correction:
        # Hiệu chỉnh Yates: giảm sai số xấp xỉ liên tục cho phân phối nhị thức
        corrected_diff = max(0.0, float(diff - 1.0))
        statistic = float((corrected_diff**2) / (b + c))
    else:
        statistic = float((diff**2) / (b + c))

    # Tính p-value từ phân phối Chi-bình phương với bậc tự do df = 1
    if HAS_SCIPY:
        p_value = float(stats.chi2.sf(statistic, df=1))
    else:
        # Fallback giải tích chính xác khi không có scipy:
        # Với Chi-square bậc tự do 1: P(Chi_1^2 >= x) = erfc(sqrt(x / 2))
        p_value = float(math.erfc(math.sqrt(statistic / 2.0)))

    return (statistic, p_value)


def compute_mcnemar_test(
    y_true: Sequence[Any],
    y_pred_a: Sequence[Any],
    y_pred_b: Sequence[Any],
    continuity_correction: bool = True,
) -> tuple[float, float]:
    """
    Kiểm định McNemar so sánh tính khác biệt có ý nghĩa thống kê giữa hai mô hình A và B.

    Tham số:
        y_true: Nhãn thực tế ground-truth.
        y_pred_a: Nhãn dự đoán của Mô hình A.
        y_pred_b: Nhãn dự đoán của Mô hình B.
        continuity_correction: Áp dụng hiệu chỉnh liên tục Yates (mặc định True).

    Trả về:
        Tuple (statistic, p_value).
    """
    n = len(y_true)
    if len(y_pred_a) != n or len(y_pred_b) != n:
        raise ValueError(
            f"Kích thước y_true ({n}), y_pred_a ({len(y_pred_a)}), y_pred_b ({len(y_pred_b)}) không khớp."
        )

    b = 0  # Model A đúng, Model B sai
    c = 0  # Model A sai, Model B đúng

    for yt, ya, yb in zip(y_true, y_pred_a, y_pred_b):
        a_correct = ya == yt
        b_correct = yb == yt

        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1

    return mcnemar_test_from_counts(b=b, c=c, continuity_correction=continuity_correction)


# =====================================================================
# 7. Sai số hiệu chuẩn kỳ vọng (Expected Calibration Error - ECE)
# =====================================================================


def compute_ece_details(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    confidences: Sequence[float],
    num_bins: int = DEFAULT_ECE_BINS,
) -> ECEResult:
    """
    Tính toán chi tiết sai số hiệu chuẩn kỳ vọng (ECE) cùng các thống kê theo từng bin phân chia đều.

    Tham chiếu học thuật:
        Guo, Pleiss, Sun, Weinberger (ICML 2017), 'On Calibration of Modern Neural Networks'.

    Công thức:
        ECE = \\sum_{m=1}^M \\frac{|B_m|}{N} |\\text{acc}(B_m) - \\text{conf}(B_m)|
        trong đó:
            \\text{acc}(B_m) = \\frac{1}{|B_m|} \\sum_{i \\in B_m} \\mathbf{1}(y_i = \\hat{y}_i)
            \\text{conf}(B_m) = \\frac{1}{|B_m|} \\sum_{i \\in B_m} \\hat{p}_i

    Tham số:
        y_true: Nhãn ground-truth.
        y_pred: Nhãn dự đoán.
        confidences: Độ tin cậy dự đoán (xác suất softmax của lớp được chọn) trong đoạn [0.0, 1.0].
        num_bins: Số lượng bin chia đều (mặc định 10).

    Trả về:
        Đối tượng ECEResult chi tiết.
    """
    n = len(y_true)
    if len(y_pred) != n or len(confidences) != n:
        raise ValueError(
            f"Kích thước y_true ({n}), y_pred ({len(y_pred)}), confidences ({len(confidences)}) không khớp."
        )
    if num_bins < 1:
        raise ValueError(f"num_bins phải >= 1, nhận được {num_bins}.")

    bin_edges = [float(i / num_bins) for i in range(num_bins + 1)]

    if n == 0:
        return ECEResult(
            ece=0.0,
            bin_accuracies=[0.0] * num_bins,
            bin_confidences=[0.0] * num_bins,
            bin_counts=[0] * num_bins,
            bin_edges=bin_edges,
            num_bins=num_bins,
            total_samples=0,
        )

    # Khởi tạo các thùng chứa cho từng bin
    bin_correct = [0] * num_bins
    bin_conf_sum = [0.0] * num_bins
    bin_counts = [0] * num_bins

    for yt, yp, conf_raw in zip(y_true, y_pred, confidences):
        # Giới hạn độ tin cậy trong khoảng hợp lệ [0.0, 1.0]
        conf = max(0.0, min(1.0, float(conf_raw)))

        # Xác định chỉ số bin với các khoảng bằng nhau: [0, 1/M), [1/M, 2/M), ..., [(M-1)/M, 1]
        bin_idx = min(int(conf * num_bins), num_bins - 1)

        bin_counts[bin_idx] += 1
        bin_conf_sum[bin_idx] += conf
        if yt == yp:
            bin_correct[bin_idx] += 1

    bin_accuracies: list[float] = []
    bin_confidences: list[float] = []
    ece = 0.0

    for i in range(num_bins):
        count = bin_counts[i]
        if count > 0:
            acc = float(bin_correct[i] / count)
            avg_conf = float(bin_conf_sum[i] / count)
            bin_accuracies.append(acc)
            bin_confidences.append(avg_conf)
            ece += (count / n) * abs(acc - avg_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    return ECEResult(
        ece=float(ece),
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        bin_edges=bin_edges,
        num_bins=num_bins,
        total_samples=n,
    )


def compute_ece(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    confidences: Sequence[float],
    num_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """
    Tính chỉ số sai số hiệu chuẩn kỳ vọng (ECE) dạng số thực.
    """
    result = compute_ece_details(
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        num_bins=num_bins,
    )
    return result.ece


def compute_expected_calibration_error(
    y_true: Sequence[Any],
    y_pred: Sequence[Any] | None = None,
    confidences: Sequence[float] | None = None,
    probs: Sequence[Sequence[float]] | None = None,
    num_bins: int = DEFAULT_ECE_BINS,
) -> ECEResult:
    """
    Tính toán chi tiết sai số hiệu chuẩn kỳ vọng (Expected Calibration Error - ECE).

    Tham chiếu học thuật:
        Guo, Pleiss, Sun, Weinberger (ICML 2017), 'On Calibration of Modern Neural Networks'.

    Hỗ trợ linh hoạt:
    - Nhận confidences trực tiếp cùng y_true, y_pred.
    - Hoặc nhận ma trận phân phối xác suất softmax probs (tự động trích xuất độ tin cậy lớn nhất).
    """
    if confidences is None:
        if probs is None:
            raise ValueError("Cần cung cấp confidences hoặc probs để tính ECE.")
        confidences = [float(max(p_row)) if p_row else 1.0 for p_row in probs]
    if y_pred is None:
        if probs is not None:
            import numpy as np

            y_pred = [int(np.argmax(p_row)) if hasattr(p_row, "__len__") else 0 for p_row in probs]
        else:
            raise ValueError("Cần cung cấp y_pred hoặc probs để tính ECE.")
    return compute_ece_details(
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        num_bins=num_bins,
    )


# =====================================================================
# 8. Sinh biểu đồ ma trận nhầm lẫn (Confusion Matrix Figure)
# =====================================================================


def generate_confusion_matrix_figure(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    output_path: str | Path | None = None,
    title: str = "Ma Trận Nhầm Lẫn Phân Loại Ý Định (Confusion Matrix)",
    normalize: str | None = None,
    max_classes_display: int = 30,
) -> Path:
    """
    Khởi tạo và lưu biểu đồ ma trận nhầm lẫn bằng matplotlib vào thư mục report/figures/.

    Tham số:
        y_true: Danh sách nhãn thực tế.
        y_pred: Danh sách nhãn dự đoán.
        labels: Danh sách lớp hiển thị (nếu None, tự động trích xuất các lớp có trong mẫu).
        output_path: Đường dẫn tệp hình ảnh đầu ra (mặc định report/figures/confusion_matrix.png).
        title: Tiêu đề biểu đồ.
        normalize: Tùy chọn chuẩn hóa ('true', 'pred', 'all', hoặc None).
        max_classes_display: Giới hạn số lượng lớp tối đa để vẽ nhằm đảm bảo độ rõ nét trực quan.

    Trả về:
        Đường dẫn Path tới tệp hình ảnh PNG đã được lưu thành công.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Kích thước y_true ({len(y_true)}) và y_pred ({len(y_pred)}) không đồng nhất."
        )

    # Xác định đường dẫn đích an toàn
    save_path = Path(output_path) if output_path is not None else DEFAULT_CONFUSION_MATRIX_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Xác định danh sách nhãn lớp cần trực quan hóa
    if labels is None:
        unique_labels = sorted({str(y) for y in y_true} | {str(y) for y in y_pred})
    else:
        unique_labels = [str(l) for l in labels]

    # Nếu số lượng lớp quá lớn (ví dụ 189 intents), chọn top N lớp phổ biến nhất trong ground-truth
    if len(unique_labels) > max_classes_display:
        label_frequency: dict[str, int] = {}
        for y in y_true:
            sy = str(y)
            label_frequency[sy] = label_frequency.get(sy, 0) + 1
        sorted_labels = sorted(
            unique_labels,
            key=lambda lbl: label_frequency.get(lbl, 0),
            reverse=True,
        )
        display_labels = sorted(sorted_labels[:max_classes_display])
    else:
        display_labels = unique_labels

    n_classes = len(display_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(display_labels)}

    # Tính toán ma trận số đếm nhầm lẫn
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for yt_raw, yp_raw in zip(y_true, y_pred):
        yt = str(yt_raw)
        yp = str(yp_raw)
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1.0

    # Chuẩn hóa nếu có yêu cầu
    if normalize == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    elif normalize == "pred":
        col_sums = cm.sum(axis=0, keepdims=True)
        cm = np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums > 0)
    elif normalize == "all":
        total_sum = cm.sum()
        if total_sum > 0:
            cm = cm / total_sum

    # Khởi tạo kích thước canvas linh hoạt theo số lượng lớp
    fig_size = max(8.0, min(24.0, n_classes * 0.5 + 4.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    cmap = plt.get_cmap("Blues")
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted Label)", fontsize=12, labelpad=10)
    ax.set_ylabel("Nhãn Thực Tế (True Label)", fontsize=12, labelpad=10)

    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)

    # Hiển thị số liệu lên từng ô nếu kích thước ma trận đủ nhỏ để đọc được
    if n_classes <= 20:
        threshold = cm.max() / 2.0 if cm.size > 0 and cm.max() > 0 else 0.5
        fmt = ".2f" if normalize else ".0f"
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm[i, j]
                text_color = "white" if val > threshold else "black"
                ax.text(
                    j,
                    i,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


# =====================================================================
# 9. CLI và Điểm thực thi chính (Main Entrypoint)
# =====================================================================


def create_cli_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh cho module đánh giá."""
    parser = argparse.ArgumentParser(
        description="Bộ công cụ tính toán chỉ số đánh giá học thuật và phân tích thống kê PhoBERT."
    )
    parser.add_argument(
        "--intents-path",
        type=str,
        default=str(DEFAULT_INTENTS_PATH),
        help="Đường dẫn tới tệp intents.json schema.",
    )
    parser.add_argument(
        "--output-fig",
        type=str,
        default=str(DEFAULT_CONFUSION_MATRIX_PATH),
        help="Đường dẫn lưu biểu đồ ma trận nhầm lẫn PNG.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Hạt giống ngẫu nhiên thiết lập cho thực nghiệm.",
    )
    return parser


def main(args: Sequence[str] | None = None) -> int:
    """Hàm thực thi chính khi chạy module độc lập."""
    parser = create_cli_parser()
    parsed_args = parser.parse_args(args)

    set_seed(parsed_args.seed)

    intents_path = Path(parsed_args.intents_path)
    if intents_path.exists():
        labels = load_intent_labels(intents_path)
        print(f"Đã nạp thành công {len(labels)} ý định từ {intents_path}")
    else:
        print(f"Lưu ý: Không tìm thấy tệp {intents_path}, bỏ qua tải schema.")

    # Kiểm tra tạo một ma trận nhầm lẫn mẫu minh họa
    sample_true = ["LATEX_TAO_BANG", "MATH_MATH_BASIC_PHAN_SO", "LATEX_TAO_BANG"]
    sample_pred = ["LATEX_TAO_BANG", "MATH_MATH_BASIC_PHAN_SO", "MATH_MATH_BASIC_PHAN_SO"]
    out_path = generate_confusion_matrix_figure(
        sample_true, sample_pred, output_path=parsed_args.output_fig
    )
    print(f"Đã xuất biểu đồ ma trận nhầm lẫn kiểm chứng tại: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
