"""
Module kiểm thực tập kiểm thử và phân tích thống kê thực nghiệm cắt bỏ (Ablation Study Statistical Utilities).

Tệp tin: train/ablation_stats.py
Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Identifiers in English, docstrings và comments bằng tiếng Việt giải thích lý do (WHY) và công thức bài báo khoa học.
- Toàn bộ hằng số cấu hình được nạp từ core/config.py (không dùng số ma thuật).
- Thiết lập seed tất định thông qua core.config.set_seed.
- Trung thực số liệu (Honesty Rules):
  + Honesty Rule #2: Không bịa đặt chỉ số, mọi chỉ số đều được tính toán thực tế.
  + Honesty Rule #3: Bất biến tập kiểm thử (Test Set Invariance). Lưu và kiểm chứng mã băm SHA-256
    của data/test.csv đối chiếu với PROGRESS.md trước mọi quá trình thực thi đánh giá cuối cùng.
    Tuyệt đối không tinh chỉnh trên test set, không chỉnh sửa, tự sinh hay thêm bớt bừa bãi vào test.csv.

Các tính năng chính:
1. Kiểm thực toàn vẹn tập kiểm thử (Test Set Integrity Verification):
   - Tính toán mã băm SHA-256 của data/test.csv theo khối (chunked binary reading).
   - Trích xuất mã băm SHA-256 được cam kết trong PROGRESS.md (KA-05).
   - Kiểm tra đối chiếu trước khi thực thi, phát hiện và ngăn chặn mọi sự thay đổi dữ liệu kiểm thử.
2. Ước lượng khoảng tin cậy Bootstrap 95% (Non-parametric Bootstrap CI - B=1000):
   - Ước lượng khoảng tin cậy phân vị (Percentile Bootstrap CI - Efron & Tibshirani, 1993)
     cho Accuracy và Macro-F1 (trên schema 189 ý định).
3. Kiểm định McNemar so sánh các mô hình cắt bỏ (Ablation Model Comparison):
   - Đánh giá ý nghĩa thống kê của sự cải thiện (McNemar, 1947; Dietterich, 1998)
     với hiệu chỉnh liên tục Yates và xử lý trường hợp bất đồng b + c = 0.
4. Tổng hợp đa hạt giống (Multi-Seed Aggregation):
   - Tính toán trung bình mẫu, độ lệch chuẩn mẫu hiệu chỉnh Bessel (ddof=1) và sai số chuẩn (SEM)
     trên các hạt giống huấn luyện độc lập (ví dụ seeds 42, 43, 44).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

# Đảm bảo đường dẫn gốc dự án có trong sys.path khi thực thi trực tiếp script
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.config import (
    DEFAULT_BOOTSTRAP_ROUNDS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_INTENTS_PATH,
    DEFAULT_MCNEMAR_ALPHA,
    DEFAULT_SEED,
    DEFAULT_ZERO_DIVISION,
    PROGRESS_PATH,
    TEST_CSV_PATH,
    TEST_CSV_SHA256,
    TEST_CSV_SHA256_REGEX,
    set_seed,
)
from train.evaluate import (
    compute_accuracy,
    compute_macro_f1,
    compute_mean,
    compute_sample_std,
    load_intent_labels,
    mcnemar_test_from_counts,
)

# Kích thước khối đọc tệp tin nhị phân (64 KB)
BUFFER_SIZE: Final[int] = 65536


# =====================================================================
# 1. Ngoại lệ và Cấu trúc Dữ liệu Kết quả (Data Classes & Exceptions)
# =====================================================================


class TestSetChecksumMismatchError(RuntimeError):
    """
    Ngoại lệ phát sinh khi mã băm SHA-256 thực tế của data/test.csv không khớp
    với giá trị được cam kết trong PROGRESS.md hoặc core/config.py.

    Tuân thủ nghiêm ngặt Honesty Rule #3 (AGENTS.md).
    """

    __test__ = False  # Ngăn pytest hiểu nhầm là test class vì tiền tố 'Test'

    def __init__(self, actual_sha256: str, expected_sha256: str, test_csv_path: Path | str) -> None:
        self.actual_sha256 = actual_sha256
        self.expected_sha256 = expected_sha256
        self.test_csv_path = str(test_csv_path)
        message = (
            f"Phát hiện vi phạm tính toàn vẹn tập kiểm thử (Honesty Rule #3)!\n"
            f"Tệp kiểm thử: {self.test_csv_path}\n"
            f"Mã băm SHA-256 thực tế:   {self.actual_sha256}\n"
            f"Mã băm cam kết PROGRESS: {self.expected_sha256}\n"
            f"Tuyệt đối không sửa đổi, tự sinh hoặc tinh chỉnh trên data/test.csv."
        )
        super().__init__(message)


@dataclass
class BootstrapCIResult:
    """
    Kết quả ước lượng khoảng tin cậy Bootstrap phi tham số (Percentile Bootstrap CI).

    Tham chiếu: Efron & Tibshirani (1993), 'An Introduction to the Bootstrap'.
    """

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    num_bootstrap: int = DEFAULT_BOOTSTRAP_ROUNDS

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary."""
        return asdict(self)

    def format_ci(self, percentage: bool = False, precision: int = 4) -> str:
        """
        Định dạng chuỗi biểu diễn khoảng tin cậy: point_estimate [ci_lower, ci_upper].
        """
        scale = 100.0 if percentage else 1.0
        suffix = "%" if percentage else ""
        fmt = f".{precision}f"
        pe_str = f"{self.point_estimate * scale:{fmt}}{suffix}"
        low_str = f"{self.ci_lower * scale:{fmt}}{suffix}"
        high_str = f"{self.ci_upper * scale:{fmt}}{suffix}"
        return f"{pe_str} [{low_str}, {high_str}]"


@dataclass
class McNemarComparisonResult:
    """
    Kết quả kiểm định McNemar so sánh thống kê giữa hai mô hình trong nghiên cứu cắt bỏ.

    Tham chiếu: McNemar (1947), Dietterich (1998) 'Approximate Statistical Tests for
    Comparing Supervised Classification Learning Algorithms'.
    """

    model_a_name: str
    model_b_name: str
    statistic: float
    p_value: float
    b: int  # Model A đúng, Model B sai
    c: int  # Model A sai, Model B đúng
    alpha: float
    is_significant: bool
    winner: str | None
    contingency_table: dict[str, int]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary."""
        return asdict(self)


@dataclass
class MultiSeedMetricSummary:
    """
    Tóm tắt thống kê đa hạt giống cho một chỉ số đo lường đơn lẻ (Multi-Seed Metric Summary).
    """

    metric_name: str
    mean: float
    std: float
    sem: float  # Sai số chuẩn của giá trị trung bình (Standard Error of the Mean)
    min: float
    max: float
    n_seeds: int
    values: list[float]
    formatted: str

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary."""
        return asdict(self)

    def format_summary(self, percentage: bool = False, precision: int = 2) -> str:
        """
        Định dạng tóm tắt Mean ± Std (ví dụ: '85.20 ± 0.45%').
        """
        scale = 100.0 if percentage else 1.0
        suffix = "%" if percentage else ""
        fmt = f".{precision}f"
        return f"{self.mean * scale:{fmt}} ± {self.std * scale:{fmt}}{suffix}"


@dataclass
class MultiSeedRunAggregation:
    """
    Kết quả tổng hợp đa hạt giống cho một cấu hình mô hình (Multi-Seed Run Aggregation).
    """

    model_name: str
    seeds: list[int]
    n_runs: int
    metrics: dict[str, MultiSeedMetricSummary]

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary."""
        return {
            "model_name": self.model_name,
            "seeds": self.seeds,
            "n_runs": self.n_runs,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    def format_table(self, percentage: bool = False, precision: int = 2) -> str:
        """
        Tạo bảng định dạng ASCII hiển thị kết quả các chỉ số qua nhiều hạt giống.
        """
        lines = [
            f"=== Multi-Seed Aggregation: {self.model_name} (N={self.n_runs} seeds: {self.seeds}) ===",
            f"{'Metric':<20} | {'Mean ± Std':<18} | {'SEM':<10} | {'[Min, Max]':<20}",
            "-" * 76,
        ]
        scale = 100.0 if percentage else 1.0
        suffix = "%" if percentage else ""
        fmt = f".{precision}f"
        for name, summary in self.metrics.items():
            mean_std = f"{summary.mean * scale:{fmt}} ± {summary.std * scale:{fmt}}{suffix}"
            sem_str = f"{summary.sem * scale:{fmt}}{suffix}"
            min_max = f"[{summary.min * scale:{fmt}}, {summary.max * scale:{fmt}}]{suffix}"
            lines.append(f"{name:<20} | {mean_std:<18} | {sem_str:<10} | {min_max:<20}")
        return "\n".join(lines)


# =====================================================================
# 2. Kiểm thực Tính toàn vẹn Tập kiểm thử (Test Set Integrity Routines)
# =====================================================================


def compute_file_sha256(file_path: Path | str) -> str:
    """
    Tính toán mã băm SHA-256 tất định của tệp tin theo khối nhị phân 64 KB.

    Lý do thiết kế:
        Đọc theo khối nhị phân giúp tiết kiệm bộ nhớ RAM khi tính toán băm cho các tệp dữ liệu lớn,
        đồng thời đảm bảo tính bất biến tuyệt đối của mã băm trên mọi nền tảng hệ điều hành.

    Tham số:
        file_path: Đường dẫn tệp tin cần tính mã băm.

    Trả về:
        Chuỗi mã băm SHA-256 dạng thập lục phân thường (64 ký tự hex).

    Ngoại lệ:
        FileNotFoundError nếu tệp tin không tồn tại.
    """
    path_obj = Path(file_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp tin để tính toán SHA-256: {path_obj}")

    hasher = hashlib.sha256()
    with open(path_obj, "rb") as f:
        while chunk := f.read(BUFFER_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def extract_test_sha256_from_progress(
    progress_path: Path | str = PROGRESS_PATH,
) -> str:
    """
    Trích xuất mã băm SHA-256 của data/test.csv được cam kết trong tệp theo dõi PROGRESS.md.

    Lý do:
        Honesty Rule #3 quy định: 'Store sha256 of data/test.csv in PROGRESS.md at KA-05
        and verify it before every final evaluation.'
        Hàm này đọc tệp PROGRESS.md và tìm dòng chứa biểu thức quy chuẩn 'sha256 test.csv: <hash>'.

    Tham số:
        progress_path: Đường dẫn tệp PROGRESS.md (mặc định lấy từ core/config.py).

    Trả về:
        Chuỗi mã băm SHA-256 (64 ký tự).

    Ngoại lệ:
        FileNotFoundError nếu tệp PROGRESS.md không tồn tại.
        ValueError nếu không tìm thấy mẫu mã băm hợp lệ trong PROGRESS.md.
    """
    path_obj = Path(progress_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp theo dõi tiến độ: {path_obj}")

    content = path_obj.read_text(encoding="utf-8")
    match = re.search(TEST_CSV_SHA256_REGEX, content, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Không tìm thấy mẫu mã băm SHA-256 của data/test.csv trong tệp {path_obj.name}. "
            f"Kỳ vọng biểu thức regex: '{TEST_CSV_SHA256_REGEX}'."
        )

    expected_hash = match.group(1).strip().lower()
    if len(expected_hash) != 64:
        raise ValueError(
            f"Mã băm trích xuất được từ PROGRESS.md không hợp lệ (chiều dài {len(expected_hash)} != 64): '{expected_hash}'"
        )
    return expected_hash


def verify_test_set_checksum(
    test_csv_path: Path | str = TEST_CSV_PATH,
    progress_path: Path | str = PROGRESS_PATH,
    expected_sha256: str | None = None,
    raise_on_error: bool = True,
) -> tuple[bool, str, str]:
    """
    Xác minh mã băm SHA-256 của tập kiểm thử data/test.csv đối chiếu với PROGRESS.md
    hoặc giá trị băm dự kiến được chỉ định trước khi thực thi đánh giá.

    Tuân thủ Honesty Rule #3 (AGENTS.md):
        Ngăn chặn mọi hành vi rò rỉ dữ liệu, chỉnh sửa vô tình, tự sinh hoặc tinh chỉnh siêu tham số
        trên tập kiểm thử.

    Tham số:
        test_csv_path: Đường dẫn tệp CSV kiểm thử (data/test.csv).
        progress_path: Đường dẫn tệp PROGRESS.md ghi nhận cam kết băm.
        expected_sha256: Mã băm kỳ vọng (nếu None, tự động trích xuất từ progress_path;
                         nếu trích xuất thất bại, sử dụng hằng số TEST_CSV_SHA256).
        raise_on_error: Nếu True, ném ngoại lệ TestSetChecksumMismatchError khi không khớp;
                        nếu False, trả về cờ boolean False.

    Trả về:
        Tuple (is_valid, actual_sha256, expected_sha256).

    Ngoại lệ:
        TestSetChecksumMismatchError nếu không khớp và raise_on_error=True.
        FileNotFoundError nếu tệp test_csv_path không tồn tại.
    """
    test_path = Path(test_csv_path)
    actual_sha256 = compute_file_sha256(test_path)

    target_expected: str
    if expected_sha256 is not None:
        target_expected = expected_sha256.strip().lower()
    else:
        try:
            target_expected = extract_test_sha256_from_progress(progress_path)
        except (FileNotFoundError, ValueError, OSError):
            # Dự phòng an toàn tới hằng số cấu hình chuẩn được lưu trong core/config.py
            target_expected = TEST_CSV_SHA256.strip().lower()

    is_valid = actual_sha256 == target_expected

    if not is_valid and raise_on_error:
        raise TestSetChecksumMismatchError(
            actual_sha256=actual_sha256,
            expected_sha256=target_expected,
            test_csv_path=test_path,
        )

    return (is_valid, actual_sha256, target_expected)


def ensure_test_set_integrity(
    test_csv_path: Path | str = TEST_CSV_PATH,
    progress_path: Path | str = PROGRESS_PATH,
) -> str:
    """
    Rút gọn kiểm tra bắt buộc (Pre-execution Verification Routine) trước khi chạy đánh giá test set.
    Trả về mã băm SHA-256 đã được xác thực, hoặc ném ngoại lệ chặn đứng tiến trình.

    Tham số:
        test_csv_path: Đường dẫn tệp test.csv.
        progress_path: Đường dẫn tệp PROGRESS.md.

    Trả về:
        Chuỗi SHA-256 của tập kiểm thử hợp lệ.
    """
    _, actual_sha256, _ = verify_test_set_checksum(
        test_csv_path=test_csv_path,
        progress_path=progress_path,
        raise_on_error=True,
    )
    return actual_sha256


# =====================================================================
# 3. Khoảng Tin cậy Bootstrap 95% (Accuracy & Macro-F1, B=1000)
# =====================================================================


def compute_accuracy_bootstrap_ci(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    num_bootstrap: int = DEFAULT_BOOTSTRAP_ROUNDS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
) -> BootstrapCIResult:
    """
    Ước lượng khoảng tin cậy 95% Bootstrap phi tham số (B=1000) cho độ chính xác Accuracy.

    Tham chiếu học thuật:
        Efron & Tibshirani (1993), 'An Introduction to the Bootstrap', Chapter 13 (Percentile Method).

    Công thức:
        - Lấy mẫu có hoàn lại (sampling with replacement) B lần từ tập chỉ mục {0, ..., N-1}.
        - Tính Accuracy_b trên từng mẫu bootstrap.
        - Khoảng tin cậy [alpha/2, 1 - alpha/2] với alpha = 1 - confidence_level (ví dụ [2.5%, 97.5%]).

    Tham số:
        y_true: Chuỗi nhãn ground-truth thực tế.
        y_pred: Chuỗi nhãn dự đoán từ mô hình.
        num_bootstrap: Số vòng lặp bootstrap (mặc định B=1000).
        confidence_level: Mức độ tin cậy (mặc định 0.95 tương ứng 95% CI).
        seed: Hạt giống ngẫu nhiên đảm bảo tính tái lập.

    Trả về:
        Đối tượng BootstrapCIResult chứa giá trị ước lượng điểm và khoảng [ci_lower, ci_upper].
    """
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError(f"Kích thước y_true ({n}) và y_pred ({len(y_pred)}) không đồng nhất.")
    if n == 0:
        raise ValueError("Dữ liệu y_true và y_pred rỗng, không thể ước lượng Bootstrap.")
    if num_bootstrap < 1:
        raise ValueError(f"Số vòng lặp num_bootstrap phải >= 1, nhận được {num_bootstrap}.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level phải trong khoảng (0, 1), nhận được {confidence_level}.")

    # Tính ước lượng điểm trên toàn bộ tập dữ liệu gốc
    point_estimate = compute_accuracy(y_true=y_true, y_pred=y_pred)

    if n == 1:
        return BootstrapCIResult(
            metric_name="accuracy",
            point_estimate=point_estimate,
            ci_lower=point_estimate,
            ci_upper=point_estimate,
            confidence_level=confidence_level,
            num_bootstrap=num_bootstrap,
        )

    # Chuyển đổi thành mảng boolean so khớp nhanh: correct[i] = 1 nếu y_true[i] == y_pred[i]
    correct_array = np.array([yt == yp for yt, yp in zip(y_true, y_pred)], dtype=np.float64)

    rng = np.random.default_rng(seed)
    bootstrap_accuracies = np.empty(num_bootstrap, dtype=np.float64)

    for i in range(num_bootstrap):
        sampled_indices = rng.choice(n, size=n, replace=True)
        bootstrap_accuracies[i] = float(np.mean(correct_array[sampled_indices]))

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - (alpha / 2.0))

    ci_lower = float(np.percentile(bootstrap_accuracies, lower_pct))
    ci_upper = float(np.percentile(bootstrap_accuracies, upper_pct))

    return BootstrapCIResult(
        metric_name="accuracy",
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        num_bootstrap=num_bootstrap,
    )


def compute_macro_f1_bootstrap_ci(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    intents_path: Path | str = DEFAULT_INTENTS_PATH,
    num_bootstrap: int = DEFAULT_BOOTSTRAP_ROUNDS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
) -> BootstrapCIResult:
    """
    Ước lượng khoảng tin cậy 95% Bootstrap phi tham số (B=1000) cho điểm Macro-F1
    trên không gian nhãn schema kịch bản cố định (189 ý định).

    Lý do thiết kế:
        Macro-F1 đánh giá bình quân F1 trên toàn bộ các lớp của schema. Để phản ánh chính xác
        độ bao phủ, danh sách lớp mục tiêu được cố định theo schema, các lớp không xuất hiện
        trong mẫu con bootstrap sẽ nhận F1 = 0.0 theo chuẩn Van Rijsbergen (1979).

    Tham số:
        y_true: Danh sách nhãn thực tế.
        y_pred: Danh sách nhãn dự đoán.
        labels: Danh sách lớp mục tiêu (nếu None, tự động nạp từ intents_path).
        intents_path: Đường dẫn intents.json khi labels=None.
        num_bootstrap: Số lượt lấy mẫu (mặc định B=1000).
        confidence_level: Mức độ tin cậy (mặc định 0.95).
        seed: Hạt giống ngẫu nhiên đảm bảo tái lập kết quả.

    Trả về:
        Đối tượng BootstrapCIResult cho chỉ số Macro-F1.
    """
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError(f"Kích thước y_true ({n}) và y_pred ({len(y_pred)}) không đồng nhất.")
    if n == 0:
        raise ValueError("Dữ liệu y_true và y_pred rỗng, không thể ước lượng Bootstrap.")
    if num_bootstrap < 1:
        raise ValueError(f"Số vòng lặp num_bootstrap phải >= 1, nhận được {num_bootstrap}.")

    target_labels: list[str]
    if labels is not None:
        target_labels = [str(x) for x in labels]
    else:
        target_labels = load_intent_labels(intents_path=intents_path)

    # Ước lượng điểm Macro-F1 trên dữ liệu gốc
    point_estimate = compute_macro_f1(
        y_true=y_true,
        y_pred=y_pred,
        labels=target_labels,
        zero_division=DEFAULT_ZERO_DIVISION,
    )

    if n == 1:
        return BootstrapCIResult(
            metric_name="macro_f1",
            point_estimate=point_estimate,
            ci_lower=point_estimate,
            ci_upper=point_estimate,
            confidence_level=confidence_level,
            num_bootstrap=num_bootstrap,
        )

    paired_data = list(zip(y_true, y_pred))
    rng = np.random.default_rng(seed)
    bootstrap_f1s: list[float] = []

    for _ in range(num_bootstrap):
        sampled_indices = rng.choice(n, size=n, replace=True)
        sub_true = [paired_data[idx][0] for idx in sampled_indices]
        sub_pred = [paired_data[idx][1] for idx in sampled_indices]
        f1_val = compute_macro_f1(
            y_true=sub_true,
            y_pred=sub_pred,
            labels=target_labels,
            zero_division=DEFAULT_ZERO_DIVISION,
        )
        bootstrap_f1s.append(f1_val)

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - (alpha / 2.0))

    ci_lower = float(np.percentile(bootstrap_f1s, lower_pct))
    ci_upper = float(np.percentile(bootstrap_f1s, upper_pct))

    return BootstrapCIResult(
        metric_name="macro_f1",
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        num_bootstrap=num_bootstrap,
    )


def compute_ablation_bootstrap_cis(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
    intents_path: Path | str = DEFAULT_INTENTS_PATH,
    num_bootstrap: int = DEFAULT_BOOTSTRAP_ROUNDS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
) -> dict[str, BootstrapCIResult]:
    """
    Tính đồng thời khoảng tin cậy Bootstrap 95% cho cả Accuracy và Macro-F1.

    Trả về:
        Dictionary ánh xạ {"accuracy": BootstrapCIResult, "macro_f1": BootstrapCIResult}.
    """
    acc_ci = compute_accuracy_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        num_bootstrap=num_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    f1_ci = compute_macro_f1_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        intents_path=intents_path,
        num_bootstrap=num_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "accuracy": acc_ci,
        "macro_f1": f1_ci,
    }


# =====================================================================
# 4. Kiểm định McNemar So sánh Các Mô hình (Ablation Model Comparison)
# =====================================================================


def compare_models_mcnemar(
    y_true: Sequence[Any],
    y_pred_a: Sequence[Any],
    y_pred_b: Sequence[Any],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    alpha: float = DEFAULT_MCNEMAR_ALPHA,
    continuity_correction: bool = True,
) -> McNemarComparisonResult:
    """
    Thực hiện kiểm định McNemar có hiệu chỉnh liên tục Yates để so sánh thống kê
    giữa hai biến thể mô hình (Model A vs Model B) trong nghiên cứu thực nghiệm cắt bỏ (Ablation Study).

    Tham chiếu học thuật:
        - McNemar, Q. (1947), 'Note on the sampling error of the difference between correlated
          proportions or percentages', Psychometrika, 12(2), pp. 153-157.
        - Dietterich, T. G. (1998), 'Approximate Statistical Tests for Comparing Supervised
          Classification Learning Algorithms', Neural Computation, 10(7), pp. 1895-1923.

    Bảng liên kết 2x2 (Contingency Table):
        - n11: Cả Model A và Model B đều ĐÚNG.
        - n10 (b): Model A ĐÚNG, Model B SAI.
        - n01 (c): Model A SAI, Model B ĐÚNG.
        - n00: Cả hai mô hình đều SAI.

    Công thức thống kê Chi-Square (Yates' correction):
        \\chi^2 = \\frac{ (\\max(0, |b - c| - 1))^2 }{ b + c }
        Khi b + c = 0: statistic = 0.0, p_value = 1.0 (hai mô hình hoàn toàn tương đồng về phân loại).

    Tham số:
        y_true: Nhãn thực tế ground-truth.
        y_pred_a: Dự đoán của Model A.
        y_pred_b: Dự đoán của Model B.
        model_a_name: Tên nhận diện Model A.
        model_b_name: Tên nhận diện Model B.
        alpha: Ngưỡng mức ý nghĩa thống kê (mặc định 0.05).
        continuity_correction: Áp dụng hiệu chỉnh Yates (mặc định True).

    Trả về:
        Đối tượng McNemarComparisonResult chứa đầy đủ số đo bất đồng, thống kê chi2, p-value,
        cờ ý nghĩa và kết luận mô hình chiến thắng.
    """
    n = len(y_true)
    if len(y_pred_a) != n or len(y_pred_b) != n:
        raise ValueError(
            f"Kích thước y_true ({n}), y_pred_a ({len(y_pred_a)}), y_pred_b ({len(y_pred_b)}) không đồng nhất."
        )
    if n == 0:
        raise ValueError("Tập dữ liệu so sánh rỗng.")

    n11 = 0  # Both correct
    b = 0    # A correct, B incorrect
    c = 0    # A incorrect, B correct
    n00 = 0  # Both incorrect

    for yt, ya, yb in zip(y_true, y_pred_a, y_pred_b):
        a_correct = ya == yt
        b_correct = yb == yt

        if a_correct and b_correct:
            n11 += 1
        elif a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1
        else:
            n00 += 1

    statistic, p_value = mcnemar_test_from_counts(
        b=b,
        c=c,
        continuity_correction=continuity_correction,
    )

    is_significant = bool(p_value < alpha)
    winner: str | None = None
    if is_significant:
        if b > c:
            winner = model_a_name
        elif c > b:
            winner = model_b_name

    # Xây dựng văn bản tóm tắt giải thích khoa học bằng tiếng Việt
    contingency = {
        "both_correct": n11,
        "a_only": b,
        "b_only": c,
        "both_incorrect": n00,
    }

    if is_significant:
        summary = (
            f"Sự khác biệt giữa {model_a_name} và {model_b_name} CÓ Ý NGHĨA THỐNG KÊ "
            f"(McNemar chi2 = {statistic:.4f}, p = {p_value:.4e} < {alpha}). "
            f"Mô hình vượt trội: {winner} (bất đồng: {b} vs {c})."
        )
    else:
        summary = (
            f"Sự khác biệt giữa {model_a_name} và {model_b_name} KHÔNG CÓ Ý NGHĨA THỐNG KÊ "
            f"(McNemar chi2 = {statistic:.4f}, p = {p_value:.4f} >= {alpha}). "
            f"Không đủ bằng chứng để khẳng định một mô hình tốt hơn (bất đồng: {b} vs {c})."
        )

    return McNemarComparisonResult(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        statistic=statistic,
        p_value=p_value,
        b=b,
        c=c,
        alpha=alpha,
        is_significant=is_significant,
        winner=winner,
        contingency_table=contingency,
        summary=summary,
    )


# =====================================================================
# 5. Tổng hợp Đa Hạt Giống (Multi-Seed Aggregation)
# =====================================================================


def aggregate_seed_values(
    values: Sequence[float],
    metric_name: str = "metric",
    percentage: bool = False,
    precision: int = 4,
) -> MultiSeedMetricSummary:
    """
    Tổng hợp dãy giá trị của một chỉ số thu được qua nhiều hạt giống ngẫu nhiên độc lập.

    Công thức:
        - Giá trị trung bình: \\bar{x} = \\frac{1}{S} \\sum_{s=1}^S x_s
        - Độ lệch chuẩn mẫu hiệu chỉnh Bessel (ddof=1): s = \\sqrt{ \\frac{1}{S-1} \\sum (x_s - \\bar{x})^2 }
        - Sai số chuẩn (SEM - Standard Error of the Mean): \\text{SEM} = \\frac{s}{\\sqrt{S}}

    Tham số:
        values: Chuỗi các số thực thu được qua các seed.
        metric_name: Tên định danh của chỉ số đo lường.
        percentage: Hiển thị dưới dạng phần trăm (nhân 100).
        precision: Số chữ số sau dấu phẩy trong chuỗi định dạng formatted.

    Trả về:
        Đối tượng MultiSeedMetricSummary.
    """
    if not values:
        raise ValueError(f"Dãy giá trị values cho chỉ số '{metric_name}' không được rỗng.")

    vals = [float(v) for v in values]
    n_seeds = len(vals)
    mean_val = compute_mean(vals)
    std_val = compute_sample_std(vals, ddof=1)
    sem_val = float(std_val / math.sqrt(n_seeds)) if n_seeds > 0 else 0.0
    min_val = min(vals)
    max_val = max(vals)

    scale = 100.0 if percentage else 1.0
    suffix = "%" if percentage else ""
    fmt = f".{precision}f"
    formatted = f"{mean_val * scale:{fmt}} ± {std_val * scale:{fmt}}{suffix}"

    return MultiSeedMetricSummary(
        metric_name=metric_name,
        mean=mean_val,
        std=std_val,
        sem=sem_val,
        min=min_val,
        max=max_val,
        n_seeds=n_seeds,
        values=vals,
        formatted=formatted,
    )


def aggregate_multi_seed_results(
    seed_runs: Sequence[dict[str, Any]] | dict[int, dict[str, Any]],
    model_name: str = "Model",
    seeds: Sequence[int] | None = None,
) -> MultiSeedRunAggregation:
    """
    Tổng hợp toàn diện các chỉ số đánh giá từ danh sách kết quả chạy qua nhiều hạt giống.

    Định dạng đầu vào hỗ trợ:
        1. Danh sách các dictionary kết quả: `[{"accuracy": 0.85, "macro_f1": 0.83}, ...]`
        2. Dictionary ánh xạ seed -> metrics: `{42: {"accuracy": 0.85}, 43: {"accuracy": 0.86}}`

    Tham số:
        seed_runs: Tập kết quả chạy thực nghiệm trên các seed.
        model_name: Tên mô hình đang tổng hợp.
        seeds: Danh sách nhãn seed tương ứng (tùy chọn).

    Trả về:
        Đối tượng MultiSeedRunAggregation chứa tóm tắt thống kê cho từng chỉ số số học.
    """
    run_list: list[dict[str, Any]] = []
    inferred_seeds: list[int] = []

    if isinstance(seed_runs, dict):
        for s_key, metrics_dict in seed_runs.items():
            run_list.append(metrics_dict)
            try:
                inferred_seeds.append(int(s_key))
            except (ValueError, TypeError):
                inferred_seeds.append(len(inferred_seeds))
    else:
        run_list = list(seed_runs)
        if seeds is not None:
            inferred_seeds = [int(s) for s in seeds]
        else:
            inferred_seeds = list(range(len(run_list)))

    if not run_list:
        raise ValueError("seed_runs rỗng, không thể tổng hợp đa hạt giống.")

    # Tìm tập hợp tất cả các trường có giá trị số (float, int)
    numeric_keys: set[str] = set()
    for run in run_list:
        for k, v in run.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_keys.add(k)

    aggregated_metrics: dict[str, MultiSeedMetricSummary] = {}
    for key in sorted(numeric_keys):
        # Trích xuất giá trị cho key này từ các run có chứa key
        vals = [float(run[key]) for run in run_list if key in run and isinstance(run[key], (int, float))]
        if vals:
            # Tự động định dạng phần trăm cho accuracy và f1 nếu giá trị trong khoảng [0, 1]
            is_pct = any(m in key.lower() for m in ("acc", "f1", "precision", "recall")) and max(vals) <= 1.0
            aggregated_metrics[key] = aggregate_seed_values(
                values=vals,
                metric_name=key,
                percentage=is_pct,
                precision=2 if is_pct else 4,
            )

    return MultiSeedRunAggregation(
        model_name=model_name,
        seeds=inferred_seeds,
        n_runs=len(run_list),
        metrics=aggregated_metrics,
    )


# =====================================================================
# 6. Giao diện Dòng lệnh và Trình diễn Thực thi (CLI)
# =====================================================================


def run_test_set_verification_cli(
    test_csv_path: Path | str = TEST_CSV_PATH,
    progress_path: Path | str = PROGRESS_PATH,
) -> int:
    """
    Thực hiện kiểm tra tính toàn vẹn của tập kiểm thử từ giao diện dòng lệnh.
    In báo cáo chi tiết và trả về mã thoát exit code (0: Thành công, 1: Thất bại).
    """
    print("=" * 70)
    print("VIETNAMESE LATEX KNOWLEDGE ASSISTANT - TEST SET INTEGRITY VERIFIER")
    print("Tuân thủ Honesty Rule #3 (AGENTS.md)")
    print("=" * 70)

    try:
        is_valid, actual, expected = verify_test_set_checksum(
            test_csv_path=test_csv_path,
            progress_path=progress_path,
            raise_on_error=False,
        )
        print(f"  Đường dẫn tệp kiểm thử : {Path(test_csv_path).resolve()}")
        print(f"  Mã băm thực tế (Actual) : {actual}")
        print(f"  Mã băm quy định (PROGRESS): {expected}")

        if is_valid:
            print("  Trạng thái xác thực     : [ĐẠT CHUẨN] HỢP LỆ (VERIFIED)")
            print("=" * 70)
            return 0
        else:
            print("  Trạng thái xác thực     : [CẢNH BÁO] SAI LỆCH MÃ BĂM (MISMATCH)")
            print("  Phát hiện nguy cơ vi phạm Honesty Rule #3!")
            print("=" * 70)
            return 1
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        print(f"  Lỗi trong quá trình kiểm thực: {e}")
        print("=" * 70)
        return 1


def main() -> None:
    """Hàm điểm nhập CLI chính của module train/ablation_stats.py."""
    parser = argparse.ArgumentParser(
        description="Tiện ích kiểm thực tập kiểm thử và phân tích thống kê ablation."
    )
    parser.add_argument(
        "--verify-test",
        action="store_true",
        help="Kiểm thực mã băm SHA-256 của data/test.csv đối chiếu với PROGRESS.md.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(TEST_CSV_PATH),
        help=f"Đường dẫn tới tệp test.csv (mặc định: {TEST_CSV_PATH}).",
    )
    parser.add_argument(
        "--progress-path",
        type=str,
        default=str(PROGRESS_PATH),
        help=f"Đường dẫn tới tệp PROGRESS.md (mặc định: {PROGRESS_PATH}).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Chạy kịch bản mô phỏng kiểm định McNemar, Bootstrap CI và tổng hợp hạt giống.",
    )

    args = parser.parse_args()

    # Nếu gọi cờ verify-test
    if args.verify_test:
        code = run_test_set_verification_cli(
            test_csv_path=args.test_csv,
            progress_path=args.progress_path,
        )
        sys.exit(code)

    if args.demo:
        set_seed(DEFAULT_SEED)
        print("\n--- DEMO: BOOTSTRAP CI ---")
        y_true = [f"I_{i % 5}" for i in range(100)]
        y_pred = [f"I_{i % 5}" if i % 10 != 0 else "I_0" for i in range(100)]
        res_ci = compute_accuracy_bootstrap_ci(y_true, y_pred, num_bootstrap=500)
        print(f"Accuracy: {res_ci.format_ci(percentage=True)}")

        print("\n--- DEMO: MCNEMAR TEST ---")
        y_pred_b = [f"I_{i % 5}" if i % 4 != 0 else "I_1" for i in range(100)]
        res_mcnemar = compare_models_mcnemar(y_true, y_pred, y_pred_b, "Model A", "Model B")
        print(res_mcnemar.summary)

        print("\n--- DEMO: MULTI-SEED AGGREGATION ---")
        mock_runs = [
            {"accuracy": 0.895, "macro_f1": 0.880},
            {"accuracy": 0.902, "macro_f1": 0.887},
            {"accuracy": 0.908, "macro_f1": 0.892},
        ]
        agg = aggregate_multi_seed_results(mock_runs, model_name="PhoBERT Baseline", seeds=[42, 43, 44])
        print(agg.format_table(percentage=True))
        sys.exit(0)

    # Mặc định thực hiện kiểm thực test set
    code = run_test_set_verification_cli(
        test_csv_path=args.test_csv,
        progress_path=args.progress_path,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
