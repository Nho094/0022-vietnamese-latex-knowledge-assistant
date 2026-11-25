"""
Module hiệu chuẩn ngưỡng tin cậy và co giãn nhiệt độ (Threshold Calibration and Temperature Scaling).

Tệp tin: train/calibrate_tau.py
Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Identifiers bằng tiếng Anh; chú thích (comments/docstrings) bằng tiếng Việt giải thích lý do (WHY) và công thức bài báo khoa học.
- Toàn bộ hằng số cấu hình được nạp từ core/config.py (không dùng số ma thuật).
- Thiết lập seed ngẫu nhiên tất định thông qua core.config.set_seed.
- Quy tắc Trung thực #3 (Honesty Rule #3):
  + Hiệu chuẩn ngưỡng tau, theta và nhiệt độ T ĐƯỢC ĐÁNH GIÁ NGHIÊM NGẶT TRÊN TẬP VALIDATION (data/val.csv).
  + Tuyệt đối KHÔNG đọc, tinh chỉnh hay can thiệp vào tập kiểm thử data/test.csv.
  + Kiểm thực phòng vệ ném lỗi ngay lập tức nếu phát hiện đường dẫn hoặc mã băm của test.csv.
- Cơ sở lý thuyết học thuật:
  1. Co giãn nhiệt độ (Temperature Scaling) theo Guo et al., ICML 2017 ('On Calibration of Modern Neural Networks'):
     - z_i(T) = z_i / T với T > 0 được tối ưu hóa để cực tiểu hóa hàm mất mát Negative Log-Likelihood (NLL).
     - Đánh giá sự cải thiện của Expected Calibration Error (ECE) trước và sau khi co giãn nhiệt độ.
  2. Phân loại có chọn lọc (Selective Classification) theo Geifman & El-Yaniv, NIPS 2017 và Chow, 1970:
     - Hàm chọn lọc g(x) = 1(max_k p_k >= tau).
     - Hiệu chuẩn ngưỡng tau để đạt độ chính xác chọn lọc mục tiêu (Target Selective Accuracy: 95%).
     - Tính toán đường cong Đánh đổi Rủi ro - Độ bao phủ (Risk-Coverage Curve).
  3. Hiệu chuẩn ngưỡng truy hồi dự phòng theta (Retrieval Fallback Threshold):
     - Đánh giá trên các trường hợp fallback (p < tau) và dữ liệu validation.
     - Cân bằng giữa độ chính xác truy hồi kịch bản đúng và tỷ lệ loại bỏ yêu cầu ngoài phạm vi (out-of-scope).
  4. Đo lường tỷ lệ loại bỏ ngoài phạm vi (Out-of-Scope Rejection Rate) và tỷ lệ từ chối sai (False Rejection Rate).
  5. Xuất biểu đồ đường cong Risk-Coverage ra report/figures/risk_coverage.png và tham số ra models/thresholds.json.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import sys
from collections.abc import Sequence
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
import scipy.optimize

from core.config import (
    DEFAULT_CALIBRATION_TAU_STEPS,
    DEFAULT_CALIBRATION_THETA_STEPS,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_ECE_BINS,
    DEFAULT_RISK_COVERAGE_PATH,
    DEFAULT_SEED,
    DEFAULT_TARGET_SELECTIVE_ACCURACY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEMPERATURE_BOUNDS,
    DEFAULT_THRESHOLDS_PATH,
    OUT_OF_SCOPE_INTENT,
    SMOKE_THRESHOLDS_PATH,
    TAU_INTENT_THRESHOLD,
    TEST_CSV_PATH,
    TEST_CSV_SHA256,
    THETA_RETRIEVAL_THRESHOLD,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    checkpoint_is_smoke,
    checkpoint_sha256,
    set_seed,
)
from core.preprocess import preprocess_fingerprint
from core.retrieval import Retriever
from train.evaluate import compute_ece_details

logger = logging.getLogger(__name__)


# =====================================================================
# Cấu trúc dữ liệu kết quả hiệu chuẩn (Data Classes)
# =====================================================================


@dataclass
class TemperatureOptimizationResult:
    """
    Kết quả tối ưu hóa tham số nhiệt độ T theo Guo et al. (ICML 2017).

    Thuộc tính:
        optimal_temperature: Giá trị T* tối ưu làm cực tiểu hóa hàm NLL trên tập Validation.
        nll_before: Giá trị mất mát Negative Log-Likelihood trước khi hiệu chuẩn (T = 1.0).
        nll_after: Giá trị mất mát NLL sau khi hiệu chuẩn (T = T*).
        ece_before: Sai số hiệu chuẩn kỳ vọng ECE trước khi co giãn nhiệt độ (T = 1.0).
        ece_after: Sai số hiệu chuẩn kỳ vọng ECE sau khi co giãn nhiệt độ (T = T*).
        converged: Cờ xác nhận thuật toán tối ưu hóa hội tụ thành công.
    """

    optimal_temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary."""
        return asdict(self)


@dataclass
class TauCalibrationResult:
    """
    Kết quả hiệu chuẩn ngưỡng tin cậy tau cho phân loại có chọn lọc (Selective Classification).
    Tham chiếu: Geifman & El-Yaniv (NIPS 2017), Chow (1970).

    Thuộc tính:
        optimal_tau: Ngưỡng tau* tối ưu đạt hoặc vượt độ chính xác chọn lọc mục tiêu.
        target_accuracy: Độ chính xác chọn lọc mục tiêu cần đạt (mặc định 0.95).
        selective_accuracy: Độ chính xác thực tế trên các mẫu được chấp nhận tại tau*.
        coverage: Tỷ lệ bao phủ (tỷ lệ mẫu được chấp nhận vượt ngưỡng tau*).
        risk: Rủi ro chọn lọc (1 - selective_accuracy, tức tỷ lệ lỗi trên mẫu chấp nhận).
        accepted_samples: Số lượng mẫu được chấp nhận tại tau*.
        total_samples: Tổng số mẫu đánh giá trên tập Validation.
        target_met: Cờ xác nhận có đạt được độ chính xác mục tiêu hay không.
    """

    optimal_tau: float
    target_accuracy: float
    selective_accuracy: float
    coverage: float
    risk: float
    accepted_samples: int
    total_samples: int
    target_met: bool

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary."""
        return asdict(self)


@dataclass
class RiskCoverageCurveResult:
    """
    Dữ liệu đường cong Đánh đổi Rủi ro - Độ bao phủ (Risk-Coverage Profile).

    Thuộc tính:
        taus: Danh sách các ngưỡng tau được đánh giá từ 0.0 đến 1.0.
        coverages: Tỷ lệ bao phủ tương ứng với từng ngưỡng tau.
        risks: Rủi ro chọn lọc tương ứng với từng ngưỡng tau.
        accuracies: Độ chính xác chọn lọc tương ứng với từng ngưỡng tau.
        accepted_counts: Số lượng mẫu được chấp nhận cho mỗi tau.
        auc_rc: Diện tích dưới đường cong Rủi ro - Độ bao phủ (Area under Risk-Coverage Curve).
    """

    taus: list[float]
    coverages: list[float]
    risks: list[float]
    accuracies: list[float]
    accepted_counts: list[int]
    auc_rc: float

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary."""
        return asdict(self)


@dataclass
class ThetaCalibrationResult:
    """
    Kết quả hiệu chuẩn ngưỡng truy hồi dự phòng theta trên các trường hợp fallback (p < tau).

    Thuộc tính:
        optimal_theta: Ngưỡng tương đồng cosine theta* tối ưu.
        fallback_accuracy: Độ chính xác truy hồi đúng kịch bản trên các trường hợp fallback được chấp nhận.
        fallback_coverage: Tỷ lệ các trường hợp fallback được truy hồi chấp nhận (score >= theta).
        oos_rejection_rate: Tỷ lệ câu hỏi ngoài phạm vi (OUT_OF_SCOPE) bị loại bỏ đúng (score < theta).
        false_rejection_rate: Tỷ lệ câu hỏi trong phạm vi bị từ chối sai (score < theta).
        total_fallback_samples: Tổng số mẫu fallback được đánh giá.
    """

    optimal_theta: float
    fallback_accuracy: float
    fallback_coverage: float
    oos_rejection_rate: float
    false_rejection_rate: float
    total_fallback_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary."""
        return asdict(self)


@dataclass
class RejectionRateResult:
    """
    Đo lường tỷ lệ loại bỏ yêu cầu ngoài phạm vi và tỷ lệ từ chối sai trên toàn bộ hệ thống.

    Thuộc tính:
        classifier_oos_rejection_rate: Tỷ lệ OOS bị từ chối ở tầng phân loại PhoBERT (p < tau).
        classifier_false_rejection_rate: Tỷ lệ in-scope bị rơi vào fallback ở tầng phân loại (p < tau).
        system_oos_rejection_rate: Tỷ lệ OOS bị loại bỏ hoàn toàn bởi toàn hệ thống (báo không hiểu).
        system_false_rejection_rate: Tỷ lệ in-scope bị loại bỏ sai bởi toàn hệ thống (báo không hiểu).
        num_oos_samples: Số lượng mẫu ngoài phạm vi trên tập Validation.
        num_in_scope_samples: Số lượng mẫu trong phạm vi trên tập Validation.
    """

    classifier_oos_rejection_rate: float
    classifier_false_rejection_rate: float
    system_oos_rejection_rate: float
    system_false_rejection_rate: float
    num_oos_samples: int
    num_in_scope_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary."""
        return asdict(self)


@dataclass
class CalibrationPipelineResult:
    """
    Kết quả tổng thể toàn diện của quy trình hiệu chuẩn ngưỡng và co giãn nhiệt độ.
    """

    temperature_result: TemperatureOptimizationResult
    tau_result: TauCalibrationResult
    curve_result: RiskCoverageCurveResult
    theta_result: ThetaCalibrationResult
    rejection_result: RejectionRateResult
    thresholds_path: Path
    figure_path: Path
    val_samples_total: int
    val_samples_in_scope: int
    val_samples_out_of_scope: int
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi toàn bộ kết quả sang dictionary."""
        return {
            "tau": self.tau_result.optimal_tau,
            "theta": self.theta_result.optimal_theta,
            "temperature": self.temperature_result.optimal_temperature,
            "target_selective_accuracy": self.tau_result.target_accuracy,
            "val_selective_accuracy": self.tau_result.selective_accuracy,
            "val_coverage": self.tau_result.coverage,
            "val_risk": self.tau_result.risk,
            "ece_before": self.temperature_result.ece_before,
            "ece_after": self.temperature_result.ece_after,
            "nll_before": self.temperature_result.nll_before,
            "nll_after": self.temperature_result.nll_after,
            "classifier_oos_rejection_rate": self.rejection_result.classifier_oos_rejection_rate,
            "classifier_false_rejection_rate": self.rejection_result.classifier_false_rejection_rate,
            "system_oos_rejection_rate": self.rejection_result.system_oos_rejection_rate,
            "system_false_rejection_rate": self.rejection_result.system_false_rejection_rate,
            "val_samples_total": self.val_samples_total,
            "val_samples_in_scope": self.val_samples_in_scope,
            "val_samples_out_of_scope": self.val_samples_out_of_scope,
            "timestamp": self.timestamp,
            "source_dataset": "data/val.csv",
            "temperature_details": self.temperature_result.to_dict(),
            "tau_details": self.tau_result.to_dict(),
            "theta_details": self.theta_result.to_dict(),
            "rejection_details": self.rejection_result.to_dict(),
            "auc_rc": self.curve_result.auc_rc,
        }


# =====================================================================
# 1. Kiểm thực phòng vệ nghiêm ngặt chống rò rỉ dữ liệu Test (Honesty Rule #3)
# =====================================================================


def verify_not_test_dataset(dataset_path: Path | str) -> None:
    """
    Kiểm thực phòng vệ nghiêm ngặt chống rò rỉ hoặc tinh chỉnh trên tập kiểm thử (test.csv)
    theo Honesty Rule #3 (AGENTS.md).

    Quy tắc:
    - Tuyệt đối không tinh chỉnh siêu tham số trên test set.
    - Ném ngoại lệ ValueError nếu phát hiện đường dẫn hoặc mã băm trùng khớp với test.csv.

    Tham số:
        dataset_path: Đường dẫn tệp dữ liệu cần kiểm tra.

    Ngoại lệ:
        ValueError: Nếu tệp là data/test.csv hoặc có mã băm SHA-256 của tập kiểm thử.
    """
    p = Path(dataset_path).resolve()
    test_p = TEST_CSV_PATH.resolve()

    # Kiểm tra tên tệp và đường dẫn phân giải tuyệt đối
    if p == test_p or p.name == "test.csv":
        raise ValueError(
            f"Vi phạm Honesty Rule #3 (AGENTS.md): Tuyệt đối không tinh chỉnh trên tập kiểm thử '{p}'. "
            f"Quá trình hiệu chuẩn ngưỡng chỉ được thực thi trên tập Validation (val.csv)."
        )

    # Kiểm tra mã băm nội dung tệp nếu tệp tồn tại
    if p.exists() and p.is_file():
        try:
            with p.open("rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash.lower() == TEST_CSV_SHA256.lower():
                raise ValueError(
                    f"Vi phạm Honesty Rule #3 (AGENTS.md): Tệp dữ liệu '{p}' có mã băm SHA-256 trùng khớp "
                    f"với tập kiểm thử test.csv ({file_hash}). Nghiêm cấm hiệu chuẩn ngưỡng trên tập kiểm thử!"
                )
        except OSError as e:
            logger.warning("Không thể đọc tệp để tính mã băm kiểm thực: %s", e)


# =====================================================================
# 2. Co giãn nhiệt độ và Tối ưu hóa NLL (Guo et al., ICML 2017)
# =====================================================================


def compute_nll(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float = 1.0,
) -> float:
    """
    Tính toán giá trị mất mát Negative Log-Likelihood (NLL) với nhiệt độ T.

    Công thức:
        NLL(T) = - 1/N * sum_{i=1}^N log p_{i, y_i}(T)
        trong đó p_{i, c}(T) = exp(z_{i, c} / T) / sum_j exp(z_{i, j} / T).

    Kỹ thuật số học:
        Sử dụng mẹo Log-Sum-Exp ổn định tránh tràn số (numerical overflow):
        log sum exp(z/T) = m/T + log sum exp((z - m)/T) với m = max(z).

    Tham số:
        logits: Ma trận logits chưa chuẩn hóa kích thước (N, K).
        labels: Mảng chỉ số nhãn ground-truth kích thước (N,).
        temperature: Tham số nhiệt độ T > 0.

    Trả về:
        Giá trị mất mát NLL dạng số thực.
    """
    if temperature <= 0.0:
        raise ValueError(f"Tham số nhiệt độ T phải > 0.0, nhận được: {temperature}")

    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    # Log-Sum-Exp trick ổn định số học
    max_logits = np.max(logits, axis=-1, keepdims=True)
    scaled_diff = (logits - max_logits) / temperature
    log_sum_exp = np.log(np.sum(np.exp(scaled_diff), axis=-1)) + (
        max_logits.squeeze(-1) / temperature
    )

    # Lấy logit của lớp ground-truth
    gt_logits = logits[np.arange(n_samples), labels] / temperature
    nll = -np.mean(gt_logits - log_sum_exp)
    return float(nll)


def apply_temperature_softmax(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Áp dụng co giãn nhiệt độ lên logits và tính phân phối xác suất softmax chuẩn hóa.

    Công thức:
        p_{i, c}(T) = exp( (z_{i, c} - max(z_i)) / T ) / sum_j exp( (z_{i, j} - max(z_i)) / T )

    Tham số:
        logits: Ma trận logits kích thước (N, K).
        temperature: Tham số nhiệt độ T > 0.

    Trả về:
        Ma trận xác suất đã chuẩn hóa kích thước (N, K) với tổng mỗi hàng bằng 1.0.
    """
    if temperature <= 0.0:
        raise ValueError(f"Tham số nhiệt độ T phải > 0.0, nhận được: {temperature}")

    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_scaled = np.exp((logits - max_logits) / temperature)
    sum_exp = np.sum(exp_scaled, axis=-1, keepdims=True)
    probs = exp_scaled / np.maximum(sum_exp, 1e-12)
    return probs


def optimize_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float] = DEFAULT_TEMPERATURE_BOUNDS,
    num_bins: int = DEFAULT_ECE_BINS,
) -> TemperatureOptimizationResult:
    """
    Tối ưu hóa tham số nhiệt độ T để cực tiểu hóa NLL trên tập Validation.

    Tham chiếu học thuật:
        Guo, Pleiss, Sun, Weinberger (ICML 2017) 'On Calibration of Modern Neural Networks'.

    Quy trình:
    1. Tính NLL và ECE tại nhiệt độ gốc T = 1.0.
    2. Sử dụng thuật toán tối ưu hóa 1 chiều scipy.optimize.minimize_scalar (bounded)
       để tìm T* trong khoảng bounds (mặc định [0.05, 10.0]).
    3. Tính NLL và ECE sau khi áp dụng T*.

    Tham số:
        logits: Ma trận logits kích thước (N, K).
        labels: Mảng chỉ số nhãn thực tế kích thước (N,).
        bounds: Cặp giới hạn nhiệt độ (min_T, max_T).
        num_bins: Số lượng bin chia đều để tính ECE.

    Trả về:
        Đối tượng TemperatureOptimizationResult chứa kết quả chi tiết.
    """
    n_samples = len(labels)
    if n_samples == 0:
        return TemperatureOptimizationResult(
            optimal_temperature=DEFAULT_TEMPERATURE,
            nll_before=0.0,
            nll_after=0.0,
            ece_before=0.0,
            ece_after=0.0,
            converged=True,
        )

    # 1. Tính toán trước khi hiệu chuẩn (T = 1.0)
    nll_before = compute_nll(logits, labels, temperature=DEFAULT_TEMPERATURE)
    probs_before = apply_temperature_softmax(logits, temperature=DEFAULT_TEMPERATURE)
    preds_before = np.argmax(probs_before, axis=-1)
    confs_before = np.max(probs_before, axis=-1)

    ece_res_before = compute_ece_details(
        y_true=labels.tolist(),
        y_pred=preds_before.tolist(),
        confidences=confs_before.tolist(),
        num_bins=num_bins,
    )

    # 2. Tối ưu hóa hàm NLL theo T
    def objective_fn(t_val: float) -> float:
        return compute_nll(logits, labels, temperature=t_val)

    opt_res = scipy.optimize.minimize_scalar(
        objective_fn,
        bounds=bounds,
        method="bounded",
    )

    optimal_t = float(opt_res.x) if opt_res.success else DEFAULT_TEMPERATURE
    # Giới hạn trong khoảng biên an toàn
    optimal_t = max(bounds[0], min(bounds[1], optimal_t))

    # 3. Tính toán sau khi co giãn nhiệt độ (T = T*)
    nll_after = compute_nll(logits, labels, temperature=optimal_t)
    probs_after = apply_temperature_softmax(logits, temperature=optimal_t)
    preds_after = np.argmax(probs_after, axis=-1)
    confs_after = np.max(probs_after, axis=-1)

    ece_res_after = compute_ece_details(
        y_true=labels.tolist(),
        y_pred=preds_after.tolist(),
        confidences=confs_after.tolist(),
        num_bins=num_bins,
    )

    return TemperatureOptimizationResult(
        optimal_temperature=optimal_t,
        nll_before=nll_before,
        nll_after=nll_after,
        ece_before=ece_res_before.ece,
        ece_after=ece_res_after.ece,
        converged=bool(opt_res.success),
    )


# =====================================================================
# 3. Phân loại có chọn lọc và Hiệu chuẩn ngưỡng tau (Selective Classification)
# =====================================================================


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    num_steps: int = DEFAULT_CALIBRATION_TAU_STEPS,
) -> RiskCoverageCurveResult:
    """
    Tính toán đường cong Đánh đổi Rủi ro - Độ bao phủ (Risk-Coverage Profile) trên dải ngưỡng tau.

    Tham chiếu học thuật:
        Geifman & El-Yaniv (NIPS 2017) 'Selective Classification for Deep Neural Networks'.

    Công thức:
        Coverage(tau) = |{i : conf_i >= tau}| / N
        SelectiveAccuracy(tau) = sum_{i: conf_i >= tau} 1(y_i == pred_i) / |{i: conf_i >= tau}|
        SelectiveRisk(tau) = 1.0 - SelectiveAccuracy(tau)

    Tham số:
        confidences: Mảng độ tin cậy dự đoán (sau co giãn nhiệt độ) trong đoạn [0, 1].
        y_true: Nhãn thực tế ground-truth.
        y_pred: Nhãn dự đoán của mô hình.
        num_steps: Số lượng bước chia trên đoạn [0.0, 1.0] (mặc định 101).

    Trả về:
        Đối tượng RiskCoverageCurveResult chứa các mảng coverages, risks, accuracies và AUC-RC.
    """
    n = len(y_true)
    if n == 0 or len(y_pred) != n or len(confidences) != n:
        raise ValueError(
            f"Kích thước dữ liệu không khớp hoặc rỗng: n={n}, y_pred={len(y_pred)}, confs={len(confidences)}"
        )

    taus = np.linspace(0.0, 1.0, num_steps).tolist()
    conf_arr = np.asarray(confidences, dtype=np.float64)
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)

    coverages: list[float] = []
    risks: list[float] = []
    accuracies: list[float] = []
    accepted_counts: list[int] = []

    for tau in taus:
        mask = conf_arr >= tau
        count_accepted = int(np.sum(mask))
        coverage = float(count_accepted / n)

        if count_accepted > 0:
            correct = int(np.sum(true_arr[mask] == pred_arr[mask]))
            acc = float(correct / count_accepted)
            risk = float(1.0 - acc)
        else:
            # Khi không chấp nhận mẫu nào, quy ước rủi ro bằng 0.0 và độ chính xác tuyệt đối 1.0
            acc = 1.0
            risk = 0.0

        coverages.append(coverage)
        risks.append(risk)
        accuracies.append(acc)
        accepted_counts.append(count_accepted)

    # Tính toán diện tích dưới đường cong Risk-Coverage (AUC-RC) theo phương pháp hình thang
    # Lưu ý: Sắp xếp theo Coverage tăng dần để tích phân số học chính xác
    sorted_pairs = sorted(zip(coverages, risks), key=lambda pair: pair[0])
    sorted_cov = [p[0] for p in sorted_pairs]
    sorted_risk = [p[1] for p in sorted_pairs]

    try:
        # numpy >= 2.0 sử dụng trapezoid, tương thích ngược trapz
        if hasattr(np, "trapezoid"):
            auc_rc = float(np.trapezoid(sorted_risk, sorted_cov))
        else:
            auc_rc = float(np.trapz(sorted_risk, sorted_cov))
    except (TypeError, ValueError, FloatingPointError):
        auc_rc = 0.0

    return RiskCoverageCurveResult(
        taus=taus,
        coverages=coverages,
        risks=risks,
        accuracies=accuracies,
        accepted_counts=accepted_counts,
        auc_rc=auc_rc,
    )


def calibrate_tau_selective_accuracy(
    confidences: np.ndarray,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    target_accuracy: float = DEFAULT_TARGET_SELECTIVE_ACCURACY,
    num_steps: int = DEFAULT_CALIBRATION_TAU_STEPS,
) -> TauCalibrationResult:
    """
    Hiệu chuẩn ngưỡng tin cậy tau* cho bộ phân loại có chọn lọc để đạt độ chính xác mục tiêu.

    Mục tiêu tối ưu hóa:
        tau* = min { tau in [0, 1] : SelectiveAccuracy(tau) >= target_accuracy }
        (Chọn ngưỡng tau nhỏ nhất thỏa mãn độ chính xác để tối đa hóa độ bao phủ Coverage).

    Nếu không có ngưỡng nào đạt target_accuracy (ví dụ mô hình chưa đủ mạnh):
        Chọn tau làm cực đại hóa độ chính xác chọn lọc hoặc điểm điều hòa Harmonic F1 giữa Acc và Coverage.

    Tham số:
        confidences: Độ tin cậy dự đoán của mô hình.
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        target_accuracy: Độ chính xác chọn lọc mục tiêu (mặc định 0.95 theo DEFAULT_TARGET_SELECTIVE_ACCURACY).
        num_steps: Số điểm lưới quét ngưỡng tau.

    Trả về:
        Đối tượng TauCalibrationResult.
    """
    curve_res = compute_risk_coverage_curve(
        confidences=confidences,
        y_true=y_true,
        y_pred=y_pred,
        num_steps=num_steps,
    )

    n = len(y_true)
    target_met = False
    valid_candidates: list[tuple[float, float, float, int]] = []

    for tau, cov, acc, cnt in zip(
        curve_res.taus,
        curve_res.coverages,
        curve_res.accuracies,
        curve_res.accepted_counts,
    ):
        # Yêu cầu phải chấp nhận ít nhất 1 mẫu và đạt độ chính xác mục tiêu
        if cnt > 0 and acc >= target_accuracy:
            valid_candidates.append((tau, cov, acc, cnt))

    if valid_candidates:
        # Chọn tau nhỏ nhất để đạt Coverage lớn nhất
        target_met = True
        best_candidate = min(valid_candidates, key=lambda c: c[0])
        optimal_tau, cov, acc, cnt = best_candidate
    else:
        # Nếu không có ngưỡng nào đạt target_accuracy:
        # Tìm tau tối đa hóa điểm F1 điều hòa giữa Accuracy và Coverage
        best_f1 = -1.0
        best_candidate = (TAU_INTENT_THRESHOLD, 0.0, 0.0, 0)
        for tau, cov, acc, cnt in zip(
            curve_res.taus,
            curve_res.coverages,
            curve_res.accuracies,
            curve_res.accepted_counts,
        ):
            if cnt > 0:
                f1_harmonic = (2 * acc * cov) / (acc + cov + 1e-12)
                if f1_harmonic > best_f1:
                    best_f1 = f1_harmonic
                    best_candidate = (tau, cov, acc, cnt)

        optimal_tau, cov, acc, cnt = best_candidate

    risk = float(1.0 - acc)
    return TauCalibrationResult(
        optimal_tau=float(optimal_tau),
        target_accuracy=float(target_accuracy),
        selective_accuracy=float(acc),
        coverage=float(cov),
        risk=risk,
        accepted_samples=int(cnt),
        total_samples=n,
        target_met=target_met,
    )


# =====================================================================
# 4. Hiệu chuẩn ngưỡng truy hồi dự phòng theta trên các trường hợp Fallback
# =====================================================================


def calibrate_theta_fallback(
    fallback_scores: Sequence[float],
    fallback_preds: Sequence[Any],
    y_true: Sequence[Any],
    num_steps: int = DEFAULT_CALIBRATION_THETA_STEPS,
) -> ThetaCalibrationResult:
    """
    Hiệu chuẩn ngưỡng tương đồng cosine theta* trên các trường hợp fallback (p < tau).

    Nguyên lý:
    - Khi điểm truy hồi s >= theta: hệ thống coi như 'understood' (gợi ý người dùng lựa chọn kịch bản).
    - Khi điểm truy hồi s < theta: hệ thống 'say not understood' (báo không hiểu).
    - Cần cân bằng:
      + Độ chính xác in-scope: truy hồi đúng intent ground-truth khi chấp nhận.
      + Khả năng loại bỏ out-of-scope: từ chối các câu hỏi nằm ngoài phạm vi hỗ trợ (OUT_OF_SCOPE).

    Hàm mục tiêu:
        Tối ưu hóa điểm cân bằng Harmonic F1 giữa độ chính xác truy hồi đúng (In-Scope Precision)
        và tỷ lệ loại bỏ câu ngoài phạm vi (OOS Rejection Rate).

    Tham số:
        fallback_scores: Điểm số cosine similarity cao nhất của truy hồi cho từng mẫu fallback.
        fallback_preds: Intent được truy hồi dự đoán hàng đầu.
        y_true: Nhãn thực tế (bao gồm cả in-scope và OUT_OF_SCOPE).
        num_steps: Số lượng bước chia trên lưới theta.

    Trả về:
        Đối tượng ThetaCalibrationResult.
    """
    n = len(fallback_scores)
    if n == 0 or len(fallback_preds) != n or len(y_true) != n:
        return ThetaCalibrationResult(
            optimal_theta=THETA_RETRIEVAL_THRESHOLD,
            fallback_accuracy=0.0,
            fallback_coverage=0.0,
            oos_rejection_rate=1.0,
            false_rejection_rate=0.0,
            total_fallback_samples=0,
        )

    thetas = np.linspace(0.0, 1.0, num_steps).tolist()
    scores_arr = np.asarray(fallback_scores, dtype=np.float64)
    preds_arr = np.asarray(fallback_preds)
    trues_arr = np.asarray(y_true)

    is_oos = trues_arr == OUT_OF_SCOPE_INTENT
    num_oos = int(np.sum(is_oos))
    num_in_scope = n - num_oos

    best_score = -1.0
    best_theta = THETA_RETRIEVAL_THRESHOLD
    best_acc = 0.0
    best_cov = 0.0
    best_oos_rr = 1.0 if num_oos == 0 else 0.0
    best_frr = 0.0

    for theta in thetas:
        accepted_mask = scores_arr >= theta
        rejected_mask = ~accepted_mask

        n_accepted = int(np.sum(accepted_mask))
        coverage = float(n_accepted / n)

        # Tính toán trên nhóm Out-of-Scope
        if num_oos > 0:
            oos_rejected = int(np.sum(rejected_mask & is_oos))
            oos_rr = float(oos_rejected / num_oos)
        else:
            oos_rr = 1.0

        # Tính toán trên nhóm In-Scope
        if num_in_scope > 0:
            in_accepted = accepted_mask & (~is_oos)
            in_correct = int(np.sum(in_accepted & (preds_arr == trues_arr)))
            in_accepted_count = int(np.sum(in_accepted))

            in_acc = float(in_correct / in_accepted_count) if in_accepted_count > 0 else 0.0
            in_rejected = int(np.sum(rejected_mask & (~is_oos)))
            frr = float(in_rejected / num_in_scope)
        else:
            in_acc = 1.0
            frr = 0.0

        # Điểm cân bằng tổng thể (Harmonic Score)
        if num_oos > 0:
            harmonic = (2 * in_acc * oos_rr) / (in_acc + oos_rr + 1e-12)
        else:
            harmonic = (2 * in_acc * coverage) / (in_acc + coverage + 1e-12)

        if harmonic > best_score:
            best_score = harmonic
            best_theta = theta
            best_acc = in_acc
            best_cov = coverage
            best_oos_rr = oos_rr
            best_frr = frr

    return ThetaCalibrationResult(
        optimal_theta=float(best_theta),
        fallback_accuracy=float(best_acc),
        fallback_coverage=float(best_cov),
        oos_rejection_rate=float(best_oos_rr),
        false_rejection_rate=float(best_frr),
        total_fallback_samples=n,
    )


# =====================================================================
# 5. Đo lường tỷ lệ loại bỏ yêu cầu ngoài phạm vi và tỷ lệ từ chối sai
# =====================================================================


def compute_rejection_rates(
    clf_confidences: Sequence[float],
    tau: float,
    fallback_scores: Sequence[float],
    theta: float,
    y_true: Sequence[str],
    clf_preds: Sequence[str] | None = None,
    fallback_preds: Sequence[str] | None = None,
) -> RejectionRateResult:
    """
    Tính toán tỷ lệ loại bỏ yêu cầu ngoài phạm vi (OOS Rejection Rate)
    và tỷ lệ từ chối sai (False Rejection Rate) ở cả cấp độ phân loại và hệ thống.

    Định nghĩa:
    1. Classifier-Level:
       - Mẫu bị loại bỏ / rơi vào fallback nếu:
         (pred == OUT_OF_SCOPE) HOẶC (confidence < tau).
       - Classifier OOS Rejection Rate = |{i in OOS : clf_rejected}| / |OOS|.
       - Classifier False Rejection Rate = |{i in In-Scope : clf_rejected}| / |In-Scope|.

    2. System-Level (Toàn hệ thống kết hợp PhoBERT + Fallback Retrieval):
       - Mẫu bị từ chối hoàn toàn (báo không hiểu / ngoài phạm vi) nếu:
         Classifier từ chối VÀ Fallback từ chối (score < theta hoặc fallback_pred == OUT_OF_SCOPE).
       - System OOS Rejection Rate = |{i in OOS : sys_rejected}| / |OOS|.
       - System False Rejection Rate = |{i in In-Scope : sys_rejected}| / |In-Scope|.

    Tham số:
        clf_confidences: Độ tin cậy phân loại từ PhoBERT (sau co giãn nhiệt độ).
        tau: Ngưỡng chấp nhận ý định tau.
        fallback_scores: Điểm số truy hồi cosine similarity (nếu không chạy fallback thì gán 0.0).
        theta: Ngưỡng truy hồi theta.
        y_true: Danh sách nhãn thực tế của toàn bộ tập đánh giá.
        clf_preds: Danh sách nhãn dự đoán từ classifier (tùy chọn).
        fallback_preds: Danh sách nhãn dự đoán từ fallback retrieval (tùy chọn).

    Trả về:
        Đối tượng RejectionRateResult.
    """
    n = len(y_true)
    if len(clf_confidences) != n or len(fallback_scores) != n:
        raise ValueError("Kích thước các danh sách đánh giá không đồng nhất.")

    confs = np.asarray(clf_confidences, dtype=np.float64)
    scores = np.asarray(fallback_scores, dtype=np.float64)
    trues = np.asarray(y_true)

    is_oos = trues == OUT_OF_SCOPE_INTENT
    num_oos = int(np.sum(is_oos))
    num_in_scope = n - num_oos

    # Xác định điều kiện từ chối của classifier
    if clf_preds is not None:
        clf_pred_arr = np.asarray(clf_preds)
        clf_rejected_mask = (clf_pred_arr == OUT_OF_SCOPE_INTENT) | (confs < tau)
    else:
        clf_rejected_mask = confs < tau

    # Xác định điều kiện từ chối của fallback
    if fallback_preds is not None:
        fb_pred_arr = np.asarray(fallback_preds)
        fb_rejected_mask = (
            (scores < theta) | (fb_pred_arr == OUT_OF_SCOPE_INTENT) | (fb_pred_arr == "UNKNOWN")
        )
    else:
        fb_rejected_mask = scores < theta

    # Toàn hệ thống từ chối khi cả classifier và fallback đều từ chối
    sys_rejected_mask = clf_rejected_mask & fb_rejected_mask

    if num_oos == 0:
        clf_oos_rr = 1.0
        sys_oos_rr = 1.0
    else:
        clf_oos_rejected = int(np.sum(is_oos & clf_rejected_mask))
        clf_oos_rr = float(clf_oos_rejected / num_oos)

        sys_oos_rejected = int(np.sum(is_oos & sys_rejected_mask))
        sys_oos_rr = float(sys_oos_rejected / num_oos)

    if num_in_scope == 0:
        clf_frr = 0.0
        sys_frr = 0.0
    else:
        clf_in_rejected = int(np.sum((~is_oos) & clf_rejected_mask))
        clf_frr = float(clf_in_rejected / num_in_scope)

        sys_in_rejected = int(np.sum((~is_oos) & sys_rejected_mask))
        sys_frr = float(sys_in_rejected / num_in_scope)

    return RejectionRateResult(
        classifier_oos_rejection_rate=clf_oos_rr,
        classifier_false_rejection_rate=clf_frr,
        system_oos_rejection_rate=sys_oos_rr,
        system_false_rejection_rate=sys_frr,
        num_oos_samples=num_oos,
        num_in_scope_samples=num_in_scope,
    )


# =====================================================================
# 6. Trực quan hóa Đường cong Risk-Coverage (Figure Generation)
# =====================================================================


def plot_risk_coverage_curve(
    curve_result: RiskCoverageCurveResult,
    calibrated_tau: float,
    target_accuracy: float = DEFAULT_TARGET_SELECTIVE_ACCURACY,
    output_path: Path | str = DEFAULT_RISK_COVERAGE_PATH,
) -> Path:
    """
    Vẽ và lưu biểu đồ đường cong Đánh đổi Rủi ro - Độ bao phủ (Risk-Coverage Curve).

    Cấu trúc hình ảnh:
    - Bảng bên trái: Risk vs Coverage curve, đường rủi ro mục tiêu (1 - target_accuracy),
      và điểm hoạt động tối ưu tại tau* (Calibrated Operating Point).
    - Bảng bên phải: Động lực học ngưỡng - Độ chính xác chọn lọc và Độ bao phủ theo ngưỡng tau.

    Tham số:
        curve_result: Dữ liệu đường cong RiskCoverageCurveResult.
        calibrated_tau: Giá trị ngưỡng tau* đã hiệu chuẩn.
        target_accuracy: Độ chính xác mục tiêu.
        output_path: Đường dẫn tệp PNG cần lưu.

    Trả về:
        Đường dẫn Path tới tệp hình ảnh đã lưu.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    target_risk = 1.0 - target_accuracy

    # Tìm chỉ số tương ứng với calibrated_tau
    tau_arr = np.asarray(curve_result.taus)
    idx_opt = int(np.argmin(np.abs(tau_arr - calibrated_tau)))
    opt_cov = curve_result.coverages[idx_opt]
    opt_risk = curve_result.risks[idx_opt]
    opt_acc = curve_result.accuracies[idx_opt]

    # Khởi tạo biểu đồ 2 panel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # ----------------- PANEL 1: Risk-Coverage Curve -----------------
    ax1.plot(
        curve_result.coverages,
        curve_result.risks,
        color="#1f77b4",
        linewidth=2.5,
        label=f"Risk-Coverage Profile (AUC = {curve_result.auc_rc:.4f})",
    )
    ax1.axhline(
        y=target_risk,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Target Risk <= {target_risk:.2f} (Acc >= {target_accuracy * 100:.0f}%)",
    )

    # Đánh dấu điểm hoạt động hiệu chuẩn
    ax1.scatter(
        [opt_cov],
        [opt_risk],
        color="#d62728",
        s=150,
        zorder=5,
        marker="*",
        label=f"Calibrated Point (tau={calibrated_tau:.2f})",
    )

    ax1.annotate(
        f"tau* = {calibrated_tau:.2f}\nCov = {opt_cov * 100:.1f}%\nAcc = {opt_acc * 100:.1f}%",
        xy=(opt_cov, opt_risk),
        xytext=(min(0.9, opt_cov + 0.08), max(0.05, opt_risk + 0.08)),
        arrowprops={"facecolor": "#d62728", "shrink": 0.08, "width": 1.5, "headwidth": 8},
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#fff2f2", "edgecolor": "#d62728"},
    )

    ax1.set_xlabel("Coverage (Tỷ lệ bao phủ)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Selective Risk (Rủi ro chọn lọc)", fontsize=11, fontweight="bold")
    ax1.set_title("Risk-Coverage Tradeoff (Geifman & El-Yaniv, 2017)", fontsize=12, fontweight="bold")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, max(0.5, max(curve_result.risks) * 1.1))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=9)

    # ----------------- PANEL 2: Metrics vs Tau -----------------
    ax2.plot(
        curve_result.taus,
        curve_result.accuracies,
        color="#2ca02c",
        linewidth=2.0,
        label="Selective Accuracy",
    )
    ax2.plot(
        curve_result.taus,
        curve_result.coverages,
        color="#1f77b4",
        linewidth=2.0,
        linestyle="-.",
        label="Coverage",
    )
    ax2.axhline(
        y=target_accuracy,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        label=f"Target Accuracy ({target_accuracy * 100:.0f}%)",
    )
    ax2.axvline(
        x=calibrated_tau,
        color="#9467bd",
        linestyle=":",
        linewidth=2.0,
        label=f"Calibrated tau* = {calibrated_tau:.2f}",
    )

    ax2.set_xlabel("Confidence Threshold tau (Ngưỡng tin cậy)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Metric Value (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax2.set_title("Threshold Dynamics vs Accuracy & Coverage", fontsize=12, fontweight="bold")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close(fig)
    return out_p


# =====================================================================
# 7. Xuất tham số hiệu chuẩn ra models/thresholds.json
# =====================================================================


def export_calibrated_parameters(
    tau: float,
    theta: float,
    temperature: float,
    metadata: dict[str, Any] | None = None,
    output_path: Path | str = DEFAULT_THRESHOLDS_PATH,
    smoke_output_path: Path | str = SMOKE_THRESHOLDS_PATH,
) -> Path:
    """
    Xuất các tham số hiệu chuẩn (tau, theta, temperature) cùng siêu dữ liệu ra tệp JSON.
    Tương thích hoàn toàn với hàm nạp cấu hình `core.config.load_calibrated_thresholds`.

    Tham số:
        tau: Ngưỡng xác suất softmax chấp nhận intent trực tiếp.
        theta: Ngưỡng tương đồng cosine tối thiểu trong bước truy hồi fallback.
        temperature: Tham số nhiệt độ T sau co giãn nhiệt độ.
        metadata: Bổ sung các thông số đánh giá thực nghiệm (tùy chọn).
        output_path: Đường dẫn tệp JSON đầu ra (mặc định models/thresholds.json).

    Trả về:
        Đường dẫn Path tới tệp JSON đã lưu.
    """
    out_p = Path(output_path)
    is_smoke = bool(metadata and metadata.get("smoke_mode")) or float(tau) == 0.0
    if is_smoke or float(tau) == 0.0:
        out_p = Path(smoke_output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "tau": round(float(tau), 4),
        "theta": round(float(theta), 4),
        "temperature": round(float(temperature), 4),
    }

    if metadata:
        safe_metadata = dict(metadata)
        if is_smoke:
            safe_metadata["smoke_mode"] = True
        for k, v in safe_metadata.items():
            if k not in data:
                if isinstance(v, float):
                    data[k] = round(v, 4)
                else:
                    data[k] = v

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return out_p


# =====================================================================
# 8. Bộ cung cấp Logits và Dự đoán mô hình (Logits Provider)
# =====================================================================


def load_dataset_samples(csv_path: Path | str) -> list[dict[str, str]]:
    """
    Nạp dữ liệu từ tệp CSV chuẩn (text, label, category), xử lý an toàn UTF-8 BOM.
    """
    verify_not_test_dataset(csv_path)
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Tệp dữ liệu không tồn tại: '{p}'")

    with open(p, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_validation_logits_and_predictions(
    val_samples: list[dict[str, str]],
    train_samples: list[dict[str, str]] | None = None,
    checkpoint_dir: Path | str | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """
    Thu nhận ma trận logits, danh sách nhãn dự đoán và nhãn thực tế trên tập Validation.

    Quy trình phân giải nguồn logits:
    1. Nếu có checkpoint PhoBERT hoàn chỉnh tại checkpoint_dir hỗ trợ phân loại nhãn hợp lệ:
       Sử dụng IntentClassifier để tính toán logits mạng neural.
    2. Nếu không có checkpoint hoặc checkpoint thu nhỏ trong smoke-run:
       Huấn luyện nhanh mô hình cơ sở TF-IDF Linear/Logistic trên tập huấn luyện (train.csv)
       để thu nhận ma trận logits thực tế khách quan trên tập Validation (val.csv).

    Tham số:
        val_samples: Danh sách mẫu tập kiểm thực.
        train_samples: Danh sách mẫu tập huấn luyện (dùng huấn luyện mô hình nếu cần).
        checkpoint_dir: Đường dẫn checkpoint PhoBERT (tùy chọn).

    Trả về:
        Tuple: (logits, labels_vocab, y_true, y_pred)
    """
    val_texts = [s["text"] for s in val_samples]
    val_trues = [s["label"] for s in val_samples]

    # Kiểm tra nếu có thể nạp IntentClassifier từ checkpoint_dir
    if checkpoint_dir is not None and Path(checkpoint_dir).exists():
        try:
            from core.classifier import IntentClassifier

            clf = IntentClassifier.from_pretrained(
                checkpoint_dir,
                verify_fingerprint=False,
            )
            # Kiểm tra xem nhãn của checkpoint có khớp với tập validation không
            clf_labels = set(clf.id2label.values())
            val_unique_labels = set(val_trues)
            if len(clf_labels.intersection(val_unique_labels)) > 5:
                # Checkpoint hợp lệ cho bài toán
                preds_res = clf.predict_batch(val_texts)
                y_pred = [res.intent for res in preds_res]

                # Xây dựng logits từ all_probabilities hoặc decision
                label_list = sorted(clf_labels)
                lbl2idx = {lbl: i for i, lbl in enumerate(label_list)}
                logits_mat = np.zeros((len(val_samples), len(label_list)), dtype=np.float64)
                for i, res in enumerate(preds_res):
                    for lbl, prob in res.all_probabilities.items():
                        if lbl in lbl2idx:
                            # Chuyển ngược từ xác suất sang log-odds
                            logits_mat[i, lbl2idx[lbl]] = np.log(max(1e-6, prob))
                return logits_mat, label_list, val_trues, y_pred
        except (ImportError, OSError, ValueError, RuntimeError, KeyError) as e:
            logger.info("Không dùng checkpoint PhoBERT, chuyển sang baseline: %s", e)

    # Sử dụng baseline huấn luyện trên train.csv
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    if train_samples is None:
        if TRAIN_CSV_PATH.exists():
            train_samples = load_dataset_samples(TRAIN_CSV_PATH)
        else:
            raise FileNotFoundError("Cần tập train.csv để huấn luyện mô hình sinh logits validation.")

    train_texts = [s["text"] for s in train_samples]
    train_labels = [s["label"] for s in train_samples]

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    x_train = vec.fit_transform(train_texts)

    clf = LogisticRegression(max_iter=200, C=1.0, random_state=DEFAULT_SEED)
    clf.fit(x_train, train_labels)

    x_val = vec.transform(val_texts)
    decision = clf.decision_function(x_val)

    if decision.ndim == 1:
        # Trường hợp phân loại 2 lớp
        logits_mat = np.column_stack([-decision, decision])
        classes_list = list(clf.classes_)
    else:
        logits_mat = np.asarray(decision, dtype=np.float64)
        classes_list = list(clf.classes_)

    # Dự đoán nhãn
    pred_indices = np.argmax(logits_mat, axis=-1)
    y_pred = [classes_list[idx] for idx in pred_indices]

    return logits_mat, classes_list, val_trues, y_pred


# =====================================================================
# 9. Đường ống thực thi hiệu chuẩn hoàn chỉnh (Calibration Pipeline)
# =====================================================================


def run_calibration_pipeline(
    val_csv_path: Path | str = VAL_CSV_PATH,
    train_csv_path: Path | str = TRAIN_CSV_PATH,
    checkpoint_dir: Path | str | None = None,
    target_accuracy: float = DEFAULT_TARGET_SELECTIVE_ACCURACY,
    output_thresholds_path: Path | str = DEFAULT_THRESHOLDS_PATH,
    output_figure_path: Path | str = DEFAULT_RISK_COVERAGE_PATH,
    seed: int = DEFAULT_SEED,
    smoke_mode: bool = False,
    output_smoke_path: Path | str = SMOKE_THRESHOLDS_PATH,
) -> CalibrationPipelineResult:
    """
    Thực thi toàn bộ quy trình hiệu chuẩn ngưỡng tin cậy và co giãn nhiệt độ trên tập Validation.

    Tuân thủ nghiêm ngặt:
    - Đánh giá tuyệt đối trên tập Validation (val.csv).
    - Ngăn chặn triệt để mọi truy cập vào tập Test (test.csv) theo Honesty Rule #3.

    Các bước xử lý:
    1. Kiểm thực đường dẫn chống rò rỉ dữ liệu test.
    2. Nạp dữ liệu validation và thu nhận ma trận logits của mô hình.
    3. Tối ưu hóa tham số nhiệt độ T* trên các mẫu in-scope để cực tiểu hóa NLL và cải thiện ECE.
    4. Áp dụng T* để tính toán xác suất và độ tin cậy được hiệu chuẩn.
    5. Hiệu chuẩn ngưỡng tau* cho Selective Classification đạt target_accuracy (95%).
    6. Tính toán đường cong Risk-Coverage và lưu biểu đồ ra report/figures/risk_coverage.png.
    7. Trích xuất các trường hợp Fallback (p < tau*) và hiệu chuẩn ngưỡng theta* bằng SbertRetrievalFallback.
    8. Tính toán tỷ lệ loại bỏ Out-of-Scope và tỷ lệ từ chối sai (False Rejection Rate).
    9. Xuất các tham số hiệu chuẩn (tau, theta, temperature) ra models/thresholds.json.

    Tham số:
        val_csv_path: Đường dẫn tệp CSV tập kiểm thực (mặc định data/val.csv).
        train_csv_path: Đường dẫn tệp CSV tập huấn luyện (mặc định data/train.csv).
        checkpoint_dir: Đường dẫn checkpoint mô hình nếu có.
        target_accuracy: Độ chính xác chọn lọc mục tiêu (mặc định 0.95).
        output_thresholds_path: Đường dẫn xuất thresholds.json (mặc định models/thresholds.json).
        output_figure_path: Đường dẫn lưu hình ảnh risk_coverage.png.
        seed: Hạt giống ngẫu nhiên thiết lập cho thực nghiệm.
        smoke_mode: Cờ kích hoạt chế độ chạy thử nhanh với tập mẫu thu nhỏ phục vụ kiểm thử.

    Trả về:
        Đối tượng CalibrationPipelineResult chứa toàn bộ chỉ số thực tế.
    """
    set_seed(seed)

    effective_checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    real_checkpoint = (
        effective_checkpoint.is_dir()
        and (effective_checkpoint / "config.json").is_file()
        and any((effective_checkpoint / name).is_file()
                for name in ("pytorch_model.bin", "model.safetensors"))
        and not checkpoint_is_smoke(effective_checkpoint)
    )
    if not smoke_mode and not real_checkpoint:
        raise RuntimeError(
            "BLOCKED: hiệu chuẩn thật yêu cầu classifier checkpoint PhoBERT không-smoke. "
            "Chạy scripts/run_real_models.sh sau khi có trọng số."
        )
    checkpoint_hash = checkpoint_sha256(effective_checkpoint) if real_checkpoint else ""
    smoke_artifact = smoke_mode or not real_checkpoint

    # 1. Kiểm thực phòng vệ Honesty Rule #3
    verify_not_test_dataset(val_csv_path)

    # 2. Nạp dữ liệu
    val_samples = load_dataset_samples(val_csv_path)
    train_samples = load_dataset_samples(train_csv_path) if Path(train_csv_path).exists() else None

    if smoke_mode:
        # Đảm bảo tập smoke 30 mẫu luôn bao gồm cả mẫu in-scope lẫn OUT_OF_SCOPE (nếu có)
        in_rows = [s for s in val_samples if s["label"] != OUT_OF_SCOPE_INTENT]
        oos_rows = [s for s in val_samples if s["label"] == OUT_OF_SCOPE_INTENT]
        picked_oos = oos_rows[: min(5, len(oos_rows))]
        val_samples = in_rows[: 30 - len(picked_oos)] + picked_oos

    val_total = len(val_samples)
    val_oos_count = sum(1 for s in val_samples if s["label"] == OUT_OF_SCOPE_INTENT)
    val_in_scope_count = val_total - val_oos_count

    # 3. Thu nhận logits trên tập validation
    logits_mat, class_labels, y_true, _y_pred_raw = get_validation_logits_and_predictions(
        val_samples=val_samples,
        train_samples=train_samples,
        checkpoint_dir=effective_checkpoint,
    )

    lbl2id = {lbl: i for i, lbl in enumerate(class_labels)}

    # Lọc các mẫu in-scope để tối ưu hóa nhiệt độ T
    in_scope_indices: list[int] = []
    in_scope_labels: list[int] = []

    for i, s in enumerate(val_samples):
        lbl = s["label"]
        if lbl != OUT_OF_SCOPE_INTENT and lbl in lbl2id:
            in_scope_indices.append(i)
            in_scope_labels.append(lbl2id[lbl])

    # 4. Tối ưu hóa nhiệt độ T* (Temperature Scaling - Guo et al., 2017)
    if in_scope_indices:
        in_scope_logits = logits_mat[in_scope_indices]
        in_scope_y = np.asarray(in_scope_labels, dtype=np.int64)

        temp_res = optimize_temperature(
            logits=in_scope_logits,
            labels=in_scope_y,
            bounds=DEFAULT_TEMPERATURE_BOUNDS,
            num_bins=DEFAULT_ECE_BINS,
        )
    else:
        temp_res = TemperatureOptimizationResult(
            optimal_temperature=DEFAULT_TEMPERATURE,
            nll_before=0.0,
            nll_after=0.0,
            ece_before=0.0,
            ece_after=0.0,
            converged=True,
        )

    # 5. Áp dụng nhiệt độ T* lên toàn bộ logits tập Validation
    calibrated_probs = apply_temperature_softmax(
        logits=logits_mat,
        temperature=temp_res.optimal_temperature,
    )
    calibrated_confs = np.max(calibrated_probs, axis=-1)
    calibrated_pred_idx = np.argmax(calibrated_probs, axis=-1)
    y_pred_calibrated = [class_labels[idx] for idx in calibrated_pred_idx]

    # 6. Hiệu chuẩn ngưỡng tau* và tính toán đường cong Risk-Coverage
    # Đánh giá Selective Classification trên các mẫu in-scope
    if in_scope_indices:
        in_confs = calibrated_confs[in_scope_indices]
        in_trues = [y_true[i] for i in in_scope_indices]
        in_preds = [y_pred_calibrated[i] for i in in_scope_indices]

        tau_res = calibrate_tau_selective_accuracy(
            confidences=in_confs,
            y_true=in_trues,
            y_pred=in_preds,
            target_accuracy=target_accuracy,
            num_steps=DEFAULT_CALIBRATION_TAU_STEPS,
        )
        curve_res = compute_risk_coverage_curve(
            confidences=in_confs,
            y_true=in_trues,
            y_pred=in_preds,
            num_steps=DEFAULT_CALIBRATION_TAU_STEPS,
        )
    else:
        tau_res = calibrate_tau_selective_accuracy(
            confidences=calibrated_confs,
            y_true=y_true,
            y_pred=y_pred_calibrated,
            target_accuracy=target_accuracy,
            num_steps=DEFAULT_CALIBRATION_TAU_STEPS,
        )
        curve_res = compute_risk_coverage_curve(
            confidences=calibrated_confs,
            y_true=y_true,
            y_pred=y_pred_calibrated,
            num_steps=DEFAULT_CALIBRATION_TAU_STEPS,
        )

    # Lưu biểu đồ đường cong Risk-Coverage
    saved_figure_path = plot_risk_coverage_curve(
        curve_result=curve_res,
        calibrated_tau=tau_res.optimal_tau,
        target_accuracy=target_accuracy,
        output_path=output_figure_path,
    )

    # 7. Xử lý các trường hợp Fallback và hiệu chuẩn ngưỡng theta*
    # Mẫu fallback là mẫu có độ tin cậy phân loại dưới ngưỡng tau*
    fallback_retriever = Retriever()

    fallback_scores: list[float] = []
    fallback_preds: list[str] = []
    fallback_trues: list[str] = []

    # Thu thập điểm số và nhãn fallback cho toàn bộ mẫu phục vụ đo lường tỷ lệ loại bỏ
    all_fallback_scores: list[float] = []
    all_fallback_preds: list[str] = []

    for i, s in enumerate(val_samples):
        text = s["text"]
        true_lbl = s["label"]
        conf = float(calibrated_confs[i])
        pred_lbl = y_pred_calibrated[i]

        # Chỉ cần truy hồi nếu mẫu có khả năng rơi vào fallback hoặc là mẫu OOS
        if (
            conf < tau_res.optimal_tau
            or pred_lbl == OUT_OF_SCOPE_INTENT
            or true_lbl == OUT_OF_SCOPE_INTENT
        ):
            fb_res = fallback_retriever.retrieve(text)
            top_score = fb_res.top_candidate.score if fb_res.top_candidate else 0.0
            top_intent = fb_res.top_candidate.intent if fb_res.top_candidate else "UNKNOWN"

            fallback_scores.append(top_score)
            fallback_preds.append(top_intent)
            fallback_trues.append(true_lbl)
            all_fallback_scores.append(top_score)
            all_fallback_preds.append(top_intent)
        else:
            all_fallback_scores.append(1.0)
            all_fallback_preds.append(pred_lbl)

    theta_res = calibrate_theta_fallback(
        fallback_scores=fallback_scores,
        fallback_preds=fallback_preds,
        y_true=fallback_trues,
        num_steps=DEFAULT_CALIBRATION_THETA_STEPS,
    )

    # 8. Tính toán tỷ lệ loại bỏ yêu cầu ngoài phạm vi và tỷ lệ từ chối sai
    rejection_res = compute_rejection_rates(
        clf_confidences=calibrated_confs.tolist(),
        tau=tau_res.optimal_tau,
        fallback_scores=all_fallback_scores,
        theta=theta_res.optimal_theta,
        y_true=y_true,
        clf_preds=y_pred_calibrated,
        fallback_preds=all_fallback_preds,
    )

    # 9. Xuất tham số hiệu chuẩn ra JSON
    now_str = datetime.datetime.now(datetime.UTC).isoformat()
    pipeline_result = CalibrationPipelineResult(
        temperature_result=temp_res,
        tau_result=tau_res,
        curve_result=curve_res,
        theta_result=theta_res,
        rejection_result=rejection_res,
        thresholds_path=Path(output_thresholds_path),
        figure_path=Path(saved_figure_path),
        val_samples_total=val_total,
        val_samples_in_scope=val_in_scope_count,
        val_samples_out_of_scope=val_oos_count,
        timestamp=now_str,
    )

    exported_threshold_path = export_calibrated_parameters(
        tau=tau_res.optimal_tau,
        theta=theta_res.optimal_theta,
        temperature=temp_res.optimal_temperature,
        metadata={
            **pipeline_result.to_dict(),
            "classifier_checkpoint_sha256": checkpoint_hash,
            "preprocessing_fingerprint": preprocess_fingerprint(),
            "smoke_mode": smoke_artifact,
        },
        output_path=output_thresholds_path,
        smoke_output_path=output_smoke_path,
    )
    pipeline_result.thresholds_path = exported_threshold_path

    return pipeline_result


# =====================================================================
# 10. Giao diện Dòng lệnh (CLI)
# =====================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Xây dựng bộ phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Hiệu chuẩn ngưỡng tin cậy tau, theta và co giãn nhiệt độ T trên tập Validation."
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=str(VAL_CSV_PATH),
        help=f"Đường dẫn tập CSV kiểm thực (mặc định: {VAL_CSV_PATH}).",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(TRAIN_CSV_PATH),
        help=f"Đường dẫn tập CSV huấn luyện (mặc định: {TRAIN_CSV_PATH}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Đường dẫn thư mục checkpoint PhoBERT (nếu có).",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=DEFAULT_TARGET_SELECTIVE_ACCURACY,
        help=f"Độ chính xác chọn lọc mục tiêu (mặc định: {DEFAULT_TARGET_SELECTIVE_ACCURACY}).",
    )
    parser.add_argument(
        "--output-thresholds",
        type=str,
        default=str(DEFAULT_THRESHOLDS_PATH),
        help=f"Đường dẫn lưu kết quả JSON hiệu chuẩn (mặc định: {DEFAULT_THRESHOLDS_PATH}).",
    )
    parser.add_argument(
        "--output-smoke-thresholds",
        type=str,
        default=str(SMOKE_THRESHOLDS_PATH),
        help="Đường dẫn lưu artifact hiệu chuẩn smoke thay vì thresholds production.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default=str(DEFAULT_RISK_COVERAGE_PATH),
        help=f"Đường dẫn lưu hình ảnh đường cong Risk-Coverage (mặc định: {DEFAULT_RISK_COVERAGE_PATH}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Hạt giống ngẫu nhiên tất định (mặc định: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help="Chế độ chạy thử nhanh với số lượng mẫu thu nhỏ.",
    )
    return parser


def main() -> None:
    """Hàm điểm nhập chính cho giao diện dòng lệnh của module train/calibrate_tau.py."""
    parser = build_arg_parser()
    args = parser.parse_args()

    print("=" * 75)
    print("VIETNAMESE LATEX KNOWLEDGE ASSISTANT - THRESHOLD CALIBRATION & TEMPERATURE SCALING")
    print("Tuân thủ kiến trúc và Honesty Rule #3 (AGENTS.md)")
    print("=" * 75)
    print(f"  Tập dữ liệu kiểm thực : {args.val_path}")
    print(f"  Độ chính xác mục tiêu  : {args.target_accuracy * 100:.1f}%")
    print(f"  Tệp xuất tham số       : {args.output_thresholds}")
    print(f"  Tệp xuất biểu đồ       : {args.output_figure}")
    print("=" * 75)

    try:
        res = run_calibration_pipeline(
            val_csv_path=args.val_path,
            train_csv_path=args.train_path,
            checkpoint_dir=args.checkpoint_dir,
            target_accuracy=args.target_accuracy,
            output_thresholds_path=args.output_thresholds,
            output_figure_path=args.output_figure,
            seed=args.seed,
            smoke_mode=args.smoke_mode,
            output_smoke_path=args.output_smoke_thresholds,
        )

        print("\n[KẾT QUẢ HIỆU CHUẨN THÀNH CÔNG]")
        print("  1. Co giãn nhiệt độ (Guo et al., 2017):")
        print(f"     - Nhiệt độ tối ưu T*    : {res.temperature_result.optimal_temperature:.4f}")
        print(f"     - NLL trước / sau       : {res.temperature_result.nll_before:.4f} -> {res.temperature_result.nll_after:.4f}")
        print(f"     - ECE trước / sau       : {res.temperature_result.ece_before * 100:.2f}% -> {res.temperature_result.ece_after * 100:.2f}%")
        print("  2. Hiệu chuẩn ngưỡng tin cậy tau* (Selective Classification):")
        print(f"     - Ngưỡng tối ưu tau*    : {res.tau_result.optimal_tau:.4f}")
        print(f"     - Selective Accuracy    : {res.tau_result.selective_accuracy * 100:.2f}% (Mục tiêu >= {args.target_accuracy * 100:.1f}%)")
        print(f"     - Coverage (Độ bao phủ) : {res.tau_result.coverage * 100:.2f}%")
        print(f"     - Selective Risk        : {res.tau_result.risk * 100:.2f}%")
        print(f"     - AUC-RC                : {res.curve_result.auc_rc:.4f}")
        print("  3. Hiệu chuẩn ngưỡng truy hồi fallback theta*:")
        print(f"     - Ngưỡng tối ưu theta*  : {res.theta_result.optimal_theta:.4f}")
        print(f"     - Fallback Accuracy     : {res.theta_result.fallback_accuracy * 100:.2f}%")
        print(f"     - Fallback Coverage     : {res.theta_result.fallback_coverage * 100:.2f}%")
        print("  4. Tỷ lệ loại bỏ yêu cầu:")
        print(f"     - Classifier OOS RR     : {res.rejection_result.classifier_oos_rejection_rate * 100:.2f}%")
        print(f"     - Classifier FRR        : {res.rejection_result.classifier_false_rejection_rate * 100:.2f}%")
        print(f"     - System OOS RR         : {res.rejection_result.system_oos_rejection_rate * 100:.2f}%")
        print(f"     - System FRR            : {res.rejection_result.system_false_rejection_rate * 100:.2f}%")
        print("  5. Tệp xuất:")
        print(f"     - Tham số JSON          : {res.thresholds_path}")
        print(f"     - Biểu đồ PNG           : {res.figure_path}")
        print("=" * 75)
        sys.exit(0)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, FileNotFoundError) as e:
        print(f"\n[LỖI THỰC THI HIỆU CHUẨN]: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
