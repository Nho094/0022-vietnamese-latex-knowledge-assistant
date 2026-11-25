"""
Bộ kiểm thử đơn vị tự động cho module hiệu chuẩn ngưỡng và co giãn nhiệt độ (train/calibrate_tau.py).

Tệp tin: tests/test_calibrate_tau.py
Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Identifiers bằng tiếng Anh; chú thích docstrings/comments bằng tiếng Việt giải thích lý do (WHY).
- Kiểm thực nghiêm ngặt Honesty Rule #3 (nghiêm cấm can thiệp hoặc tinh chỉnh trên test set).
- Đối chuẩn tính tay (hand-computed ground truth) cho các công thức:
  + Negative Log-Likelihood (NLL) và Softmax co giãn nhiệt độ (Guo et al., 2017).
  + Tỷ lệ bao phủ Coverage, Rủi ro chọn lọc Selective Risk và Độ chính xác chọn lọc (Geifman & El-Yaniv, 2017).
  + Hiệu chuẩn ngưỡng tau* tối ưu hóa độ bao phủ dưới ràng buộc rủi ro.
  + Hiệu chuẩn ngưỡng theta* trên các trường hợp Fallback (cân bằng In-Scope Precision và OOS Rejection Rate).
  + Đo lường tỷ lệ loại bỏ yêu cầu ngoài phạm vi (OOS Rejection Rate) và từ chối sai (False Rejection Rate).
  + Sinh biểu đồ đường cong Risk-Coverage ra report/figures/ và xuất models/thresholds.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.config import (
    DEFAULT_CALIBRATION_TAU_STEPS,
    DEFAULT_CALIBRATION_THETA_STEPS,
    DEFAULT_TARGET_SELECTIVE_ACCURACY,
    DEFAULT_TEMPERATURE_BOUNDS,
    OUT_OF_SCOPE_INTENT,
    TEST_CSV_PATH,
    THETA_RETRIEVAL_THRESHOLD,
    VAL_CSV_PATH,
    load_calibrated_thresholds,
)
from train.calibrate_tau import (
    RejectionRateResult,
    RiskCoverageCurveResult,
    TauCalibrationResult,
    TemperatureOptimizationResult,
    ThetaCalibrationResult,
    apply_temperature_softmax,
    calibrate_tau_selective_accuracy,
    calibrate_theta_fallback,
    compute_nll,
    compute_rejection_rates,
    compute_risk_coverage_curve,
    export_calibrated_parameters,
    optimize_temperature,
    plot_risk_coverage_curve,
    run_calibration_pipeline,
    verify_not_test_dataset,
)

# =====================================================================
# 1. Kiểm thực Honesty Rule #3 (Cấm dùng tập kiểm thử test.csv)
# =====================================================================


def test_verify_not_test_dataset_blocks_test_path() -> None:
    """Kiểm tra chặn truy cập trực tiếp bằng đường dẫn test.csv."""
    with pytest.raises(ValueError, match="Vi phạm Honesty Rule #3"):
        verify_not_test_dataset(TEST_CSV_PATH)

    with pytest.raises(ValueError, match="Vi phạm Honesty Rule #3"):
        verify_not_test_dataset("data/test.csv")


def test_verify_not_test_dataset_blocks_test_checksum(tmp_path: Path) -> None:
    """Kiểm tra chặn tệp có tên khác nhưng mang mã băm nội dung của test.csv."""
    fake_csv = tmp_path / "sneaky_dataset.csv"
    if TEST_CSV_PATH.exists():
        fake_csv.write_bytes(TEST_CSV_PATH.read_bytes())
        with pytest.raises(ValueError, match="Vi phạm Honesty Rule #3"):
            verify_not_test_dataset(fake_csv)


def test_verify_not_test_dataset_accepts_val_dataset() -> None:
    """Kiểm tra tập validation val.csv được chấp nhận hợp lệ."""
    if VAL_CSV_PATH.exists():
        # Không được ném ngoại lệ
        verify_not_test_dataset(VAL_CSV_PATH)


# =====================================================================
# 2. Co giãn nhiệt độ và NLL (Guo et al., ICML 2017)
# =====================================================================


def test_compute_nll_invalid_temperature() -> None:
    """Tham số nhiệt độ T <= 0 phải ném ngoại lệ ValueError."""
    logits = np.array([[2.0, 1.0]], dtype=np.float64)
    labels = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match="Tham số nhiệt độ T phải > 0.0"):
        compute_nll(logits, labels, temperature=0.0)

    with pytest.raises(ValueError, match="Tham số nhiệt độ T phải > 0.0"):
        compute_nll(logits, labels, temperature=-1.0)


def test_compute_nll_hand_computed_ground_truth() -> None:
    """
    Đối chuẩn giá trị NLL tính tay:
    Logits: z = [[2.0, 0.0]] -> softmax p_0 = exp(2)/(exp(2)+exp(0)) = 7.389056 / 8.389056 = 0.880797
    label = 0 -> NLL = -log(0.880797) = 0.126928.
    Với T = 2.0: z' = [[1.0, 0.0]] -> p_0 = exp(1)/(exp(1)+1) = 2.71828 / 3.71828 = 0.731058
    label = 0 -> NLL = -log(0.731058) = 0.313261.
    """
    logits = np.array([[2.0, 0.0]], dtype=np.float64)
    labels = np.array([0], dtype=np.int64)

    nll_t1 = compute_nll(logits, labels, temperature=1.0)
    expected_t1 = -np.log(np.exp(2.0) / (np.exp(2.0) + np.exp(0.0)))
    assert pytest.approx(nll_t1, rel=1e-5) == expected_t1

    nll_t2 = compute_nll(logits, labels, temperature=2.0)
    expected_t2 = -np.log(np.exp(1.0) / (np.exp(1.0) + np.exp(0.0)))
    assert pytest.approx(nll_t2, rel=1e-5) == expected_t2


def test_apply_temperature_softmax_properties() -> None:
    """Kiểm tra các tính chất toán học của phân phối xác suất softmax sau co giãn nhiệt độ."""
    logits = np.array(
        [
            [1.5, 2.5, -0.5],
            [10.0, 2.0, 1.0],
            [-5.0, -2.0, -1.0],
        ],
        dtype=np.float64,
    )

    probs = apply_temperature_softmax(logits, temperature=1.5)

    # 1. Tổng mỗi hàng phải bằng 1.0
    row_sums = np.sum(probs, axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(3), rtol=1e-5)

    # 2. Toàn bộ xác suất nằm trong [0.0, 1.0]
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    # 3. Bảo toàn thứ hạng (Rank-preserving): argmax không đổi theo nhiệt độ
    original_argmax = np.argmax(logits, axis=-1)
    scaled_argmax = np.argmax(probs, axis=-1)
    np.testing.assert_array_equal(original_argmax, scaled_argmax)


def test_optimize_temperature_minimizes_nll() -> None:
    """
    Kiểm tra thuật toán tối ưu hóa nhiệt độ T* tìm được nhiệt độ cực tiểu hóa NLL.
    Mô phỏng mô hình nơ-ron bị overconfident trên một số mẫu sai lệch.
    """
    logits = np.array(
        [
            [8.0, 1.0],  # Đúng (rất tự tin)
            [7.0, 2.0],  # Đúng
            [1.0, 9.0],  # Đúng
            [6.0, 5.0],  # Sai: mô hình dự đoán 0 nhưng nhãn thực tế là 1
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    res: TemperatureOptimizationResult = optimize_temperature(
        logits=logits,
        labels=labels,
        bounds=DEFAULT_TEMPERATURE_BOUNDS,
    )

    assert res.converged is True
    assert res.optimal_temperature > 0.0
    # NLL sau hiệu chuẩn bắt buộc phải nhỏ hơn hoặc bằng NLL ban đầu
    assert res.nll_after <= res.nll_before + 1e-6
    # ECE được tính toán đầy đủ
    assert 0.0 <= res.ece_before <= 1.0
    assert 0.0 <= res.ece_after <= 1.0


# =====================================================================
# 3. Phân loại có chọn lọc và Đường cong Risk-Coverage (Geifman & El-Yaniv, 2017)
# =====================================================================


def test_compute_risk_coverage_curve_monotonicity() -> None:
    """
    Độ bao phủ Coverage phải là hàm đơn điệu giảm (không tăng) theo ngưỡng tin cậy tau:
    tau_1 < tau_2 -> Coverage(tau_1) >= Coverage(tau_2).
    """
    confs = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25], dtype=np.float64)
    y_true = ["A", "B", "C", "D", "A", "B", "C", "D"]
    y_pred = ["A", "B", "C", "D", "X", "Y", "C", "Z"]

    curve_res: RiskCoverageCurveResult = compute_risk_coverage_curve(
        confidences=confs,
        y_true=y_true,
        y_pred=y_pred,
        num_steps=DEFAULT_CALIBRATION_TAU_STEPS,
    )

    assert len(curve_res.taus) == DEFAULT_CALIBRATION_TAU_STEPS
    assert len(curve_res.coverages) == DEFAULT_CALIBRATION_TAU_STEPS
    assert len(curve_res.risks) == DEFAULT_CALIBRATION_TAU_STEPS

    # Kiểm tra tính đơn điệu không tăng của Coverage
    for i in range(len(curve_res.coverages) - 1):
        assert curve_res.coverages[i] >= curve_res.coverages[i + 1] - 1e-9

    # Tại tau = 0.0, Coverage = 1.0 (toàn bộ mẫu được chấp nhận)
    assert pytest.approx(curve_res.coverages[0], abs=1e-5) == 1.0
    # AUC-RC phải là số thực không âm
    assert curve_res.auc_rc >= 0.0


def test_calibrate_tau_selective_accuracy_reaches_target() -> None:
    """
    Hiệu chuẩn tau* với dữ liệu tổng hợp:
    4 mẫu tự tin cao (>= 0.80) đều đúng -> tại tau = 0.80, selective accuracy = 100% >= 95%.
    4 mẫu tự tin thấp (< 0.80) có lỗi -> khi giảm tau thì accuracy giảm.
    Thuật toán phải chọn tau* nhỏ nhất thỏa mãn accuracy >= target_accuracy để cực đại hóa Coverage.
    """
    confs = np.array([0.98, 0.92, 0.88, 0.82, 0.70, 0.60, 0.50, 0.40], dtype=np.float64)
    y_true = ["INT_1", "INT_2", "INT_3", "INT_4", "INT_5", "INT_6", "INT_7", "INT_8"]
    y_pred = ["INT_1", "INT_2", "INT_3", "INT_4", "WRONG", "WRONG", "WRONG", "WRONG"]

    tau_res: TauCalibrationResult = calibrate_tau_selective_accuracy(
        confidences=confs,
        y_true=y_true,
        y_pred=y_pred,
        target_accuracy=DEFAULT_TARGET_SELECTIVE_ACCURACY,
        num_steps=101,
    )

    assert tau_res.target_met is True
    assert tau_res.selective_accuracy >= DEFAULT_TARGET_SELECTIVE_ACCURACY
    assert tau_res.risk == pytest.approx(1.0 - tau_res.selective_accuracy, abs=1e-5)
    # Ngưỡng tau* phải nằm trong khoảng chấp nhận được 4 mẫu đúng hàng đầu
    assert 0.70 < tau_res.optimal_tau <= 0.82
    assert tau_res.coverage == 0.5  # 4/8 mẫu
    assert tau_res.accepted_samples == 4


def test_calibrate_tau_fallback_when_target_unreachable() -> None:
    """Khi dữ liệu không thể đạt 95% accuracy ở bất kỳ ngưỡng nào, thuật toán tìm giải pháp cân bằng tối ưu."""
    confs = np.array([0.90, 0.80, 0.70, 0.60], dtype=np.float64)
    y_true = ["A", "B", "C", "D"]
    y_pred = ["WRONG", "WRONG", "WRONG", "WRONG"]  # 0% accuracy

    tau_res: TauCalibrationResult = calibrate_tau_selective_accuracy(
        confidences=confs,
        y_true=y_true,
        y_pred=y_pred,
        target_accuracy=0.95,
    )

    assert tau_res.target_met is False
    assert 0.0 <= tau_res.optimal_tau <= 1.0


# =====================================================================
# 4. Hiệu chuẩn ngưỡng truy hồi fallback theta
# =====================================================================


def test_calibrate_theta_fallback_balanced_score() -> None:
    """
    Kiểm tra hiệu chuẩn ngưỡng theta* trên tập fallback:
    - 2 mẫu in-scope có score cao (0.85, 0.75) và dự đoán đúng.
    - 2 mẫu out-of-scope có score thấp (0.25, 0.15).
    Ngưỡng theta* tối ưu phải nằm giữa 0.25 và 0.75 để vừa giữ in-scope vừa loại bỏ OOS.
    """
    fallback_scores = [0.85, 0.75, 0.25, 0.15]
    fallback_preds = ["LATEX_MATH", "LATEX_DOC", "LATEX_RANDOM", "LATEX_RANDOM"]
    fallback_trues = ["LATEX_MATH", "LATEX_DOC", OUT_OF_SCOPE_INTENT, OUT_OF_SCOPE_INTENT]

    theta_res: ThetaCalibrationResult = calibrate_theta_fallback(
        fallback_scores=fallback_scores,
        fallback_preds=fallback_preds,
        y_true=fallback_trues,
        num_steps=DEFAULT_CALIBRATION_THETA_STEPS,
    )

    assert 0.25 < theta_res.optimal_theta <= 0.75
    assert theta_res.fallback_accuracy == 1.0
    assert theta_res.oos_rejection_rate == 1.0
    assert theta_res.total_fallback_samples == 4


def test_calibrate_theta_fallback_empty_data() -> None:
    """Kiểm tra xử lý an toàn khi không có mẫu fallback nào."""
    theta_res: ThetaCalibrationResult = calibrate_theta_fallback(
        fallback_scores=[],
        fallback_preds=[],
        y_true=[],
    )
    assert theta_res.optimal_theta == THETA_RETRIEVAL_THRESHOLD
    assert theta_res.total_fallback_samples == 0


# =====================================================================
# 5. Đo lường tỷ lệ loại bỏ ngoài phạm vi và từ chối sai
# =====================================================================


def test_compute_rejection_rates_hand_computed() -> None:
    """
    Đối chuẩn tính tay tỷ lệ loại bỏ OOS và từ chối sai:
    Tập gồm 4 mẫu:
    - 2 mẫu In-Scope:
      + Mẫu 0: conf=0.9 (>= 0.8 tau), score=0.0 -> Được chấp nhận ở classifier
      + Mẫu 1: conf=0.5 (< 0.8), score=0.4 (>= 0.3 theta) -> Chấp nhận ở fallback
    - 2 mẫu Out-of-Scope:
      + Mẫu 2: conf=0.2 (< 0.8), score=0.1 (< 0.3) -> Bị loại bỏ hoàn toàn
      + Mẫu 3: conf=0.3 (< 0.8), score=0.2 (< 0.3) -> Bị loại bỏ hoàn toàn
    """
    clf_confs = [0.9, 0.5, 0.2, 0.3]
    fallback_scores = [0.0, 0.4, 0.1, 0.2]
    y_true = ["IN_1", "IN_2", OUT_OF_SCOPE_INTENT, OUT_OF_SCOPE_INTENT]

    rej_res: RejectionRateResult = compute_rejection_rates(
        clf_confidences=clf_confs,
        tau=0.8,
        fallback_scores=fallback_scores,
        theta=0.3,
        y_true=y_true,
    )

    # 1. Classifier OOS Rejection Rate: 2/2 mẫu OOS có conf < 0.8 -> 100%
    assert rej_res.classifier_oos_rejection_rate == 1.0

    # 2. Classifier False Rejection Rate: 1/2 mẫu In-Scope có conf < 0.8 -> 50%
    assert rej_res.classifier_false_rejection_rate == 0.5

    # 3. System OOS Rejection Rate: 2/2 mẫu OOS có conf < 0.8 VÀ score < 0.3 -> 100%
    assert rej_res.system_oos_rejection_rate == 1.0

    # 4. System False Rejection Rate: 0/2 mẫu In-Scope bị từ chối cả 2 tầng -> 0%
    assert rej_res.system_false_rejection_rate == 0.0

    assert rej_res.num_oos_samples == 2
    assert rej_res.num_in_scope_samples == 2


# =====================================================================
# 6. Biểu đồ Risk-Coverage và Xuất JSON tham số
# =====================================================================


def test_plot_risk_coverage_curve(tmp_path: Path) -> None:
    """Kiểm tra chức năng vẽ và lưu tệp hình ảnh PNG của đường cong Risk-Coverage."""
    taus = [0.0, 0.5, 1.0]
    coverages = [1.0, 0.6, 0.0]
    risks = [0.2, 0.05, 0.0]
    accuracies = [0.8, 0.95, 1.0]
    accepted = [10, 6, 0]

    curve_res = RiskCoverageCurveResult(
        taus=taus,
        coverages=coverages,
        risks=risks,
        accuracies=accuracies,
        accepted_counts=accepted,
        auc_rc=0.08,
    )

    out_fig = tmp_path / "figures" / "test_risk_coverage.png"
    saved_path = plot_risk_coverage_curve(
        curve_result=curve_res,
        calibrated_tau=0.5,
        target_accuracy=0.95,
        output_path=out_fig,
    )

    assert saved_path.exists()
    assert saved_path.is_file()
    assert saved_path.stat().st_size > 1000  # PNG hợp lệ


def test_export_calibrated_parameters(tmp_path: Path) -> None:
    """
    Kiểm tra xuất thresholds.json và tính tương thích với hàm
    `core.config.load_calibrated_thresholds`.
    """
    out_json = tmp_path / "models" / "thresholds.json"
    metadata = {
        "val_selective_accuracy": 0.962,
        "coverage": 0.85,
        "ece_before": 0.08,
        "ece_after": 0.02,
    }

    saved_path = export_calibrated_parameters(
        tau=0.825,
        theta=0.35,
        temperature=1.25,
        metadata=metadata,
        output_path=out_json,
    )

    assert saved_path.exists()
    with open(saved_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["tau"] == 0.825
    assert data["theta"] == 0.35
    assert data["temperature"] == 1.25
    assert data["val_selective_accuracy"] == 0.962

    # Kiểm tra nạp lại qua hàm chuẩn của core/config.py
    loaded = load_calibrated_thresholds(saved_path)
    assert loaded["tau"] == 0.825
    assert loaded["theta"] == 0.35
    assert loaded["temperature"] == 1.25


def test_smoke_threshold_export_is_redirected(tmp_path: Path) -> None:
    official = tmp_path / "models" / "thresholds.json"
    smoke = tmp_path / "models" / "smoke" / "thresholds_smoke.json"
    saved = export_calibrated_parameters(
        tau=0.7, theta=0.3, temperature=1.0,
        metadata={"smoke_mode": True}, output_path=official, smoke_output_path=smoke,
    )
    assert saved == smoke
    assert smoke.is_file()
    assert not official.exists()


def test_zero_tau_threshold_export_is_redirected(tmp_path: Path) -> None:
    official = tmp_path / "models" / "thresholds.json"
    smoke = tmp_path / "models" / "smoke" / "thresholds_smoke.json"
    saved = export_calibrated_parameters(
        tau=0.0, theta=0.3, temperature=1.0,
        metadata={"smoke_mode": False}, output_path=official, smoke_output_path=smoke,
    )
    assert saved == smoke
    assert not official.exists()
    assert json.loads(smoke.read_text(encoding="utf-8"))["smoke_mode"] is True


# =====================================================================
# 7. Kiểm thử tích hợp toàn bộ đường ống (Integration Smoke Test)
# =====================================================================


def test_calibration_pipeline_smoke_run(tmp_path: Path) -> None:
    """
    Kiểm thử tích hợp đường ống hiệu chuẩn run_calibration_pipeline
    ở chế độ smoke_mode trên tập validation thực tế.
    """
    out_thresh = tmp_path / "thresholds.json"
    out_fig = tmp_path / "risk_coverage.png"

    res = run_calibration_pipeline(
        val_csv_path=VAL_CSV_PATH,
        output_thresholds_path=out_thresh,
        output_figure_path=out_fig,
        smoke_mode=True,
        output_smoke_path=out_thresh,
    )

    assert res.val_samples_total == 30
    assert 0.0 <= res.tau_result.optimal_tau <= 1.0
    assert 0.0 < res.theta_result.optimal_theta <= 1.0
    assert res.temperature_result.optimal_temperature > 0.0
    assert out_thresh.exists()
    assert json.loads(out_thresh.read_text(encoding="utf-8"))["smoke_mode"] is True
    assert out_fig.exists()
