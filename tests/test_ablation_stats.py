"""
Bộ kiểm thử đơn vị cho module kiểm thực tập kiểm thử và phân tích thống kê ablation (tests/test_ablation_stats.py).

Tuân thủ kiến trúc và quy chuẩn AGENTS.md:
- Python 3.11, type hints, ruff-clean, pytest.
- Identifiers in English; docstrings và comments giải thích lý do (WHY) bằng tiếng Việt.
- Kiểm thử toàn diện các yêu cầu:
  1. Kiểm thực mã băm SHA-256 của data/test.csv đối chiếu với PROGRESS.md trước khi thực thi.
  2. Phát hiện và ngăn chặn mọi sự sai lệch, sửa đổi đối với tập kiểm thử (Honesty Rule #3).
  3. Ước lượng khoảng tin cậy 95% Bootstrap phi tham số (B=1000) cho Accuracy và Macro-F1.
  4. Kiểm định McNemar so sánh các mô hình với hiệu chỉnh Yates và trường hợp biên b + c = 0.
  5. Tổng hợp đa hạt giống (Multi-Seed Aggregation): Mean, Bessel Sample Std (ddof=1), SEM.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Final

import pytest

from core.config import (
    DEFAULT_INTENTS_PATH,
    DEFAULT_SEED,
    PROGRESS_PATH,
    TEST_CSV_PATH,
    TEST_CSV_SHA256,
)
from train.ablation_stats import (
    BootstrapCIResult,
    McNemarComparisonResult,
    MultiSeedMetricSummary,
    MultiSeedRunAggregation,
    TestSetChecksumMismatchError,
    aggregate_multi_seed_results,
    aggregate_seed_values,
    compare_models_mcnemar,
    compute_ablation_bootstrap_cis,
    compute_accuracy_bootstrap_ci,
    compute_file_sha256,
    compute_macro_f1_bootstrap_ci,
    ensure_test_set_integrity,
    extract_test_sha256_from_progress,
    main,
    run_test_set_verification_cli,
    verify_test_set_checksum,
)

TOLERANCE: Final[float] = 1e-6


# =====================================================================
# 1. Kiểm thử Tính toán Mã băm SHA-256 (File SHA-256 Computation)
# =====================================================================


def test_compute_file_sha256_deterministic(tmp_path: Path) -> None:
    """
    Kiểm tra hàm compute_file_sha256 tính toán mã băm SHA-256 chính xác và tất định.
    """
    test_content = b"vietnamese latex knowledge assistant - test set verification\n"
    expected_hash = hashlib.sha256(test_content).hexdigest()

    file_path = tmp_path / "sample.csv"
    file_path.write_bytes(test_content)

    actual_hash = compute_file_sha256(file_path)
    assert actual_hash == expected_hash
    assert len(actual_hash) == 64


def test_compute_file_sha256_file_not_found() -> None:
    """Kiểm tra ngoại lệ FileNotFoundError khi tệp không tồn tại."""
    with pytest.raises(FileNotFoundError, match="Không tìm thấy tệp tin"):
        compute_file_sha256(Path("duong_dan_khong_ton_tai_12345.csv"))


# =====================================================================
# 2. Kiểm thử Trích xuất Băm từ PROGRESS.md và Xác thực Toàn vẹn
# =====================================================================


def test_extract_test_sha256_from_progress_real_file() -> None:
    """
    Kiểm tra trích xuất mã băm SHA-256 từ tệp PROGRESS.md thực tế của dự án.
    Mã băm trích xuất phải khớp với TEST_CSV_SHA256 trong core/config.py.
    """
    assert PROGRESS_PATH.is_file(), f"PROGRESS.md không tồn tại tại: {PROGRESS_PATH}"
    extracted_hash = extract_test_sha256_from_progress(PROGRESS_PATH)
    assert extracted_hash == TEST_CSV_SHA256
    assert len(extracted_hash) == 64


def test_extract_test_sha256_from_progress_custom_valid(tmp_path: Path) -> None:
    """Kiểm tra trích xuất mã băm từ tệp markdown mẫu hợp lệ."""
    fake_hash = "a" * 64
    progress_file = tmp_path / "PROGRESS.md"
    progress_file.write_text(f"# Log\n| KA-05 | Pipeline | DONE | ... | sha256 test.csv: {fake_hash} |\n")

    extracted = extract_test_sha256_from_progress(progress_file)
    assert extracted == fake_hash


def test_extract_test_sha256_from_progress_missing_pattern(tmp_path: Path) -> None:
    """Kiểm tra ngoại lệ ValueError khi tệp không chứa mẫu băm sha256 test.csv."""
    bad_file = tmp_path / "PROGRESS.md"
    bad_file.write_text("# Progress Tracking\nChưa có thông tin băm kiểm thử.\n")

    with pytest.raises(ValueError, match="Không tìm thấy mẫu mã băm"):
        extract_test_sha256_from_progress(bad_file)


def test_verify_test_set_checksum_on_real_dataset() -> None:
    """
    Xác minh mã băm của tệp dữ liệu thực tế data/test.csv.
    Phải khớp hoàn toàn với cam kết trong PROGRESS.md (Honesty Rule #3).
    """
    assert TEST_CSV_PATH.is_file(), f"Tệp kiểm thử data/test.csv không tồn tại: {TEST_CSV_PATH}"
    is_valid, actual_hash, expected_hash = verify_test_set_checksum(
        test_csv_path=TEST_CSV_PATH,
        progress_path=PROGRESS_PATH,
        raise_on_error=True,
    )
    assert is_valid is True
    assert actual_hash == TEST_CSV_SHA256
    assert expected_hash == TEST_CSV_SHA256


def test_verify_test_set_checksum_detects_tampering(tmp_path: Path) -> None:
    """
    Kiểm tra cơ chế bảo vệ phát hiện khi tệp test.csv bị chỉnh sửa hoặc giả mạo.
    - Với raise_on_error=True: Ném ngoại lệ TestSetChecksumMismatchError.
    - Với raise_on_error=False: Trả về is_valid=False.
    """
    tampered_test = tmp_path / "tampered_test.csv"
    tampered_test.write_text("text,label,category\nVi phạm tính toàn vẹn,L1,C1\n", encoding="utf-8")

    # 1. Kiểm tra với raise_on_error=True
    with pytest.raises(TestSetChecksumMismatchError) as exc_info:
        verify_test_set_checksum(
            test_csv_path=tampered_test,
            expected_sha256=TEST_CSV_SHA256,
            raise_on_error=True,
        )
    assert "Phát hiện vi phạm tính toàn vẹn tập kiểm thử" in str(exc_info.value)
    assert exc_info.value.actual_sha256 != TEST_CSV_SHA256
    assert exc_info.value.expected_sha256 == TEST_CSV_SHA256

    # 2. Kiểm tra với raise_on_error=False
    is_valid, actual, expected = verify_test_set_checksum(
        test_csv_path=tampered_test,
        expected_sha256=TEST_CSV_SHA256,
        raise_on_error=False,
    )
    assert is_valid is False
    assert actual != expected


def test_ensure_test_set_integrity_returns_hash() -> None:
    """Kiểm tra hàm helper ensure_test_set_integrity trả về mã băm khi xác thực thành công."""
    verified_hash = ensure_test_set_integrity(
        test_csv_path=TEST_CSV_PATH,
        progress_path=PROGRESS_PATH,
    )
    assert verified_hash == TEST_CSV_SHA256


# =====================================================================
# 3. Kiểm thử Khoảng Tin cậy 95% Bootstrap (Accuracy & Macro-F1)
# =====================================================================


def test_accuracy_bootstrap_ci_perfect_accuracy() -> None:
    """
    Kiểm tra tính đúng đắn khi mô hình đạt 100% độ chính xác:
    Point estimate = 1.0, CI = [1.0, 1.0].
    """
    y_true = ["A", "B", "C", "D", "E"] * 10
    y_pred = list(y_true)

    res = compute_accuracy_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        num_bootstrap=200,
        confidence_level=0.95,
        seed=DEFAULT_SEED,
    )

    assert isinstance(res, BootstrapCIResult)
    assert res.metric_name == "accuracy"
    assert res.point_estimate == 1.0
    assert res.ci_lower == 1.0
    assert res.ci_upper == 1.0
    assert "100.0000%" in res.format_ci(percentage=True)


def test_accuracy_bootstrap_ci_zero_accuracy() -> None:
    """
    Kiểm tra tính đúng đắn khi mô hình đoán sai 100%:
    Point estimate = 0.0, CI = [0.0, 0.0].
    """
    y_true = ["A"] * 20
    y_pred = ["B"] * 20

    res = compute_accuracy_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        num_bootstrap=200,
        confidence_level=0.95,
        seed=DEFAULT_SEED,
    )

    assert res.point_estimate == 0.0
    assert res.ci_lower == 0.0
    assert res.ci_upper == 0.0


def test_accuracy_bootstrap_ci_mixed_data_bounds() -> None:
    """
    Kiểm tra ước lượng khoảng tin cậy 95% trên dữ liệu hỗn hợp:
    - 0.0 <= ci_lower <= point_estimate <= ci_upper <= 1.0.
    """
    y_true = [f"Class_{i % 4}" for i in range(100)]
    # Dự đoán đúng 80%
    y_pred = [f"Class_{i % 4}" if i % 5 != 0 else "Class_99" for i in range(100)]

    res = compute_accuracy_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        num_bootstrap=500,
        confidence_level=0.95,
        seed=DEFAULT_SEED,
    )

    assert res.point_estimate == pytest.approx(0.80, abs=1e-4)
    assert 0.65 <= res.ci_lower <= res.point_estimate
    assert res.point_estimate <= res.ci_upper <= 0.95
    assert res.num_bootstrap == 500
    assert res.confidence_level == 0.95


def test_macro_f1_bootstrap_ci_with_labels() -> None:
    """
    Kiểm tra ước lượng Bootstrap CI cho Macro-F1 với tập nhãn cụ thể.
    """
    labels = ["A", "B"]
    y_true = ["A", "A", "B", "B"] * 25
    y_pred = ["A", "A", "B", "A"] * 25

    res = compute_macro_f1_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        num_bootstrap=300,
        confidence_level=0.95,
        seed=DEFAULT_SEED,
    )

    assert isinstance(res, BootstrapCIResult)
    assert res.metric_name == "macro_f1"
    assert 0.0 <= res.ci_lower <= res.point_estimate <= res.ci_upper <= 1.0


def test_compute_ablation_bootstrap_cis_dict() -> None:
    """
    Kiểm tra hàm tiện ích compute_ablation_bootstrap_cis trả về dict chứa cả accuracy và macro_f1.
    """
    labels = ["L1", "L2"]
    y_true = ["L1", "L2"] * 20
    y_pred = ["L1", "L2"] * 20

    res_dict = compute_ablation_bootstrap_cis(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        intents_path=DEFAULT_INTENTS_PATH,
        num_bootstrap=100,
        seed=DEFAULT_SEED,
    )

    assert "accuracy" in res_dict
    assert "macro_f1" in res_dict
    assert isinstance(res_dict["accuracy"], BootstrapCIResult)
    assert isinstance(res_dict["macro_f1"], BootstrapCIResult)
    assert res_dict["accuracy"].point_estimate == 1.0
    assert res_dict["macro_f1"].point_estimate == 1.0


def test_bootstrap_ci_input_validation() -> None:
    """Kiểm tra xử lý lỗi khi tham số đầu vào không hợp lệ."""
    with pytest.raises(ValueError, match="không đồng nhất"):
        compute_accuracy_bootstrap_ci(["A", "B"], ["A"])

    with pytest.raises(ValueError, match="rỗng"):
        compute_accuracy_bootstrap_ci([], [])

    with pytest.raises(ValueError, match="num_bootstrap phải >= 1"):
        compute_accuracy_bootstrap_ci(["A"], ["A"], num_bootstrap=0)

    with pytest.raises(ValueError, match="confidence_level phải trong khoảng"):
        compute_accuracy_bootstrap_ci(["A"], ["A"], confidence_level=1.5)


# =====================================================================
# 4. Kiểm định McNemar So sánh Các Biến thể Mô hình (Ablation Comparison)
# =====================================================================


def test_compare_models_mcnemar_identical_models() -> None:
    """
    Kiểm thử trường hợp biên khi hai mô hình có dự đoán hoàn toàn đồng thuận (b = 0, c = 0).
    Theo yêu cầu hệ thống: statistic = 0.0, p_value = 1.0, is_significant = False, winner = None.
    """
    y_true = ["A", "B", "C", "D"] * 10
    y_pred_a = list(y_true)
    y_pred_b = list(y_true)

    result = compare_models_mcnemar(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        model_a_name="PhoBERT Baseline",
        model_b_name="PhoBERT + Retrieval",
    )

    assert isinstance(result, McNemarComparisonResult)
    assert result.b == 0
    assert result.c == 0
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.is_significant is False
    assert result.winner is None
    assert "KHÔNG CÓ Ý NGHĨA THỐNG KÊ" in result.summary


def test_compare_models_mcnemar_statistically_significant() -> None:
    """
    Kiểm thử trường hợp một mô hình vượt trội rõ rệt (b = 25, c = 0).
    Hiệu chỉnh Yates: chi2 = (|25 - 0| - 1)^2 / 25 = 24^2 / 25 = 576 / 25 = 23.04.
    p-value < 0.001 -> có ý nghĩa thống kê, winner = "Full Pipeline".
    """
    y_true = ["A"] * 50
    # Model A đúng 40 câu
    y_pred_a = ["A"] * 40 + ["B"] * 10
    # Model B đúng 15 câu (chỉ đúng 15 câu đầu trong 40 câu của A)
    y_pred_b = ["A"] * 15 + ["B"] * 35

    # b = 25 (A đúng, B sai), c = 0 (A sai, B đúng)
    result = compare_models_mcnemar(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        model_a_name="Full Pipeline",
        model_b_name="PhoBERT Only",
        alpha=0.05,
        continuity_correction=True,
    )

    assert result.b == 25
    assert result.c == 0
    expected_chi2 = (abs(25 - 0) - 1.0) ** 2 / 25.0  # 23.04
    assert result.statistic == pytest.approx(expected_chi2, abs=TOLERANCE)
    assert result.p_value < 0.001
    assert result.is_significant is True
    assert result.winner == "Full Pipeline"
    assert "CÓ Ý NGHĨA THỐNG KÊ" in result.summary


def test_compare_models_mcnemar_symmetrical_ties() -> None:
    """
    Kiểm thử trường hợp bất đồng đối xứng (b = 10, c = 10).
    |b - c| = 0 -> statistic = 0.0, p_value = 1.0, is_significant = False.
    """
    y_true = ["A"] * 20
    # A đúng 10 mẫu đầu, sai 10 mẫu sau
    y_pred_a = ["A"] * 10 + ["B"] * 10
    # B sai 10 mẫu đầu, đúng 10 mẫu sau
    y_pred_b = ["B"] * 10 + ["A"] * 10

    result = compare_models_mcnemar(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        model_a_name="Variant 1",
        model_b_name="Variant 2",
    )

    assert result.b == 10
    assert result.c == 10
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.is_significant is False
    assert result.winner is None


def test_compare_models_mcnemar_validation() -> None:
    """Kiểm tra kiểm thực tham số đầu vào của hàm compare_models_mcnemar."""
    with pytest.raises(ValueError, match="không đồng nhất"):
        compare_models_mcnemar(["A"], ["A"], ["A", "B"])

    with pytest.raises(ValueError, match="rỗng"):
        compare_models_mcnemar([], [], [])


# =====================================================================
# 5. Kiểm thử Tổng hợp Đa Hạt Giống (Multi-Seed Aggregation)
# =====================================================================


def test_aggregate_seed_values_hand_computed() -> None:
    """
    Kiểm chứng đối chuẩn tính tay (hand-computed) cho aggregate_seed_values:
    Dữ liệu: [0.80, 0.85, 0.90] (3 hạt giống)
    - Mean = (0.80 + 0.85 + 0.90) / 3 = 0.85
    - Sample variance (ddof=1) = [(-0.05)^2 + 0^2 + (0.05)^2] / 2 = 0.005 / 2 = 0.0025
    - Sample Std = sqrt(0.0025) = 0.05
    - SEM = 0.05 / sqrt(3) ~= 0.0288675
    """
    values = [0.80, 0.85, 0.90]
    summary = aggregate_seed_values(values=values, metric_name="accuracy", percentage=True, precision=2)

    assert isinstance(summary, MultiSeedMetricSummary)
    assert summary.metric_name == "accuracy"
    assert summary.mean == pytest.approx(0.85, abs=TOLERANCE)
    assert summary.std == pytest.approx(0.05, abs=TOLERANCE)
    assert summary.sem == pytest.approx(0.05 / (3**0.5), abs=TOLERANCE)
    assert summary.min == pytest.approx(0.80, abs=TOLERANCE)
    assert summary.max == pytest.approx(0.90, abs=TOLERANCE)
    assert summary.n_seeds == 3
    assert summary.formatted == "85.00 ± 5.00%"


def test_aggregate_seed_values_single_seed() -> None:
    """
    Kiểm tra khi chỉ có 1 seed (n=1):
    Mean = value, Std = 0.0, SEM = 0.0.
    """
    summary = aggregate_seed_values([0.925], metric_name="macro_f1")
    assert summary.mean == 0.925
    assert summary.std == 0.0
    assert summary.sem == 0.0
    assert summary.n_seeds == 1


def test_aggregate_seed_values_empty_raises() -> None:
    """Kiểm tra ngoại lệ ValueError khi danh sách seed rỗng."""
    with pytest.raises(ValueError, match="không được rỗng"):
        aggregate_seed_values([])


def test_aggregate_multi_seed_results_list_format() -> None:
    """
    Kiểm tra tổng hợp danh sách các lần chạy hạt giống:
    [{"accuracy": 0.89, "macro_f1": 0.87}, {"accuracy": 0.91, "macro_f1": 0.89}]
    """
    runs = [
        {"accuracy": 0.89, "macro_f1": 0.87, "non_numeric": "ignore_me"},
        {"accuracy": 0.91, "macro_f1": 0.89, "non_numeric": "ignore_me"},
    ]
    agg = aggregate_multi_seed_results(runs, model_name="PhoBERT Baseline", seeds=[42, 43])

    assert isinstance(agg, MultiSeedRunAggregation)
    assert agg.model_name == "PhoBERT Baseline"
    assert agg.n_runs == 2
    assert agg.seeds == [42, 43]
    assert "accuracy" in agg.metrics
    assert "macro_f1" in agg.metrics
    assert "non_numeric" not in agg.metrics

    acc_sum = agg.metrics["accuracy"]
    assert acc_sum.mean == pytest.approx(0.90, abs=TOLERANCE)
    assert acc_sum.min == pytest.approx(0.89, abs=TOLERANCE)
    assert acc_sum.max == pytest.approx(0.91, abs=TOLERANCE)

    # Kiểm tra phương thức xuất bảng ASCII format_table
    table_text = agg.format_table(percentage=True)
    assert "Multi-Seed Aggregation: PhoBERT Baseline" in table_text
    assert "accuracy" in table_text
    assert "macro_f1" in table_text


def test_aggregate_multi_seed_results_dict_format() -> None:
    """
    Kiểm tra tổng hợp khi đầu vào là dictionary ánh xạ seed -> metrics.
    """
    runs_dict = {
        42: {"accuracy": 0.95, "loss": 0.12},
        43: {"accuracy": 0.97, "loss": 0.10},
        44: {"accuracy": 0.96, "loss": 0.11},
    }
    agg = aggregate_multi_seed_results(runs_dict, model_name="Bi-Encoder Mode A")

    assert agg.n_runs == 3
    assert agg.seeds == [42, 43, 44]
    assert "accuracy" in agg.metrics
    assert "loss" in agg.metrics

    loss_sum = agg.metrics["loss"]
    assert loss_sum.mean == pytest.approx(0.11, abs=TOLERANCE)
    assert loss_sum.std == pytest.approx(0.01, abs=TOLERANCE)


# =====================================================================
# 6. Kiểm thử Giao diện Dòng lệnh (CLI Entrypoint)
# =====================================================================


def test_run_test_set_verification_cli_success() -> None:
    """Kiểm tra hàm run_test_set_verification_cli trả về mã thoát 0 khi tệp hợp lệ."""
    exit_code = run_test_set_verification_cli(
        test_csv_path=TEST_CSV_PATH,
        progress_path=PROGRESS_PATH,
    )
    assert exit_code == 0


def test_run_test_set_verification_cli_failure(tmp_path: Path) -> None:
    """Kiểm tra hàm run_test_set_verification_cli trả về mã thoát 1 khi tệp bị sửa đổi."""
    fake_csv = tmp_path / "fake_test.csv"
    fake_csv.write_text("text,label,category\nSai băm,L1,C1\n", encoding="utf-8")

    exit_code = run_test_set_verification_cli(
        test_csv_path=fake_csv,
        progress_path=PROGRESS_PATH,
    )
    assert exit_code == 1


def test_main_cli_verify_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kiểm tra CLI main() với cờ --verify-test."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train/ablation_stats.py",
            "--verify-test",
            "--test-csv",
            str(TEST_CSV_PATH),
            "--progress-path",
            str(PROGRESS_PATH),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_main_cli_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kiểm tra CLI main() với cờ --demo."""
    monkeypatch.setattr(sys, "argv", ["train/ablation_stats.py", "--demo"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
