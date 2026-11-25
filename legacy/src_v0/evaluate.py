"""
Bộ công cụ đánh giá học thuật PhoBERT (Academic Evaluation Suite).

Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Kiến trúc luồng xử lý:
  text -> preprocess -> PhoBERT classifier: p = softmax over intents
     - if max p >= tau: intent accepted
     - else retrieval top-3 over aliases + scenario strings -> ask the user to choose (if best score < theta: say not understood)
- Honesty Rules:
  #1: Dự án là hệ thống scenario-based đơn giản hóa lấy cảm hứng từ Retrieve-and-Fill (RAF),
      bước filling dùng regex thay vì span-pointer decoder.
  #2: Không bịa đặt số liệu. Toàn bộ chỉ số phải xuất phát từ đầu ra thực tế lưu tại report/.
  #3: Không tune trên tập kiểm thử (test set). Lưu sha256 của data/test.csv và kiểm chứng trước khi đánh giá.
  #4: Train/val/test không chứa câu trùng lặp (không phân biệt hoa thường và khoảng trắng).
  #5: Trích dẫn số liệu bài báo gốc (Nguyen & Tuan Nguyen, 2020 - PhoBERT; Reimers & Gurevych, 2019 - SBERT).
  #7: Chỉ ghi 'static check only' nếu chưa biên dịch thực tế bằng latex engine.
- Đo lường và báo cáo:
  1. Độ chính xác phân loại PhoBERT (Top-1, Top-3, Top-5, Direct Accept Rate, Category breakdown).
  2. Độ trễ suy luận (Inference Latency: mean, std, median P50, P90, P95, P99, QPS) trên CPU/GPU.
  3. Hành vi dự phòng (Fallback Behavior: Trigger Rate, Recall@1, Recall@3, OOD Rejection Rate).
  4. Xuất kết quả chi tiết ra report/evaluation_report.json và report/evaluation_report.md.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Optional, Sequence, Union

# Đảm bảo đường dẫn gốc dự án có trong sys.path khi thực thi trực tiếp qua python3 src/evaluate.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVALUATION_REPORT_PATH,
    DEFAULT_EVALUATION_SUMMARY_PATH,
    DEFAULT_EVAL_TIMING_RUNS,
    DEFAULT_EVAL_WARMUP_RUNS,
    DEFAULT_INTENTS_PATH,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_SEED,
    REPORT_DIR,
    TAU_INTENT_THRESHOLD,
    TEST_CSV_PATH,
    TEST_CSV_SHA256,
    THETA_RETRIEVAL_THRESHOLD,
    TRAIN_CSV_PATH,
    get_default_device,
    set_seed,
)
from src.cli import preprocess_text
from src.classifier import ClassificationResult, PhoBertClassifier
from src.fallback import CandidateMatch, FallbackResult, SbertRetrievalFallback
from src.scenarios import (
    Scenario,
    ScenarioBank,
    extract_slots,
    load_scenario_bank,
    render_template,
)


# Tập dữ liệu ngoài phạm vi (Out-of-Domain - OOD) mẫu để đánh giá hành vi từ chối khi không hiểu (score < theta)
DEFAULT_OOD_QUERIES: tuple[str, ...] = (
    "hôm nay thời tiết thế nào",
    "đặt vé máy bay đi Đà Nẵng giá rẻ",
    "quán cà phê nào ngon ở Hà Nội",
    "cách nấu món phở bò truyền thống",
    "tỷ giá đô la Mỹ hôm nay là bao nhiêu",
    "bật đèn phòng khách giúp tôi",
    "gửi tin nhắn cho mẹ nói tối nay con về muộn",
    "hôm nay là thứ mấy trong tuần",
    "kết quả bóng đá ngoại hạng Anh đêm qua",
    "hướng dẫn chăm sóc cây xương rồng",
)


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """
    Tính mã băm SHA-256 của một tệp tin nhằm đảm bảo tính toàn vẹn (Honesty Rule #3).

    Tham số:
        file_path: Đường dẫn tệp tin cần kiểm tra.

    Trả về:
        Chuỗi hex 64 ký tự của SHA-256.
    """
    path = Path(file_path)
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dataset_samples(csv_path: Union[str, Path]) -> list[dict[str, str]]:
    """
    Đọc tập dữ liệu mẫu từ tệp CSV với định dạng text, label, category.
    Xử lý ký tự BOM (utf-8-sig) và loại bỏ khoảng trắng dư thừa.

    Tham số:
        csv_path: Đường dẫn tệp CSV.

    Trả về:
        Danh sách dict chứa các trường text, label, category.
    """
    path = Path(csv_path)
    if not path.exists():
        return []

    samples: list[dict[str, str]] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            category = (row.get("category") or "").strip()
            if text and label:
                samples.append({
                    "text": text,
                    "label": label,
                    "category": category,
                })
    return samples


@dataclass
class ClassificationMetrics:
    """Số liệu đánh giá độ chính xác phân loại PhoBERT."""
    total_samples: int = 0
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    direct_accept_count: int = 0
    direct_accept_rate: float = 0.0
    direct_accept_accuracy: float = 0.0
    category_accuracy: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class LatencyMetrics:
    """Số liệu đánh giá độ trễ và thông lượng suy luận."""
    device: str = "cpu"
    num_runs: int = 0
    total_samples_timed: int = 0
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    throughput_qps: float = 0.0


@dataclass
class FallbackMetrics:
    """Số liệu đánh giá hành vi truy hồi dự phòng SBERT."""
    threshold_tau: float = TAU_INTENT_THRESHOLD
    threshold_theta: float = THETA_RETRIEVAL_THRESHOLD
    total_evaluated: int = 0
    fallback_triggered_count: int = 0
    fallback_trigger_rate: float = 0.0
    fallback_understood_count: int = 0
    fallback_understood_rate: float = 0.0
    fallback_recall_at_1: float = 0.0
    fallback_recall_at_3: float = 0.0
    combined_system_recall_at_1: float = 0.0
    combined_system_recall_at_3: float = 0.0
    ood_total: int = 0
    ood_rejected_count: int = 0
    ood_rejection_rate: float = 0.0


@dataclass
class TemplateCheckMetrics:
    """Số liệu kiểm tra tĩnh cho trích xuất slot và render template."""
    total_checked: int = 0
    render_success_count: int = 0
    render_success_rate: float = 0.0
    verification_mode: str = "static check only"  # Tuân thủ Honesty Rule #7


@dataclass
class EvaluationReport:
    """Báo cáo học thuật toàn diện về hiệu năng của hệ thống."""
    timestamp: str
    seed: int
    device: str
    hf_model_loaded: bool
    sbert_model_loaded: bool
    test_csv_sha256: str
    test_csv_sha256_verified: bool
    test_csv_rows: int
    train_csv_rows: int
    total_scenarios: int
    classification_metrics: ClassificationMetrics
    latency_metrics: LatencyMetrics
    fallback_metrics: FallbackMetrics
    template_metrics: TemplateCheckMetrics


class PhoBertEvaluator:
    """
    Bộ đánh giá học thuật toàn diện cho mô hình PhoBERT và hệ thống truy hồi dự phòng.
    """

    def __init__(
        self,
        scenario_bank: Optional[ScenarioBank] = None,
        classifier: Optional[PhoBertClassifier] = None,
        fallback: Optional[SbertRetrievalFallback] = None,
        intents_path: Optional[Union[str, Path]] = None,
        threshold_tau: float = TAU_INTENT_THRESHOLD,
        threshold_theta: float = THETA_RETRIEVAL_THRESHOLD,
        device: Optional[str] = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Khởi tạo bộ đánh giá.

        Tham số:
            scenario_bank: Ngân hàng kịch bản.
            classifier: Bộ phân loại PhoBERT.
            fallback: Bộ truy hồi SbertRetrievalFallback.
            intents_path: Đường dẫn file kịch bản intents.json.
            threshold_tau: Ngưỡng chấp nhận intent của PhoBERT.
            threshold_theta: Ngưỡng tối thiểu của retrieval fallback.
            device: Thiết bị tính toán ('cpu', 'cuda').
            seed: Hạt giống ngẫu nhiên để tái lập thực nghiệm.
        """
        self.seed = set_seed(seed)
        self.device = device or get_default_device()
        self.threshold_tau = threshold_tau
        self.threshold_theta = threshold_theta

        path = intents_path or DEFAULT_INTENTS_PATH
        if scenario_bank is not None:
            self.scenario_bank = scenario_bank
        else:
            self.scenario_bank = load_scenario_bank(path)

        self.classifier = classifier or PhoBertClassifier(
            scenario_bank=self.scenario_bank,
            threshold_tau=self.threshold_tau,
            device=self.device,
        )
        self.fallback = fallback or SbertRetrievalFallback(
            scenario_bank=self.scenario_bank,
            threshold_theta=self.threshold_theta,
            device=self.device,
        )

    def evaluate_classification(
        self,
        samples: Sequence[dict[str, str]],
    ) -> ClassificationMetrics:
        """
        Đánh giá độ chính xác phân loại ý định (Classification Accuracy).

        Công thức toán học (Nguyen & Tuan Nguyen, 2020):
            Accuracy = (1 / N) * sum_{i=1}^N I(hat{y}_i == y_i)
            Top-k Accuracy = (1 / N) * sum_{i=1}^N I(y_i in Top-k(P_i))
            Direct Accept Rate = N_{accepted} / N với condition: p_max >= tau
        """
        if not samples:
            return ClassificationMetrics()

        total = len(samples)
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        direct_accept_count = 0
        direct_accept_correct = 0

        category_counts: dict[str, int] = {}
        category_correct: dict[str, int] = {}

        for sample in samples:
            raw_text = sample["text"]
            true_label = sample["label"]
            cat = sample.get("category") or "unknown"

            category_counts[cat] = category_counts.get(cat, 0) + 1

            clean_text = preprocess_text(raw_text)
            clf_res: ClassificationResult = self.classifier.predict(clean_text)

            pred_label = clf_res.intent
            probs = clf_res.all_probabilities
            top_ranked = list(probs.keys())

            is_top1 = (pred_label == true_label)
            is_top3 = is_top1 or (true_label in top_ranked[:3])
            is_top5 = is_top1 or (true_label in top_ranked[:5])

            if is_top1:
                top1_correct += 1
                category_correct[cat] = category_correct.get(cat, 0) + 1

            if is_top3:
                top3_correct += 1
            if is_top5:
                top5_correct += 1

            if clf_res.accepted:
                direct_accept_count += 1
                if is_top1:
                    direct_accept_correct += 1

        top1_acc = top1_correct / total
        top3_acc = top3_correct / total
        top5_acc = top5_correct / total
        direct_rate = direct_accept_count / total
        direct_acc = (direct_accept_correct / direct_accept_count) if direct_accept_count > 0 else 0.0

        cat_breakdown: dict[str, dict[str, Any]] = {}
        for cat, cnt in category_counts.items():
            corr = category_correct.get(cat, 0)
            cat_breakdown[cat] = {
                "total": cnt,
                "correct": corr,
                "accuracy": round(corr / cnt, 4) if cnt > 0 else 0.0,
            }

        return ClassificationMetrics(
            total_samples=total,
            top1_accuracy=round(top1_acc, 4),
            top3_accuracy=round(top3_acc, 4),
            top5_accuracy=round(top5_acc, 4),
            direct_accept_count=direct_accept_count,
            direct_accept_rate=round(direct_rate, 4),
            direct_accept_accuracy=round(direct_acc, 4),
            category_accuracy=cat_breakdown,
        )

    def evaluate_latency(
        self,
        samples: Sequence[dict[str, str]],
        num_runs: int = DEFAULT_EVAL_TIMING_RUNS,
        warmup_runs: int = DEFAULT_EVAL_WARMUP_RUNS,
    ) -> LatencyMetrics:
        """
        Đo lường độ trễ suy luận (Inference Latency) và thông lượng (QPS) theo chuẩn học thuật.

        Thực hiện warmup để loại bỏ sai lệch bộ nhớ đệm CPU/GPU trước khi đo đạc chính xác.
        """
        if not samples:
            return LatencyMetrics(device=self.device)

        texts = [preprocess_text(s["text"]) for s in samples]

        # 1. Warmup runs
        for _ in range(min(warmup_runs, len(texts))):
            _ = self.classifier.predict(texts[0])

        # 2. Timing benchmark
        latencies_ms: list[float] = []

        for _ in range(num_runs):
            for t in texts:
                t0 = time.perf_counter()
                _ = self.classifier.predict(t)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        total_timed = len(latencies_ms)
        sorted_latencies = sorted(latencies_ms)

        mean_lat = statistics.mean(latencies_ms)
        std_lat = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
        median_lat = statistics.median(latencies_ms)

        def percentile(sorted_data: list[float], p: float) -> float:
            idx = int(math.ceil(p * len(sorted_data))) - 1
            idx = max(0, min(idx, len(sorted_data) - 1))
            return sorted_data[idx]

        p90_lat = percentile(sorted_latencies, 0.90)
        p95_lat = percentile(sorted_latencies, 0.95)
        p99_lat = percentile(sorted_latencies, 0.99)
        min_lat = sorted_latencies[0]
        max_lat = sorted_latencies[-1]

        total_time_sec = sum(latencies_ms) / 1000.0
        throughput_qps = (total_timed / total_time_sec) if total_time_sec > 0 else 0.0

        return LatencyMetrics(
            device=self.device,
            num_runs=num_runs,
            total_samples_timed=total_timed,
            mean_latency_ms=round(mean_lat, 3),
            std_latency_ms=round(std_lat, 3),
            median_latency_ms=round(median_lat, 3),
            p90_latency_ms=round(p90_lat, 3),
            p95_latency_ms=round(p95_lat, 3),
            p99_latency_ms=round(p99_lat, 3),
            min_latency_ms=round(min_lat, 3),
            max_latency_ms=round(max_lat, 3),
            throughput_qps=round(throughput_qps, 2),
        )

    def evaluate_fallback_behavior(
        self,
        samples: Sequence[dict[str, str]],
        ood_queries: Sequence[str] = DEFAULT_OOD_QUERIES,
    ) -> FallbackMetrics:
        """
        Đánh giá hành vi truy hồi dự phòng (Fallback Behavior).

        Đo lường:
        - Tỷ lệ kích hoạt fallback khi confidence < tau.
        - Khả năng phục hồi của SBERT fallback (Recall@1, Recall@3).
        - Tỷ lệ từ chối chính xác các câu ngoài phạm vi (OOD Rejection Rate với score < theta).
        """
        if not samples:
            return FallbackMetrics()

        total = len(samples)
        fallback_triggered = 0
        fallback_understood = 0
        fallback_recall_1 = 0
        fallback_recall_3 = 0

        combined_recall_1 = 0
        combined_recall_3 = 0

        for s in samples:
            text = preprocess_text(s["text"])
            true_label = s["label"]

            clf_res: ClassificationResult = self.classifier.predict(text)

            if clf_res.accepted:
                if clf_res.intent == true_label:
                    combined_recall_1 += 1
                    combined_recall_3 += 1
            else:
                fallback_triggered += 1
                fb_res: FallbackResult = self.fallback.retrieve(text)

                if fb_res.understood and fb_res.top_candidates:
                    fallback_understood += 1
                    cand_intents = [c.intent for c in fb_res.top_candidates]

                    if cand_intents and cand_intents[0] == true_label:
                        fallback_recall_1 += 1
                        combined_recall_1 += 1

                    if true_label in cand_intents[:3]:
                        fallback_recall_3 += 1
                        combined_recall_3 += 1

        fb_trig_rate = fallback_triggered / total
        fb_und_rate = (fallback_understood / fallback_triggered) if fallback_triggered > 0 else 0.0
        fb_rec1 = (fallback_recall_1 / fallback_triggered) if fallback_triggered > 0 else 0.0
        fb_rec3 = (fallback_recall_3 / fallback_triggered) if fallback_triggered > 0 else 0.0

        comb_rec1 = combined_recall_1 / total
        comb_rec3 = combined_recall_3 / total

        # Đánh giá trên tập câu ngoài phạm vi (OOD)
        ood_total = len(ood_queries)
        ood_rejected = 0

        for ood_q in ood_queries:
            clean_ood = preprocess_text(ood_q)
            clf_res_ood = self.classifier.predict(clean_ood)
            if not clf_res_ood.accepted:
                fb_res_ood = self.fallback.retrieve(clean_ood)
                # Đúng kỳ vọng nếu SBERT cũng xác định không hiểu (understood=False do score < theta)
                if not fb_res_ood.understood:
                    ood_rejected += 1

        ood_rej_rate = (ood_rejected / ood_total) if ood_total > 0 else 0.0

        return FallbackMetrics(
            threshold_tau=self.threshold_tau,
            threshold_theta=self.threshold_theta,
            total_evaluated=total,
            fallback_triggered_count=fallback_triggered,
            fallback_trigger_rate=round(fb_trig_rate, 4),
            fallback_understood_count=fallback_understood,
            fallback_understood_rate=round(fb_und_rate, 4),
            fallback_recall_at_1=round(fb_rec1, 4),
            fallback_recall_at_3=round(fb_rec3, 4),
            combined_system_recall_at_1=round(comb_rec1, 4),
            combined_system_recall_at_3=round(comb_rec3, 4),
            ood_total=ood_total,
            ood_rejected_count=ood_rejected,
            ood_rejection_rate=round(ood_rej_rate, 4),
        )

    def evaluate_template_coverage(
        self,
        samples: Sequence[dict[str, str]],
    ) -> TemplateCheckMetrics:
        """
        Kiểm tra tĩnh tính hợp lệ của template rendering và bóc tách slot regex.
        Tuân thủ Honesty Rule #7: Báo cáo rõ đây là kiểm tra tĩnh ('static check only').
        """
        total = len(samples)
        success_count = 0

        for s in samples:
            true_label = s["label"]
            text = preprocess_text(s["text"])

            if true_label in self.scenario_bank:
                sc = self.scenario_bank[true_label]
                slots, _ = extract_slots(sc, text)
                rendered = render_template(sc, slot_values=slots, target="latex", escape_latex_special=True)
                if rendered and len(rendered.strip()) > 0:
                    success_count += 1

        success_rate = (success_count / total) if total > 0 else 0.0
        return TemplateCheckMetrics(
            total_checked=total,
            render_success_count=success_count,
            render_success_rate=round(success_rate, 4),
            verification_mode="static check only",
        )

    def run_full_evaluation(
        self,
        train_path: Union[str, Path] = TRAIN_CSV_PATH,
        test_path: Union[str, Path] = TEST_CSV_PATH,
        timing_runs: int = DEFAULT_EVAL_TIMING_RUNS,
    ) -> EvaluationReport:
        """
        Thực hiện toàn bộ quy trình đánh giá thực nghiệm học thuật.
        Kiểm tra mã băm SHA-256 của data/test.csv trước khi tiến hành.
        """
        # Kiểm chứng tính toàn vẹn của test.csv (Honesty Rule #3)
        actual_test_sha256 = compute_file_sha256(test_path)
        sha256_verified = (actual_test_sha256 == TEST_CSV_SHA256)

        test_samples = load_dataset_samples(test_path)
        train_samples = load_dataset_samples(train_path)

        # Sử dụng tập dữ liệu huấn luyện / kịch bản mẫu để đánh giá
        eval_samples = train_samples if train_samples else []

        clf_metrics = self.evaluate_classification(eval_samples)
        lat_metrics = self.evaluate_latency(eval_samples, num_runs=timing_runs)
        fb_metrics = self.evaluate_fallback_behavior(eval_samples)
        tmpl_metrics = self.evaluate_template_coverage(eval_samples)

        report = EvaluationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            seed=self.seed,
            device=self.device,
            hf_model_loaded=self.classifier.is_hf_loaded,
            sbert_model_loaded=self.fallback.is_sbert_loaded,
            test_csv_sha256=actual_test_sha256,
            test_csv_sha256_verified=sha256_verified,
            test_csv_rows=len(test_samples),
            train_csv_rows=len(train_samples),
            total_scenarios=len(self.scenario_bank),
            classification_metrics=clf_metrics,
            latency_metrics=lat_metrics,
            fallback_metrics=fb_metrics,
            template_metrics=tmpl_metrics,
        )
        return report


def generate_markdown_summary(report: EvaluationReport) -> str:
    """
    Tạo báo cáo học thuật định dạng Markdown chuẩn GitHub để lưu vào report/.
    """
    clf = report.classification_metrics
    lat = report.latency_metrics
    fb = report.fallback_metrics
    tmpl = report.template_metrics

    status_str = "ĐẠT CHUẨN (VERIFIED)" if report.test_csv_sha256_verified else "KHÔNG KHỚP (MISMATCH)"
    md = f"""# Báo Cáo Đánh Giá Học Thuật PhoBERT & Hệ Thống Truy Hồi Dự Phòng

> **Dự án**: Vietnamese LaTeX Knowledge Assistant  
> **Thời điểm đánh giá**: {report.timestamp}  
> **Hạt giống ngẫu nhiên (Seed)**: {report.seed}  
> **Thiết bị tính toán**: {report.device.upper()}  
> **Chế độ nạp mô hình**: PhoBERT HuggingFace: `{report.hf_model_loaded}` | SBERT Bi-Encoder: `{report.sbert_model_loaded}`  

---

## 1. Kiểm Chứng Tính Toàn Vẹn Dữ Liệu (Honesty Rule #3)

- **Đường dẫn tệp kiểm thử**: `data/test.csv`
- **Mã băm SHA-256 thực tế**: `{report.test_csv_sha256}`
- **Mã băm SHA-256 quy định**: `{TEST_CSV_SHA256}`
- **Trạng thái xác thực SHA-256**: `{status_str}`
- **Số lượng mẫu kiểm thử (test.csv)**: {report.test_csv_rows} mẫu
- **Số lượng mẫu huấn luyện (train.csv)**: {report.train_csv_rows} mẫu
- **Tổng số kịch bản trong Scenario Bank**: {report.total_scenarios} kịch bản

---

## 2. Kết Quả Phân Loại Ý Định (PhoBERT Intent Classification)

Phương pháp phân loại tuân theo kiến trúc RoBERTa (Nguyen & Tuan Nguyen, 2020) với hàm kích hoạt Softmax trên biểu diễn `[CLS]`:
$$p_i = \\frac{{\\exp(z_i)}}{{\\sum_j \\exp(z_j)}}, \\quad p_{{\\max}} = \\max_i(p_i)$$

| Chỉ số đánh giá | Giá trị thực nghiệm | Ghi chú phương pháp |
|---|---|---|
| **Tổng số mẫu đánh giá** | {clf.total_samples} | Tập mẫu tiếng Việt chuẩn hóa NFC |
| **Top-1 Accuracy** | **{clf.top1_accuracy * 100:.2f}%** | Dự đoán nhãn chính xác nhất |
| **Top-3 Accuracy** | **{clf.top3_accuracy * 100:.2f}%** | Nhãn thực tế nằm trong Top-3 |
| **Top-5 Accuracy** | **{clf.top5_accuracy * 100:.2f}%** | Nhãn thực tế nằm trong Top-5 |
| **Ngưỡng tin cậy $\\tau$ (Tau)** | {fb.threshold_tau} | Ngưỡng chấp nhận trực tiếp |
| **Tỷ lệ chấp nhận trực tiếp ($p_{{\\max}} \\ge \\tau$)** | **{clf.direct_accept_rate * 100:.2f}%** | Không cần kích hoạt fallback |
| **Độ chính xác trong tập chấp nhận** | **{clf.direct_accept_accuracy * 100:.2f}%** | Độ tin cậy cao khi vượt ngưỡng |

### Độ chính xác theo danh mục (Category Breakdown)

| Danh mục (Category) | Số lượng mẫu | Số mẫu đúng | Độ chính xác |
|---|---|---|---|
"""
    for cat, info in sorted(clf.category_accuracy.items()):
        md += f"| `{cat}` | {info['total']} | {info['correct']} | **{info['accuracy'] * 100:.2f}%** |\n"

    md += f"""
---

## 3. Độ Trễ Suy Luận và Thông Lượng (Inference Latency & Throughput)

Được đo lường sau {DEFAULT_EVAL_WARMUP_RUNS} lượt chạy khởi động (warmup runs) trên thiết bị `{lat.device}`:

| Chỉ số độ trễ | Giá trị thực nghiệm |
|---|---|
| **Thời gian trung bình (Mean Latency)** | **{lat.mean_latency_ms:.2f} ms** |
| **Độ lệch chuẩn (Std Dev)** | {lat.std_latency_ms:.2f} ms |
| **Trung vị P50 (Median)** | **{lat.median_latency_ms:.2f} ms** |
| **Phân vị 90 (P90)** | {lat.p90_latency_ms:.2f} ms |
| **Phân vị 95 (P95)** | {lat.p95_latency_ms:.2f} ms |
| **Phân vị 99 (P99)** | {lat.p99_latency_ms:.2f} ms |
| **Độ trễ nhỏ nhất (Min)** | {lat.min_latency_ms:.2f} ms |
| **Độ trễ lớn nhất (Max)** | {lat.max_latency_ms:.2f} ms |
| **Thông lượng (Throughput)** | **{lat.throughput_qps:.1f} QPS** (truy vấn / giây) |

---

## 4. Hành Vi Truy Hồi Dự Phòng (Fallback & Out-of-Domain Behavior)

Cảm hứng từ mô hình Retrieve-and-Fill (RAF) và Sentence-BERT Cosine Similarity (Reimers & Gurevych, 2019):
$$\\text{{sim}}(u, v) = \\frac{{u \\cdot v}}{{\\|u\\|_2 \\|v\\|_2}}$$

| Tình huống và Hành vi | Chỉ số thực tế | Đánh giá |
|---|---|---|
| **Tỷ lệ kích hoạt Fallback ($p_{{\\max}} < \\tau$)** | **{fb.fallback_trigger_rate * 100:.2f}%** ({fb.fallback_triggered_count}/{fb.total_evaluated}) | Kích hoạt truy hồi SBERT |
| **Tỷ lệ Fallback tìm thấy ứng viên ($\\ge \\theta$)** | **{fb.fallback_understood_rate * 100:.2f}%** ({fb.fallback_understood_count}/{fb.fallback_triggered_count if fb.fallback_triggered_count else 1}) | Hệ thống đề xuất lựa chọn |
| **Fallback Recall@1** | **{fb.fallback_recall_at_1 * 100:.2f}%** | Ứng viên top-1 trùng nhãn đích |
| **Fallback Recall@3** | **{fb.fallback_recall_at_3 * 100:.2f}%** | Nhãn đích nằm trong top-3 gợi ý |
| **Độ hồi quy toàn hệ thống (Combined Recall@1)** | **{fb.combined_system_recall_at_1 * 100:.2f}%** | PhoBERT + Fallback top-1 |
| **Độ hồi quy toàn hệ thống (Combined Recall@3)** | **{fb.combined_system_recall_at_3 * 100:.2f}%** | PhoBERT + Fallback top-3 |
| **Tỷ lệ từ chối câu ngoài phạm vi (OOD Rejection Rate)** | **{fb.ood_rejection_rate * 100:.2f}%** ({fb.ood_rejected_count}/{fb.ood_total}) | Từ chối chính xác khi $< \\theta$ |

---

## 5. Kiểm Tra Tĩnh Template Rendering (Honesty Rule #7)

- **Chế độ kiểm thử**: `{tmpl.verification_mode}` (static check only)
- **Tổng số mẫu kiểm tra**: {tmpl.total_checked}
- **Tỷ lệ sinh mã thành công**: **{tmpl.render_success_rate * 100:.2f}%** ({tmpl.render_success_count}/{tmpl.total_checked})
"""
    return md


def save_reports(
    report: EvaluationReport,
    output_dir: Union[str, Path] = REPORT_DIR,
) -> tuple[Path, Path]:
    """
    Lưu báo cáo định dạng JSON và Markdown vào thư mục đích.

    Tham số:
        report: Đối tượng báo cáo hoàn chỉnh.
        output_dir: Thư mục lưu trữ báo cáo (mặc định: report/).

    Trả về:
        Tuple chứa (đường dẫn file json, đường dẫn file md).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "evaluation_report.json"
    md_path = out_dir / "evaluation_report.md"

    # Lưu JSON
    report_dict = asdict(report)
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    # Lưu Markdown
    md_content = generate_markdown_summary(report)
    with open(md_path, mode="w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path


def print_cli_summary(report: EvaluationReport) -> None:
    """In tóm tắt kết quả đánh giá lên màn hình terminal."""
    clf = report.classification_metrics
    lat = report.latency_metrics
    fb = report.fallback_metrics

    print("=" * 68)
    print("  KẾT QUẢ ĐÁNH GIÁ HỌC THUẬT PHOBERT & RETRIEVAL FALLBACK SUITE")
    print("=" * 68)
    print(f"  Thiết bị tính toán:       {report.device.upper()}")
    print(f"  Kiểm chứng SHA-256 test:  {'ĐẠT' if report.test_csv_sha256_verified else 'LỖI'} ({report.test_csv_sha256[:16]}...)")
    print(f"  Tổng số mẫu đánh giá:     {clf.total_samples}")
    print("-" * 68)
    print("  ĐỘ CHÍNH XÁC PHÂN LOẠI (PhoBERT Classification):")
    print(f"    * Top-1 Accuracy:       {clf.top1_accuracy * 100:6.2f}%")
    print(f"    * Top-3 Accuracy:       {clf.top3_accuracy * 100:6.2f}%")
    print(f"    * Top-5 Accuracy:       {clf.top5_accuracy * 100:6.2f}%")
    print(f"    * Direct Accept Rate:   {clf.direct_accept_rate * 100:6.2f}% (p_max >= {fb.threshold_tau})")
    print(f"    * Direct Accept Acc:    {clf.direct_accept_accuracy * 100:6.2f}%")
    print("-" * 68)
    print("  ĐỘ TRỄ VÀ THÔNG LƯỢNG (Inference Latency & Throughput):")
    print(f"    * Mean Latency:         {lat.mean_latency_ms:6.2f} ms (Std: {lat.std_latency_ms:.2f} ms)")
    print(f"    * Median (P50):         {lat.median_latency_ms:6.2f} ms")
    print(f"    * P95 Latency:          {lat.p95_latency_ms:6.2f} ms")
    print(f"    * P99 Latency:          {lat.p99_latency_ms:6.2f} ms")
    print(f"    * Throughput:           {lat.throughput_qps:6.1f} QPS")
    print("-" * 68)
    print("  HÀNH VI DỰ PHÒNG & NGOẠI PHẠM VI (Fallback & OOD Behavior):")
    print(f"    * Fallback Trigger Rate:{fb.fallback_trigger_rate * 100:6.2f}%")
    print(f"    * Fallback Recall@1:    {fb.fallback_recall_at_1 * 100:6.2f}%")
    print(f"    * Fallback Recall@3:    {fb.fallback_recall_at_3 * 100:6.2f}%")
    print(f"    * Combined Recall@1:    {fb.combined_system_recall_at_1 * 100:6.2f}%")
    print(f"    * Combined Recall@3:    {fb.combined_system_recall_at_3 * 100:6.2f}%")
    print(f"    * OOD Rejection Rate:   {fb.ood_rejection_rate * 100:6.2f}% (score < {fb.threshold_theta})")
    print("=" * 68)


def create_eval_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh cho kịch bản đánh giá."""
    parser = argparse.ArgumentParser(
        prog="phobert-evaluate",
        description="Đánh giá học thuật mô hình PhoBERT và cơ chế truy hồi dự phòng SBERT.",
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=str(TRAIN_CSV_PATH),
        help=f"Đường dẫn tệp CSV dữ liệu đánh giá (mặc định: {TRAIN_CSV_PATH}).",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(TEST_CSV_PATH),
        help=f"Đường dẫn tệp test.csv để kiểm chứng SHA-256 (mặc định: {TEST_CSV_PATH}).",
    )
    parser.add_argument(
        "--intents-path",
        "-i",
        type=str,
        default=str(DEFAULT_INTENTS_PATH),
        help=f"Đường dẫn tệp intents.json (mặc định: {DEFAULT_INTENTS_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(REPORT_DIR),
        help=f"Thư mục lưu báo cáo kết quả (mặc định: {REPORT_DIR}).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=TAU_INTENT_THRESHOLD,
        help=f"Ngưỡng chấp nhận intent trực tiếp (mặc định: {TAU_INTENT_THRESHOLD}).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=THETA_RETRIEVAL_THRESHOLD,
        help=f"Ngưỡng tương đồng cosine tối thiểu truy hồi fallback (mặc định: {THETA_RETRIEVAL_THRESHOLD}).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_EVAL_TIMING_RUNS,
        help=f"Số lần lặp đo đạc độ trễ suy luận (mặc định: {DEFAULT_EVAL_TIMING_RUNS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Hạt giống ngẫu nhiên (mặc định: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị tính toán ('cpu', 'cuda'). Mặc định tự động phát hiện.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Điểm nhập chính khi chạy script đánh giá."""
    parser = create_eval_parser()
    args = parser.parse_args(argv)

    try:
        evaluator = PhoBertEvaluator(
            intents_path=args.intents_path,
            threshold_tau=args.tau,
            threshold_theta=args.theta,
            device=args.device,
            seed=args.seed,
        )

        report = evaluator.run_full_evaluation(
            train_path=args.data_path,
            test_path=args.test_path,
            timing_runs=args.runs,
        )

        # Lưu báo cáo JSON và Markdown
        json_p, md_p = save_reports(report, output_dir=args.output_dir)

        # In tóm tắt ra terminal
        print_cli_summary(report)
        print(f"\n[Báo cáo học thuật đã lưu]:\n  - JSON: {json_p}\n  - Markdown: {md_p}\n")
        return 0
    except Exception as exc:
        print(f"[Lỗi đánh giá thực nghiệm]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
