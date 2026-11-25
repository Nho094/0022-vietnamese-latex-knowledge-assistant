"""
Đường ống huấn luyện và đánh giá các mô hình đường cơ sở (Baseline Models Pipeline).

Bao gồm:
1. Mô hình phân loại thống kê TF-IDF n-gram (unigrams + bigrams) kết hợp LinearSVC (Vapnik, 1995; Fan et al., 2008).
2. Mô hình khớp mờ (Fuzzy String Matching) dựa trên danh sách bí danh (aliases) từ ngân hàng kịch bản intents.json
   sử dụng thuật toán Token Set Ratio (rapidfuzz/fuzzywuzzy).

Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Identifiers bằng tiếng Anh; chú thích (comments/docstrings) bằng tiếng Việt giải thích lý do (WHY) và công thức bài báo khoa học.
- Toàn bộ hằng số cấu hình được nạp từ core/config.py (không dùng số ma thuật).
- Thiết lập seed ngẫu nhiên tất định thông qua core.config.set_seed.
- Quy tắc Trung thực #3 (Honesty Rule #3):
  + Tinh chỉnh siêu tham số regularization C từ [0.1, 1, 10] CHỈ dựa trên hiệu năng tập Validation.
  + Tuyệt đối không kiểm tra dữ liệu Test trong quá trình tìm C.
  + Kiểm tra và đối chiếu mã băm SHA-256 của data/test.csv trước khi đánh giá kiểm thử một lần duy nhất.
- Hỗ trợ cờ lựa chọn mục tiêu phân loại: 'intent' (189 lớp) hoặc 'category' (12 nhóm).
- Tính toán đầy đủ chỉ số đánh giá, khoảng tin cậy Bootstrap 95%, kiểm định McNemar và trực quan hóa ma trận nhầm lẫn
  bằng module train/evaluate.py, xuất kết quả ra report/baseline_results.json và report/figures/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# Đảm bảo đường dẫn gốc dự án có trong sys.path khi chạy script trực tiếp
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

try:
    from rapidfuzz import fuzz

    HAS_RAPIDFUZZ: bool = True
except ImportError:
    HAS_RAPIDFUZZ = False

from core.config import (
    BASELINE_C_CANDIDATES,
    DEFAULT_BASELINE_C,
    DEFAULT_BASELINE_FUZZY_CM_PATH,
    DEFAULT_BASELINE_RESULTS_PATH,
    DEFAULT_BASELINE_SVM_CM_PATH,
    DEFAULT_BOOTSTRAP_ROUNDS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_INTENTS_PATH,
    DEFAULT_SEED,
    DEFAULT_SVM_MAX_ITER,
    DEFAULT_TFIDF_NGRAM_RANGE,
    DEFAULT_TFIDF_SUBLINEAR_TF,
    DEFAULT_ZERO_DIVISION,
    SUPPORTED_CLASSIFICATION_TARGETS,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    set_seed,
)
from core.preprocess import normalize, prepare_for_phobert, preprocess_fingerprint
from train.evaluate import (
    compute_accuracy,
    compute_classification_metrics,
    compute_macro_f1,
    compute_mcnemar_test,
    compute_paired_bootstrap_ci,
    compute_weighted_f1,
    generate_confusion_matrix_figure,
    load_intent_labels,
)

# =====================================================================
# 1. Cấu trúc dữ liệu và Nạp tập dữ liệu (Dataset Loading)
# =====================================================================


@dataclass
class BaselineSample:
    """
    Biểu diễn một mẫu quan sát dữ liệu văn bản tiếng Việt kèm nhãn ý định và danh mục.
    """

    text: str
    label: str
    category: str


def compute_file_sha256(file_path: Path | str) -> str:
    """
    Tính mã băm SHA-256 tất định của tệp dữ liệu.

    Tuân thủ Honesty Rule #3 (AGENTS.md):
    Lưu vết và kiểm tra tính toàn vẹn của tệp data/test.csv đảm bảo không bị can thiệp.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_baseline_samples(
    csv_path: Path | str,
    max_samples: int | None = None,
) -> list[BaselineSample]:
    """
    Đọc danh sách mẫu từ tệp CSV với mã hóa UTF-8-SIG (xử lý an toàn BOM \\ufeff).

    Tham số:
        csv_path: Đường dẫn tệp CSV dữ liệu.
        max_samples: Giới hạn số lượng mẫu tối đa (dùng cho chế độ smoke test).

    Trả về:
        Danh sách các đối tượng BaselineSample.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Tệp CSV không tồn tại: {path}")

    samples: list[BaselineSample] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label = row.get("label", "").strip()
            category = row.get("category", "").strip()
            if text and label:
                samples.append(BaselineSample(text=text, label=label, category=category))
                if max_samples is not None and len(samples) >= max_samples:
                    break
    return samples


def preprocess_texts_for_baseline(
    texts: Sequence[str],
    lowercase: bool = True,
) -> list[str]:
    """
    Tiền xử lý chuỗi văn bản đầu vào qua chuẩn hóa chữ thường và phân đoạn từ tiếng Việt.

    Quy trình xử lý:
    1. Chuẩn hóa Unicode dựng sẵn NFC và quy đổi các ký hiệu toán học tương đồng (math look-alikes).
    2. Chuyển thành chữ thường (lowercase=True) theo yêu cầu bài toán baseline để giảm chiều không gian từ vựng.
    3. Phân đoạn từ tiếng Việt bằng thư viện pyvi với cơ chế bảo vệ biểu thức toán học và lệnh LaTeX.

    Tham số:
        texts: Danh sách chuỗi văn bản thô.
        lowercase: Cờ chuyển đổi chữ thường (mặc định True cho TF-IDF).

    Trả về:
        Danh sách chuỗi văn bản sau tiền xử lý và phân đoạn từ.
    """
    return [prepare_for_phobert(t, lowercase=lowercase) for t in texts]


# =====================================================================
# 2. Mô hình Baseline 1: TF-IDF Vectorizer + LinearSVC
# =====================================================================


class TfidfLinearSvcClassifier:
    """
    Bộ phân loại văn bản kết hợp trích xuất đặc trưng TF-IDF n-gram (sublinear TF)
    và bộ phân loại Support Vector Classifier tuyến tính (LinearSVC - Fan et al., 2008).

    Cơ sở lý thuyết và công thức:
    - TF dưới tuyến tính (Sublinear Term Frequency scaling):
        tf_{sublinear}(t, d) = 1 + \\log(tf(t, d)) \\text{ nếu } tf(t, d) > 0, \\text{ ngược lại } 0
      Giúp giảm ảnh hưởng quá mức của các từ xuất hiện lặp lại nhiều lần trong câu dài.
    - Đặc trưng n-gram: kết hợp unigrams và bigrams (ngram_range=(1, 2)) để nắm bắt cả từ đơn
      và cặp từ ghép tiếng Việt (ví dụ: 'đạo_hàm', 'đạo_hàm riêng').
    - Hàm mất mát LinearSVC: Hinge loss l2-regularized cực tiểu hóa:
        \\min_w \\frac{1}{2} w^T w + C \\sum_{i=1}^n \\max(0, 1 - y_i w^T x_i)
    """

    def __init__(
        self,
        c: float = DEFAULT_BASELINE_C,
        ngram_range: tuple[int, int] = DEFAULT_TFIDF_NGRAM_RANGE,
        sublinear_tf: bool = DEFAULT_TFIDF_SUBLINEAR_TF,
        max_iter: int = DEFAULT_SVM_MAX_ITER,
        random_state: int = DEFAULT_SEED,
    ) -> None:
        """
        Khởi tạo pipeline vector hóa TF-IDF và LinearSVC.

        Tham số:
            c: Hệ số phạt điều chuẩn (regularization parameter C).
            ngram_range: Khoảng n-gram đặc trưng (mặc định (1, 2)).
            sublinear_tf: Áp dụng co giãn logarit tần suất từ (sublinear tf).
            max_iter: Số vòng lặp tối đa của bộ giải LinearSVC.
            random_state: Hạt giống ngẫu nhiên đảm bảo tính tái lập.
        """
        self.c = c
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.max_iter = max_iter
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.classifier = LinearSVC(
            C=self.c,
            max_iter=self.max_iter,
            random_state=self.random_state,
            dual="auto",
        )
        self.is_fitted: bool = False

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> TfidfLinearSvcClassifier:
        """
        Huấn luyện bộ vector hóa TF-IDF và mô hình LinearSVC trên tập dữ liệu truyền vào.

        Tham số:
            texts: Danh sách văn bản đã qua tiền xử lý.
            labels: Danh sách nhãn mục tiêu tương ứng.

        Trả về:
            Chính đối tượng sau khi khớp trọng số.
        """
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_fitted = True
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        """
        Dự đoán nhãn cho danh sách văn bản kiểm thử.

        Tham số:
            texts: Danh sách văn bản cần suy luận.

        Trả về:
            Danh sách nhãn dự đoán dạng chuỗi.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình TfidfLinearSvcClassifier chưa được huấn luyện qua phương thức fit().")
        X = self.vectorizer.transform(texts)
        preds: np.ndarray = self.classifier.predict(X)
        return [str(p) for p in preds]


def tune_linear_svc_c(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    val_texts: Sequence[str],
    val_labels: Sequence[str],
    c_candidates: Sequence[float] = BASELINE_C_CANDIDATES,
    eval_labels: Sequence[str] | None = None,
    ngram_range: tuple[int, int] = DEFAULT_TFIDF_NGRAM_RANGE,
    sublinear_tf: bool = DEFAULT_TFIDF_SUBLINEAR_TF,
    max_iter: int = DEFAULT_SVM_MAX_ITER,
    random_state: int = DEFAULT_SEED,
) -> tuple[float, dict[str, dict[str, float]]]:
    """
    Tìm kiếm siêu tham số điều chuẩn C cho LinearSVC strictly trên tập Validation.

    Tuân thủ Honesty Rule #3 (AGENTS.md):
    - Khớp TF-IDF trên tập Train và chuyển đổi văn bản tập Validation.
    - Duyệt qua từng giá trị C trong danh sách c_candidates [0.1, 1, 10].
    - Đo lường độ chính xác Accuracy và Macro-F1 trên tập Validation.
    - Lựa chọn C tối ưu dựa trên điểm Validation Macro-F1 (phù hợp với bài toán phân loại đa lớp mất cân bằng).
    - Tuyệt đối không kiểm tra dữ liệu Test trong bước này.

    Tham số:
        train_texts: Văn bản tập huấn luyện (đã tiền xử lý).
        train_labels: Nhãn tập huấn luyện.
        val_texts: Văn bản tập kiểm thực (đã tiền xử lý).
        val_labels: Nhãn tập kiểm thực.
        c_candidates: Danh sách các giá trị C ứng viên.
        eval_labels: Toàn bộ danh sách nhãn đánh giá (cho Macro-F1 schema).
        ngram_range: Cấu hình n-gram TF-IDF.
        sublinear_tf: Cờ sublinear TF.
        max_iter: Số vòng lặp tối đa.
        random_state: Hạt giống ngẫu nhiên.

    Trả về:
        Tuple gồm (best_c, tuning_history_dict).
    """
    # Khớp trước bộ trích xuất đặc trưng TF-IDF trên tập Train
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        token_pattern=r"(?u)\b\w+\b",
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    tuning_history: dict[str, dict[str, float]] = {}
    best_c: float = float(c_candidates[0])
    best_val_macro_f1: float = -1.0
    best_val_accuracy: float = -1.0

    for c_val in c_candidates:
        c_float = float(c_val)
        svc = LinearSVC(
            C=c_float,
            max_iter=max_iter,
            random_state=random_state,
            dual="auto",
        )
        svc.fit(X_train, train_labels)
        val_preds_arr: np.ndarray = svc.predict(X_val)
        val_preds = [str(p) for p in val_preds_arr]

        acc = compute_accuracy(val_labels, val_preds)
        macro_f1 = compute_macro_f1(val_labels, val_preds, labels=eval_labels)
        weighted_f1 = compute_weighted_f1(val_labels, val_preds, labels=eval_labels)

        tuning_history[str(c_float)] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        }

        # Tiêu chí chọn C: Ưu tiên Macro-F1 cao nhất, nếu hòa thì xét Accuracy
        if (macro_f1 > best_val_macro_f1) or (
            abs(macro_f1 - best_val_macro_f1) < 1e-9 and acc > best_val_accuracy
        ):
            best_val_macro_f1 = macro_f1
            best_val_accuracy = acc
            best_c = c_float

    return best_c, tuning_history


# =====================================================================
# 3. Mô hình Baseline 2: Khớp Mờ Chuỗi Bí Danh (Fuzzy Matching Baseline)
# =====================================================================


class FuzzyAliasClassifier:
    """
    Bộ phân loại cơ sở sử dụng kỹ thuật so khớp mờ (Fuzzy String Matching)
    đối chiếu câu truy vấn đầu vào với danh sách bí danh (aliases) trong intents.json.

    Lý do thiết kế và cơ sở thuật toán (WHY):
    1. Token Set Ratio (rapidfuzz.fuzz.token_set_ratio):
       Tách chuỗi truy vấn và bí danh thành các tập hợp token riêng biệt:
       Intersection T_{inter} = S_1 \\cap S_2, Diff T_1 = S_1 \\setminus S_2, Diff T_2 = S_2 \\setminus S_1.
       Tạo các chuỗi kết hợp:
       u_0 = T_{inter}, u_1 = T_{inter} + T_1, u_2 = T_{inter} + T_2.
       Điểm tương đồng là giá trị lớn nhất:
       \\text{score} = \\max( \\text{ratio}(u_0, u_1), \\text{ratio}(u_0, u_2), \\text{ratio}(u_1, u_2) )
       Thuật toán này cực kỳ bền vững trước các từ đệm mở đầu câu đàm thoại tiếng Việt
       (ví dụ: 'tôi muốn tạo', 'hãy giúp mình') khi đối chiếu với các alias ngắn gọn của kịch bản.
    2. Cơ chế giải quyết thế hòa tất định (Deterministic Tie-Breaking):
       Khi nhiều alias đạt cùng điểm token_set_ratio cực đại (thường gặp khi alias là tập con hoàn hảo của câu),
       sử dụng tỷ lệ Levenshtein cơ bản (fuzz.ratio) làm trọng số phụ để ưu tiên alias có độ tương đồng
       chiều dài và cấu trúc sát nhất với câu truy vấn.
    """

    def __init__(self, intents_json_path: Path | str = DEFAULT_INTENTS_PATH) -> None:
        """
        Khởi tạo bộ dữ liệu bí danh từ tệp cấu hình kịch bản intents.json.

        Tham số:
            intents_json_path: Đường dẫn tệp JSON kịch bản.
        """
        self.intents_json_path = Path(intents_json_path)
        if not self.intents_json_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp intent schema: {self.intents_json_path}")

        with open(self.intents_json_path, mode="r", encoding="utf-8") as f:
            intents_data: list[dict[str, Any]] = json.load(f)

        # Lưu trữ cấu trúc bản ghi bí danh: (normalized_alias, intent, category)
        self.alias_records: list[tuple[str, str, str]] = []
        for item in intents_data:
            intent = str(item.get("intent", "")).strip()
            category = str(item.get("category", "")).strip()
            aliases = item.get("aliases", [])
            for alias in aliases:
                norm_alias = normalize(str(alias).strip(), lowercase=True)
                if norm_alias:
                    self.alias_records.append((norm_alias, intent, category))

        # Danh sách chuỗi bí danh phục vụ truy vấn
        self.alias_strings: list[str] = [rec[0] for rec in self.alias_records]

    def predict_single(self, text: str, target: str = "intent") -> str:
        """
        Dự đoán nhãn (intent hoặc category) cho một câu truy vấn văn bản.

        Tham số:
            text: Chuỗi văn bản tiếng Việt.
            target: Mục tiêu phân loại ('intent' hoặc 'category').

        Trả về:
            Tên nhãn dự đoán tốt nhất.
        """
        if not self.alias_records:
            return ""

        query_norm = normalize(text, lowercase=True)

        best_score: float = -1.0
        best_intent: str = self.alias_records[0][1]
        best_category: str = self.alias_records[0][2]

        if HAS_RAPIDFUZZ:
            # Quét tìm kiếm kết hợp token_set_ratio và ratio phụ
            for norm_alias, intent, category in self.alias_records:
                primary_score = float(fuzz.token_set_ratio(query_norm, norm_alias))
                # Trọng số phụ 1e-4 phá vỡ thế hòa ưu tiên chiều dài khớp
                secondary_score = float(fuzz.ratio(query_norm, norm_alias))
                total_score = primary_score + 1e-4 * secondary_score

                if total_score > best_score:
                    best_score = total_score
                    best_intent = intent
                    best_category = category
        else:
            # Fallback đơn giản dựa trên tập token khi thiếu rapidfuzz
            query_tokens = set(query_norm.split())
            for norm_alias, intent, category in self.alias_records:
                alias_tokens = set(norm_alias.split())
                if not alias_tokens:
                    continue
                intersection = len(query_tokens & alias_tokens)
                jaccard = float(intersection / len(query_tokens | alias_tokens))
                if jaccard > best_score:
                    best_score = jaccard
                    best_intent = intent
                    best_category = category

        return best_intent if target == "intent" else best_category

    def predict(self, texts: Sequence[str], target: str = "intent") -> list[str]:
        """
        Dự đoán hàng loạt cho danh sách câu văn bản.

        Tham số:
            texts: Danh sách văn bản truy vấn.
            target: 'intent' hoặc 'category'.

        Trả về:
            Danh sách nhãn dự đoán.
        """
        return [self.predict_single(t, target=target) for t in texts]


# =====================================================================
# 4. Đường ống Thực nghiệm Hoàn chỉnh (End-to-End Pipeline)
# =====================================================================


def run_baseline_pipeline(
    target: str = "intent",
    c_candidates: Sequence[float] = BASELINE_C_CANDIDATES,
    train_path: Path | str = TRAIN_CSV_PATH,
    val_path: Path | str = VAL_CSV_PATH,
    test_path: Path | str = TEST_CSV_PATH,
    intents_path: Path | str = DEFAULT_INTENTS_PATH,
    results_path: Path | str = DEFAULT_BASELINE_RESULTS_PATH,
    svm_cm_path: Path | str = DEFAULT_BASELINE_SVM_CM_PATH,
    fuzzy_cm_path: Path | str = DEFAULT_BASELINE_FUZZY_CM_PATH,
    num_bootstrap: int = DEFAULT_BOOTSTRAP_ROUNDS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
    smoke_mode: bool = False,
    smoke_samples: int = 32,
) -> dict[str, Any]:
    """
    Thực thi toàn bộ đường ống huấn luyện mô hình đường cơ sở TF-IDF LinearSVC
    và Fuzzy Matching, tính toán độ đo thống kê và xuất báo cáo kết quả.

    Tuân thủ quy trình kiểm thử nghiêm ngặt:
    1. Kiểm tra tính toàn vẹn và ghi nhận SHA-256 của tập data/test.csv (Honesty Rule #3).
    2. Nạp dữ liệu Train, Validation, Test.
    3. Tiền xử lý dữ liệu (chuẩn hóa NFC, chuyển chữ thường, tách từ pyvi bảo vệ toán/LaTeX).
    4. Tìm kiếm siêu tham số C trên tập Validation strictly mà không chạm đến tập Test.
    5. Khớp mô hình LinearSVC tối ưu trên tập Train và đánh giá MỘT LẦN DUY NHẤT trên tập Test.
    6. Đánh giá mô hình Fuzzy Matching đối chiếu bí danh trên tập Test.
    7. Tính toán Classification Metrics (Accuracy, Macro-F1, Weighted-F1, Per-class metrics)
       và Bootstrap 95% Confidence Intervals (Efron & Tibshirani, 1993).
    8. Thực hiện kiểm định McNemar (1947) có hiệu chỉnh liên tục Yates so sánh 2 mô hình đường cơ sở.
    9. Vẽ và lưu trữ biểu đồ Confusion Matrix của cả hai mô hình vào report/figures/.
    10. Ghi dữ liệu ra tệp JSON report/baseline_results.json.

    Tham số:
        target: Mục tiêu phân loại ('intent' hoặc 'category').
        c_candidates: Danh sách giá trị C điều chuẩn cần thử nghiệm.
        train_path, val_path, test_path: Đường dẫn các tệp CSV phân chia dữ liệu.
        intents_path: Đường dẫn tệp JSON ngân hàng kịch bản.
        results_path: Đường dẫn tệp kết quả JSON đầu ra.
        svm_cm_path, fuzzy_cm_path: Đường dẫn lưu ảnh ma trận nhầm lẫn PNG.
        num_bootstrap: Số lượt lấy mẫu Bootstrap ước lượng khoảng tin cậy.
        confidence_level: Mức độ tin cậy Bootstrap (mặc định 0.95).
        seed: Hạt giống ngẫu nhiên tất định.
        smoke_mode: Chế độ kiểm thử nhanh với tập mẫu thu nhỏ.
        smoke_samples: Số lượng mẫu thu nhỏ trong chế độ smoke mode.

    Trả về:
        Dictionary chứa toàn bộ chỉ số đo lường học thuật và siêu tham số thực nghiệm.
    """
    if target not in SUPPORTED_CLASSIFICATION_TARGETS:
        raise ValueError(
            f"Mục tiêu phân loại '{target}' không được hỗ trợ. "
            f"Danh sách hợp lệ: {list(SUPPORTED_CLASSIFICATION_TARGETS)}"
        )

    # Thiết lập hạt giống ngẫu nhiên tất định
    set_seed(seed)

    train_file = Path(train_path)
    val_file = Path(val_path)
    test_file = Path(test_path)
    intents_file = Path(intents_path)
    out_results = Path(results_path)
    out_svm_cm = Path(svm_cm_path)
    out_fuzzy_cm = Path(fuzzy_cm_path)

    # 1. Tính toán mã băm SHA-256 của tập kiểm thử theo Honesty Rule #3
    test_csv_sha256 = compute_file_sha256(test_file)

    # 2. Đọc dữ liệu mẫu từ CSV
    max_limit = smoke_samples if smoke_mode else None
    train_samples = read_baseline_samples(train_file, max_samples=max_limit)
    val_samples = read_baseline_samples(val_file, max_samples=max_limit)
    test_samples = read_baseline_samples(test_file, max_samples=max_limit)

    # Xác định danh sách nhãn đánh giá tương ứng mục tiêu
    if target == "intent":
        schema_labels: list[str] = load_intent_labels(intents_file)
        eval_labels: list[str] = schema_labels
    else:
        # Category: tập hợp 12 danh mục ngữ nghĩa được sắp xếp bảng chữ cái
        cat_set: set[str] = set()
        for s in train_samples + val_samples + test_samples:
            if s.category:
                cat_set.add(s.category)
        eval_labels = sorted(cat_set)

    # 3. Tiền xử lý văn bản: chuẩn hóa chữ thường và phân đoạn từ pyvi bảo vệ toán/LaTeX
    train_raw_texts = [s.text for s in train_samples]
    val_raw_texts = [s.text for s in val_samples]
    test_raw_texts = [s.text for s in test_samples]

    train_labels = [s.label if target == "intent" else s.category for s in train_samples]
    val_labels = [s.label if target == "intent" else s.category for s in val_samples]
    test_labels = [s.label if target == "intent" else s.category for s in test_samples]

    train_preprocessed = preprocess_texts_for_baseline(train_raw_texts, lowercase=True)
    val_preprocessed = preprocess_texts_for_baseline(val_raw_texts, lowercase=True)
    test_preprocessed = preprocess_texts_for_baseline(test_raw_texts, lowercase=True)

    # Ghi nhận vân tay tiền xử lý (preprocess fingerprint)
    fingerprint = preprocess_fingerprint(lowercase=True)

    # 4. Tinh chỉnh tham số C của LinearSVC strictly trên tập Validation
    best_c, c_tuning_history = tune_linear_svc_c(
        train_texts=train_preprocessed,
        train_labels=train_labels,
        val_texts=val_preprocessed,
        val_labels=val_labels,
        c_candidates=c_candidates,
        eval_labels=eval_labels,
        ngram_range=DEFAULT_TFIDF_NGRAM_RANGE,
        sublinear_tf=DEFAULT_TFIDF_SUBLINEAR_TF,
        max_iter=DEFAULT_SVM_MAX_ITER,
        random_state=seed,
    )

    # 5. Huấn luyện mô hình LinearSVC hoàn chỉnh trên tập Train với best_c và đánh giá test
    svm_baseline = TfidfLinearSvcClassifier(
        c=best_c,
        ngram_range=DEFAULT_TFIDF_NGRAM_RANGE,
        sublinear_tf=DEFAULT_TFIDF_SUBLINEAR_TF,
        max_iter=DEFAULT_SVM_MAX_ITER,
        random_state=seed,
    )
    svm_baseline.fit(train_preprocessed, train_labels)

    # Đánh giá DUY NHẤT một lần trên tập kiểm thử Test
    svm_test_preds = svm_baseline.predict(test_preprocessed)

    # 6. Đánh giá mô hình Fuzzy Alias Matching trên tập kiểm thử Test
    fuzzy_baseline = FuzzyAliasClassifier(intents_json_path=intents_file)
    fuzzy_test_preds = fuzzy_baseline.predict(test_raw_texts, target=target)

    # 7. Tính toán chỉ số phân loại chi tiết (Classification Metrics)
    svm_metrics = compute_classification_metrics(
        y_true=test_labels,
        y_pred=svm_test_preds,
        labels=eval_labels,
        zero_division=DEFAULT_ZERO_DIVISION,
    )
    fuzzy_metrics = compute_classification_metrics(
        y_true=test_labels,
        y_pred=fuzzy_test_preds,
        labels=eval_labels,
        zero_division=DEFAULT_ZERO_DIVISION,
    )

    # 8. Ước lượng khoảng tin cậy Bootstrap 95% phi tham số (Percentile Method)
    # Số vòng bootstrap thu nhỏ trong chế độ smoke mode để tối ưu thời gian chạy
    eff_bootstrap_rounds = 20 if smoke_mode else num_bootstrap

    svm_acc_ci = compute_paired_bootstrap_ci(
        y_true=test_labels,
        y_pred=svm_test_preds,
        metric_fn=compute_accuracy,
        num_bootstrap=eff_bootstrap_rounds,
        confidence_level=confidence_level,
        seed=seed,
    )
    svm_macro_f1_ci = compute_paired_bootstrap_ci(
        y_true=test_labels,
        y_pred=svm_test_preds,
        metric_fn=lambda yt, yp: compute_macro_f1(yt, yp, labels=eval_labels),
        num_bootstrap=eff_bootstrap_rounds,
        confidence_level=confidence_level,
        seed=seed,
    )
    svm_weighted_f1_ci = compute_paired_bootstrap_ci(
        y_true=test_labels,
        y_pred=svm_test_preds,
        metric_fn=lambda yt, yp: compute_weighted_f1(yt, yp, labels=eval_labels),
        num_bootstrap=eff_bootstrap_rounds,
        confidence_level=confidence_level,
        seed=seed,
    )

    fuzzy_acc_ci = compute_paired_bootstrap_ci(
        y_true=test_labels,
        y_pred=fuzzy_test_preds,
        metric_fn=compute_accuracy,
        num_bootstrap=eff_bootstrap_rounds,
        confidence_level=confidence_level,
        seed=seed,
    )
    fuzzy_macro_f1_ci = compute_paired_bootstrap_ci(
        y_true=test_labels,
        y_pred=fuzzy_test_preds,
        metric_fn=lambda yt, yp: compute_macro_f1(yt, yp, labels=eval_labels),
        num_bootstrap=eff_bootstrap_rounds,
        confidence_level=confidence_level,
        seed=seed,
    )
    fuzzy_weighted_f1_ci = compute_paired_bootstrap_ci(
        y_true=test_labels,
        y_pred=fuzzy_test_preds,
        metric_fn=lambda yt, yp: compute_weighted_f1(yt, yp, labels=eval_labels),
        num_bootstrap=eff_bootstrap_rounds,
        confidence_level=confidence_level,
        seed=seed,
    )

    # 9. Kiểm định McNemar so sánh TF-IDF SVM vs Fuzzy Matching
    mcnemar_stat, mcnemar_p = compute_mcnemar_test(
        y_true=test_labels,
        y_pred_a=svm_test_preds,
        y_pred_b=fuzzy_test_preds,
        continuity_correction=True,
    )
    b_count = sum(
        ya == yt and yb != yt
        for yt, ya, yb in zip(test_labels, svm_test_preds, fuzzy_test_preds)
    )
    c_count = sum(
        ya != yt and yb == yt
        for yt, ya, yb in zip(test_labels, svm_test_preds, fuzzy_test_preds)
    )

    # 10. Trực quan hóa biểu đồ ma trận nhầm lẫn (Confusion Matrix Figures)
    out_svm_cm.parent.mkdir(parents=True, exist_ok=True)
    out_fuzzy_cm.parent.mkdir(parents=True, exist_ok=True)

    svm_cm_title = (
        f"Ma Trận Nhầm Lẫn TF-IDF LinearSVC ({target.capitalize()})"
    )
    fuzzy_cm_title = (
        f"Ma Trận Nhầm Lẫn Fuzzy Matching ({target.capitalize()})"
    )

    generate_confusion_matrix_figure(
        y_true=test_labels,
        y_pred=svm_test_preds,
        labels=eval_labels,
        output_path=out_svm_cm,
        title=svm_cm_title,
    )
    generate_confusion_matrix_figure(
        y_true=test_labels,
        y_pred=fuzzy_test_preds,
        labels=eval_labels,
        output_path=out_fuzzy_cm,
        title=fuzzy_cm_title,
    )

    # 11. Đóng gói kết quả đầu ra
    results_payload: dict[str, Any] = {
        "target": target,
        "test_csv_sha256": test_csv_sha256,
        "preprocessing_fingerprint": fingerprint,
        "data_split_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "tfidf_svm": {
            "c_tuning_validation_history": c_tuning_history,
            "best_c": best_c,
            "sublinear_tf": DEFAULT_TFIDF_SUBLINEAR_TF,
            "ngram_range": list(DEFAULT_TFIDF_NGRAM_RANGE),
            "test_metrics": {
                "accuracy": svm_metrics.accuracy,
                "macro_f1": svm_metrics.macro_f1,
                "weighted_f1": svm_metrics.weighted_f1,
                "total_classes": svm_metrics.total_classes,
                "total_samples": svm_metrics.total_samples,
                "per_class": svm_metrics.per_class,
            },
            "test_bootstrap_ci_95": {
                "accuracy": [svm_acc_ci[0], svm_acc_ci[1]],
                "macro_f1": [svm_macro_f1_ci[0], svm_macro_f1_ci[1]],
                "weighted_f1": [svm_weighted_f1_ci[0], svm_weighted_f1_ci[1]],
            },
            "confusion_matrix_figure": str(out_svm_cm),
        },
        "fuzzy_matching": {
            "method": "rapidfuzz_token_set_ratio",
            "test_metrics": {
                "accuracy": fuzzy_metrics.accuracy,
                "macro_f1": fuzzy_metrics.macro_f1,
                "weighted_f1": fuzzy_metrics.weighted_f1,
                "total_classes": fuzzy_metrics.total_classes,
                "total_samples": fuzzy_metrics.total_samples,
                "per_class": fuzzy_metrics.per_class,
            },
            "test_bootstrap_ci_95": {
                "accuracy": [fuzzy_acc_ci[0], fuzzy_acc_ci[1]],
                "macro_f1": [fuzzy_macro_f1_ci[0], fuzzy_macro_f1_ci[1]],
                "weighted_f1": [fuzzy_weighted_f1_ci[0], fuzzy_weighted_f1_ci[1]],
            },
            "confusion_matrix_figure": str(out_fuzzy_cm),
        },
        "statistical_comparison": {
            "mcnemar_test": {
                "statistic": float(mcnemar_stat),
                "p_value": float(mcnemar_p),
                "b": int(b_count),
                "c": int(c_count),
            },
        },
    }

    # Xuất ra tệp JSON
    out_results.parent.mkdir(parents=True, exist_ok=True)
    with open(out_results, mode="w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    return results_payload


# =====================================================================
# 5. Giao diện Dòng lệnh (CLI Entry Point)
# =====================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh cho kịch bản baseline."""
    parser = argparse.ArgumentParser(
        description="Đường ống huấn luyện và đánh giá mô hình Baseline TF-IDF LinearSVC & Fuzzy Matching.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="intent",
        choices=SUPPORTED_CLASSIFICATION_TARGETS,
        help="Mục tiêu phân loại ('intent' hoặc 'category'). Mặc định là 'intent'.",
    )
    parser.add_argument(
        "--c-candidates",
        type=float,
        nargs="+",
        default=list(BASELINE_C_CANDIDATES),
        help="Danh sách các giá trị regularization C cần tìm kiếm trên tập Validation.",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(TRAIN_CSV_PATH),
        help="Đường dẫn tệp CSV tập huấn luyện.",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=str(VAL_CSV_PATH),
        help="Đường dẫn tệp CSV tập kiểm thực.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(TEST_CSV_PATH),
        help="Đường dẫn tệp CSV tập kiểm thử.",
    )
    parser.add_argument(
        "--intents-path",
        type=str,
        default=str(DEFAULT_INTENTS_PATH),
        help="Đường dẫn tệp JSON ngân hàng kịch bản intents.json.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=str(DEFAULT_BASELINE_RESULTS_PATH),
        help="Đường dẫn lưu kết quả đo lường JSON.",
    )
    parser.add_argument(
        "--svm-cm-path",
        type=str,
        default=str(DEFAULT_BASELINE_SVM_CM_PATH),
        help="Đường dẫn lưu biểu đồ confusion matrix của TF-IDF LinearSVC.",
    )
    parser.add_argument(
        "--fuzzy-cm-path",
        type=str,
        default=str(DEFAULT_BASELINE_FUZZY_CM_PATH),
        help="Đường dẫn lưu biểu đồ confusion matrix của Fuzzy Matching.",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_ROUNDS,
        help="Số lượt lấy mẫu lặp lại Bootstrap (mặc định 1000).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help="Mức độ tin cậy ước lượng Bootstrap (mặc định 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Hạt giống ngẫu nhiên tất định.",
    )
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help="Chế độ chạy thử nhanh với tập mẫu thu nhỏ phục vụ kiểm thử đơn vị.",
    )
    return parser


def main(args: Sequence[str] | None = None) -> int:
    """Điểm nhập thực thi chính của script dòng lệnh."""
    parser = build_arg_parser()
    parsed_args = parser.parse_args(args)

    start_time = time.time()
    print("=" * 70)
    print("KHỞI CHẠY ĐƯỜNG ỐNG ĐÁNH GIÁ MÔ HÌNH BASELINE (TF-IDF SVM & FUZZY MATCHING)")
    print(f"  Mục tiêu (Target):         {parsed_args.target}")
    print(f"  Ứng viên C:                {parsed_args.c_candidates}")
    print(f"  Hạt giống (Seed):          {parsed_args.seed}")
    print(f"  Số vòng Bootstrap:         {parsed_args.num_bootstrap}")
    print(f"  Tập Train:                 {parsed_args.train_path}")
    print(f"  Tập Val:                   {parsed_args.val_path}")
    print(f"  Tập Test:                  {parsed_args.test_path}")
    print("=" * 70)

    results = run_baseline_pipeline(
        target=parsed_args.target,
        c_candidates=parsed_args.c_candidates,
        train_path=parsed_args.train_path,
        val_path=parsed_args.val_path,
        test_path=parsed_args.test_path,
        intents_path=parsed_args.intents_path,
        results_path=parsed_args.results_path,
        svm_cm_path=parsed_args.svm_cm_path,
        fuzzy_cm_path=parsed_args.fuzzy_cm_path,
        num_bootstrap=parsed_args.num_bootstrap,
        confidence_level=parsed_args.confidence_level,
        seed=parsed_args.seed,
        smoke_mode=parsed_args.smoke_mode,
    )

    elapsed = time.time() - start_time
    svm_res = results["tfidf_svm"]["test_metrics"]
    svm_ci = results["tfidf_svm"]["test_bootstrap_ci_95"]
    fuzzy_res = results["fuzzy_matching"]["test_metrics"]
    fuzzy_ci = results["fuzzy_matching"]["test_bootstrap_ci_95"]
    mcnemar = results["statistical_comparison"]["mcnemar_test"]

    print("\n" + "=" * 70)
    print("KẾT QUẢ THỰC NGHIỆM ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ (TEST SET):")
    print(f"  Mã băm SHA-256 test.csv:   {results['test_csv_sha256']}")
    print(f"  Vân tay tiền xử lý:        {results['preprocessing_fingerprint']}")
    print("-" * 70)
    print(f"1. TF-IDF LinearSVC (Best C = {results['tfidf_svm']['best_c']}):")
    print(
        f"   - Accuracy:    {svm_res['accuracy']:.4f} "
        f"[95% CI: {svm_ci['accuracy'][0]:.4f} - {svm_ci['accuracy'][1]:.4f}]"
    )
    print(
        f"   - Macro-F1:    {svm_res['macro_f1']:.4f} "
        f"[95% CI: {svm_ci['macro_f1'][0]:.4f} - {svm_ci['macro_f1'][1]:.4f}]"
    )
    print(
        f"   - Weighted-F1: {svm_res['weighted_f1']:.4f} "
        f"[95% CI: {svm_ci['weighted_f1'][0]:.4f} - {svm_ci['weighted_f1'][1]:.4f}]"
    )
    print(f"   - Biểu đồ CM:  {results['tfidf_svm']['confusion_matrix_figure']}")
    print("-" * 70)
    print("2. Fuzzy Alias Matching (rapidfuzz token_set_ratio):")
    print(
        f"   - Accuracy:    {fuzzy_res['accuracy']:.4f} "
        f"[95% CI: {fuzzy_ci['accuracy'][0]:.4f} - {fuzzy_ci['accuracy'][1]:.4f}]"
    )
    print(
        f"   - Macro-F1:    {fuzzy_res['macro_f1']:.4f} "
        f"[95% CI: {fuzzy_ci['macro_f1'][0]:.4f} - {fuzzy_ci['macro_f1'][1]:.4f}]"
    )
    print(
        f"   - Weighted-F1: {fuzzy_res['weighted_f1']:.4f} "
        f"[95% CI: {fuzzy_ci['weighted_f1'][0]:.4f} - {fuzzy_ci['weighted_f1'][1]:.4f}]"
    )
    print(f"   - Biểu đồ CM:  {results['fuzzy_matching']['confusion_matrix_figure']}")
    print("-" * 70)
    print("3. Kiểm định McNemar (TF-IDF LinearSVC vs Fuzzy Matching):")
    print(
        f"   - Statistic:   {mcnemar['statistic']:.4f}, p-value: {mcnemar['p_value']:.4e} "
        f"(b={mcnemar['b']}, c={mcnemar['c']})"
    )
    print(f"  Tệp JSON lưu kết quả:      {parsed_args.results_path}")
    print(f"  Tổng thời gian thực thi:   {elapsed:.2f} giây")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
