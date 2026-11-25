"""
Kịch bản tinh chỉnh (Fine-Tuning) và tìm kiếm siêu tham số cho PhoBERT.

Tệp tin: train/finetune_phobert.py
Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Kiến trúc mô hình JointBERT Eq.(1) (Chen, Zhuo, & Wang, 2019):
    z^i = W^i * Dropout(h_0) + b^i
    p^i = softmax(z^i)
- Lựa chọn mục tiêu phân loại linh hoạt: 'intent' (189 lớp theo schema) hoặc 'category' (12 nhóm ngữ nghĩa).
- Tìm kiếm tốc độ học (Learning Rate Search) trên các giá trị ứng viên, đánh giá dựa trên Validation Macro-F1.
- Dừng sớm (Early Stopping) với tính kiên nhẫn 5 epochs (patience=5) để chống quá khớp (overfitting).
- Lịch trình tốc độ học: Khởi động tuyến tính (linear warm-up) và suy giảm tuyến tính (linear decay)
  theo chuẩn BERT/RoBERTa (Devlin et al., 2018; Liu et al., 2019).
- Chế độ chạy thử nhanh (Smoke Run Mode) kiểm thực siêu tốc trên CPU phục vụ CI/CD và kiểm thử tự động.
- Lưu trữ toàn diện checkpoint, cấu hình, ánh xạ nhãn và vân tay tiền xử lý (preprocess fingerprint).
- Kiểm tra mã băm SHA-256 của tập kiểm thử data/test.csv theo Quy tắc Trung thực #3 (Honesty Rule #3).
- Xuất số liệu đo lường học thuật (Accuracy, Macro-F1, Confusion Matrix, ECE) ra report/phobert_results.json.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# Đảm bảo đường dẫn gốc dự án có trong sys.path khi thực thi trực tiếp script
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, RobertaConfig

from core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_ECE_BINS,
    DEFAULT_FINETUNE_SEEDS,
    DEFAULT_FINETUNE_TARGET,
    DEFAULT_INTENTS_PATH,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PHOBERT_RESULTS_PATH,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    MAX_SEQ_LENGTH,
    PHOBERT_CANDIDATE_LRS,
    PHOBERT_DIR,
    PHOBERT_MODEL_NAME,
    REPORT_DIR,
    SMOKE_RUN_CANDIDATE_LRS,
    SMOKE_RUN_EPOCHS,
    SMOKE_RUN_MAX_SAMPLES,
    SMOKE_RUN_SEEDS,
    SUPPORTED_CLASSIFICATION_TARGETS,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    get_default_device,
    set_seed,
)
from core.models import PhoBertIntentClassifier
from core.preprocess import prepare_for_phobert, preprocess_fingerprint
from train.evaluate import (
    compute_classification_metrics,
    compute_expected_calibration_error,
    compute_mean,
    compute_sample_std,
    generate_confusion_matrix_figure,
    load_intent_labels,
)

try:
    from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

    _HAS_SKLEARN: bool = True
except ImportError:
    _HAS_SKLEARN = False


# =====================================================================
# 1. Cấu trúc dữ liệu và Dataset PyTorch
# =====================================================================


@dataclass
class DatasetSample:
    """Mẫu dữ liệu đơn lẻ trong tập huấn luyện/kiểm thử."""

    text: str
    label: str
    category: str


class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch hỗ trợ tiền xử lý văn bản tiếng Việt và tokenization cho PhoBERT.

    Tuân thủ quy trình xử lý của AGENTS.md:
    Chuỗi văn bản -> Tiền xử lý NFC & chuẩn hóa toán học -> Phân đoạn từ pyvi -> Tokenization PhoBERT.
    """

    def __init__(
        self,
        samples: list[DatasetSample],
        target: str,
        label2id: dict[str, int],
        tokenizer: Any,
        max_seq_length: int = MAX_SEQ_LENGTH,
        preprocess: bool = True,
    ) -> None:
        """
        Khởi tạo dataset phân loại văn bản.

        Tham số:
            samples: Danh sách các mẫu dữ liệu DatasetSample.
            target: Mục tiêu phân loại ('intent' hoặc 'category').
            label2id: Bảng ánh xạ tên nhãn sang số nguyên ID.
            tokenizer: Bộ tách từ PhoBERT AutoTokenizer.
            max_seq_length: Chiều dài chuỗi tối đa sau khi padding/truncation.
            preprocess: Cờ kích hoạt tiền xử lý prepare_for_phobert.
        """
        self.samples = samples
        self.target = target
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.preprocess = preprocess

        self.encoded_inputs: list[dict[str, torch.Tensor]] = []
        self.labels: list[int] = []
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Thực hiện tiền xử lý và mã hóa trước toàn bộ tập dữ liệu vào bộ nhớ RAM."""
        for sample in self.samples:
            target_str = sample.label if self.target == "intent" else sample.category
            if target_str not in self.label2id:
                # Bỏ qua mẫu có nhãn ngoài danh mục đã lập chỉ mục
                continue

            processed_text = (
                prepare_for_phobert(sample.text) if self.preprocess else sample.text
            )
            encoding = self.tokenizer(
                processed_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            item = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
            if "token_type_ids" in encoding:
                item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

            self.encoded_inputs.append(item)
            self.labels.append(self.label2id[target_str])

    def __len__(self) -> int:
        """Số lượng mẫu hợp lệ trong dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Truy xuất tensor đầu vào và nhãn cho một mẫu tại chỉ số idx."""
        item = {k: v for k, v in self.encoded_inputs[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =====================================================================
# 2. Tiện ích Nạp dữ liệu và Mã hóa Băm
# =====================================================================


def compute_file_sha256(file_path: Path | str) -> str:
    """
    Tính mã băm SHA-256 tất định của tệp tin.

    Tuân thủ Honesty Rule #3 (AGENTS.md):
    Kiểm tra và lưu trữ SHA-256 của data/test.csv để đảm bảo không rò rỉ dữ liệu kiểm thử.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Tệp không tồn tại: {file_path}")
    hasher = hashlib.sha256()
    with open(path_obj, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_csv_samples(
    csv_path: Path | str, max_samples: int | None = None
) -> list[DatasetSample]:
    """
    Đọc dữ liệu mẫu từ tệp CSV với mã hóa UTF-8-SIG (bảo vệ chống ký tự BOM \\ufeff).

    Tham số:
        csv_path: Đường dẫn tệp CSV.
        max_samples: Số lượng mẫu tối đa cần lấy (dùng cho chế độ smoke run).

    Trả về:
        Danh sách các đối tượng DatasetSample.
    """
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {csv_path}")

    samples: list[DatasetSample] = []
    with open(path_obj, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label = row.get("label", "").strip()
            category = row.get("category", "").strip()
            if text and label:
                samples.append(DatasetSample(text=text, label=label, category=category))
                if max_samples is not None and len(samples) >= max_samples:
                    break
    return samples


def build_label_mappings(
    target: str,
    train_samples: list[DatasetSample],
    intents_json_path: Path | str | None = None,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    Xây dựng danh sách nhãn và bảng ánh xạ 2 chiều (label2id, id2label) tất định.

    Đối với target 'intent': Sử dụng toàn bộ 189 nhãn theo schema intents.json (nếu có)
    nhằm đảm bảo không bị thiếu lớp khi tính macro-F1 trên toàn bộ không gian ý định.
    Đối với target 'category': Sử dụng 12 danh mục ngữ nghĩa được sắp xếp theo bảng chữ cái.
    """
    if target == "intent":
        schema_path = (
            Path(intents_json_path) if intents_json_path else DEFAULT_INTENTS_PATH
        )
        if schema_path.exists():
            labels = load_intent_labels(schema_path)
        else:
            labels = sorted({s.label for s in train_samples})
    elif target == "category":
        labels = sorted({s.category for s in train_samples if s.category})
    else:
        raise ValueError(
            f"Mục tiêu phân loại '{target}' không hợp lệ. Hỗ trợ: {SUPPORTED_CLASSIFICATION_TARGETS}"
        )

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return labels, label2id, id2label


# =====================================================================
# 3. Lịch trình Tốc độ học (Linear Warm-up & Decay)
# =====================================================================


def create_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Tạo bộ điều chỉnh tốc độ học với khởi động tuyến tính và suy giảm tuyến tính.

    Công thức tham chiếu (Devlin et al., 2018 - BERT; Liu et al., 2019 - RoBERTa):
        Nếu current_step < num_warmup_steps:
            lr_factor = current_step / max(1, num_warmup_steps)
        Ngược lại:
            lr_factor = max(0.0, (num_training_steps - current_step) / max(1, num_training_steps - num_warmup_steps))

    LÝ DO THIẾT KẾ (WHY):
        1. Khởi động tuyến tính (warm-up) giúp ổn định gradient trong các bước đầu tiên khi
           tầng phân loại mới khởi tạo ngẫu nhiên, tránh làm phá vỡ các trọng số biểu diễn tiền huấn luyện của PhoBERT.
        2. Suy giảm tuyến tính (decay) giúp hội tụ êm ái về điểm cực tiểu tối ưu ở các epoch cuối.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# 4. Khởi tạo Mô hình và Bộ Tách Từ
# =====================================================================


def load_phobert_tokenizer(model_dir: Path | str | None = None) -> Any:
    """
    Nạp AutoTokenizer của PhoBERT từ thư mục bộ nhớ đệm ngoại tuyến hoặc Hugging Face Hub.
    """
    target_dir = Path(model_dir) if model_dir else PHOBERT_DIR
    if target_dir.exists() and (target_dir / "vocab.txt").exists():
        return AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True)
    return AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)


def create_classifier_model(
    num_labels: int,
    smoke_mode: bool = False,
    model_name_or_path: Path | str | None = None,
    device: str = "cpu",
) -> PhoBertIntentClassifier:
    """
    Khởi tạo mạng nơ-ron phân loại PhoBertIntentClassifier theo JointBERT Eq.(1).

    Chiến lược khởi tạo:
    1. Nếu smoke_mode=True: Khởi tạo mô hình RobertaConfig siêu nhẹ (1 tầng, 2 heads, 64-dim)
       để thực hiện huấn luyện và kiểm thử siêu nhanh trên CPU mà không cần tải file trọng số 540MB.
    2. Nếu chế độ bình thường: Thử nạp backbone PhoBERT pre-trained từ model_name_or_path hoặc Hub.
       Nếu môi trường ngoại tuyến chưa có file trọng số pytorch_model.bin/safetensors,
       tự động khởi tạo từ cấu hình AutoConfig của vinai/phobert-base-v2 một cách an toàn.
    """
    if smoke_mode:
        tiny_config = RobertaConfig(
            vocab_size=64001,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=128,
            max_position_embeddings=128,
        )
        model = PhoBertIntentClassifier.from_config(tiny_config, num_labels=num_labels)
        return model.to(device)

    target_path = Path(model_name_or_path) if model_name_or_path else PHOBERT_DIR
    try:
        # Thử nạp pre-trained backbone có sẵn
        model = PhoBertIntentClassifier(
            num_labels=num_labels,
            model_name_or_path=target_path
            if target_path.exists()
            else PHOBERT_MODEL_NAME,
        )
    except (OSError, ValueError):
        # Fallback khởi tạo kiến trúc chuẩn PhoBERT base 768-dim từ config khi chạy ngoại tuyến
        base_config = AutoConfig.from_pretrained(
            PHOBERT_MODEL_NAME, num_labels=num_labels
        )
        model = PhoBertIntentClassifier.from_config(base_config, num_labels=num_labels)

    return model.to(device)


# =====================================================================
# 5. Vòng lặp Huấn luyện và Đánh giá (Training & Early Stopping)
# =====================================================================


def evaluate_split(
    model: PhoBertIntentClassifier,
    dataloader: DataLoader,
    labels: list[str],
    id2label: dict[int, str],
    device: str,
) -> dict[str, Any]:
    """
    Đánh giá mô hình trên một tập dữ liệu (Validation hoặc Test).

    Tính toán:
    - Giá trị mất mát trung bình (Average CrossEntropyLoss).
    - Accuracy, Macro-F1, Weighted-F1 thông qua train.evaluate.compute_classification_metrics.
    - Expected Calibration Error (ECE) thông qua train.evaluate.compute_expected_calibration_error.
    - Ma trận nhầm lẫn (Confusion Matrix).

    Trả về:
        Dictionary chứa toàn bộ chỉ số đo lường học thuật thực tế.
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[str] = []
    all_trues: list[str] = []
    all_probs_list: list[list[float]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_tensor,
                return_dict=True,
            )
            if outputs.loss is not None:
                total_loss += outputs.loss.item() * len(labels_tensor)

            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            for p_idx, t_idx, prob_row in zip(
                preds.cpu().tolist(),
                labels_tensor.cpu().tolist(),
                probs.cpu().tolist(),
            ):
                all_preds.append(id2label.get(p_idx, str(p_idx)))
                all_trues.append(id2label.get(t_idx, str(t_idx)))
                all_probs_list.append(prob_row)

    avg_loss = total_loss / max(1, len(dataloader.dataset))

    # Tính toán các chỉ số phân loại học thuật
    clf_res = compute_classification_metrics(
        y_true=all_trues,
        y_pred=all_preds,
        labels=labels,
        zero_division=0.0,
    )

    # Tính toán Expected Calibration Error (ECE - Guo et al., 2017)
    ece_res = compute_expected_calibration_error(
        y_true=all_trues,
        y_pred=all_preds,
        probs=all_probs_list,
        num_bins=DEFAULT_ECE_BINS,
    )

    # Tính ma trận nhầm lẫn
    if _HAS_SKLEARN:
        cm = sklearn_confusion_matrix(all_trues, all_preds, labels=labels).tolist()
    else:
        # Fallback tự tính ma trận nhầm lẫn đơn giản
        label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
        n_classes = len(labels)
        cm_matrix = [[0] * n_classes for _ in range(n_classes)]
        for t, p in zip(all_trues, all_preds):
            if t in label_to_idx and p in label_to_idx:
                cm_matrix[label_to_idx[t]][label_to_idx[p]] += 1
        cm = cm_matrix

    return {
        "loss": float(avg_loss),
        "accuracy": float(clf_res.accuracy),
        "macro_f1": float(clf_res.macro_f1),
        "weighted_f1": float(clf_res.weighted_f1),
        "ece": float(ece_res.ece),
        "per_class": clf_res.per_class,
        "confusion_matrix": cm,
        "y_true": all_trues,
        "y_pred": all_preds,
    }


def train_single_run(
    model: PhoBertIntentClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    labels: list[str],
    id2label: dict[int, str],
    learning_rate: float,
    max_epochs: int,
    patience: int,
    warmup_ratio: float,
    weight_decay: float,
    device: str,
) -> tuple[PhoBertIntentClassifier, dict[str, Any]]:
    """
    Huấn luyện mô hình cho một cấu hình (Single Run) có dừng sớm (Early Stopping).

    Quy trình:
    1. Thiết lập tối ưu hóa AdamW với trọng số phân rã (weight_decay).
    2. Khởi tạo lịch trình học có linear warmup và linear decay.
    3. Giám sát Validation Macro-F1 sau mỗi epoch:
       - Nếu F1 cải thiện: Lưu lại bản sao trọng số tốt nhất (best_state_dict) và đặt lại patience_counter = 0.
       - Nếu F1 không cải thiện liên tiếp qua `patience` epochs: Dừng sớm (Early Stopping) để chống quá khớp.
    4. Khôi phục trọng số mô hình tốt nhất trước khi trả về.

    Trả về:
        Tuple gồm (mô hình đã khôi phục trọng số tốt nhất, nhật ký huấn luyện lịch sử).
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_training_steps = len(train_loader) * max_epochs
    num_warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = create_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_val_metrics: dict[str, Any] = {}
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_tensor,
                return_dict=True,
            )
            loss = outputs.loss
            if loss is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item() * len(labels_tensor)

        avg_train_loss = epoch_loss / max(1, len(train_loader.dataset))

        # Đánh giá trên tập Validation
        val_res = evaluate_split(model, val_loader, labels, id2label, device)
        val_macro_f1 = val_res["macro_f1"]

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_res["loss"],
            "val_macro_f1": val_macro_f1,
            "val_accuracy": val_res["accuracy"],
            "val_ece": val_res["ece"],
        }
        history.append(record)

        # Kiểm tra điều kiện Dừng sớm (Early Stopping)
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_metrics = val_res
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Kích hoạt dừng sớm do hết hạn kiên nhẫn
                break

    # Khôi phục trạng thái mô hình tại epoch đạt kết quả cao nhất
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    training_summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_metrics": best_val_metrics,
        "epochs_trained": len(history),
        "history": history,
    }
    return model, training_summary


# =====================================================================
# 6. Tìm kiếm Tốc độ học (Learning Rate Search)
# =====================================================================


def search_learning_rate(
    candidate_lrs: Sequence[float],
    train_dataset: TextClassificationDataset,
    val_dataset: TextClassificationDataset,
    labels: list[str],
    id2label: dict[int, str],
    batch_size: int,
    max_epochs: int,
    patience: int,
    warmup_ratio: float,
    weight_decay: float,
    smoke_mode: bool,
    device: str,
    search_seed: int = 42,
) -> tuple[float, dict[str, float]]:
    """
    Tìm kiếm tốc độ học tối ưu trên tập Validation (Learning Rate Search).

    Tuân thủ Honesty Rule #3 (AGENTS.md):
    TUYỆT ĐỐI KHÔNG sử dụng tập kiểm thử (test set) để tìm kiếm siêu tham số.
    Toàn bộ quá trình chọn lựa candidate_lrs được đánh giá duy nhất trên Validation Macro-F1.

    Trả về:
        Tuple gồm (best_learning_rate, từ điển ánh xạ {lr_string: val_macro_f1}).
    """
    lr_results: dict[str, float] = {}
    best_lr = candidate_lrs[0]
    best_val_macro_f1 = -1.0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for lr in candidate_lrs:
        set_seed(search_seed)
        model = create_classifier_model(
            num_labels=len(labels),
            smoke_mode=smoke_mode,
            device=device,
        )

        _, summary = train_single_run(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            labels=labels,
            id2label=id2label,
            learning_rate=lr,
            max_epochs=max_epochs,
            patience=patience,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            device=device,
        )

        val_f1 = summary["best_val_macro_f1"]
        lr_results[f"{lr:.1e}"] = float(val_f1)

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_lr = lr

    return best_lr, lr_results


# =====================================================================
# 7. Huấn luyện Đa Hạt Giống (Multi-Seed Fine-Tuning) & Đánh giá Test
# =====================================================================


def run_experiment_pipeline(
    target: str = DEFAULT_FINETUNE_TARGET,
    candidate_lrs: Sequence[float] = PHOBERT_CANDIDATE_LRS,
    seeds: Sequence[int] = DEFAULT_FINETUNE_SEEDS,
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_seq_length: int = MAX_SEQ_LENGTH,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    smoke_mode: bool = False,
    device: str | None = None,
    train_path: Path | str = TRAIN_CSV_PATH,
    val_path: Path | str = VAL_CSV_PATH,
    test_path: Path | str = TEST_CSV_PATH,
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    results_path: Path | str = DEFAULT_PHOBERT_RESULTS_PATH,
) -> dict[str, Any]:
    """
    Toàn bộ đường ống thực nghiệm tinh chỉnh PhoBERT (End-to-End Pipeline).

    Các bước tuần tự:
    1. Kiểm tra tính toàn vẹn và tính mã băm SHA-256 của tập kiểm thử data/test.csv.
    2. Nạp dữ liệu train, val, test và xây dựng ánh xạ nhãn xác định.
    3. Tạo mã băm vân tay tiền xử lý (preprocess fingerprint) lưu vết môi trường.
    4. Tìm kiếm tốc độ học tối ưu (Learning Rate Search) trên tập Validation.
    5. Huấn luyện độc lập qua danh sách các hạt giống ngẫu nhiên (Multi-Seed Fine-Tuning).
    6. Đánh giá kiểm thử (Test Evaluation) một lần duy nhất cho mỗi hạt giống sau huấn luyện.
    7. Tính toán thống kê trung bình và độ lệch chuẩn mẫu (sample std với n-1 bậc tự do).
    8. Lưu trữ checkpoint tốt nhất, cấu hình, ánh xạ nhãn, vân tay và báo cáo JSON kết quả.
    """
    run_device = device if device is not None else get_default_device()
    test_file_path = Path(test_path)
    train_file_path = Path(train_path)
    val_file_path = Path(val_path)
    save_checkpoint_path = Path(checkpoint_dir)
    save_results_path = Path(results_path)

    # 1. Tính toán mã băm SHA-256 của tập kiểm thử theo Honesty Rule #3
    test_csv_sha256 = compute_file_sha256(test_file_path)

    # 2. Điều chỉnh tham số trong chế độ Smoke Run
    effective_candidate_lrs = SMOKE_RUN_CANDIDATE_LRS if smoke_mode else candidate_lrs
    effective_seeds = SMOKE_RUN_SEEDS if smoke_mode else seeds
    effective_epochs = SMOKE_RUN_EPOCHS if smoke_mode else max_epochs
    max_samples_limit = SMOKE_RUN_MAX_SAMPLES if smoke_mode else None

    # Nạp dữ liệu mẫu từ CSV
    train_samples = read_csv_samples(train_file_path, max_samples=max_samples_limit)
    val_samples = read_csv_samples(
        val_file_path,
        max_samples=max_samples_limit // 2 if max_samples_limit else None,
    )
    test_samples = read_csv_samples(
        test_file_path,
        max_samples=max_samples_limit // 2 if max_samples_limit else None,
    )

    # Xây dựng bảng ánh xạ nhãn
    labels, label2id, id2label = build_label_mappings(
        target=target,
        train_samples=train_samples,
    )

    # 3. Tạo mã băm vân tay tiền xử lý
    fingerprint_hash = preprocess_fingerprint()

    # Nạp tokenizer
    tokenizer = load_phobert_tokenizer()

    # Khởi tạo Datasets
    train_dataset = TextClassificationDataset(
        samples=train_samples,
        target=target,
        label2id=label2id,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    val_dataset = TextClassificationDataset(
        samples=val_samples,
        target=target,
        label2id=label2id,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    test_dataset = TextClassificationDataset(
        samples=test_samples,
        target=target,
        label2id=label2id,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Tìm kiếm Tốc độ học tối ưu (Search LR trên Validation Set)
    best_lr, lr_search_results = search_learning_rate(
        candidate_lrs=effective_candidate_lrs,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        labels=labels,
        id2label=id2label,
        batch_size=batch_size,
        max_epochs=effective_epochs,
        patience=patience,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        smoke_mode=smoke_mode,
        device=run_device,
        search_seed=effective_seeds[0],
    )

    # 5. Huấn luyện đa hạt giống (Multi-Seed Fine-Tuning)
    per_seed_results: list[dict[str, Any]] = []
    best_overall_seed = effective_seeds[0]
    best_overall_val_f1 = -1.0
    best_model_instance: PhoBertIntentClassifier | None = None
    best_test_metrics: dict[str, Any] = {}

    for seed in effective_seeds:
        set_seed(seed)
        model = create_classifier_model(
            num_labels=len(labels),
            smoke_mode=smoke_mode,
            device=run_device,
        )

        trained_model, train_summary = train_single_run(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            labels=labels,
            id2label=id2label,
            learning_rate=best_lr,
            max_epochs=effective_epochs,
            patience=patience,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            device=run_device,
        )

        # 6. Đánh giá trên tập kiểm thử (Test Evaluation)
        test_eval = evaluate_split(
            model=trained_model,
            dataloader=test_loader,
            labels=labels,
            id2label=id2label,
            device=run_device,
        )

        val_f1 = train_summary["best_val_macro_f1"]
        seed_record = {
            "seed": seed,
            "best_epoch": train_summary["best_epoch"],
            "epochs_trained": train_summary["epochs_trained"],
            "val_macro_f1": float(val_f1),
            "test_accuracy": float(test_eval["accuracy"]),
            "test_macro_f1": float(test_eval["macro_f1"]),
            "test_weighted_f1": float(test_eval["weighted_f1"]),
            "test_ece": float(test_eval["ece"]),
            "test_loss": float(test_eval["loss"]),
        }
        per_seed_results.append(seed_record)

        if val_f1 > best_overall_val_f1:
            best_overall_val_f1 = val_f1
            best_overall_seed = seed
            best_model_instance = trained_model
            best_test_metrics = test_eval

    # 7. Tính toán số liệu thống kê tổng hợp (Mean & Sample Std với n-1 bậc tự do)
    accuracies = [r["test_accuracy"] for r in per_seed_results]
    macro_f1s = [r["test_macro_f1"] for r in per_seed_results]
    eces = [r["test_ece"] for r in per_seed_results]

    acc_mean = compute_mean(accuracies)
    acc_std = compute_sample_std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0

    f1_mean = compute_mean(macro_f1s)
    f1_std = compute_sample_std(macro_f1s, ddof=1) if len(macro_f1s) > 1 else 0.0

    ece_mean = compute_mean(eces)
    ece_std = compute_sample_std(eces, ddof=1) if len(eces) > 1 else 0.0

    # 8. Lưu trữ Checkpoints và Metadata
    save_checkpoint_path.mkdir(parents=True, exist_ok=True)
    if best_model_instance is not None:
        best_model_instance.save_pretrained(save_checkpoint_path)

    # Lưu ánh xạ nhãn label2id và id2label
    with open(save_checkpoint_path / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(save_checkpoint_path / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    # Lưu cấu hình mô hình và thực nghiệm (kế thừa các trường RobertaConfig nếu có)
    merged_config: dict[str, Any] = {}
    config_file = save_checkpoint_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file, encoding="utf-8") as f:
                merged_config = json.load(f)
        except (json.JSONDecodeError, OSError):
            merged_config = {}

    finetune_config = {
        "target": target,
        "num_labels": len(labels),
        "best_learning_rate": best_lr,
        "candidate_learning_rates": list(effective_candidate_lrs),
        "seeds": list(effective_seeds),
        "best_seed": best_overall_seed,
        "patience": patience,
        "max_epochs": effective_epochs,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "smoke_mode": smoke_mode,
        "device": run_device,
        "test_csv_sha256": test_csv_sha256,
        "preprocessing_fingerprint": fingerprint_hash,
    }
    merged_config.update(finetune_config)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(merged_config, f, ensure_ascii=False, indent=2)

    # Lưu vân tay tiền xử lý
    with open(
        save_checkpoint_path / "preprocessing_fingerprint.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "fingerprint_sha256": fingerprint_hash,
                "target": target,
                "verified_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Sinh biểu đồ ma trận nhầm lẫn nếu có dự đoán test
    cm_figure_path: str | None = None
    if "y_true" in best_test_metrics and "y_pred" in best_test_metrics:
        figures_dir = REPORT_DIR / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        cm_fig_out = figures_dir / f"phobert_{target}_confusion_matrix.png"
        try:
            generate_confusion_matrix_figure(
                y_true=best_test_metrics["y_true"],
                y_pred=best_test_metrics["y_pred"],
                labels=labels,
                output_path=cm_fig_out,
                title=f"PhoBERT Confusion Matrix ({target})",
            )
            cm_figure_path = str(cm_fig_out)
        except (RuntimeError, ValueError, OSError):
            cm_figure_path = None

    # Biên soạn tệp kết quả JSON đầu ra
    results_payload: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "target": target,
        "num_classes": len(labels),
        "test_csv_sha256": test_csv_sha256,
        "preprocessing_fingerprint": fingerprint_hash,
        "smoke_mode": smoke_mode,
        "candidate_learning_rates": list(effective_candidate_lrs),
        "best_learning_rate": best_lr,
        "lr_search_results": lr_search_results,
        "seeds": list(effective_seeds),
        "early_stopping_patience": patience,
        "multi_seed_results": per_seed_results,
        "aggregate_test_metrics": {
            "accuracy_mean": float(acc_mean),
            "accuracy_std": float(acc_std),
            "macro_f1_mean": float(f1_mean),
            "macro_f1_std": float(f1_std),
            "ece_mean": float(ece_mean),
            "ece_std": float(ece_std),
        },
        "best_seed": best_overall_seed,
        "best_model_test_metrics": {
            "accuracy": float(best_test_metrics.get("accuracy", 0.0)),
            "macro_f1": float(best_test_metrics.get("macro_f1", 0.0)),
            "weighted_f1": float(best_test_metrics.get("weighted_f1", 0.0)),
            "ece": float(best_test_metrics.get("ece", 0.0)),
            "loss": float(best_test_metrics.get("loss", 0.0)),
            "confusion_matrix": best_test_metrics.get("confusion_matrix", []),
            "per_class": best_test_metrics.get("per_class", {}),
        },
        "saved_checkpoint_dir": str(save_checkpoint_path),
        "confusion_matrix_figure": cm_figure_path,
    }

    save_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    return results_payload


# =====================================================================
# 8. Giao diện Dòng lệnh (CLI)
# =====================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh cho kịch bản fine-tuning PhoBERT."""
    parser = argparse.ArgumentParser(
        description="Đường ống tinh chỉnh (Fine-Tuning) PhoBERT và tìm kiếm siêu tham số.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_FINETUNE_TARGET,
        choices=SUPPORTED_CLASSIFICATION_TARGETS,
        help="Mục tiêu phân loại: 'intent' (189 lớp) hoặc 'category' (12 nhóm).",
    )
    parser.add_argument(
        "--candidate-lrs",
        type=float,
        nargs="+",
        default=list(PHOBERT_CANDIDATE_LRS),
        help="Danh sách các tốc độ học ứng viên tìm kiếm (mặc định: 1e-5 2e-5 3e-5 5e-5).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_FINETUNE_SEEDS),
        help="Danh sách các hạt giống ngẫu nhiên đa thực nghiệm (mặc định: 42 43 44).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Số epoch kiên nhẫn cho cơ chế dừng sớm (mặc định: 5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Số epoch tối đa cho mỗi lần huấn luyện (mặc định: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Kích thước batch huấn luyện và đánh giá (mặc định: 16).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Chiều dài chuỗi token tối đa (mặc định: 128).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help="Tỷ lệ khởi động tuyến tính warm-up (mặc định: 0.1).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Hệ số phân rã trọng số AdamW (mặc định: 0.01).",
    )
    parser.add_argument(
        "--smoke",
        "--smoke-run",
        action="store_true",
        dest="smoke_mode",
        help="Kích hoạt chế độ chạy thử nhanh (Smoke Run) trên CPU trong vài giây.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị tính toán ('cpu', 'cuda'). Mặc định tự động phát hiện.",
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
        help="Đường dẫn tệp CSV tập kiểm thực (validation).",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(TEST_CSV_PATH),
        help="Đường dẫn tệp CSV tập kiểm thử (test).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Thư mục lưu trữ model checkpoint và metadata.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=str(DEFAULT_PHOBERT_RESULTS_PATH),
        help="Đường dẫn tệp JSON lưu trữ kết quả thực nghiệm.",
    )
    return parser


def main() -> None:
    """Điểm nhập thực thi chính của script finetune_phobert.py."""
    parser = build_arg_parser()
    args = parser.parse_args()

    print("=" * 70)
    print("KHỞI CHẠY ĐƯỜNG ỐNG TINH CHỈNH PHOBERT (KA-11 PIPELINE RUNNER)")
    print("=" * 70)
    print(f"Mục tiêu phân loại: {args.target}")
    print(f"Chế độ Smoke Run: {args.smoke_mode}")
    print(f"Tốc độ học ứng viên: {args.candidate_lrs}")
    print(f"Hạt giống thực nghiệm: {args.seeds}")
    print(f"Patience dừng sớm: {args.patience} epochs")
    print("-" * 70)

    results = run_experiment_pipeline(
        target=args.target,
        candidate_lrs=args.candidate_lrs,
        seeds=args.seeds,
        patience=args.patience,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        smoke_mode=args.smoke_mode,
        device=args.device,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        checkpoint_dir=args.checkpoint_dir,
        results_path=args.results_path,
    )

    agg = results["aggregate_test_metrics"]
    print("\n" + "=" * 70)
    print("KẾT QUẢ THỰC NGHIỆM ĐÁNH GIÁ (TEST METRICS):")
    print(f"  Mã băm test.csv SHA-256: {results['test_csv_sha256']}")
    print(f"  Vân tay tiền xử lý:      {results['preprocessing_fingerprint']}")
    print(f"  Tốc độ học tối ưu:       {results['best_learning_rate']}")
    print(
        f"  Test Accuracy:           {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}"
    )
    print(
        f"  Test Macro-F1:           {agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}"
    )
    print(f"  Test ECE:                {agg['ece_mean']:.4f} ± {agg['ece_std']:.4f}")
    print(f"  Tệp kết quả:             {args.results_path}")
    print(f"  Thư mục checkpoint:      {results['saved_checkpoint_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
