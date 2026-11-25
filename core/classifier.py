"""
Bộ máy suy luận phân loại ý định (Intent Classifier Inference Engine) và kiểm thực vân tay tiền xử lý.

Tệp tin: core/classifier.py
Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Kiến trúc suy luận:
    text -> preprocess (NFC, chuẩn hóa ký hiệu toán, tách từ pyvi bảo vệ toán/LaTeX cho PhoBERT)
         -> PhoBERT classifier: p = softmax(z / T) trên không gian ý định
            - if max p >= tau: ý định được chấp nhận (accepted = True)
            - else: chuyển sang truy hồi fallback top-3 (accepted = False)
- Cơ sở lý thuyết và công thức tham chiếu:
    1. JointBERT Eq.(1) (Chen, Zhuo, & Wang, 2019 - arXiv:1902.10909):
           z^i = W^i * Dropout(h_0) + b^i
           p^i = softmax(z^i)
           y^i = argmax_k(p^i_k)
    2. Temperature Scaling Hook (Guo, Pleiss, Sun, & Weinberger, 2017 - "On Calibration of Modern Neural Networks", ICML 2017):
           p_i(T) = exp(z_i / T) / sum_j exp(z_j / T)
       với tham số nhiệt độ T > 0 (mặc định T = 1.0). Khi T > 1.0, phân phối xác suất
       được làm mượt (giảm overconfidence); khi T < 1.0, phân phối sắc nét hơn.
    3. Chuẩn hóa xác suất (Probability Normalization):
           sum_k p_k = 1.0, p_k in [0.0, 1.0]
       Đảm bảo độ ổn định số học (numerical stability) cho mọi kích thước batch.
    4. Kiểm thực vân tay tiền xử lý tại thời điểm chạy (Runtime Preprocessing Fingerprint Verification):
       So sánh mã băm SHA-256 tất định giữa môi trường runtime (preprocess_fingerprint()) và
       metadata lưu trữ trong artifact của mô hình (preprocessing_fingerprint.json hoặc config.json),
       ngăn ngừa triệt để lỗi suy giảm độ chính xác do trôi dạt phiên bản phân đoạn từ (segmenter drift).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import AutoTokenizer

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_TEMPERATURE,
    FINGERPRINT_METADATA_FILENAME,
    MAX_SEQ_LENGTH,
    PHOBERT_MODEL_NAME,
    resolve_phobert_tokenizer_dir,
    TAU_INTENT_THRESHOLD,
    get_default_device,
)
from core.models import PhoBertIntentClassifier
from core.preprocess import prepare_for_phobert, preprocess_fingerprint

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Các ngoại lệ tùy biến (Custom Exceptions)
# =====================================================================


class FingerprintMismatchError(RuntimeError):
    """
    Ngoại lệ phát sinh khi vân tay tiền xử lý (preprocessing fingerprint) thời điểm chạy
    không trùng khớp với metadata được lưu trong artifact của mô hình.
    """


class ModelArtifactError(FileNotFoundError):
    """
    Ngoại lệ phát sinh khi thiếu các tệp tin cấu hình hoặc trọng số cần thiết trong thư mục artifact.
    """


# =====================================================================
# 2. Cấu trúc kết quả phân loại (Classification Data Structures)
# =====================================================================


@dataclass
class ClassificationResult:
    """
    Kết quả suy luận phân loại ý định cho một câu văn bản đầu vào.

    Thuộc tính:
        intent: Nhãn ý định được dự đoán (hoặc nhãn có xác suất cao nhất).
        confidence: Điểm tin cậy (xác suất softmax đã chuẩn hóa, p_max in [0.0, 1.0]).
        accepted: Trạng thái chấp nhận ý định (True nếu confidence >= tau, ngược lại False).
        category: Nhóm ngữ nghĩa (ví dụ: 'math_expression', 'tables', 'tex_misc').
        all_probabilities: Phân phối xác suất chuẩn hóa trên toàn bộ không gian ý định {nhãn: xác suất}.
        raw_text: Chuỗi văn bản thô ban đầu do người dùng nhập.
        preprocessed_text: Chuỗi văn bản sau khi qua prepare_for_phobert (chuẩn hóa NFC, toán & tách từ).
    """

    intent: str
    confidence: float
    accepted: bool
    category: str | None = None
    all_probabilities: dict[str, float] = field(default_factory=dict)
    raw_text: str = ""
    preprocessed_text: str = ""

    @property
    def probabilities(self) -> dict[str, float]:
        """Bí danh truy cập thuận tiện cho all_probabilities."""
        return self.all_probabilities

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi kết quả sang kiểu dữ liệu dict thuận tiện cho logging và xuất JSON."""
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "accepted": self.accepted,
            "category": self.category,
            "all_probabilities": self.all_probabilities,
            "raw_text": self.raw_text,
            "preprocessed_text": self.preprocessed_text,
        }


# Bí danh tương thích
IntentPrediction = ClassificationResult


# =====================================================================
# 3. Tiện ích kiểm thực và lưu trữ vân tay tiền xử lý (Fingerprint Validator)
# =====================================================================


def verify_preprocessing_fingerprint(
    artifact_dir: str | Path,
    strict: bool = True,
    expected_fingerprint: str | None = None,
) -> bool:
    """
    Kiểm thực tính toàn vẹn của vân tay tiền xử lý (preprocessing fingerprint) giữa môi trường
    thực thi hiện tại và metadata được lưu trong artifact của mô hình.

    Quy trình kiểm tra:
    1. Tìm kiếm tệp 'preprocessing_fingerprint.json' trong artifact_dir:
       Trích xuất trường 'fingerprint_sha256' hoặc 'preprocessing_fingerprint'.
    2. Nếu không có, tìm kiếm trường 'preprocessing_fingerprint' trong tệp 'config.json'.
    3. Đối chiếu mã băm với expected_fingerprint (mặc định lấy từ preprocess_fingerprint()).
    4. Nếu có sự sai lệch (mismatch):
       - Nếu strict=True: Phát sinh ngoại lệ FingerprintMismatchError chặn quá trình suy luận sai lệch.
       - Nếu strict=False: Ghi log cảnh báo và trả về False.

    Tham số:
        artifact_dir: Thư mục chứa artifact mô hình đã huấn luyện.
        strict: Cờ kiểm tra nghiêm ngặt (mặc định True).
        expected_fingerprint: Mã băm kỳ vọng (nếu None sẽ tự sinh từ preprocess_fingerprint()).

    Trả về:
        True nếu vân tay trùng khớp hoàn toàn, False nếu sai lệch (khi strict=False).

    Ngoại lệ:
        FingerprintMismatchError: Khi phát hiện sai lệch vân tay hoặc thiếu metadata (khi strict=True).
    """
    path_obj = Path(artifact_dir)
    if not path_obj.exists() or not path_obj.is_dir():
        if strict:
            raise ModelArtifactError(
                f"Thư mục artifact mô hình không tồn tại: '{path_obj.resolve()}'"
            )
        return False

    runtime_fp = (
        expected_fingerprint if expected_fingerprint is not None else preprocess_fingerprint()
    )
    stored_fp: str | None = None

    # 1. Kiểm tra preprocessing_fingerprint.json
    fp_file = path_obj / FINGERPRINT_METADATA_FILENAME
    if fp_file.exists():
        try:
            with open(fp_file, encoding="utf-8") as f:
                data = json.load(f)
                stored_fp = data.get("fingerprint_sha256") or data.get("preprocessing_fingerprint")
        except (json.JSONDecodeError, OSError) as err:
            logger.warning("Không thể đọc tệp %s: %s", fp_file, err)

    # 2. Dự phòng: Kiểm tra trường trong config.json
    if stored_fp is None:
        config_file = path_obj / "config.json"
        if config_file.exists():
            try:
                with open(config_file, encoding="utf-8") as f:
                    data = json.load(f)
                    stored_fp = data.get("preprocessing_fingerprint")
            except (json.JSONDecodeError, OSError) as err:
                logger.warning("Không thể đọc tệp %s: %s", config_file, err)

    # 3. Xử lý trường hợp không tìm thấy vân tay trong metadata
    if stored_fp is None:
        msg = (
            f"Không tìm thấy metadata vân tay tiền xử lý (fingerprint) trong thư mục '{path_obj}'. "
            "Artifact mô hình có thể bị thiếu tệp 'preprocessing_fingerprint.json' hoặc 'config.json'."
        )
        if strict:
            raise FingerprintMismatchError(msg)
        logger.warning(msg)
        return False

    # 4. So sánh đối chiếu mã băm
    if stored_fp != runtime_fp:
        msg = (
            f"Lệch vân tay tiền xử lý (Preprocessing Fingerprint Mismatch)!\n"
            f"  - Vân tay lưu trữ trong artifact: {stored_fp}\n"
            f"  - Vân tay môi trường chạy hiện tại: {runtime_fp}\n"
            "Nguyên nhân: Phiên bản phân đoạn từ pyvi hoặc các quy tắc chuẩn hóa văn bản đã bị thay đổi. "
            "Điều này có thể làm giảm sút nghiêm trọng độ chính xác phân loại ý định của mô hình."
        )
        if strict:
            raise FingerprintMismatchError(msg)
        logger.warning(msg)
        return False

    return True


def save_artifact_fingerprint(
    artifact_dir: str | Path,
    target: str = "intent",
    fingerprint_hash: str | None = None,
) -> Path:
    """
    Lưu trữ thông tin vân tay tiền xử lý vào tệp preprocessing_fingerprint.json trong artifact.

    Tham số:
        artifact_dir: Thư mục artifact đích.
        target: Mục tiêu phân loại ('intent' hoặc 'category').
        fingerprint_hash: Mã băm SHA-256 (nếu None sẽ tự sinh qua preprocess_fingerprint()).

    Trả về:
        Đường dẫn Path dẫn tới tệp metadata vừa được ghi.
    """
    path_obj = Path(artifact_dir)
    path_obj.mkdir(parents=True, exist_ok=True)
    out_file = path_obj / FINGERPRINT_METADATA_FILENAME

    fp = fingerprint_hash if fingerprint_hash is not None else preprocess_fingerprint()
    payload = {
        "fingerprint_sha256": fp,
        "target": target,
        "nfc": True,
        "math_lookalikes": True,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_file


# =====================================================================
# 4. Bộ máy suy luận chính (IntentClassifier Inference Engine)
# =====================================================================


class IntentClassifier:
    """
    Bộ máy suy luận sản xuất (Production Inference Engine) phân loại ý định dựa trên PhoBERT.

    Đặc tính kỹ thuật:
    - Hỗ trợ phương thức dự đoán đơn lẻ `predict(text)` và theo lô `predict_batch(texts)`.
    - Tiền xử lý tự động qua `prepare_for_phobert(text)` (chuẩn hóa NFC, chuyển look-alikes toán học,
      và phân đoạn từ pyvi kết hợp cơ chế bảo vệ biểu thức toán/LaTeX).
    - Chuẩn hóa phân phối xác suất softmax (Probability Normalization) đảm bảo tổng xác suất bằng 1.0.
    - Điểm móc co giãn nhiệt độ (Temperature Scaling Hook) theo chuẩn học thuật (Guo et al., 2017).
    - Tự động kiểm thực vân tay tiền xử lý (Preprocessing Fingerprint Verification) khi nạp artifact.
    """

    def __init__(
        self,
        model: PhoBertIntentClassifier | None = None,
        tokenizer: Any | None = None,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        device: str | None = None,
        threshold_tau: float = TAU_INTENT_THRESHOLD,
        temperature: float = DEFAULT_TEMPERATURE,
        max_seq_length: int = MAX_SEQ_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        category_map: dict[str, str] | None = None,
        temperature_scaling_hook: Callable[[torch.Tensor, float], torch.Tensor] | None = None,
    ) -> None:
        """
        Khởi tạo IntentClassifier với mô hình và bộ tách từ đã sẵn sàng.

        Tham số:
            model: Thể hiện của mô hình PhoBertIntentClassifier (đã huấn luyện hoặc từ cấu hình).
            tokenizer: Thể hiện bộ tách từ PhoBERT AutoTokenizer (hoặc mock đối tượng kiểm thử).
            id2label: Ánh xạ từ chỉ số số nguyên sang tên nhãn ý định {0: 'INTENT_A'}.
            label2id: Ánh xạ từ tên nhãn ý định sang chỉ số { 'INTENT_A': 0}.
            device: Thiết bị tính toán ('cpu', 'cuda'). Mặc định tự phát hiện.
            threshold_tau: Ngưỡng xác suất softmax chấp nhận ý định (mặc định TAU_INTENT_THRESHOLD=0.8).
            temperature: Tham số nhiệt độ ban đầu T > 0 (mặc định DEFAULT_TEMPERATURE=1.0).
            max_seq_length: Chiều dài chuỗi token tối đa (mặc định MAX_SEQ_LENGTH=128).
            batch_size: Kích thước batch mặc định cho suy luận theo lô (mặc định DEFAULT_BATCH_SIZE=16).
            category_map: Ánh xạ từ nhãn ý định sang danh mục ngữ nghĩa {intent: category}.
            temperature_scaling_hook: Hàm tùy biến xử lý co giãn nhiệt độ (tùy chọn).
        """
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "Mô-đun core.classifier yêu cầu PyTorch và Transformers. "
                "Vui lòng cài đặt qua `pip install -r requirements.txt`."
            )

        self.device = device if device is not None else get_default_device()
        self.threshold_tau = threshold_tau
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.category_map: dict[str, str] = category_map or {}

        # Khởi tạo tham số nhiệt độ T và kiểm tra tính hợp lệ (T > 0)
        self._temperature: float = 1.0
        self.set_temperature(temperature)

        # Hook co giãn nhiệt độ tùy biến (Guo et al., 2017)
        self._temperature_scaling_hook: Callable[[torch.Tensor, float], torch.Tensor] | None = (
            temperature_scaling_hook
        )

        # Thiết lập mô hình
        self.model = model
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

        # Thiết lập tokenizer
        self.tokenizer = tokenizer

        # Đồng bộ ánh xạ nhãn
        if id2label is not None and label2id is not None:
            self.id2label: dict[int, str] = {int(k): str(v) for k, v in id2label.items()}
            self.label2id: dict[str, int] = {str(k): int(v) for k, v in label2id.items()}
        elif id2label is not None:
            self.id2label = {int(k): str(v) for k, v in id2label.items()}
            self.label2id = {v: k for k, v in self.id2label.items()}
        elif label2id is not None:
            self.label2id = {str(k): int(v) for k, v in label2id.items()}
            self.id2label = {v: k for k, v in self.label2id.items()}
        elif self.model is not None and hasattr(self.model, "num_labels"):
            num_labels = self.model.num_labels
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = {f"LABEL_{i}": i for i in range(num_labels)}
        else:
            self.id2label = {}
            self.label2id = {}

    @property
    def num_labels(self) -> int:
        """Số lượng nhãn lớp ý định trong không gian phân loại."""
        return len(self.id2label)

    @property
    def temperature(self) -> float:
        """Giá trị tham số nhiệt độ hiện tại."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Thiết lập tham số nhiệt độ thông qua property setter."""
        self.set_temperature(value)

    def set_temperature(self, temperature: float) -> None:
        """
        Thiết lập tham số nhiệt độ T > 0 cho cơ chế Temperature Scaling (Guo et al., 2017).

        Công thức:
            p_i = softmax(z_i / T)
        - Khi T = 1.0: Giữ nguyên logits ban đầu từ mạng nơ-ron (Standard Softmax).
        - Khi T > 1.0: Tăng entropy, làm mịn phân phối xác suất (chống overconfidence).
        - Khi T < 1.0: Giảm entropy, làm sắc nhọn phân phối xác suất.

        Tham số:
            temperature: Giá trị nhiệt độ (bắt buộc > 0.0).

        Ngoại lệ:
            ValueError: Nếu temperature <= 0.0.
        """
        if temperature <= 0.0:
            raise ValueError(
                f"Tham số nhiệt độ temperature phải lớn hơn 0 (> 0.0), nhận giá trị: {temperature}"
            )
        self._temperature = float(temperature)

    def set_temperature_scaling_hook(
        self, hook: Callable[[torch.Tensor, float], torch.Tensor] | None
    ) -> None:
        """
        Đăng ký hoặc hủy bỏ hàm hook co giãn nhiệt độ tùy biến.

        Hàm hook nhận vào: (logits: torch.Tensor, temperature: float) và trả về logits đã co giãn.
        """
        self._temperature_scaling_hook = hook

    def _apply_temperature_scaling(self, logits: torch.Tensor, temp: float) -> torch.Tensor:
        """
        Áp dụng co giãn nhiệt độ lên logits theo Guo et al., 2017.
        """
        if temp <= 0.0:
            raise ValueError(f"Tham số nhiệt độ temperature phải > 0.0, nhận giá trị: {temp}")

        if self._temperature_scaling_hook is not None:
            return self._temperature_scaling_hook(logits, temp)

        # Mặc định: z_scaled = z / T
        return logits / temp

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        device: str | None = None,
        threshold_tau: float = TAU_INTENT_THRESHOLD,
        temperature: float = DEFAULT_TEMPERATURE,
        verify_fingerprint: bool = True,
        strict_fingerprint: bool = True,
        tokenizer_path: str | Path | None = None,
        category_map: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> IntentClassifier:
        """
        Nạp mô hình suy luận từ thư mục checkpoint artifact hoàn chỉnh.

        Các bước tải và khởi tạo:
        1. Kiểm thực tính toàn vẹn của vân tay tiền xử lý (runtime fingerprint vs artifact metadata).
        2. Nạp cấu trúc ánh xạ nhãn id2label và label2id từ label2id.json / id2label.json.
        3. Nạp trọng số mô hình PhoBertIntentClassifier.from_pretrained.
        4. Nạp bộ tách từ AutoTokenizer (từ checkpoint, bộ nhớ đệm offline PHOBERT_DIR, hoặc HF Hub).
        5. Đưa mô hình vào trạng thái eval() và chuyển sang thiết bị tương ứng (CPU/GPU).

        Tham số:
            model_path: Đường dẫn thư mục checkpoint chứa artifact.
            device: Thiết bị chạy ('cpu', 'cuda').
            threshold_tau: Ngưỡng chấp nhận ý định tau (mặc định 0.8).
            temperature: Tham số nhiệt độ T (mặc định 1.0).
            verify_fingerprint: Có kiểm tra vân tay tiền xử lý hay không (mặc định True).
            strict_fingerprint: Chế độ nghiêm ngặt ném ngoại lệ nếu sai lệch (mặc định True).
            tokenizer_path: Đường dẫn bộ tách từ riêng (tùy chọn).
            category_map: Bảng ánh xạ nhãn ý định sang nhóm (tùy chọn).

        Trả về:
            Thể hiện IntentClassifier sẵn sàng cho suy luận.
        """
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("Cần PyTorch và Transformers để nạp mô hình pre-trained.")

        checkpoint_dir = Path(model_path)
        if not checkpoint_dir.exists():
            raise ModelArtifactError(
                f"Thư mục checkpoint mô hình không tồn tại: '{checkpoint_dir.resolve()}'"
            )

        # 1. Kiểm thực vân tay tiền xử lý
        if verify_fingerprint:
            verify_preprocessing_fingerprint(checkpoint_dir, strict=strict_fingerprint)

        # 2. Nạp ánh xạ nhãn
        id2label: dict[int, str] = {}
        label2id: dict[str, int] = {}

        id2label_path = checkpoint_dir / "id2label.json"
        label2id_path = checkpoint_dir / "label2id.json"

        if id2label_path.exists():
            with open(id2label_path, encoding="utf-8") as f:
                raw_id2label = json.load(f)
                id2label = {int(k): str(v) for k, v in raw_id2label.items()}
        if label2id_path.exists():
            with open(label2id_path, encoding="utf-8") as f:
                label2id = json.load(f)

        if not id2label and label2id:
            id2label = {v: k for k, v in label2id.items()}
        if not label2id and id2label:
            label2id = {v: k for k, v in id2label.items()}

        num_labels = len(id2label) if id2label else 2

        # 3. Nạp trọng số mô hình PhoBertIntentClassifier
        model = PhoBertIntentClassifier.from_pretrained(
            checkpoint_dir,
            num_labels=num_labels,
            **kwargs,
        )

        # 4. Nạp tokenizer
        tok_dir: Path | str
        if tokenizer_path is not None:
            tok_dir = tokenizer_path
        elif (checkpoint_dir / "vocab.txt").is_file() and (checkpoint_dir / "bpe.codes").is_file():
            tok_dir = checkpoint_dir
        elif resolve_phobert_tokenizer_dir() is not None:
            tok_dir = resolve_phobert_tokenizer_dir()
        else:
            tok_dir = PHOBERT_MODEL_NAME

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(tok_dir), local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1"
            )
        except (OSError, ValueError, RuntimeError):
            if os.environ.get("HF_HUB_OFFLINE") == "1":
                raise
            tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))

        return cls(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            label2id=label2id,
            device=device,
            threshold_tau=threshold_tau,
            temperature=temperature,
            category_map=category_map,
        )

    @classmethod
    def from_config(
        cls,
        config: Any,
        num_labels: int = 2,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        tokenizer: Any | None = None,
        device: str | None = None,
        threshold_tau: float = TAU_INTENT_THRESHOLD,
        temperature: float = DEFAULT_TEMPERATURE,
        category_map: dict[str, str] | None = None,
        temperature_scaling_hook: Callable[[torch.Tensor, float], torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> IntentClassifier:
        """
        Khởi tạo IntentClassifier từ đối tượng cấu hình PretrainedConfig (không tải trọng số từ đĩa).
        Hữu ích cho unit testing và kiểm thử tích hợp nội bộ siêu tốc.
        """
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("Cần PyTorch và Transformers để khởi tạo từ cấu hình.")

        model = PhoBertIntentClassifier.from_config(config, num_labels=num_labels, **kwargs)
        return cls(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            label2id=label2id,
            device=device,
            threshold_tau=threshold_tau,
            temperature=temperature,
            category_map=category_map,
            temperature_scaling_hook=temperature_scaling_hook,
        )

    def verify_artifact_fingerprint(
        self,
        artifact_dir: str | Path | None = None,
        strict: bool = True,
    ) -> bool:
        """
        Phương thức tiện ích kiểm tra vân tay tiền xử lý của một artifact đối với môi trường hiện tại.
        """
        target_dir = Path(artifact_dir) if artifact_dir is not None else DEFAULT_CHECKPOINT_DIR
        return verify_preprocessing_fingerprint(target_dir, strict=strict)

    def predict(
        self,
        text: str,
        temperature: float | None = None,
        threshold_tau: float | None = None,
    ) -> ClassificationResult:
        """
        Thực hiện suy luận phân loại ý định cho một chuỗi câu văn bản tiếng Việt đơn lẻ.

        Quy trình xử lý:
        1. Chuẩn hóa và phân đoạn từ tiếng Việt qua prepare_for_phobert(text).
        2. Nếu văn bản rỗng hoặc chỉ chứa khoảng trắng: Trả về kết quả UNKNOWN an toàn.
        3. Mã hóa token và chuyển tiếp qua mạng nơ-ron PhoBERT.
        4. Áp dụng co giãn nhiệt độ (temperature scaling hook) lên logits.
        5. Chuẩn hóa xác suất softmax: sum_k p_k = 1.0.
        6. So sánh điểm xác suất lớn nhất max(p) với ngưỡng tau.

        Tham số:
            text: Chuỗi văn bản tiếng Việt từ người dùng.
            temperature: Tham số nhiệt độ T áp dụng riêng cho lượt gọi này (nếu None dùng self.temperature).
            threshold_tau: Ngưỡng tau áp dụng riêng cho lượt gọi này (nếu None dùng self.threshold_tau).

        Trả về:
            Đối tượng ClassificationResult chứa nhãn ý định, độ tin cậy và phân phối xác suất.
        """
        results = self.predict_batch(
            texts=[text],
            batch_size=1,
            temperature=temperature,
            threshold_tau=threshold_tau,
        )
        return results[0]

    def predict_batch(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        temperature: float | None = None,
        threshold_tau: float | None = None,
    ) -> list[ClassificationResult]:
        """
        Thực hiện suy luận phân loại ý định theo lô (batch inference) tối ưu hóa hiệu năng tính toán.

        Tham số:
            texts: Danh sách các chuỗi văn bản tiếng Việt cần phân loại.
            batch_size: Kích thước lô (nếu None sử dụng self.batch_size).
            temperature: Tham số nhiệt độ T áp dụng cho lô (nếu None dùng self.temperature).
            threshold_tau: Ngưỡng tin cậy tau (nếu None dùng self.threshold_tau).

        Trả về:
            Danh sách các đối tượng ClassificationResult theo đúng thứ tự câu đầu vào.
        """
        if not texts:
            return []

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "IntentClassifier chưa được nạp mô hình hoặc bộ tách từ (model/tokenizer is None). "
                "Vui lòng khởi tạo với mô hình hợp lệ hoặc sử dụng `IntentClassifier.from_pretrained()`."
            )

        eff_temp = temperature if temperature is not None else self._temperature
        if eff_temp <= 0.0:
            raise ValueError(f"Tham số nhiệt độ temperature phải > 0.0, nhận giá trị: {eff_temp}")

        eff_tau = threshold_tau if threshold_tau is not None else self.threshold_tau
        eff_batch_size = max(1, batch_size if batch_size is not None else self.batch_size)

        num_texts = len(texts)
        results: list[ClassificationResult] = [None] * num_texts  # type: ignore[list-item]

        # Bước 1: Tiền xử lý toàn bộ các chuỗi văn bản bằng prepare_for_phobert
        preprocessed_texts: list[str] = []
        valid_indices: list[int] = []

        for idx, text in enumerate(texts):
            raw_str = str(text) if text is not None else ""
            if not raw_str.strip():
                # Xử lý văn bản rỗng an toàn: không đưa vào GPU tính toán
                results[idx] = ClassificationResult(
                    intent="UNKNOWN",
                    confidence=0.0,
                    accepted=False,
                    category=None,
                    all_probabilities={},
                    raw_text=raw_str,
                    preprocessed_text="",
                )
                continue

            # Phân đoạn từ ngữ tiếng Việt bảo vệ ký hiệu toán học và lệnh LaTeX
            seg_text = prepare_for_phobert(raw_str)
            preprocessed_texts.append(seg_text)
            valid_indices.append(idx)

        # Nếu toàn bộ các câu trong batch đều rỗng
        if not valid_indices:
            return results

        # Bước 2: Chia lô tính toán suy luận trên các mẫu hợp lệ
        total_valid = len(valid_indices)
        for start_idx in range(0, total_valid, eff_batch_size):
            end_idx = min(start_idx + eff_batch_size, total_valid)
            batch_slice_indices = valid_indices[start_idx:end_idx]
            batch_preprocessed = [preprocessed_texts[i] for i in range(start_idx, end_idx)]

            # Mã hóa token đầu vào bằng PhoBERT tokenizer
            encoded = self.tokenizer(
                batch_preprocessed,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )

            # Chuyển dữ liệu sang thiết bị thực thi (CPU/CUDA)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Lan truyền xuôi không lưu gradient
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                logits = outputs.logits

                # Áp dụng Temperature Scaling hook: z_calibrated = logits / T
                scaled_logits = self._apply_temperature_scaling(logits, eff_temp)

                # Chuẩn hóa xác suất Softmax: p_i = exp(z_i) / sum_j exp(z_j)
                probabilities = torch.softmax(scaled_logits, dim=-1)

                # Đảm bảo tính chuẩn hóa số học tuyệt đối: sum_k p_k = 1.0
                prob_sums = probabilities.sum(dim=-1, keepdim=True)
                probabilities = probabilities / prob_sums

                # Trích xuất nhãn dự đoán tốt nhất và xác suất tương ứng
                max_confidences, pred_label_ids = torch.max(probabilities, dim=-1)

            # Bước 3: Đóng gói kết quả ClassificationResult
            for b_idx, orig_idx in enumerate(batch_slice_indices):
                pred_id = int(pred_label_ids[b_idx].item())
                confidence = float(max_confidences[b_idx].item())
                intent_name = self.id2label.get(pred_id, f"INTENT_{pred_id}")
                accepted = confidence >= eff_tau

                # Lấy danh mục ngữ nghĩa nếu có
                category = self.category_map.get(intent_name)

                # Thu thập toàn bộ phân phối xác suất
                all_probs: dict[str, float] = {}
                row_probs = probabilities[b_idx]
                for c_idx in range(self.num_labels):
                    c_name = self.id2label.get(c_idx, f"INTENT_{c_idx}")
                    all_probs[c_name] = float(row_probs[c_idx].item())

                results[orig_idx] = ClassificationResult(
                    intent=intent_name,
                    confidence=confidence,
                    accepted=accepted,
                    category=category,
                    all_probabilities=all_probs,
                    raw_text=str(texts[orig_idx]),
                    preprocessed_text=batch_preprocessed[b_idx],
                )

        return results
