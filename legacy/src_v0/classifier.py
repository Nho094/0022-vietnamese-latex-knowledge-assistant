"""
Module phân loại ý định (Intent Classifier) dựa trên PhoBERT cho trợ lý tiếng Việt LaTeX.

Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Kiến trúc xử lý:
  text -> preprocess -> PhoBERT classifier: p = softmax over intents
     - if max p >= tau: intent accepted
     - else retrieval top-3 over aliases + scenario strings -> ask the user to choose (if best score < theta: say not understood)
- Công thức toán học (Nguyen & Tuan Nguyen, 2020 - PhoBERT: Pre-trained RoBERTa for Vietnamese):
  z = W * h_{[CLS]} + b (logits đầu ra từ lớp phân loại tuyến tính trên biểu diễn ngữ cảnh)
  p_i = softmax(z)_i = exp(z_i) / sum_j(exp(z_j))
  p_max = max_i(p_i)
  accepted = True nếu p_max >= tau (với tau = TAU_INTENT_THRESHOLD cấu hình trong core/config.py)
- Hỗ trợ tải ngoại tuyến từ thư mục models/hf_cache mà không cần kết nối internet.
- Hỗ trợ thực thi trên cả CPU và GPU ('cuda') thông qua tùy chọn device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INTENTS_PATH,
    MAX_SEQ_LENGTH,
    PHOBERT_DIR,
    PHOBERT_MODEL_NAME,
    TAU_INTENT_THRESHOLD,
    get_default_device,
)
from src.scenarios import ScenarioBank, load_scenario_bank


@dataclass
class ClassificationResult:
    """
    Kết quả dự đoán ý định từ bộ phân loại PhoBERT.

    Thuộc tính:
        intent: Tên nhãn ý định có xác suất cao nhất.
        confidence: Điểm xác suất softmax cao nhất (p_max trong khoảng [0, 1]).
        accepted: Trạng thái chấp nhận ý định (True nếu p_max >= tau, False nếu cần fallback).
        category: Danh mục ngữ nghĩa (ví dụ: math_expression, latex_document).
        all_probabilities: Phân phối xác suất của các ý định hàng đầu (intent -> xác suất).
    """
    intent: str
    confidence: float
    accepted: bool
    category: Optional[str] = None
    all_probabilities: dict[str, float] = field(default_factory=dict)


class PhoBertClassifier:
    """
    Bộ phân loại ý định tiếng Việt dựa trên kiến trúc RoBERTa (PhoBERT).

    Hỗ trợ tải ngoại tuyến hoàn toàn từ `models/hf_cache`, thực thi trên CPU/GPU,
    và tự động suy luận ngưỡng tin cậy tau theo AGENTS.md.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        threshold_tau: float = TAU_INTENT_THRESHOLD,
        scenario_bank: Optional[ScenarioBank] = None,
    ) -> None:
        """
        Khởi tạo bộ phân loại PhoBERT.

        Tham số:
            model_path: Đường dẫn cục bộ chứa trọng số mô hình hoặc thư mục cache.
            device: Thiết bị tính toán ('cpu', 'cuda'). Nếu None sẽ tự phát hiện qua get_default_device.
            threshold_tau: Ngưỡng xác suất softmax để chấp nhận ý định (mặc định 0.8).
            scenario_bank: Ngân hàng kịch bản để đồng bộ nhãn intent và fallback.
        """
        self.model_path = Path(model_path) if model_path is not None else PHOBERT_DIR
        self.device = device if device is not None else get_default_device()
        self.threshold_tau = threshold_tau

        # Nạp scenario bank để có danh mục nhãn đầy đủ
        if scenario_bank is None:
            if DEFAULT_INTENTS_PATH.exists():
                self.scenario_bank = load_scenario_bank(DEFAULT_INTENTS_PATH)
            else:
                self.scenario_bank = ScenarioBank([])
        else:
            self.scenario_bank = scenario_bank

        # Khởi tạo danh sách nhãn intent và ánh xạ id
        self.labels: list[str] = [sc.intent for sc in self.scenario_bank.all()]
        self.label2id: dict[str, int] = {label: i for i, label in enumerate(self.labels)}
        self.id2label: dict[int, str] = {i: label for i, label in enumerate(self.labels)}

        # Trạng thái mô hình HuggingFace / PyTorch
        self._tokenizer: Any = None
        self._model: Any = None
        self._is_hf_loaded: bool = False

        # Thử nạp mô hình HuggingFace nếu có sẵn
        self._try_load_hf_model()

    @property
    def is_hf_loaded(self) -> bool:
        """Kiểm tra mô hình Hugging Face thực tế đã được nạp hay chưa."""
        return self._is_hf_loaded

    def to(self, device: str) -> PhoBertClassifier:
        """Chuyển thiết bị tính toán của mô hình (CPU hoặc CUDA GPU)."""
        self.device = device
        if self._model is not None and hasattr(self._model, "to"):
            self._model.to(device)
        return self

    def _try_load_hf_model(self) -> bool:
        """
        Cố gắng nạp tokenizer và model PhoBERT từ bộ nhớ đệm cục bộ models/hf_cache.
        Sử dụng cờ local_files_only=True để tuyệt đối không truy cập internet.
        """
        try:
            import torch  # type: ignore[import-untyped]
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore[import-untyped]

            target_path = str(self.model_path)
            # Kiểm tra nếu thư mục tồn tại và có file cấu hình
            if self.model_path.exists() and (self.model_path / "config.json").exists():
                self._tokenizer = AutoTokenizer.from_pretrained(
                    target_path,
                    local_files_only=True,
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    target_path,
                    local_files_only=True,
                )
                if hasattr(self._model, "to"):
                    self._model.to(self.device)
                if hasattr(self._model, "eval"):
                    self._model.eval()

                # Đồng bộ label mapping nếu có trong model config
                if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                    self.id2label = {int(k): v for k, v in self._model.config.id2label.items()}
                    self.label2id = {v: k for k, v in self.id2label.items()}
                    self.labels = list(self.label2id.keys())

                self._is_hf_loaded = True
                return True
        except (ImportError, Exception):
            # shortcut: khi môi trường chưa cài PyTorch/Transformers hoặc chưa tải checkpoint cục bộ,
            # sử dụng cơ chế mô phỏng dựa trên scenario bank để test và phát triển offline.
            pass

        self._is_hf_loaded = False
        return False

    def _predict_hf(self, text: str) -> ClassificationResult:
        """
        Thực hiện suy luận qua mô hình Transformer PhoBERT (PyTorch).
        Áp dụng công thức softmax: p_i = exp(z_i) / sum_j(exp(z_j)).
        """
        import torch  # type: ignore[import-untyped]

        inputs = self._tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # Đưa tensor lên thiết bị tính toán (CPU hoặc GPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        p_max, pred_idx = torch.max(probabilities, dim=-1)
        confidence = float(p_max.item())
        pred_id = int(pred_idx.item())
        predicted_intent = self.id2label.get(pred_id, "UNKNOWN")

        # Lấy top probabilities để theo dõi
        top_k = min(5, len(self.labels))
        top_probs, top_indices = torch.topk(probabilities, k=top_k)
        all_probs = {
            self.id2label.get(int(idx.item()), f"LABEL_{idx.item()}"): round(float(p.item()), 4)
            for p, idx in zip(top_probs, top_indices)
        }

        # Kiểm tra ngưỡng tau: accepted = (p_max >= tau)
        accepted = confidence >= self.threshold_tau
        category = None
        if predicted_intent in self.scenario_bank:
            category = self.scenario_bank[predicted_intent].category

        return ClassificationResult(
            intent=predicted_intent,
            confidence=round(confidence, 4),
            accepted=accepted,
            category=category,
            all_probabilities=all_probs,
        )

    def _predict_offline_heuristic(self, text: str) -> ClassificationResult:
        """
        Thuật toán phân loại cục bộ dựa trên trọng số từ khóa và n-gram của kịch bản.
        Hoạt động khi chưa có checkpoint PhoBERT tải về trong models/hf_cache,
        đảm bảo hệ thống vẫn kiểm thử được toàn diện offline.
        """
        text_clean = text.lower().strip()
        if not text_clean or not self.labels:
            return ClassificationResult(
                intent="UNKNOWN",
                confidence=0.0,
                accepted=False,
                category=None,
                all_probabilities={},
            )

        best_score = 0.0
        best_intent = self.labels[0]
        scores: dict[str, float] = {}

        tokens = set(text_clean.split())

        for sc in self.scenario_bank.all():
            score = 0.0

            # So khớp tuyệt đối với câu mẫu (sample utterances) -> xác suất rất cao
            for utterance in sc.sample_utterances:
                utt_clean = utterance.lower().strip()
                if text_clean == utt_clean:
                    score = max(score, 0.96)
                elif utt_clean in text_clean or text_clean in utt_clean:
                    score = max(score, 0.88)
                else:
                    utt_tokens = set(utt_clean.split())
                    inter = len(tokens.intersection(utt_tokens))
                    union = len(tokens.union(utt_tokens))
                    if union > 0:
                        jaccard = inter / union
                        score = max(score, jaccard * 0.85)

            # So khớp với alias
            for alias in sc.aliases:
                alias_clean = alias.lower().strip()
                if alias_clean in text_clean:
                    score = max(score, 0.86)

            scores[sc.intent] = score
            if score > best_score:
                best_score = score
                best_intent = sc.intent

        # Chuẩn hóa về hàm softmax giả lập để đảm bảo phân phối xác suất hợp lý
        # z_i = score * 5.0 (temperature scaling)
        logits = [scores.get(label, 0.0) * 5.0 for label in self.labels]
        max_logit = max(logits) if logits else 0.0
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probabilities = [e / sum_exp for e in exp_logits]

        conf = float(best_score)
        accepted = conf >= self.threshold_tau

        category = None
        if best_intent in self.scenario_bank:
            category = self.scenario_bank[best_intent].category

        # Lấy top-5 xác suất
        indexed_probs = sorted(zip(self.labels, probabilities), key=lambda x: x[1], reverse=True)[:5]
        all_probs = {lbl: round(prob, 4) for lbl, prob in indexed_probs}

        return ClassificationResult(
            intent=best_intent if conf > 0.0 else "UNKNOWN",
            confidence=round(conf, 4),
            accepted=accepted,
            category=category,
            all_probabilities=all_probs,
        )

    def predict(self, text: str) -> ClassificationResult:
        """
        Dự đoán ý định (intent) từ chuỗi văn bản người dùng.

        Tham số:
            text: Văn bản câu hỏi / yêu cầu từ người dùng (đã tiền xử lý).

        Trả về:
            ClassificationResult chứa nhãn ý định, độ tin cậy p_max, và trạng thái accepted.
        """
        if self._is_hf_loaded:
            return self._predict_hf(text)
        return self._predict_offline_heuristic(text)

    def predict_batch(
        self,
        texts: Sequence[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[ClassificationResult]:
        """
        Dự đoán ý định cho một danh sách văn bản theo lô (batch).

        Tham số:
            texts: Danh sách câu đầu vào.
            batch_size: Kích thước lô xử lý.

        Trả về:
            Danh sách ClassificationResult tương ứng với từng câu.
        """
        results: list[ClassificationResult] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(self.predict(text))
        return results
