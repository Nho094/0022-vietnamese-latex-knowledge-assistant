"""
Kiểm thử tự động cho bộ máy suy luận IntentClassifier và kiểm thực vân tay tiền xử lý (core/classifier.py).

Tuân thủ kiến trúc và các quy tắc trung thực trong AGENTS.md:
- Kiểm tra cơ chế suy luận JointBERT Eq.(1) (Chen et al., 2019):
    z^i = W^i * Dropout(h_0) + b^i
    p^i = softmax(z^i / T)
- Kiểm tra chuẩn hóa phân phối xác suất (Probability Normalization): sum(p) == 1.0.
- Kiểm tra điểm móc co giãn nhiệt độ (Temperature Scaling Hook theo Guo et al., 2017).
- Kiểm tra kiểm thực vân tay tiền xử lý tại thời điểm chạy (Runtime Preprocessing Fingerprint Verification).
- Kiểm tra xử lý chuỗi đơn lẻ `predict` và suy luận theo lô `predict_batch`.
- Kiểm tra tiền xử lý văn bản qua `prepare_for_phobert` và xử lý an toàn chuỗi rỗng.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import RobertaConfig

from core.classifier import (
    ClassificationResult,
    FingerprintMismatchError,
    IntentClassifier,
    IntentPrediction,
    ModelArtifactError,
    save_artifact_fingerprint,
    verify_preprocessing_fingerprint,
)
from core.models import PhoBertIntentClassifier
from core.preprocess import preprocess_fingerprint

# =====================================================================
# Fixtures kiểm thử
# =====================================================================


@pytest.fixture
def tiny_roberta_config() -> RobertaConfig:
    """Cấu hình RobertaConfig siêu nhẹ phục vụ kiểm thử nhanh không cần tải trọng số nặng."""
    return RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        classifier_dropout=0.1,
    )


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Mock bộ tách từ AutoTokenizer tương thích chuẩn PyTorch Transformers."""
    tokenizer = MagicMock()

    def fake_tokenize(
        texts: list[str] | str,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        seq_len = 8
        return {
            "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }

    tokenizer.side_effect = fake_tokenize
    return tokenizer


# =====================================================================
# 1. Kiểm thử cấu trúc ClassificationResult
# =====================================================================


def test_classification_result_properties() -> None:
    """Kiểm tra thuộc tính, bí danh và phương thức to_dict của ClassificationResult."""
    res = ClassificationResult(
        intent="MATH_PHAN_SO",
        confidence=0.95,
        accepted=True,
        category="math_expression",
        all_probabilities={"MATH_PHAN_SO": 0.95, "MATH_TICH_PHAN": 0.05},
        raw_text="tạo phân số 1/2",
        preprocessed_text="tạo phân_số 1/2",
    )

    assert res.intent == "MATH_PHAN_SO"
    assert res.confidence == 0.95
    assert res.accepted is True
    assert res.category == "math_expression"
    assert res.probabilities == res.all_probabilities
    assert res.raw_text == "tạo phân số 1/2"
    assert res.preprocessed_text == "tạo phân_số 1/2"

    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["intent"] == "MATH_PHAN_SO"
    assert d["confidence"] == 0.95
    assert d["accepted"] is True
    assert d["category"] == "math_expression"

    # Bí danh IntentPrediction
    assert IntentPrediction is ClassificationResult


# =====================================================================
# 2. Kiểm thử kiểm thực vân tay tiền xử lý (Fingerprint Verification)
# =====================================================================


def test_save_and_verify_preprocessing_fingerprint_success(tmp_path: Path) -> None:
    """Kiểm tra lưu và kiểm thực vân tay tiền xử lý thành công khi khớp mã băm."""
    fp_path = save_artifact_fingerprint(tmp_path, target="intent")
    assert fp_path.exists()

    with open(fp_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "fingerprint_sha256" in data
    assert data["target"] == "intent"

    # Kiểm thực nghiêm ngặt thành công
    is_valid = verify_preprocessing_fingerprint(tmp_path, strict=True)
    assert is_valid is True


def test_verify_preprocessing_fingerprint_mismatch_raises_error(tmp_path: Path) -> None:
    """Kiểm tra phát hiện sai lệch vân tay tiền xử lý và ném ngoại lệ FingerprintMismatchError."""
    # Tạo metadata với vân tay giả mạo không khớp môi trường runtime
    tampered_fp = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    save_artifact_fingerprint(tmp_path, fingerprint_hash=tampered_fp)

    with pytest.raises(FingerprintMismatchError, match="Lệch vân tay tiền xử lý"):
        verify_preprocessing_fingerprint(tmp_path, strict=True)

    # Khi strict=False thì trả về False và không ném ngoại lệ
    is_valid = verify_preprocessing_fingerprint(tmp_path, strict=False)
    assert is_valid is False


def test_verify_preprocessing_fingerprint_fallback_to_config_json(tmp_path: Path) -> None:
    """Kiểm tra khả năng tìm kiếm vân tay dự phòng trong config.json nếu thiếu preprocessing_fingerprint.json."""
    current_fp = preprocess_fingerprint()
    config_data = {
        "num_labels": 3,
        "preprocessing_fingerprint": current_fp,
    }
    with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    assert verify_preprocessing_fingerprint(tmp_path, strict=True) is True


def test_verify_preprocessing_fingerprint_missing_metadata(tmp_path: Path) -> None:
    """Kiểm tra ném lỗi khi thư mục artifact không chứa bất kỳ tệp metadata vân tay nào."""
    with pytest.raises(FingerprintMismatchError, match="Không tìm thấy metadata vân tay"):
        verify_preprocessing_fingerprint(tmp_path, strict=True)

    assert verify_preprocessing_fingerprint(tmp_path, strict=False) is False


def test_verify_preprocessing_fingerprint_nonexistent_directory(tmp_path: Path) -> None:
    """Kiểm tra xử lý thư mục artifact không tồn tại."""
    ghost_dir = tmp_path / "ghost_artifact"
    with pytest.raises(ModelArtifactError):
        verify_preprocessing_fingerprint(ghost_dir, strict=True)

    assert verify_preprocessing_fingerprint(ghost_dir, strict=False) is False


# =====================================================================
# 3. Kiểm thử khởi tạo và cấu hình nhiệt độ (Temperature Scaling)
# =====================================================================


def test_intent_classifier_temperature_validation(
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """Kiểm tra ràng buộc giá trị nhiệt độ T > 0 (Guo et al., 2017)."""
    clf = IntentClassifier.from_config(
        tiny_roberta_config,
        num_labels=2,
        tokenizer=mock_tokenizer,
        temperature=1.0,
    )
    assert clf.temperature == 1.0

    # Cập nhật nhiệt độ hợp lệ
    clf.set_temperature(1.5)
    assert clf.temperature == 1.5

    clf.temperature = 0.8
    assert clf.temperature == 0.8

    # Nhiệt độ không hợp lệ (<= 0) phải ném ValueError
    with pytest.raises(ValueError, match="temperature phải lớn hơn 0"):
        clf.set_temperature(0.0)

    with pytest.raises(ValueError, match="temperature phải lớn hơn 0"):
        clf.set_temperature(-1.0)


def test_intent_classifier_temperature_scaling_mathematical_effect(
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """
    Kiểm tra hiệu ứng toán học của Temperature Scaling theo Guo et al., 2017:
    - Khi T tăng (T > 1.0): Entropy tăng, xác suất cực đại giảm (làm mịn).
    - Khi T giảm (T < 1.0): Entropy giảm, xác suất cực đại tăng (làm sắc).
    """
    clf = IntentClassifier.from_config(
        tiny_roberta_config,
        num_labels=3,
        tokenizer=mock_tokenizer,
        id2label={0: "INTENT_A", 1: "INTENT_B", 2: "INTENT_C"},
    )

    # Cố định logits đầu ra của mô hình để kiểm tra độc lập công thức
    fixed_logits = torch.tensor([[3.0, 1.0, 0.0]])
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = fixed_logits
    mock_model.return_value = mock_output
    clf.model = mock_model

    # 1. Chuẩn hóa tại T = 1.0 (tiêu chuẩn)
    res_t1 = clf.predict("câu kiểm tra", temperature=1.0)
    conf_t1 = res_t1.confidence

    # 2. Làm mịn tại T = 2.0 (nhiệt độ cao)
    res_t2 = clf.predict("câu kiểm tra", temperature=2.0)
    conf_t2 = res_t2.confidence
    assert conf_t2 < conf_t1, "Khi T=2.0, xác suất cực đại phải giảm so với T=1.0"

    # 3. Làm sắc tại T = 0.5 (nhiệt độ thấp)
    res_t05 = clf.predict("câu kiểm tra", temperature=0.5)
    conf_t05 = res_t05.confidence
    assert conf_t05 > conf_t1, "Khi T=0.5, xác suất cực đại phải tăng so với T=1.0"


def test_intent_classifier_custom_temperature_hook(
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """Kiểm tra đăng ký và thực thi hook co giãn nhiệt độ tùy biến (Custom Hook)."""
    hook_called = False

    def custom_hook(logits: torch.Tensor, temp: float) -> torch.Tensor:
        nonlocal hook_called
        hook_called = True
        return logits / (temp * 2.0)

    clf = IntentClassifier.from_config(
        tiny_roberta_config,
        num_labels=2,
        tokenizer=mock_tokenizer,
        temperature_scaling_hook=custom_hook,
    )

    clf.predict("câu kiểm tra")
    assert hook_called is True


# =====================================================================
# 4. Kiểm thử chuẩn hóa xác suất và dự đoán đơn lẻ (predict)
# =====================================================================


def test_intent_classifier_probability_normalization(
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """
    Kiểm tra tính chuẩn hóa xác suất (Probability Normalization):
    Tổng các xác suất p_i trên mọi lớp phải bằng 1.0 và mỗi p_i nằm trong [0.0, 1.0].
    """
    clf = IntentClassifier.from_config(
        tiny_roberta_config,
        num_labels=4,
        tokenizer=mock_tokenizer,
        id2label={0: "CLS_0", 1: "CLS_1", 2: "CLS_2", 3: "CLS_3"},
    )

    res = clf.predict("hướng dẫn tạo ma trận trong LaTeX")
    assert isinstance(res, ClassificationResult)
    assert 0.0 <= res.confidence <= 1.0

    total_prob = sum(res.all_probabilities.values())
    assert abs(total_prob - 1.0) < 1e-5, f"Tổng xác suất phải bằng 1.0, nhận được: {total_prob}"

    for p in res.all_probabilities.values():
        assert 0.0 <= p <= 1.0


def test_intent_classifier_threshold_tau_acceptance(
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """Kiểm tra điều kiện chấp nhận accepted = (confidence >= tau) theo AGENTS.md."""
    clf = IntentClassifier.from_config(
        tiny_roberta_config,
        num_labels=2,
        tokenizer=mock_tokenizer,
        threshold_tau=0.8,
    )

    # Giả lập logits mang lại confidence cao (> 0.8)
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[5.0, -5.0]])
    mock_model.return_value = mock_output
    clf.model = mock_model

    res_accepted = clf.predict("câu kiểm tra đạt ngưỡng")
    assert res_accepted.confidence > 0.8
    assert res_accepted.accepted is True

    # Giả lập logits phân vân mang lại confidence thấp (< 0.8)
    mock_output.logits = torch.tensor([[0.1, 0.0]])
    res_rejected = clf.predict("câu kiểm tra phân vân")
    assert res_rejected.confidence < 0.8
    assert res_rejected.accepted is False


def test_intent_classifier_empty_text_handling(
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """Kiểm tra xử lý văn bản rỗng hoặc chỉ chứa khoảng trắng một cách an toàn."""
    clf = IntentClassifier.from_config(
        tiny_roberta_config,
        num_labels=2,
        tokenizer=mock_tokenizer,
    )

    for empty_input in ("", "   ", "\t\n"):
        res = clf.predict(empty_input)
        assert res.intent == "UNKNOWN"
        assert res.confidence == 0.0
        assert res.accepted is False
        assert res.preprocessed_text == ""


# =====================================================================
# 5. Kiểm thử suy luận theo lô (predict_batch)
# =====================================================================


def test_intent_classifier_predict_batch_integrity_and_ordering(
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """Kiểm tra suy luận theo lô với nhiều kích thước lô, bảo toàn đúng thứ tự các câu đầu vào."""
    clf = IntentClassifier.from_config(
        tiny_roberta_config,
        num_labels=3,
        tokenizer=mock_tokenizer,
        id2label={0: "MATH_PHAN_SO", 1: "TEX_BANG", 2: "TEX_ANH"},
        category_map={
            "MATH_PHAN_SO": "math_expression",
            "TEX_BANG": "tables",
            "TEX_ANH": "figures",
        },
    )

    inputs = [
        "câu thứ nhất tạo phân số",
        "",  # Câu rỗng xen kẽ
        "câu thứ ba chèn bảng",
        "câu thứ tư chèn hình ảnh",
        "    ",  # Câu khoảng trắng xen kẽ
    ]

    # Kiểm tra với batch_size=2 để kích hoạt chia nhiều lô nhỏ
    results = clf.predict_batch(inputs, batch_size=2)
    assert len(results) == len(inputs)

    # Câu rỗng được định danh UNKNOWN chính xác tại index 1 và 4
    assert results[1].intent == "UNKNOWN"
    assert results[1].confidence == 0.0
    assert results[4].intent == "UNKNOWN"
    assert results[4].confidence == 0.0

    # Các câu hợp lệ có kết quả đầy đủ
    for i in (0, 2, 3):
        assert results[i].intent in ("MATH_PHAN_SO", "TEX_BANG", "TEX_ANH")
        assert results[i].category in ("math_expression", "tables", "figures")
        total_p = sum(results[i].all_probabilities.values())
        assert abs(total_p - 1.0) < 1e-5


def test_intent_classifier_predict_batch_empty_list(
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """Kiểm tra gọi predict_batch với danh sách rỗng."""
    clf = IntentClassifier.from_config(
        tiny_roberta_config,
        num_labels=2,
        tokenizer=mock_tokenizer,
    )
    assert clf.predict_batch([]) == []


# =====================================================================
# 6. Kiểm thử nạp từ thư mục artifact (from_pretrained)
# =====================================================================


def test_intent_classifier_from_pretrained_end_to_end(
    tmp_path: Path,
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """
    Kiểm thử quy trình from_pretrained đầy đủ:
    Lưu checkpoint mô hình, metadata vân tay, ánh xạ nhãn, sau đó nạp lại và kiểm thực.
    """
    model_dir = tmp_path / "saved_phobert_checkpoint"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Khởi tạo và lưu mô hình
    model = PhoBertIntentClassifier.from_config(tiny_roberta_config, num_labels=2)
    model.save_pretrained(model_dir)

    # 2. Lưu ánh xạ nhãn
    label2id = {"INTENT_X": 0, "INTENT_Y": 1}
    id2label = {0: "INTENT_X", 1: "INTENT_Y"}
    with open(model_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f)
    with open(model_dir / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f)

    # 3. Lưu vân tay tiền xử lý hợp lệ
    save_artifact_fingerprint(model_dir, target="intent")

    # 4. Nạp qua from_pretrained với mock tokenizer
    with patch("core.classifier.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        clf = IntentClassifier.from_pretrained(
            model_dir,
            device="cpu",
            verify_fingerprint=True,
            strict_fingerprint=True,
        )

    assert clf.num_labels == 2
    assert clf.id2label[0] == "INTENT_X"
    assert clf.id2label[1] == "INTENT_Y"

    # Kiểm tra suy luận trên mô hình vừa nạp
    res = clf.predict("kiểm thử mô hình sau khi nạp")
    assert res.intent in ("INTENT_X", "INTENT_Y")
    assert 0.0 <= res.confidence <= 1.0


def test_intent_classifier_from_pretrained_fingerprint_mismatch_blocks_loading(
    tmp_path: Path,
    tiny_roberta_config: RobertaConfig,
) -> None:
    """Kiểm tra from_pretrained chặn nạp mô hình khi phát hiện vân tay tiền xử lý bị lệch."""
    model_dir = tmp_path / "corrupted_checkpoint"
    model_dir.mkdir(parents=True, exist_ok=True)

    model = PhoBertIntentClassifier.from_config(tiny_roberta_config, num_labels=2)
    model.save_pretrained(model_dir)

    # Ghi vân tay bị lệch
    save_artifact_fingerprint(model_dir, fingerprint_hash="invalid_fingerprint_hash_1234567890")

    with pytest.raises(FingerprintMismatchError, match="Lệch vân tay tiền xử lý"):
        IntentClassifier.from_pretrained(
            model_dir,
            verify_fingerprint=True,
            strict_fingerprint=True,
        )
