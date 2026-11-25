"""
Kiểm thử tự động cho kiến trúc mạng nơ-ron phân loại ý định PhoBERT (core/models.py).

Tuân thủ kiến trúc và các quy tắc trung thực trong AGENTS.md:
- Kiểm tra kiến trúc phân loại ý định JointBERT Eq.(1) (Chen et al., 2019):
    z^i = W^i * Dropout(h_0) + b^i
- Kiểm tra tính trích xuất vector h_0 từ token đặc biệt đầu tiên (<s> tại index 0).
- Kiểm tra độ lệch kiến trúc (architectural deviation) so với RobertaClassificationHead của Hugging Face:
  xác nhận lược bỏ tầng Dense trung gian (768 -> 768) và hàm kích hoạt tanh.
- Kiểm tra tính toán hàm mất mát CrossEntropyLoss khi có labels, kiểm tra lan truyền ngược (backward pass).
- Kiểm tra hàm tiện ích predict_step, phân phối xác suất softmax, và hỗ trợ return_dict=False.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from core.config import (
    DEFAULT_CLASSIFIER_DROPOUT,
    PHOBERT_HIDDEN_DIM,
)
from core.models import (
    JointBertPhoBertClassifier,
    PhoBertIntentClassifier,
)

# Kiểm tra tính khả dụng của torch và transformers
try:
    import torch
    from torch import nn
    from transformers import RobertaConfig
    from transformers.modeling_outputs import SequenceClassifierOutput

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def test_models_module_import_and_docstring() -> None:
    """
    Kiểm tra nhập mô-đun core.models, kiểm tra tài liệu hóa tiếng Việt,
    công thức toán JointBERT Eq.(1), và bí danh JointBertPhoBertClassifier.
    Hoạt động độc lập cả khi chưa nạp trọng số thực.
    """
    assert PhoBertIntentClassifier.__doc__ is not None
    assert "JointBERT" in PhoBertIntentClassifier.__doc__
    assert "Eq.(1)" in PhoBertIntentClassifier.__doc__
    assert "tanh" in PhoBertIntentClassifier.__doc__
    assert JointBertPhoBertClassifier is PhoBertIntentClassifier


@pytest.fixture
def tiny_roberta_config() -> Any:
    """Fixture tạo cấu hình RobertaConfig siêu nhẹ phục vụ kiểm thử nhanh và độc lập."""
    if not _HAS_TORCH:
        pytest.skip("Yêu cầu PyTorch và Transformers.")
    return RobertaConfig(
        vocab_size=120,
        hidden_size=PHOBERT_HIDDEN_DIM,  # 768 chiều ẩn chuẩn của PhoBERT base
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=256,
        max_position_embeddings=128,
        classifier_dropout=DEFAULT_CLASSIFIER_DROPOUT,
    )


def test_model_initialization_with_config(tiny_roberta_config: Any) -> None:
    """
    Kiểm tra khởi tạo PhoBertIntentClassifier với cấu hình tham số chuẩn.
    Xác nhận các tầng mạng nơ-ron: AutoModel backbone, Dropout(0.1), và Linear(768, K).
    """
    if not _HAS_TORCH:
        pytest.skip("Yêu cầu PyTorch và Transformers.")
    num_labels = 12
    model = PhoBertIntentClassifier.from_config(
        config=tiny_roberta_config,
        num_labels=num_labels,
        dropout_rate=DEFAULT_CLASSIFIER_DROPOUT,
    )

    assert model.num_labels == num_labels
    assert model.hidden_size == PHOBERT_HIDDEN_DIM
    assert model.dropout_rate == DEFAULT_CLASSIFIER_DROPOUT
    assert isinstance(model.dropout, nn.Dropout)
    assert model.dropout.p == DEFAULT_CLASSIFIER_DROPOUT
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.in_features == PHOBERT_HIDDEN_DIM
    assert model.classifier.out_features == num_labels
    assert model.classifier.bias is not None


def test_from_pretrained_loads_safetensors_classifier_checkpoint(tmp_path, tiny_roberta_config: Any) -> None:
    """Xác nhận checkpoint ngoài VSIX dạng safetensors qua đúng loader PhoBERT runtime."""
    if not _HAS_TORCH:
        pytest.skip("Yêu cầu PyTorch và Transformers.")
    from safetensors.torch import save_file

    checkpoint = tmp_path / "external-checkpoint"
    checkpoint.mkdir()
    tiny_roberta_config.num_labels = 3
    tiny_roberta_config.save_pretrained(checkpoint)
    original = PhoBertIntentClassifier.from_config(tiny_roberta_config, num_labels=3)
    save_file(
        {name: tensor.detach().contiguous() for name, tensor in original.state_dict().items()},
        str(checkpoint / "model.safetensors"),
    )

    restored = PhoBertIntentClassifier.from_pretrained(checkpoint, num_labels=3)

    assert restored.num_labels == 3
    assert all(
        torch.equal(original.state_dict()[key], restored.state_dict()[key])
        for key in original.state_dict()
    )


def test_architectural_deviation_from_stock_hf_head(tiny_roberta_config: Any) -> None:
    """
    Kiểm tra độ lệch kiến trúc có chủ đích so với RobertaClassificationHead của Hugging Face:
    - Stock Hugging Face head có: dense (Linear 768->768), activation tanh, dropout, out_proj.
    - JointBERT Eq.(1) head: LƯỢC BỎ dense và tanh, chiếu trực tiếp h_0 qua Linear(768, K).
    - Kiểm tra docstring tài liệu hóa đầy đủ lý do thiết kế (WHY) và công thức JointBERT Eq.(1).
    """
    if not _HAS_TORCH:
        pytest.skip("Yêu cầu PyTorch và Transformers.")
    model = PhoBertIntentClassifier.from_config(
        config=tiny_roberta_config,
        num_labels=5,
    )

    # Xác nhận không có tầng Dense trung gian 768->768
    assert not hasattr(model, "dense")
    assert not hasattr(model, "intermediate_dense")

    # Xác nhận tầng classifier chiếu trực tiếp từ hidden_size (768) sang num_labels (5)
    assert model.classifier.in_features == PHOBERT_HIDDEN_DIM
    assert model.classifier.out_features == 5


def test_model_forward_pass_without_labels(tiny_roberta_config: Any) -> None:
    """
    Kiểm tra lan truyền xuôi khi không có nhãn (chế độ suy luận).
    Kết quả trả về đối tượng SequenceClassifierOutput với loss=None và logits có kích thước [batch_size, K].
    """
    if not _HAS_TORCH:
        pytest.skip("Yêu cầu PyTorch và Transformers.")
    batch_size = 3
    seq_len = 10
    num_labels = 7
    model = PhoBertIntentClassifier.from_config(tiny_roberta_config, num_labels=num_labels)
    model.eval()

    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert isinstance(outputs, SequenceClassifierOutput)
    assert outputs.loss is None
    assert outputs.logits is not None
    assert outputs.logits.shape == (batch_size, num_labels)


def test_model_forward_pass_with_labels_and_backward(tiny_roberta_config: Any) -> None:
    """
    Kiểm tra lan truyền xuôi khi có labels (chế độ huấn luyện/đánh giá loss):
    - Tính CrossEntropyLoss hợp lệ (> 0).
    - Lan truyền ngược loss.backward() cập nhật gradient cho cả classifier và backbone PhoBERT.
    """
    if not _HAS_TORCH:
        pytest.skip("Yêu cầu PyTorch và Transformers.")
    batch_size = 2
    seq_len = 8
    num_labels = 4
    model = PhoBertIntentClassifier.from_config(tiny_roberta_config, num_labels=num_labels)
    model.train()

    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = torch.tensor([1, 3], dtype=torch.long)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert outputs.loss is not None
    loss_val = float(outputs.loss.item())
    assert loss_val > 0.0
    assert outputs.logits.shape == (batch_size, num_labels)

    # Kiểm tra lan truyền ngược (backward pass)
    outputs.loss.backward()
    assert model.classifier.weight.grad is not None
    assert model.classifier.bias.grad is not None
    # Trọng số tầng embedding của backbone nhận gradient
    embedding_weight = model.phobert.embeddings.word_embeddings.weight
    assert embedding_weight.grad is not None


def test_h0_token_isolation() -> None:
    """
    Kiểm tra tính cô lập của vector h_0: Chỉ có token đầu tiên (<s> tại vị trí 0)
    được sử dụng để tính toán logits phân loại ý định (JointBERT Eq.(1)).
    Các token vị trí >= 1 bị biến đổi không được làm thay đổi logits nếu h_0 được giữ nguyên.
    """
    if not _HAS_TORCH:
        pytest.skip("Yêu cầu PyTorch và Transformers.")
    batch_size = 1
    seq_len = 4
    hidden_dim = PHOBERT_HIDDEN_DIM
    num_labels = 3

    # Giả lập backbone PhoBERT trả về sequence_output nhân tạo
    mock_backbone = MagicMock()

    # Tạo sequence tensor với token 0 cố định và các token sau thay đổi
    seq_out_1 = torch.zeros((batch_size, seq_len, hidden_dim))
    seq_out_1[:, 0, :] = 1.5  # h_0 token đặc biệt đầu tiên <s>
    seq_out_1[:, 1, :] = 0.5  # token vị trí 1
    seq_out_1[:, 2, :] = 0.8  # token vị trí 2

    seq_out_2 = torch.zeros((batch_size, seq_len, hidden_dim))
    seq_out_2[:, 0, :] = 1.5  # h_0 giữ nguyên tuyệt đối
    seq_out_2[:, 1, :] = 9.9  # thay đổi hoàn toàn token vị trí 1
    seq_out_2[:, 2, :] = -4.2  # thay đổi hoàn toàn token vị trí 2

    mock_outputs_1 = MagicMock()
    mock_outputs_1.__getitem__.side_effect = lambda idx: seq_out_1 if idx == 0 else None
    mock_outputs_1.hidden_states = None
    mock_outputs_1.attentions = None

    mock_outputs_2 = MagicMock()
    mock_outputs_2.__getitem__.side_effect = lambda idx: seq_out_2 if idx == 0 else None
    mock_outputs_2.hidden_states = None
    mock_outputs_2.attentions = None

    mock_backbone.side_effect = [mock_outputs_1, mock_outputs_2]

    model = PhoBertIntentClassifier(
        num_labels=num_labels,
        phobert_backbone=mock_backbone,
        dropout_rate=0.0,  # tắt dropout để kiểm tra tất định
    )
    model.eval()

    dummy_input = torch.tensor([[10, 20, 30, 40]])
    out1 = model(dummy_input)
    out2 = model(dummy_input)

    # Logits phải giống hệt nhau vì h_0 không đổi dù token 1 và token 2 thay đổi
    assert torch.allclose(out1.logits, out2.logits, atol=1e-6)


def test_predict_step_and_softmax(tiny_roberta_config: Any) -> None:
    """
    Kiểm tra hàm tiện ích predict_step:
    - Trả về nhãn argmax và phân phối xác suất softmax.
    - Tổng các xác suất theo chiều lớp phải bằng 1.0.
    """
    if not _HAS_TORCH:
        pytest.skip("Yêu cầu PyTorch và Transformers.")
    batch_size = 4
    seq_len = 6
    num_labels = 5
    model = PhoBertIntentClassifier.from_config(tiny_roberta_config, num_labels=num_labels)

    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    predictions, probabilities = model.predict_step(input_ids)

    assert predictions.shape == (batch_size,)
    assert probabilities.shape == (batch_size, num_labels)
    # Xác nhận các giá trị xác suất trong khoảng [0, 1]
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()
    # Tổng xác suất bằng 1.0
    sum_probs = probabilities.sum(dim=-1)
    assert torch.allclose(sum_probs, torch.ones(batch_size), atol=1e-5)
    # Giá trị nhãn dự đoán khớp với argmax của probabilities
    assert torch.equal(predictions, torch.argmax(probabilities, dim=-1))


def test_return_dict_false_tuple_output(tiny_roberta_config: Any) -> None:
    """
    Kiểm tra tương thích với cờ return_dict=False (trả về tuple thay vì SequenceClassifierOutput).
    """
    if not _HAS_TORCH:
        pytest.skip("Yêu cầu PyTorch và Transformers.")
    model = PhoBertIntentClassifier.from_config(tiny_roberta_config, num_labels=3)
    model.eval()

    input_ids = torch.randint(0, 100, (2, 5))
    outputs = model(input_ids, return_dict=False)

    assert isinstance(outputs, tuple)
    logits = outputs[0]
    assert logits.shape == (2, 3)

