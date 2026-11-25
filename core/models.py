"""
Kiến trúc mạng nơ-ron phân loại ý định tùy biến PhoBERT theo phong cách JointBERT.

Mô-đun này triển khai bộ phân loại ý định (Intent Classifier) dựa trên PhoBERT
tuân thủ công thức phương trình Eq.(1) của mô hình JointBERT
(Chen, Zhuo, & Wang, 2019 - "BERT for Joint Intent Classification and Slot Filling", arXiv:1902.10909):
    y^i = softmax(W^i * h_0 + b^i)

Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Trích xuất biểu diễn ngữ cảnh h_0 của token đặc biệt đầu tiên (<s> tại vị trí index 0).
- Áp dụng Dropout(0.1) và tầng chiếu tuyến tính Linear(768, K) trực tiếp lên h_0.
- Lược bỏ tầng Dense trung gian và hàm kích hoạt tanh (độ lệch kiến trúc có chủ đích
  so với RobertaClassificationHead tiêu chuẩn của Hugging Face).
- Tham số cấu hình tập trung từ core/config.py (không dùng số ma thuật).
- Chú thích kiểu dữ liệu (type hints) đầy đủ và tài liệu hóa bằng tiếng Việt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel, PretrainedConfig
    from transformers.modeling_outputs import SequenceClassifierOutput

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from core.config import (
    DEFAULT_CLASSIFIER_DROPOUT,
    PHOBERT_HIDDEN_DIM,
    PHOBERT_MODEL_NAME,
)

if _TORCH_AVAILABLE:
    _BaseModule = nn.Module
else:  # pragma: no cover
    class _BaseModule:  # type: ignore[no-redef]
        """Lớp cơ sở thay thế khi môi trường chưa cài đặt PyTorch."""


class PhoBertIntentClassifier(_BaseModule):
    """
    Mạng nơ-ron phân loại ý định tùy biến dựa trên PhoBERT theo kiến trúc JointBERT Eq.(1).

    Chi tiết công thức lý thuyết (Chen, Zhuo, & Wang, 2019 - JointBERT):
        z^i = W^i * Dropout(h_0) + b^i
        p^i = softmax(z^i)
        y^i = argmax_k(p^i_k)
    Trong đó:
        - h_0: Vector ẩn của token đặc biệt đầu tiên (<s> trong PhoBERT / RoBERTa, tương đương [CLS] trong BERT).
          Vector này đại diện cho toàn bộ ngữ cảnh ngữ nghĩa của câu đầu vào.
        - W^i in R^{K x d}, b^i in R^K: Trọng số và độ lệch của tầng chiếu tuyến tính Linear(768, K).
        - K: Số lượng lớp ý định (num_labels).
        - d = 768: Chiều không gian ẩn của PhoBERT base (hidden_size).
        - Dropout(p=0.1): Tỷ lệ ngắt kết nối ngẫu nhiên chống quá khớp (overfitting).

    GHI CHÚ VỀ ĐỘ LỆCH KIẾN TRÚC SO VỚI HUGGING FACE STOCK HEAD (ARCHITECTURAL DEVIATION):
    Head phân loại chuỗi tiêu chuẩn của Hugging Face (RobertaClassificationHead trong
    RobertaForSequenceClassification) có cấu trúc dạng mạng nhiều tầng (MLP):
        1. x = Dropout(h_0)
        2. x = Linear(hidden_size, hidden_size)(x)  [Tầng Dense trung gian 768 -> 768]
        3. x = tanh(x)                             [Hàm kích hoạt phi tuyến tanh]
        4. x = Dropout(x)
        5. logits = Linear(hidden_size, num_labels)(x)

    Trong kiến trúc PhoBertIntentClassifier của dự án này, chúng tôi CHỦ ĐỘNG LƯỢC BỎ:
        - Tầng Dense trung gian (nn.Linear(768, 768))
        - Hàm kích hoạt phi tuyến tanh (torch.tanh)
        - Lớp Dropout thứ hai

    Độ lệch kiến trúc này áp dụng trực tiếp cơ chế chiếu tuyến tính đơn giản:
        h_0 -> Dropout(0.1) -> Linear(768, K) -> logits

    LÝ DO THIẾT KẾ (WHY):
        1. Bám sát chính xác công thức toán học Eq.(1) của JointBERT (Chen et al., 2019) và
           chuẩn thực nghiệm của bài báo PhoATIS trong JointIDSF (VinAI Research).
        2. Tối giản tham số mạng: Việc lược bỏ tầng Dense (768 x 768 = 589,824 tham số trọng số
           cộng thêm 768 bias) giúp mô hình nhẹ hơn, huấn luyện nhanh hơn và đặc biệt giảm nguy cơ
           overfitting nghiêm trọng trên các tập dữ liệu ý định có quy mô mẫu vừa và nhỏ.
        3. Bảo toàn biểu diễn ngữ cảnh: Biểu diễn vector của token <s> đã được PhoBERT học sâu qua
           12 tầng Transformer đa đầu chú ý (self-attention). Phép chiếu tuyến tính trực tiếp W^i * h_0
           đóng vai trò như một siêu phẳng phân tách tuyến tính tối ưu trong không gian ngữ nghĩa 768 chiều.
    """

    def __init__(
        self,
        config: PretrainedConfig | None = None,
        num_labels: int = 2,
        dropout_rate: float = DEFAULT_CLASSIFIER_DROPOUT,
        model_name_or_path: str | Path | None = None,
        phobert_backbone: Any | None = None,
    ) -> None:
        """
        Khởi tạo mạng phân loại ý định PhoBertIntentClassifier.

        Tham số:
            config: Đối tượng PretrainedConfig của Hugging Face (tùy chọn).
            num_labels: Số lượng lớp ý định K cần dự đoán (mặc định 2).
            dropout_rate: Tỷ lệ dropout áp dụng lên h_0 trước phép chiếu tuyến tính (mặc định 0.1).
            model_name_or_path: Tên mô hình trên Hugging Face Hub hoặc đường dẫn checkpoint cục bộ.
            phobert_backbone: Đối tượng mô hình backbone AutoModel đã khởi tạo sẵn (dành cho DI/mock).
        """
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "Mô-đun core.models yêu cầu PyTorch và Hugging Face Transformers. "
                "Vui lòng cài đặt các thư viện phụ thuộc qua `pip install -r requirements.txt`."
            )

        super().__init__()
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate

        # Xác định cấu hình và chiều ẩn hidden_size (mặc định 768 cho PhoBERT base)
        self.config = config
        if config is not None and hasattr(config, "hidden_size"):
            self.hidden_size: int = config.hidden_size
        else:
            self.hidden_size = PHOBERT_HIDDEN_DIM

        # Khởi tạo mô hình backbone PhoBERT (AutoModel)
        if phobert_backbone is not None:
            self.phobert = phobert_backbone
        elif model_name_or_path is not None:
            self.phobert = AutoModel.from_pretrained(str(model_name_or_path), config=config)
        elif config is not None:
            self.phobert = AutoModel.from_config(config)
        else:
            self.phobert = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)

        # Lớp Dropout(0.1) áp dụng trên biểu diễn h_0
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Tầng chiếu tuyến tính Linear(768, K) trực tiếp từ h_0 tới K ý định
        # Tuân thủ JointBERT Eq.(1): logits = W^i * h_0 + b^i
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        # Khởi tạo trọng số cho tầng phân loại (Zero-mean Gaussian với std=0.02 theo RoBERTa chuẩn)
        self._init_classifier_weights()

    def _init_classifier_weights(self) -> None:
        """
        Khởi tạo trọng số cho tầng phân loại tuyến tính theo chuẩn RoBERTa (Normal(0, 0.02), bias=0).
        """
        if hasattr(self, "classifier") and isinstance(self.classifier, nn.Linear):
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
            if self.classifier.bias is not None:
                nn.init.zeros_(self.classifier.bias)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        num_labels: int = 2,
        dropout_rate: float = DEFAULT_CLASSIFIER_DROPOUT,
        **kwargs: Any,
    ) -> PhoBertIntentClassifier:
        """
        Nạp mô hình từ checkpoint pre-trained hoặc thư mục bộ nhớ đệm cục bộ.

        Tham số:
            pretrained_model_name_or_path: Đường dẫn checkpoint hoặc tên mô hình Hugging Face.
            num_labels: Số lượng lớp ý định K.
            dropout_rate: Tỷ lệ dropout (mặc định 0.1).
            **kwargs: Các đối số bổ sung truyền vào AutoConfig / AutoModel.
        """
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("Yêu cầu PyTorch và Transformers để nạp mô hình.")

        path_obj = Path(pretrained_model_name_or_path)
        path_str = str(pretrained_model_name_or_path)
        config = AutoConfig.from_pretrained(path_str, num_labels=num_labels, **kwargs)
        weights_file = path_obj / "pytorch_model.bin"
        safetensors_file = path_obj / "model.safetensors"
        if weights_file.exists() or safetensors_file.exists():
            backbone = AutoModel.from_config(config)
            model = cls(
                config=config,
                num_labels=num_labels,
                dropout_rate=dropout_rate,
                phobert_backbone=backbone,
            )
            if weights_file.exists():
                state_dict = torch.load(weights_file, map_location="cpu")
            else:
                try:
                    from safetensors.torch import load_file
                except ImportError as error:
                    raise ImportError("Checkpoint safetensors cần package safetensors.") from error
                state_dict = load_file(str(safetensors_file), device="cpu")
            model.load_state_dict(state_dict)
            return model

        backbone = AutoModel.from_pretrained(path_str, config=config, **kwargs)
        return cls(
            config=config,
            num_labels=num_labels,
            dropout_rate=dropout_rate,
            phobert_backbone=backbone,
        )

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """
        Lưu trọng số mô hình và cấu hình vào thư mục chỉ định.

        Tham số:
            save_directory: Đường dẫn thư mục lưu trữ checkpoint.

        Trả về:
            Đối tượng Path dẫn đến thư mục checkpoint đã lưu.
        """
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("Yêu cầu PyTorch để lưu checkpoint.")

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        if self.config is not None and hasattr(self.config, "save_pretrained"):
            self.config.save_pretrained(str(save_path))
        weights_path = save_path / "pytorch_model.bin"
        torch.save(self.state_dict(), weights_path)
        return save_path

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        num_labels: int = 2,
        dropout_rate: float = DEFAULT_CLASSIFIER_DROPOUT,
    ) -> PhoBertIntentClassifier:
        """
        Khởi tạo mô hình ngẫu nhiên từ đối tượng cấu hình PretrainedConfig (không tải trọng số).
        Rất hữu ích cho unit testing và kiểm thử kiến trúc độc lập.
        """
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("Yêu cầu PyTorch và Transformers để khởi tạo từ cấu hình.")

        backbone = AutoModel.from_config(config)
        return cls(
            config=config,
            num_labels=num_labels,
            dropout_rate=dropout_rate,
            phobert_backbone=backbone,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput | tuple[Any, ...]:
        """
        Lan truyền xuôi (Forward pass) của mạng phân loại ý định JointBERT PhoBERT.

        Quy trình xử lý theo JointBERT Eq.(1):
            1. Trích xuất đầu ra chuỗi sequence_output từ backbone PhoBERT.
            2. Lấy biểu diễn h_0 của token đặc biệt đầu tiên (<s> tại vị trí index 0):
                   h_0 = sequence_output[:, 0, :]
            3. Áp dụng Dropout(0.1):
                   h_0_drop = self.dropout(h_0)
            4. Chiếu tuyến tính Linear(768, K):
                   logits = self.classifier(h_0_drop)
            5. Tính hàm mất mát CrossEntropyLoss nếu cung cấp labels.

        Tham số:
            input_ids: Tensor mã định danh token đầu vào, kích thước [batch_size, seq_len].
            attention_mask: Tensor mặt nạ chú ý, kích thước [batch_size, seq_len].
            token_type_ids: Tensor định danh phân đoạn (segment) token (nếu có).
            position_ids: Tensor vị trí token (nếu có).
            head_mask: Mặt nạ đầu chú ý (attention head mask).
            inputs_embeds: Tensor nhúng đầu vào trực tiếp (nếu có).
            labels: Tensor nhãn lớp ý định ground truth, kích thước [batch_size].
            output_attentions: Cờ trả về trọng số chú ý attention weights.
            output_hidden_states: Cờ trả về trạng thái ẩn của tất cả các tầng.
            return_dict: Trả về đối tượng SequenceClassifierOutput hay tuple.

        Trả về:
            SequenceClassifierOutput chứa loss (nếu có labels), logits [batch_size, K],
            cùng hidden_states và attentions (nếu được yêu cầu).
        """
        use_return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "use_return_dict", True)
        )

        # Chuyển tiếp qua mô hình backbone PhoBERT
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        # sequence_output có kích thước [batch_size, seq_len, hidden_size]
        sequence_output = outputs[0]

        # Trích xuất biểu diễn h_0 của token đặc biệt đầu tiên <s> (vị trí 0 trong PhoBERT/RoBERTa)
        # JointBERT Eq.(1): h_0 = sequence_output[:, 0, :]
        h_0 = sequence_output[:, 0, :]

        # Áp dụng Dropout(0.1) chống overfitting
        h_0_drop = self.dropout(h_0)

        # Phép chiếu tuyến tính Linear(768, K) trực tiếp ra logits
        # JointBERT Eq.(1): z^i = W^i * h_0 + b^i
        logits = self.classifier(h_0_drop)

        # Tính toán hàm mất mát CrossEntropyLoss nếu có nhãn kiểm định/huấn luyện
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not use_return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def predict_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Phương thức tiện ích thực hiện suy luận nhanh và tính toán xác suất softmax.

        Công thức:
            p = softmax(logits, dim=-1)
            pred = argmax(p, dim=-1)

        Tham số:
            input_ids: Tensor token đầu vào [batch_size, seq_len].
            attention_mask: Tensor mặt nạ chú ý [batch_size, seq_len].

        Trả về:
            Tuple (predictions, probabilities):
                predictions: Tensor chứa chỉ số ý định dự đoán [batch_size].
                probabilities: Tensor chứa phân phối xác suất softmax [batch_size, num_labels].
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities


# Bí danh tương thích cho kiến trúc JointBERT
JointBertPhoBertClassifier = PhoBertIntentClassifier
