"""
Kịch bản tinh chỉnh Bi-Encoder (Sentence-BERT / Vietnamese Bi-Encoder) với các chế độ Contrastive và Triplet.

Tệp tin: train/triplet_sbert.py
Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Kiến trúc mạng Bi-Encoder linh hoạt:
  + Kiến trúc Siamese chia sẻ trọng số (Siamese shared-weight architecture) theo Sentence-BERT
    (Reimers & Gurevych, EMNLP 2019 - arXiv:1908.10084) khi shared_weights=True.
  + Kiến trúc Dual-Encoder hai nhánh độc lập (Dual-encoder architecture) theo Dense Passage Retrieval
    (DPR, Karpukhin et al., EMNLP 2020 - arXiv:2004.04906) khi shared_weights=False.
- Các chế độ hàm mất mát hỗ trợ (Training Modes):
  + Mode A (In-batch Contrastive): InfoNCE / DPR Eq.(1) (van den Oord et al., 2018; Karpukhin et al., 2020)
    kết hợp cơ chế che mẫu âm trùng lặp (Identity Masking - Khosla et al., 2020).
  + Mode A2 (Hard Negative Contrastive): Mở rộng mẫu số với top-K hard negatives theo DPR Eq.(2).
  + Mode B (Triplet Training): FaceNet Eq.(3) (Schroff, Kalenichenko, & Philbin, CVPR 2015 - arXiv:1503.03832)
    với khoảng cách Euclid và lề biên (margin).
- Tối ưu hóa và lịch trình học:
  + Tối ưu hóa AdamW (Loshchilov & Hutter, 2019) với learning rate mặc định 2e-5, batch size 16 hoặc 32.
  + Khởi động tuyến tính (linear warm-up) trên 10% tổng số bước và suy giảm tuyến tính (linear decay)
    (Devlin et al., 2018; Liu et al., 2019).
  + Dừng sớm (Early Stopping) trong khoảng 3 đến 10 epochs dựa trên chỉ số Top-1 Retrieval trên tập Validation.
- Quy tắc Trung thực #3 (Honesty Rule #3):
  + Tuyệt đối KHÔNG đọc, tinh chỉnh hay đánh giá trên tập kiểm thử data/test.csv trong quá trình huấn luyện.
  + Kiểm thực mã băm SHA-256 phòng vệ chống rò rỉ dữ liệu test set.
- Chế độ chạy thử nhanh (Smoke Mode Flag):
  + Hỗ trợ cờ --smoke phục vụ kiểm thử siêu tốc trên CPU cho CI/CD và unit testing mà không cần tải trọng số nặng.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Final

# Đảm bảo đường dẫn gốc dự án có trong sys.path khi thực thi trực tiếp script
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaConfig

from core.config import (
    BIENCODER_CANDIDATE_BATCH_SIZES,
    BIENCODER_DEFAULT_BATCH_SIZE,
    BIENCODER_DEFAULT_LR,
    BIENCODER_DEFAULT_PATIENCE,
    BIENCODER_MAX_EPOCHS,
    BIENCODER_MIN_EPOCHS,
    BIENCODER_WARMUP_RATIO,
    BIENCODER_WEIGHT_DECAY,
    DEFAULT_BIENCODER_CHECKPOINT_DIR,
    DEFAULT_BIENCODER_RESULTS_PATH,
    DEFAULT_CONTRASTIVE_SCALE,
    DEFAULT_HARD_NEGATIVE_TOP_K,
    DEFAULT_INCLUDE_ALIASES,
    DEFAULT_INTENTS_PATH,
    DEFAULT_SCENARIO_VARIANT,
    DEFAULT_SEED,
    DEFAULT_SHARED_WEIGHTS,
    DEFAULT_TRAINING_MODE,
    DEFAULT_TRIPLET_MARGIN,
    MAX_SEQ_LENGTH,
    PHOBERT_DIR,
    PHOBERT_MODEL_NAME,
    SBERT_DIR,
    SBERT_MODEL_NAME,
    SMOKE_RUN_BIENCODER_BATCH_SIZE,
    SMOKE_RUN_BIENCODER_EPOCHS,
    SMOKE_RUN_BIENCODER_MAX_SAMPLES,
    SUPPORTED_TRAINING_MODES,
    TRAIN_CSV_PATH,
    TRAINING_MODE_A,
    TRAINING_MODE_A2,
    TRAINING_MODE_B,
    VAL_CSV_PATH,
    get_default_device,
    set_seed,
)
from core.preprocess import prepare_for_embedding, preprocess_fingerprint
from core.scenario import scenario_string
from train.biencoder_dataset import (
    BiEncoderHardNegativeDataset,
    BiEncoderPairDataset,
    BiEncoderTripletDataset,
    DatasetSample,
    load_intent_bank,
    load_train_samples,
    mine_hard_negatives,
    verify_not_test_set,
)
from train.evaluate import compute_top_k_accuracy
from train.losses import (
    euclidean_triplet_loss,
    hard_negative_contrastive_loss,
    in_batch_contrastive_loss,
)

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Cấu trúc Mô hình Bi-Encoder (Siamese vs Dual-Encoder)
# =====================================================================


class TransformerEncoder(nn.Module):
    """
    Tầng mã hóa Transformer kết hợp gom cụm trung bình (Mean Pooling) theo Sentence-BERT.

    Công thức tham chiếu (Reimers & Gurevych, 2019 - Sentence-BERT Eq.(1)):
        e = sum_{t=1}^T (m_t * h_t) / sum_{t=1}^T m_t
    trong đó:
        - h_t in R^d là biểu diễn vector ẩn của token thứ t tại tầng cuối cùng (last_hidden_state).
        - m_t in {0, 1} là mặt nạ chú ý (attention_mask) loại trừ các token đệm padding.
    """

    def __init__(self, backbone: nn.Module) -> None:
        """
        Khởi tạo TransformerEncoder với mạng nơ-ron nền tảng.

        Tham số:
            backbone: Đối tượng mô hình Transformer của HuggingFace (ví dụ: AutoModel hoặc RobertaModel).
        """
        super().__init__()
        self.backbone = backbone
        # RoBERTa pooler dùng pooled_output, nhưng Sentence-BERT mean-pooling chỉ đọc
        # last_hidden_state; đóng băng pooler tránh tham số không bao giờ nhận gradient.
        pooler = getattr(backbone, "pooler", None)
        if pooler is not None:
            for parameter in pooler.parameters():
                parameter.requires_grad_(False)
        self.embedding_dim: int = getattr(
            getattr(backbone, "config", None),
            "hidden_size",
            768,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện lan truyền tiến và trích xuất vector nhúng câu qua mean pooling.

        Tham số:
            input_ids: Tensor mã token đầu vào, kích thước (batch_size, seq_len).
            attention_mask: Tensor mặt nạ chú ý, kích thước (batch_size, seq_len).

        Trả về:
            Tensor vector nhúng câu chưa chuẩn hóa L2, kích thước (batch_size, embedding_dim).
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_embeddings = outputs.last_hidden_state  # (B, L, D)

        # Áp dụng mặt nạ attention_mask để tính mean pooling chính xác
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class BiEncoderModel(nn.Module):
    """
    Kiến trúc Bi-Encoder hỗ trợ cả hai mô hình thiết kế:
    1. Siamese Shared-Weight: Hai nhánh dùng chung một bộ trọng số duy nhất (shared_weights=True)
       theo chuẩn Sentence-BERT (Reimers & Gurevych, 2019).
    2. Dual-Encoder: Hai nhánh query_encoder và doc_encoder sở hữu các tham số độc lập hoàn toàn
       (shared_weights=False) theo chuẩn Dense Passage Retrieval (Karpukhin et al., 2020).
    """

    def __init__(
        self,
        query_encoder: TransformerEncoder,
        doc_encoder: TransformerEncoder | None = None,
        shared_weights: bool = DEFAULT_SHARED_WEIGHTS,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Khởi tạo mô hình BiEncoderModel.

        Tham số:
            query_encoder: Bộ mã hóa nhánh câu hỏi / truy vấn (Query Encoder).
            doc_encoder: Bộ mã hóa nhánh tài liệu / kịch bản (Doc Encoder).
                         Nếu None và shared_weights=False, tự động sao chép sâu (deepcopy) từ query_encoder.
            shared_weights: Cờ xác định cấu trúc: True (Siamese chia sẻ trọng số) hoặc False (Dual-encoder độc lập).
            normalize_embeddings: Cờ chuẩn hóa L2 cho các vector đầu ra (||v||_2 = 1.0).
        """
        super().__init__()
        self.shared_weights = bool(shared_weights)
        self.normalize_embeddings = bool(normalize_embeddings)

        # Nhánh query encoder luôn được đăng ký chính thức
        self.query_encoder_module = query_encoder
        self.embedding_dim: int = query_encoder.embedding_dim

        if self.shared_weights:
            # Chế độ Siamese: nhánh doc dùng chung module với query
            self.doc_encoder_module: TransformerEncoder | None = None
        else:
            # Chế độ Dual-encoder: nhánh doc là một module độc lập
            if doc_encoder is not None:
                self.doc_encoder_module = doc_encoder
            else:
                self.doc_encoder_module = copy.deepcopy(query_encoder)

    @property
    def query_encoder(self) -> TransformerEncoder:
        """Thuộc tính truy xuất nhánh Query Encoder."""
        return self.query_encoder_module

    @property
    def doc_encoder(self) -> TransformerEncoder:
        """Thuộc tính truy xuất nhánh Doc Encoder (trả về query_encoder nếu shared_weights=True)."""
        if self.shared_weights or self.doc_encoder_module is None:
            return self.query_encoder_module
        return self.doc_encoder_module

    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mã hóa các câu truy vấn qua nhánh Query Encoder.

        Tham số:
            input_ids: Tensor mã token (B, L).
            attention_mask: Tensor mặt nạ chú ý (B, L).

        Trả về:
            Tensor biểu diễn vector (B, D), đã chuẩn hóa L2 nếu normalize_embeddings=True.
        """
        embeddings = self.query_encoder(input_ids, attention_mask)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_doc(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mã hóa các tài liệu/kịch bản qua nhánh Doc Encoder.

        Tham số:
            input_ids: Tensor mã token (B, L).
            attention_mask: Tensor mặt nạ chú ý (B, L).

        Trả về:
            Tensor biểu diễn vector (B, D), đã chuẩn hóa L2 nếu normalize_embeddings=True.
        """
        embeddings = self.doc_encoder(input_ids, attention_mask)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def save_pretrained(self, save_directory: Path | str) -> None:
        """
        Lưu trữ checkpoint trọng số và tệp cấu hình Bi-Encoder vào thư mục chỉ định.

        Tham số:
            save_directory: Thư mục lưu trữ.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        config_data = {
            "model_type": "biencoder",
            "shared_weights": self.shared_weights,
            "normalize_embeddings": self.normalize_embeddings,
            "embedding_dim": self.embedding_dim,
        }
        with open(save_path / "biencoder_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        save_directory: Path | str,
        backbone_factory: Any,
        device: str = "cpu",
    ) -> BiEncoderModel:
        """
        Khôi phục mô hình BiEncoderModel từ thư mục checkpoint đã lưu.

        Tham số:
            save_directory: Thư mục chứa tệp pytorch_model.bin và biencoder_config.json.
            backbone_factory: Hàm callable không đối số trả về một module Transformer backbone mới.
            device: Thiết bị đích nạp trọng số.

        Trả về:
            Đối tượng BiEncoderModel đã tải trọng số hoàn chỉnh.
        """
        save_path = Path(save_directory)
        config_file = save_path / "biencoder_config.json"
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        shared_weights = bool(config_data.get("shared_weights", True))
        normalize_embeddings = bool(config_data.get("normalize_embeddings", True))

        q_backbone = backbone_factory()
        q_encoder = TransformerEncoder(q_backbone)

        d_encoder: TransformerEncoder | None = None
        if not shared_weights:
            d_backbone = backbone_factory()
            d_encoder = TransformerEncoder(d_backbone)

        model = cls(
            query_encoder=q_encoder,
            doc_encoder=d_encoder,
            shared_weights=shared_weights,
            normalize_embeddings=normalize_embeddings,
        )
        weights_file = save_path / "pytorch_model.bin"
        state_dict = torch.load(weights_file, map_location=device)
        model.load_state_dict(state_dict)
        return model.to(device)


# =====================================================================
# 2. Khởi tạo Mô hình và Tokenizer
# =====================================================================


def load_tokenizer(model_dir: Path | str | None = None) -> Any:
    """
    Nạp bộ tách từ AutoTokenizer ưu tiên từ bộ nhớ đệm ngoại tuyến local_files_only=True.

    Tham số:
        model_dir: Đường dẫn thư mục mô hình (mặc định PHOBERT_DIR hoặc SBERT_DIR).

    Trả về:
        Đối tượng AutoTokenizer của HuggingFace.
    """
    candidates = [
        Path(model_dir) if model_dir else None,
        SBERT_DIR,
        PHOBERT_DIR,
    ]
    for cand in candidates:
        if cand and cand.exists() and (cand / "vocab.txt").exists():
            try:
                return AutoTokenizer.from_pretrained(str(cand), local_files_only=True)
            except (OSError, ValueError) as err:
                logger.debug("Không thể nạp tokenizer ngoại tuyến từ %s: %s", cand, err)
                continue

    # Fallback trực tiếp tới định danh mô hình trên Hub
    try:
        return AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    except (OSError, ValueError):
        if PHOBERT_DIR.exists():
            return AutoTokenizer.from_pretrained(str(PHOBERT_DIR))
        raise


def create_biencoder_model(
    shared_weights: bool = DEFAULT_SHARED_WEIGHTS,
    smoke_mode: bool = False,
    model_name_or_path: Path | str | None = None,
    device: str = "cpu",
) -> BiEncoderModel:
    """
    Khởi tạo mạng Bi-Encoder với cấu hình kiến trúc (Siamese vs Dual-Encoder) và chế độ thực thi.

    Chiến lược khởi tạo:
    1. Nếu smoke_mode=True: Khởi tạo mô hình RobertaConfig siêu nhẹ (1 tầng, 2 heads, 64-dim)
       phục vụ kiểm thử siêu tốc trên CPU mà không cần tải file trọng số 540MB.
    2. Nếu chế độ chuẩn: Thử nạp backbone tiền huấn luyện từ SBERT_DIR / PHOBERT_DIR hoặc Hub.
       Nếu môi trường ngoại tuyến chưa có tệp trọng số safetensors, an toàn fallback từ AutoConfig.

    Tham số:
        shared_weights: True (Siamese chia sẻ trọng số) hoặc False (Dual-encoder độc lập).
        smoke_mode: Cờ kích hoạt chế độ siêu nhẹ phục vụ CPU execution.
        model_name_or_path: Đường dẫn mô hình pre-trained hoặc None.
        device: Thiết bị tính toán ('cpu', 'cuda').

    Trả về:
        Đối tượng BiEncoderModel sẵn sàng trên thiết bị device.
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
        q_backbone = AutoModel.from_config(tiny_config)
        q_encoder = TransformerEncoder(q_backbone)

        d_encoder: TransformerEncoder | None = None
        if not shared_weights:
            d_backbone = AutoModel.from_config(tiny_config)
            d_encoder = TransformerEncoder(d_backbone)

        model = BiEncoderModel(
            query_encoder=q_encoder,
            doc_encoder=d_encoder,
            shared_weights=shared_weights,
            normalize_embeddings=True,
        )
        return model.to(device)

    # Chế độ bình thường: nạp backbone từ thư mục ngoại tuyến hoặc Hub
    target_path = Path(model_name_or_path) if model_name_or_path else None
    if target_path is None:
        if SBERT_DIR.exists() and any(SBERT_DIR.glob("*.bin")) or any(SBERT_DIR.glob("*.safetensors")):
            target_path = SBERT_DIR
        elif PHOBERT_DIR.exists() and any(PHOBERT_DIR.glob("*.bin")) or any(PHOBERT_DIR.glob("*.safetensors")):
            target_path = PHOBERT_DIR
        else:
            target_path = Path(SBERT_MODEL_NAME)

    def _build_backbone() -> nn.Module:
        try:
            return AutoModel.from_pretrained(str(target_path), local_files_only=True)
        except (OSError, ValueError, RuntimeError):
            try:
                return AutoModel.from_pretrained(str(target_path))
            except (OSError, ValueError, RuntimeError):
                # An toàn fallback khởi tạo từ config nếu ngoại tuyến thiếu weights
                config = AutoConfig.from_pretrained(PHOBERT_MODEL_NAME)
                return AutoModel.from_config(config)

    q_backbone = _build_backbone()
    q_encoder = TransformerEncoder(q_backbone)

    d_encoder = None
    if not shared_weights:
        d_backbone = _build_backbone()
        d_encoder = TransformerEncoder(d_backbone)

    model = BiEncoderModel(
        query_encoder=q_encoder,
        doc_encoder=d_encoder,
        shared_weights=shared_weights,
        normalize_embeddings=True,
    )
    return model.to(device)


# =====================================================================
# 3. Lịch trình Tốc độ học (Linear Warm-up & Linear Decay)
# =====================================================================


def create_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Tạo bộ điều chỉnh tốc độ học có khởi động tuyến tính (warm-up) và suy giảm tuyến tính (decay).

    Công thức tham chiếu (Devlin et al., 2018 - BERT; Liu et al., 2019 - RoBERTa):
        Nếu current_step < num_warmup_steps:
            lr_factor = current_step / max(1, num_warmup_steps)
        Ngược lại:
            lr_factor = max(0.0, (num_training_steps - current_step) / max(1, num_training_steps - num_warmup_steps))

    LÝ DO THIẾT KẾ (WHY):
        Khởi động tuyến tính 10% tổng số bước giúp ổn định gradient ban đầu, bảo vệ biểu diễn
        tiền huấn luyện của backbone; sau đó suy giảm dần về 0 để hội tụ tối ưu.
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
# 4. Các hàm Collate đóng gói Lô (Collate Functions)
# =====================================================================


def collate_mode_a(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> dict[str, Any]:
    """
    Đóng gói lô cho Mode A: In-batch Contrastive Learning.

    Tham số:
        batch: Danh sách các phần tử từ BiEncoderPairDataset.
        tokenizer: Đối tượng HuggingFace Tokenizer.
        max_seq_length: Chiều dài tối đa của chuỗi token.

    Trả về:
        Dictionary chứa tensor queries, positives và scenario_indices.
    """
    queries = [item["query"] for item in batch]
    positives = [item["positive"] for item in batch]
    scenario_indices = torch.tensor(
        [int(item["scenario_index"]) for item in batch],
        dtype=torch.long,
    )

    q_tok = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    p_tok = tokenizer(
        positives,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    return {
        "queries": q_tok,
        "positives": p_tok,
        "scenario_indices": scenario_indices,
    }


def collate_mode_a2(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> dict[str, Any]:
    """
    Đóng gói lô cho Mode A2: Hard Negative Contrastive Learning (DPR Eq.(2)).

    Tham số:
        batch: Danh sách các phần tử từ BiEncoderHardNegativeDataset.
        tokenizer: Tokenizer.
        max_seq_length: Độ dài chuỗi tối đa.

    Trả về:
        Dictionary chứa tensor queries, positives, hard_negatives 3D, và indices.
    """
    queries = [item["query"] for item in batch]
    positives = [item["positive"] for item in batch]
    scenario_indices = torch.tensor(
        [int(item["scenario_index"]) for item in batch],
        dtype=torch.long,
    )

    # Làm phẳng danh sách các mẫu âm khó để tokenize đồng loạt
    batch_size = len(batch)
    num_hn = len(batch[0]["hard_negatives"]) if batch else 0

    flat_hard_negatives: list[str] = []
    for item in batch:
        flat_hard_negatives.extend(item["hard_negatives"])

    hard_scenario_indices = torch.tensor(
        [item["hard_scenario_indices"] for item in batch],
        dtype=torch.long,
    )

    q_tok = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    p_tok = tokenizer(
        positives,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    hn_tok = tokenizer(
        flat_hard_negatives,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    return {
        "queries": q_tok,
        "positives": p_tok,
        "hard_negatives": hn_tok,
        "scenario_indices": scenario_indices,
        "hard_scenario_indices": hard_scenario_indices,
        "batch_size": batch_size,
        "num_hn": num_hn,
    }


def collate_mode_b(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> dict[str, Any]:
    """
    Đóng gói lô cho Mode B: Triplet Loss (FaceNet Eq.(3)).

    Tham số:
        batch: Danh sách các phần tử từ BiEncoderTripletDataset.
        tokenizer: Tokenizer.
        max_seq_length: Độ dài chuỗi tối đa.

    Trả về:
        Dictionary chứa tensor anchors, positives, negatives.
    """
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negative"] for item in batch]

    a_tok = tokenizer(
        anchors,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    p_tok = tokenizer(
        positives,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    n_tok = tokenizer(
        negatives,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    return {
        "anchors": a_tok,
        "positives": p_tok,
        "negatives": n_tok,
    }


# =====================================================================
# 5. Đánh giá Truy hồi Top-1 trên Tập Validation (Val Top-1 Retrieval)
# =====================================================================


def evaluate_validation_retrieval(
    model: BiEncoderModel,
    val_samples: list[DatasetSample],
    intent_bank: dict[str, dict[str, Any]],
    tokenizer: Any,
    device: str,
    batch_size: int = 32,
    max_seq_length: int = MAX_SEQ_LENGTH,
    mode: str = TRAINING_MODE_A,
    candidate_variant: str = DEFAULT_SCENARIO_VARIANT,
) -> dict[str, float]:
    """
    Đánh giá độ chính xác truy hồi Top-1 (Top-1 Retrieval Accuracy) trên tập Validation.

    Quy trình tính toán:
    1. Xây dựng kho kịch bản ứng viên chuẩn {s_j} cho toàn bộ các ý định trong intent_bank.
    2. Mã hóa toàn bộ kho ứng viên bằng doc_encoder thu được ma trận vector ứng viên E_cand (N_cand, D).
    3. Mã hóa toàn bộ các câu truy vấn trong val_samples bằng query_encoder thu được ma trận E_q (N_val, D).
    4. Xếp hạng ứng viên:
       - Mode A & Mode A2: Tính độ tương đồng cosine (sim = E_q @ E_cand.T) và sắp xếp giảm dần.
       - Mode B: Tính khoảng cách Euclid (dist = ||E_q - E_cand||_2) và sắp xếp tăng dần.
    5. Kiểm tra ứng viên hạng 1 (Top-1 candidate) có trùng khớp với nhãn ground-truth (val.label) hay không.
    6. Tính toán Top-1 Retrieval Accuracy và Mean Reciprocal Rank (MRR).

    Tuân thủ Honesty Rule #3:
        Hàm này chỉ nhận val_samples từ val.csv, tuyệt đối cấm dùng test.csv.

    Tham số:
        model: Mô hình BiEncoderModel.
        val_samples: Danh sách mẫu đánh giá từ val.csv.
        intent_bank: Ngân hàng kịch bản intents.json.
        tokenizer: Tokenizer.
        device: Thiết bị tính toán ('cpu', 'cuda').
        batch_size: Kích thước lô khi mã hóa embedding.
        max_seq_length: Độ dài chuỗi token tối đa.
        mode: Chế độ huấn luyện ('mode_a', 'mode_a2', 'mode_b').
        candidate_variant: Biến thể kịch bản ứng viên (mặc định 'handmade').

    Trả về:
        Dictionary chứa 'top1_accuracy', 'mrr', và 'total_samples'.
    """
    if not val_samples:
        return {"top1_accuracy": 0.0, "mrr": 0.0, "total_samples": 0.0}

    model.eval()

    # 1. Chuẩn bị danh sách các ứng viên kịch bản
    candidate_intents = sorted(intent_bank.keys())
    candidate_texts: list[str] = []
    for intent_name in candidate_intents:
        raw_str = scenario_string(intent_bank[intent_name], variant=candidate_variant)
        candidate_texts.append(prepare_for_embedding(raw_str))

    # 2. Mã hóa toàn bộ kịch bản ứng viên qua doc_encoder
    cand_embeddings_list: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(candidate_texts), batch_size):
            chunk = candidate_texts[i : i + batch_size]
            tok = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            ).to(device)
            emb = model.encode_doc(tok["input_ids"], tok["attention_mask"])
            cand_embeddings_list.append(emb)

    cand_embeddings = torch.cat(cand_embeddings_list, dim=0)  # (N_cand, D)

    # 3. Mã hóa toàn bộ các câu truy vấn validation qua query_encoder
    val_queries = [prepare_for_embedding(s.text) for s in val_samples]
    val_trues = [s.label for s in val_samples]

    query_embeddings_list: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(val_queries), batch_size):
            chunk = val_queries[i : i + batch_size]
            tok = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            ).to(device)
            emb = model.encode_query(tok["input_ids"], tok["attention_mask"])
            query_embeddings_list.append(emb)

    query_embeddings = torch.cat(query_embeddings_list, dim=0)  # (N_val, D)

    # 4. Xếp hạng ứng viên theo độ tương đồng hoặc khoảng cách Euclid
    with torch.no_grad():
        if mode in (TRAINING_MODE_A, TRAINING_MODE_A2):
            # Tích vô hướng của các vector chuẩn hóa L2 chính là Cosine Similarity
            scores = torch.matmul(query_embeddings, cand_embeddings.transpose(0, 1))  # (N_val, N_cand)
            ranked_indices = torch.argsort(scores, dim=-1, descending=True)
        else:
            # Mode B: Khoảng cách Euclid (khoảng cách càng nhỏ càng tương đồng)
            dists = torch.cdist(query_embeddings, cand_embeddings, p=2.0)  # (N_val, N_cand)
            ranked_indices = torch.argsort(dists, dim=-1, descending=False)

    ranked_indices_cpu = ranked_indices.cpu().tolist()

    # 5. Tính toán Top-1 Accuracy và MRR
    ranked_intents_all: list[list[str]] = []
    reciprocal_ranks: list[float] = []

    for row_indices, true_intent in zip(ranked_indices_cpu, val_trues):
        ranked_row = [candidate_intents[idx] for idx in row_indices]
        ranked_intents_all.append(ranked_row)

        if true_intent in ranked_row:
            rank = ranked_row.index(true_intent) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    top1_acc = compute_top_k_accuracy(y_true=val_trues, y_pred_ranked=ranked_intents_all, k=1)
    mrr = float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0

    return {
        "top1_accuracy": float(top1_acc),
        "mrr": float(mrr),
        "total_samples": float(len(val_samples)),
    }


# =====================================================================
# 6. Vòng lặp Huấn luyện Bi-Encoder với Dừng sớm (Training Loop)
# =====================================================================


def train_biencoder(
    model: BiEncoderModel,
    train_loader: DataLoader,
    val_samples: list[DatasetSample],
    intent_bank: dict[str, dict[str, Any]],
    tokenizer: Any,
    mode: str = DEFAULT_TRAINING_MODE,
    learning_rate: float = BIENCODER_DEFAULT_LR,
    min_epochs: int = BIENCODER_MIN_EPOCHS,
    max_epochs: int = BIENCODER_MAX_EPOCHS,
    patience: int = BIENCODER_DEFAULT_PATIENCE,
    warmup_ratio: float = BIENCODER_WARMUP_RATIO,
    weight_decay: float = BIENCODER_WEIGHT_DECAY,
    scale: float = DEFAULT_CONTRASTIVE_SCALE,
    triplet_margin: float = DEFAULT_TRIPLET_MARGIN,
    device: str = "cpu",
) -> tuple[BiEncoderModel, dict[str, Any]]:
    """
    Thực hiện quy trình huấn luyện Bi-Encoder với tối ưu hóa AdamW, warm-up và dừng sớm (Early Stopping).

    Quy trình:
    1. Khởi tạo optimizer AdamW trên toàn bộ tham số của model.
    2. Thiết lập lịch trình học Linear Schedule với Warmup trên 10% tổng số bước.
    3. Đánh giá Top-1 Retrieval trên tập Validation sau mỗi epoch:
       - Nếu Val Top-1 Accuracy cải thiện: Lưu bản sao state_dict tốt nhất và đặt lại patience_counter = 0.
       - Nếu không cải thiện liên tiếp qua `patience` epochs và số epoch đã vượt `min_epochs`:
         Kích hoạt Dừng sớm (Early Stopping) nhằm chống quá khớp (overfitting).
    4. Khôi phục trọng số tại epoch có kết quả cao nhất trước khi trả về.

    Tham số:
        model: Mô hình BiEncoderModel.
        train_loader: DataLoader nạp dữ liệu huấn luyện.
        val_samples: Danh sách mẫu validation.
        intent_bank: Ngân hàng kịch bản intents.json.
        tokenizer: Tokenizer.
        mode: Chế độ huấn luyện ('mode_a', 'mode_a2', 'mode_b').
        learning_rate: Tốc độ học của AdamW (mặc định 2e-5).
        min_epochs: Số epoch tối thiểu trước khi kích hoạt dừng sớm (mặc định 3).
        max_epochs: Số epoch tối đa (mặc định 10).
        patience: Số epoch kiên nhẫn khi chỉ số val không tăng (mặc định 3).
        warmup_ratio: Tỷ lệ warm-up trên tổng số bước (mặc định 0.1).
        weight_decay: Trọng số phân rã chống overfitting (mặc định 0.01).
        scale: Hệ số scale nhiệt độ cho scaled_cosine (mặc định 20.0).
        triplet_margin: Lề khoảng cách cho Triplet Loss (mặc định 1.0).
        device: Thiết bị tính toán ('cpu', 'cuda').

    Trả về:
        Tuple gồm (mô hình với trọng số tối ưu nhất, tóm tắt lịch sử huấn luyện).
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_training_steps = max(1, len(train_loader) * max_epochs)
    num_warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = create_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_val_top1 = -1.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_val_metrics: dict[str, float] = {}
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if mode == TRAINING_MODE_A:
                q_ids = batch["queries"]["input_ids"].to(device)
                q_mask = batch["queries"]["attention_mask"].to(device)
                p_ids = batch["positives"]["input_ids"].to(device)
                p_mask = batch["positives"]["attention_mask"].to(device)
                scen_idx = batch["scenario_indices"].to(device)

                q_emb = model.encode_query(q_ids, q_mask)
                p_emb = model.encode_doc(p_ids, p_mask)

                loss = in_batch_contrastive_loss(
                    queries=q_emb,
                    positives=p_emb,
                    scenario_indices=scen_idx,
                    metric="scaled_cosine",
                    scale=scale,
                )

            elif mode == TRAINING_MODE_A2:
                q_ids = batch["queries"]["input_ids"].to(device)
                q_mask = batch["queries"]["attention_mask"].to(device)
                p_ids = batch["positives"]["input_ids"].to(device)
                p_mask = batch["positives"]["attention_mask"].to(device)
                hn_ids = batch["hard_negatives"]["input_ids"].to(device)
                hn_mask = batch["hard_negatives"]["attention_mask"].to(device)
                scen_idx = batch["scenario_indices"].to(device)
                hard_scen_idx = batch["hard_scenario_indices"].to(device)

                q_emb = model.encode_query(q_ids, q_mask)
                p_emb = model.encode_doc(p_ids, p_mask)

                # hn_flat_emb kích thước (B * num_hn, D), chuyển về 3D (B, num_hn, D)
                hn_flat_emb = model.encode_doc(hn_ids, hn_mask)
                b_sz = batch["batch_size"]
                n_hn = batch["num_hn"]
                hn_emb = hn_flat_emb.view(b_sz, n_hn, -1)

                loss = hard_negative_contrastive_loss(
                    queries=q_emb,
                    positives=p_emb,
                    hard_negatives=hn_emb,
                    scenario_indices=scen_idx,
                    hard_scenario_indices=hard_scen_idx,
                    metric="scaled_cosine",
                    scale=scale,
                )

            elif mode == TRAINING_MODE_B:
                a_ids = batch["anchors"]["input_ids"].to(device)
                a_mask = batch["anchors"]["attention_mask"].to(device)
                p_ids = batch["positives"]["input_ids"].to(device)
                p_mask = batch["positives"]["attention_mask"].to(device)
                n_ids = batch["negatives"]["input_ids"].to(device)
                n_mask = batch["negatives"]["attention_mask"].to(device)

                a_emb = model.encode_query(a_ids, a_mask)
                p_emb = model.encode_doc(p_ids, p_mask)
                n_emb = model.encode_doc(n_ids, n_mask)

                loss = euclidean_triplet_loss(
                    anchor=a_emb,
                    positive=p_emb,
                    negative=n_emb,
                    margin=triplet_margin,
                )
            else:
                raise ValueError(f"Chế độ huấn luyện '{mode}' không được hỗ trợ.")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            total_batches += 1

        avg_train_loss = epoch_loss / max(1, total_batches)

        # Đánh giá trên tập Validation
        val_eval = evaluate_validation_retrieval(
            model=model,
            val_samples=val_samples,
            intent_bank=intent_bank,
            tokenizer=tokenizer,
            device=device,
            mode=mode,
        )
        val_top1 = val_eval["top1_accuracy"]

        record = {
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_top1_accuracy": float(val_top1),
            "val_mrr": float(val_eval["mrr"]),
        }
        history.append(record)

        # Kiểm tra điều kiện Dừng sớm (Early Stopping)
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_metrics = val_eval
            patience_counter = 0
        else:
            patience_counter += 1
            if epoch >= min_epochs and patience_counter >= patience:
                # Dừng sớm nếu qua min_epochs và hết tính kiên nhẫn
                break

    # Khôi phục trạng thái mô hình tại epoch đạt kết quả cao nhất
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    summary = {
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "best_val_metrics": best_val_metrics,
        "epochs_trained": len(history),
        "history": history,
    }
    return model, summary


# =====================================================================
# 7. Toàn bộ Đường ống Huấn luyện Bi-Encoder (Full Pipeline)
# =====================================================================


def run_training_pipeline(
    mode: str = DEFAULT_TRAINING_MODE,
    shared_weights: bool = DEFAULT_SHARED_WEIGHTS,
    learning_rate: float = BIENCODER_DEFAULT_LR,
    batch_size: int = BIENCODER_DEFAULT_BATCH_SIZE,
    min_epochs: int = BIENCODER_MIN_EPOCHS,
    max_epochs: int = BIENCODER_MAX_EPOCHS,
    patience: int = BIENCODER_DEFAULT_PATIENCE,
    warmup_ratio: float = BIENCODER_WARMUP_RATIO,
    weight_decay: float = BIENCODER_WEIGHT_DECAY,
    scale: float = DEFAULT_CONTRASTIVE_SCALE,
    triplet_margin: float = DEFAULT_TRIPLET_MARGIN,
    hard_negative_top_k: int = DEFAULT_HARD_NEGATIVE_TOP_K,
    smoke_mode: bool = False,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    train_path: Path | str = TRAIN_CSV_PATH,
    val_path: Path | str = VAL_CSV_PATH,
    bank_path: Path | str = DEFAULT_INTENTS_PATH,
    checkpoint_dir: Path | str = DEFAULT_BIENCODER_CHECKPOINT_DIR,
    results_path: Path | str = DEFAULT_BIENCODER_RESULTS_PATH,
) -> dict[str, Any]:
    """
    Điều phối đường ống huấn luyện tinh chỉnh Bi-Encoder hoàn chỉnh.

    Tuân thủ nghiêm ngặt Quy tắc Trung thực #3 (Honesty Rule #3):
    - Tuyệt đối không chạm vào data/test.csv.
    - Đánh giá dừng sớm trên validation top-1 retrieval từ data/val.csv.
    """
    run_device = device if device is not None else get_default_device()
    set_seed(seed)

    # 1. Kiểm tra phòng vệ Honesty Rule #3: Không chạm vào test.csv
    verify_not_test_set(train_path)
    verify_not_test_set(val_path)

    # 2. Nạp dữ liệu huấn luyện và ngân hàng kịch bản
    intent_bank = load_intent_bank(bank_path)
    train_samples_raw = load_train_samples(train_path)

    # Đọc dữ liệu validation từ val.csv
    val_samples_raw: list[DatasetSample] = []
    with open(val_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = row.get("text", "").strip()
            lbl = row.get("label", "").strip()
            cat = row.get("category", "").strip()
            if txt and lbl:
                val_samples_raw.append(DatasetSample(text=txt, label=lbl, category=cat))

    # 3. Điều chỉnh tham số cho chế độ Smoke Mode
    if smoke_mode:
        effective_epochs = SMOKE_RUN_BIENCODER_EPOCHS
        effective_min_epochs = 1
        effective_patience = 1
        effective_batch_size = SMOKE_RUN_BIENCODER_BATCH_SIZE
        effective_train_samples = train_samples_raw[:SMOKE_RUN_BIENCODER_MAX_SAMPLES]
        effective_val_samples = val_samples_raw[:SMOKE_RUN_BIENCODER_MAX_SAMPLES]
    else:
        effective_epochs = max_epochs
        effective_min_epochs = min_epochs
        effective_patience = patience
        effective_batch_size = batch_size
        effective_train_samples = train_samples_raw
        effective_val_samples = val_samples_raw

    # 4. Nạp tokenizer và khởi tạo mô hình
    tokenizer = load_tokenizer()
    model = create_biencoder_model(
        shared_weights=shared_weights,
        smoke_mode=smoke_mode,
        device=run_device,
    )

    # 5. Xây dựng Dataset và DataLoader theo từng chế độ huấn luyện
    collate_fn: Any
    if mode == TRAINING_MODE_A:
        dataset = BiEncoderPairDataset(
            samples=effective_train_samples,
            intent_bank=intent_bank,
            include_aliases=DEFAULT_INCLUDE_ALIASES,
            variant_strategy="uniform",
            dynamic_sampling=True,
            seed=seed,
            bank_path=bank_path,
        )
        collate_fn = lambda b: collate_mode_a(b, tokenizer=tokenizer)
    elif mode == TRAINING_MODE_A2:
        mode_a2_samples, _ = mine_hard_negatives(
            train_samples=effective_train_samples,
            intent_bank=intent_bank,
            top_k=hard_negative_top_k,
            bank_path=bank_path,
        )
        dataset = BiEncoderHardNegativeDataset(mode_a2_samples=mode_a2_samples)
        collate_fn = lambda b: collate_mode_a2(b, tokenizer=tokenizer)
    elif mode == TRAINING_MODE_B:
        _, triplet_samples = mine_hard_negatives(
            train_samples=effective_train_samples,
            intent_bank=intent_bank,
            top_k=1,
            bank_path=bank_path,
        )
        dataset = BiEncoderTripletDataset(triplet_samples=triplet_samples)
        collate_fn = lambda b: collate_mode_b(b, tokenizer=tokenizer)
    else:
        raise ValueError(
            f"Chế độ '{mode}' không hợp lệ. Các chế độ hợp lệ: {SUPPORTED_TRAINING_MODES}"
        )

    train_loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 6. Thực hiện huấn luyện
    start_time = time.time()
    trained_model, train_summary = train_biencoder(
        model=model,
        train_loader=train_loader,
        val_samples=effective_val_samples,
        intent_bank=intent_bank,
        tokenizer=tokenizer,
        mode=mode,
        learning_rate=learning_rate,
        min_epochs=effective_min_epochs,
        max_epochs=effective_epochs,
        patience=effective_patience,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        scale=scale,
        triplet_margin=triplet_margin,
        device=run_device,
    )
    elapsed_time = time.time() - start_time

    # 7. Lưu trữ Checkpoint và Metadata
    save_ckpt_dir = Path(checkpoint_dir)
    save_ckpt_dir.mkdir(parents=True, exist_ok=True)
    trained_model.save_pretrained(save_ckpt_dir)

    fingerprint = preprocess_fingerprint()
    results = {
        "mode": mode,
        "shared_weights": shared_weights,
        "learning_rate": learning_rate,
        "batch_size": effective_batch_size,
        "epochs_trained": train_summary["epochs_trained"],
        "best_epoch": train_summary["best_epoch"],
        "best_val_top1": train_summary["best_val_top1"],
        "best_val_metrics": train_summary["best_val_metrics"],
        "elapsed_seconds": round(elapsed_time, 2),
        "smoke_mode": smoke_mode,
        "preprocessing_fingerprint": fingerprint,
        "history": train_summary["history"],
    }

    save_res_path = Path(results_path)
    save_res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# =====================================================================
# 8. Giao diện Dòng lệnh (CLI Argument Parser)
# =====================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Xây dựng bộ phân tích cú pháp tham số dòng lệnh cho kịch bản triplet_sbert.py."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện tinh chỉnh Bi-Encoder với các chế độ Contrastive và Triplet."
    )
    parser.add_argument(
        "--mode",
        choices=list(SUPPORTED_TRAINING_MODES),
        default=DEFAULT_TRAINING_MODE,
        help="Chế độ huấn luyện: mode_a (in-batch contrastive), mode_a2 (hard negative), mode_b (triplet).",
    )
    parser.add_argument(
        "--shared-weights",
        dest="shared_weights",
        action="store_true",
        default=True,
        help="Sử dụng kiến trúc Siamese chia sẻ trọng số giữa query và doc (mặc định True).",
    )
    parser.add_argument(
        "--dual-encoder",
        dest="shared_weights",
        action="store_false",
        help="Sử dụng kiến trúc Dual-Encoder với hai nhánh query và doc độc lập.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=BIENCODER_DEFAULT_LR,
        help="Tốc độ học cho optimizer AdamW (mặc định 2e-5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=list(BIENCODER_CANDIDATE_BATCH_SIZES),
        default=BIENCODER_DEFAULT_BATCH_SIZE,
        help="Kích thước lô huấn luyện (16 hoặc 32, mặc định 16).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=BIENCODER_MIN_EPOCHS,
        help="Số epoch tối thiểu trước khi kích hoạt dừng sớm (mặc định 3).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=BIENCODER_MAX_EPOCHS,
        help="Số epoch tối đa (mặc định 10).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=BIENCODER_DEFAULT_PATIENCE,
        help="Số epoch kiên nhẫn cho Early Stopping (mặc định 3).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=BIENCODER_WARMUP_RATIO,
        help="Tỷ lệ warm-up tuyến tính (mặc định 0.1).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Chạy ở chế độ Smoke Run siêu tốc trên CPU phục vụ CI/CD và kiểm thử.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_BIENCODER_CHECKPOINT_DIR,
        help="Thư mục lưu trữ checkpoint mô hình.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_BIENCODER_RESULTS_PATH,
        help="Đường dẫn tệp JSON lưu kết quả huấn luyện.",
    )
    return parser


def main() -> None:
    """Điểm nhập thực thi chính của kịch bản triplet_sbert.py."""
    parser = build_arg_parser()
    args = parser.parse_args()

    print(
        f"==> Bắt đầu huấn luyện Bi-Encoder | Chế độ: {args.mode} | "
        f"Shared weights: {args.shared_weights} | Smoke mode: {args.smoke}"
    )
    results = run_training_pipeline(
        mode=args.mode,
        shared_weights=args.shared_weights,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        min_epochs=args.min_epochs,
        max_epochs=args.epochs,
        patience=args.patience,
        warmup_ratio=args.warmup_ratio,
        smoke_mode=args.smoke,
        checkpoint_dir=args.checkpoint_dir,
        results_path=args.results_path,
    )
    print("==> Huấn luyện hoàn tất!")
    print(
        f"Epochs trained: {results['epochs_trained']} | "
        f"Best epoch: {results['best_epoch']} | "
        f"Val Top-1: {results['best_val_top1']:.4f} | "
        f"Thời gian: {results['elapsed_seconds']}s"
    )


if __name__ == "__main__":
    main()
