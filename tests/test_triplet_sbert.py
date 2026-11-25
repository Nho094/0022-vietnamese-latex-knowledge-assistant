"""
Bộ kiểm thử đơn vị và tích hợp cho kịch bản tinh chỉnh Bi-Encoder (train/triplet_sbert.py).

Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Python 3.11+, type hints, ruff-clean, pytest.
- Identifiers bằng tiếng Anh; docstrings và comments giải thích lý do (WHY) bằng tiếng Việt.
- Kiểm thử toàn diện:
  + Kiến trúc Siamese chia sẻ trọng số (shared_weights=True) và Dual-Encoder độc lập (shared_weights=False).
  + Lưu trữ và khôi phục mô hình (save_pretrained / from_pretrained).
  + Đóng gói dữ liệu theo lô (collate_mode_a, collate_mode_a2, collate_mode_b).
  + Lan truyền tiến và tính hàm mất mát cho Mode A, Mode A2, Mode B.
  + Đánh giá truy hồi Top-1 trên tập Validation (Val Top-1 Retrieval Accuracy & MRR).
  + Quy tắc dừng sớm (Early Stopping) và khôi phục trọng số tối ưu nhất.
  + Kiểm thực phòng vệ Quy tắc Trung thực #3 (Honesty Rule #3) ngăn cản rò rỉ test.csv.
  + Lịch trình tốc độ học Linear Warm-up & Linear Decay.
  + Kiểm thử giao diện dòng lệnh (CLI Argument Parser).
  + Chạy thử thực tế hoàn chỉnh đường ống (End-to-End Smoke Run) trên CPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaConfig

from core.config import (
    BIENCODER_DEFAULT_BATCH_SIZE,
    BIENCODER_DEFAULT_LR,
    DEFAULT_CONTRASTIVE_SCALE,
    DEFAULT_TRAINING_MODE,
    DEFAULT_TRIPLET_MARGIN,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    TRAINING_MODE_A,
    TRAINING_MODE_A2,
    TRAINING_MODE_B,
    VAL_CSV_PATH,
)
from train.biencoder_dataset import (
    BiEncoderPairDataset,
    DatasetSample,
)
from train.triplet_sbert import (
    BiEncoderModel,
    build_arg_parser,
    collate_mode_a,
    collate_mode_a2,
    collate_mode_b,
    create_biencoder_model,
    create_linear_schedule_with_warmup,
    evaluate_validation_retrieval,
    load_tokenizer,
    run_training_pipeline,
    train_biencoder,
)

# =====================================================================
# Fixtures và Tiện ích Kiểm thử
# =====================================================================


@pytest.fixture(scope="module")
def shared_tokenizer() -> Any:
    """Fixture nạp tokenizer PhoBERT ngoại tuyến dùng chung cho toàn bộ module test."""
    return load_tokenizer()


def create_tiny_backbone() -> nn.Module:
    """Khởi tạo một backbone Roberta siêu nhẹ 1 tầng, 2 heads, 64-dim phục vụ test nhanh."""
    from transformers import AutoModel

    config = RobertaConfig(
        vocab_size=64001,
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    return AutoModel.from_config(config)


# =====================================================================
# 1. Kiểm thử Kiến trúc Bi-Encoder: Siamese vs Dual-Encoder
# =====================================================================


def test_biencoder_architecture_siamese_shared_weights() -> None:
    """
    Kiểm tra kiến trúc Siamese chia sẻ trọng số khi shared_weights=True.
    LÝ DO THIẾT KẾ: Theo Sentence-BERT (Reimers & Gurevych, 2019), query_encoder và doc_encoder
    dùng chung các tham số nơ-ron để ánh xạ câu hỏi và kịch bản vào cùng một không gian biểu diễn.
    """
    model = create_biencoder_model(shared_weights=True, smoke_mode=True, device="cpu")
    assert model.shared_weights is True
    assert model.embedding_dim == 64

    # Nhánh doc_encoder và query_encoder phải là cùng một module (same instance)
    assert id(model.query_encoder) == id(model.doc_encoder)

    # Tổng số tham số của mô hình phải bằng đúng số tham số của một encoder duy nhất
    q_params = sum(p.numel() for p in model.query_encoder.parameters())
    model_params = sum(p.numel() for p in model.parameters())
    assert q_params == model_params


def test_biencoder_architecture_dual_encoder_independent_weights() -> None:
    """
    Kiểm tra kiến trúc Dual-Encoder khi shared_weights=False.
    LÝ DO THIẾT KẾ: Theo DPR (Karpukhin et al., 2020), hai nhánh sở hữu tham số độc lập
    giúp chuyên biệt hóa việc biểu diễn câu hỏi người dùng và đoạn kịch bản.
    """
    model = create_biencoder_model(shared_weights=False, smoke_mode=True, device="cpu")
    assert model.shared_weights is False
    assert model.embedding_dim == 64

    # Hai nhánh encoder phải là hai đối tượng khác nhau (different instances)
    assert id(model.query_encoder) != id(model.doc_encoder)

    # Tổng số tham số mô hình phải gấp đôi số tham số của một nhánh encoder
    q_params = sum(p.numel() for p in model.query_encoder.parameters())
    d_params = sum(p.numel() for p in model.doc_encoder.parameters())
    model_params = sum(p.numel() for p in model.parameters())
    assert q_params == d_params
    assert model_params == q_params + d_params


def test_biencoder_save_and_from_pretrained() -> None:
    """Kiểm tra lưu trữ checkpoint và khôi phục mô hình BiEncoderModel từ đĩa."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        original_model = create_biencoder_model(shared_weights=True, smoke_mode=True, device="cpu")
        original_model.save_pretrained(tmp_path)

        assert (tmp_path / "pytorch_model.bin").exists()
        assert (tmp_path / "biencoder_config.json").exists()

        restored_model = BiEncoderModel.from_pretrained(
            save_directory=tmp_path,
            backbone_factory=create_tiny_backbone,
            device="cpu",
        )
        assert restored_model.shared_weights is True
        assert restored_model.embedding_dim == 64
        original_model.eval()
        restored_model.eval()

        # Kiểm tra tính đồng nhất của vector đầu ra giữa mô hình gốc và mô hình khôi phục
        input_ids = torch.randint(0, 1000, (2, 8))
        attention_mask = torch.ones((2, 8), dtype=torch.long)

        with torch.no_grad():
            orig_emb = original_model.encode_query(input_ids, attention_mask)
            rest_emb = restored_model.encode_query(input_ids, attention_mask)

        assert torch.allclose(orig_emb, rest_emb, atol=1e-5)


# =====================================================================
# 2. Kiểm thử các Hàm Collate (Mode A, Mode A2, Mode B)
# =====================================================================


def test_collate_mode_a(shared_tokenizer: Any) -> None:
    """Kiểm tra hàm collate_mode_a đóng gói lô cho Mode A."""
    batch_items = [
        {"query": "viết công thức tích phân", "positive": "Tích phân Riemann", "scenario_index": 10},
        {"query": "giải phương trình bậc hai", "positive": "Phương trình bậc hai", "scenario_index": 20},
    ]
    collated = collate_mode_a(batch_items, tokenizer=shared_tokenizer, max_seq_length=32)

    assert "queries" in collated
    assert "positives" in collated
    assert "scenario_indices" in collated

    assert collated["queries"]["input_ids"].shape[0] == 2
    assert collated["positives"]["input_ids"].shape[0] == 2
    assert torch.equal(collated["scenario_indices"], torch.tensor([10, 20], dtype=torch.long))


def test_collate_mode_a2(shared_tokenizer: Any) -> None:
    """Kiểm tra hàm collate_mode_a2 đóng gói lô mở rộng hard negatives cho Mode A2."""
    batch_items = [
        {
            "query": "câu hỏi một",
            "positive": "mẫu dương một",
            "hard_negatives": ["âm khó 1A", "âm khó 1B"],
            "scenario_index": 1,
            "hard_scenario_indices": [101, 102],
        },
        {
            "query": "câu hỏi hai",
            "positive": "mẫu dương hai",
            "hard_negatives": ["âm khó 2A", "âm khó 2B"],
            "scenario_index": 2,
            "hard_scenario_indices": [201, 202],
        },
    ]
    collated = collate_mode_a2(batch_items, tokenizer=shared_tokenizer, max_seq_length=32)

    assert "hard_negatives" in collated
    assert collated["batch_size"] == 2
    assert collated["num_hn"] == 2
    assert collated["hard_negatives"]["input_ids"].shape[0] == 4  # 2 samples * 2 hard negatives
    assert collated["hard_scenario_indices"].shape == (2, 2)


def test_collate_mode_b(shared_tokenizer: Any) -> None:
    """Kiểm tra hàm collate_mode_b đóng gói bộ ba (anchor, positive, negative) cho Mode B."""
    batch_items = [
        {"anchor": "câu mỏ neo 1", "positive": "mẫu dương 1", "negative": "mẫu âm 1"},
        {"anchor": "câu mỏ neo 2", "positive": "mẫu dương 2", "negative": "mẫu âm 2"},
    ]
    collated = collate_mode_b(batch_items, tokenizer=shared_tokenizer, max_seq_length=32)

    assert "anchors" in collated
    assert "positives" in collated
    assert "negatives" in collated
    assert collated["anchors"]["input_ids"].shape[0] == 2
    assert collated["positives"]["input_ids"].shape[0] == 2
    assert collated["negatives"]["input_ids"].shape[0] == 2


# =====================================================================
# 3. Kiểm thử Huấn luyện và Lan truyền Gradient (Forward & Backward)
# =====================================================================


def test_biencoder_forward_and_loss_mode_a(shared_tokenizer: Any) -> None:
    """Kiểm tra bước lan truyền tiến và đạo hàm ngược cho Mode A (In-batch Contrastive)."""
    model = create_biencoder_model(shared_weights=True, smoke_mode=True, device="cpu")
    model.train()

    batch_items = [
        {"query": "viết tích phân", "positive": "kịch bản tích phân", "scenario_index": 1},
        {"query": "viết đạo hàm", "positive": "kịch bản đạo hàm", "scenario_index": 2},
    ]
    batch = collate_mode_a(batch_items, tokenizer=shared_tokenizer, max_seq_length=16)

    q_ids = batch["queries"]["input_ids"]
    q_mask = batch["queries"]["attention_mask"]
    p_ids = batch["positives"]["input_ids"]
    p_mask = batch["positives"]["attention_mask"]
    scen_idx = batch["scenario_indices"]

    q_emb = model.encode_query(q_ids, q_mask)
    p_emb = model.encode_doc(p_ids, p_mask)

    from train.losses import in_batch_contrastive_loss

    loss = in_batch_contrastive_loss(
        queries=q_emb,
        positives=p_emb,
        scenario_indices=scen_idx,
        metric="scaled_cosine",
        scale=DEFAULT_CONTRASTIVE_SCALE,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    loss.backward()
    # Kiểm tra gradient đã được tính toán đầy đủ cho các tham số
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_biencoder_forward_and_loss_mode_a2(shared_tokenizer: Any) -> None:
    """Kiểm tra bước lan truyền tiến và đạo hàm ngược cho Mode A2 (Hard Negative Contrastive)."""
    model = create_biencoder_model(shared_weights=False, smoke_mode=True, device="cpu")
    model.train()

    batch_items = [
        {
            "query": "câu hỏi A",
            "positive": "kịch bản dương A",
            "hard_negatives": ["mẫu âm khó A1", "mẫu âm khó A2"],
            "scenario_index": 1,
            "hard_scenario_indices": [10, 11],
        },
        {
            "query": "câu hỏi B",
            "positive": "kịch bản dương B",
            "hard_negatives": ["mẫu âm khó B1", "mẫu âm khó B2"],
            "scenario_index": 2,
            "hard_scenario_indices": [20, 21],
        },
    ]
    batch = collate_mode_a2(batch_items, tokenizer=shared_tokenizer, max_seq_length=16)

    q_emb = model.encode_query(batch["queries"]["input_ids"], batch["queries"]["attention_mask"])
    p_emb = model.encode_doc(batch["positives"]["input_ids"], batch["positives"]["attention_mask"])

    hn_flat_emb = model.encode_doc(
        batch["hard_negatives"]["input_ids"],
        batch["hard_negatives"]["attention_mask"],
    )
    hn_emb = hn_flat_emb.view(batch["batch_size"], batch["num_hn"], -1)

    from train.losses import hard_negative_contrastive_loss

    loss = hard_negative_contrastive_loss(
        queries=q_emb,
        positives=p_emb,
        hard_negatives=hn_emb,
        scenario_indices=batch["scenario_indices"],
        hard_scenario_indices=batch["hard_scenario_indices"],
        metric="scaled_cosine",
        scale=DEFAULT_CONTRASTIVE_SCALE,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0

    loss.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_biencoder_forward_and_loss_mode_b(shared_tokenizer: Any) -> None:
    """Kiểm tra bước lan truyền tiến và đạo hàm ngược cho Mode B (Euclidean Triplet Loss)."""
    model = create_biencoder_model(shared_weights=True, smoke_mode=True, device="cpu")
    model.train()

    batch_items = [
        {"anchor": "câu mỏ neo", "positive": "kịch bản đúng", "negative": "kịch bản sai"},
        {"anchor": "câu hỏi 2", "positive": "kịch bản 2", "negative": "kịch bản sai 2"},
    ]
    batch = collate_mode_b(batch_items, tokenizer=shared_tokenizer, max_seq_length=16)

    a_emb = model.encode_query(batch["anchors"]["input_ids"], batch["anchors"]["attention_mask"])
    p_emb = model.encode_doc(batch["positives"]["input_ids"], batch["positives"]["attention_mask"])
    n_emb = model.encode_doc(batch["negatives"]["input_ids"], batch["negatives"]["attention_mask"])

    from train.losses import euclidean_triplet_loss

    loss = euclidean_triplet_loss(
        anchor=a_emb,
        positive=p_emb,
        negative=n_emb,
        margin=DEFAULT_TRIPLET_MARGIN,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0

    loss.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


# =====================================================================
# 4. Kiểm thử Đánh giá Validation Top-1 Retrieval & Early Stopping
# =====================================================================


def test_evaluate_validation_retrieval(shared_tokenizer: Any) -> None:
    """Kiểm tra tính toán độ chính xác Top-1 Retrieval trên tập validation."""
    model = create_biencoder_model(shared_weights=True, smoke_mode=True, device="cpu")

    mock_intent_bank = {
        "INTENT_INTEGRAL": {"intent": "INTENT_INTEGRAL", "describe": "Tính tích phân", "aliases": []},
        "INTENT_DERIVATIVE": {"intent": "INTENT_DERIVATIVE", "describe": "Tính đạo hàm", "aliases": []},
        "INTENT_MATRIX": {"intent": "INTENT_MATRIX", "describe": "Ma trận", "aliases": []},
    }
    val_samples = [
        DatasetSample(text="tính tích phân hàm số", label="INTENT_INTEGRAL", category="calculus"),
        DatasetSample(text="tính đạo hàm cấp một", label="INTENT_DERIVATIVE", category="calculus"),
    ]

    metrics = evaluate_validation_retrieval(
        model=model,
        val_samples=val_samples,
        intent_bank=mock_intent_bank,
        tokenizer=shared_tokenizer,
        device="cpu",
        batch_size=4,
        max_seq_length=16,
        mode=TRAINING_MODE_A,
    )

    assert "top1_accuracy" in metrics
    assert "mrr" in metrics
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert metrics["total_samples"] == 2.0


def test_evaluate_validation_retrieval_empty_samples(shared_tokenizer: Any) -> None:
    """Kiểm tra đánh giá an toàn khi tập validation rỗng."""
    model = create_biencoder_model(shared_weights=True, smoke_mode=True, device="cpu")
    metrics = evaluate_validation_retrieval(
        model=model,
        val_samples=[],
        intent_bank={},
        tokenizer=shared_tokenizer,
        device="cpu",
    )
    assert metrics["top1_accuracy"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["total_samples"] == 0.0


def test_early_stopping_behavior(shared_tokenizer: Any) -> None:
    """Kiểm tra cơ chế dừng sớm (Early Stopping) dừng sau số epoch quy định."""
    model = create_biencoder_model(shared_weights=True, smoke_mode=True, device="cpu")

    mock_intent_bank = {
        "A": {"intent": "A", "describe": "Kịch bản A", "aliases": []},
        "B": {"intent": "B", "describe": "Kịch bản B", "aliases": []},
    }
    train_samples = [
        DatasetSample(text="câu A", label="A", category="c"),
        DatasetSample(text="câu B", label="B", category="c"),
    ]
    val_samples = [
        DatasetSample(text="câu A", label="A", category="c"),
    ]

    dataset = BiEncoderPairDataset(
        samples=train_samples,
        intent_bank=mock_intent_bank,
        include_aliases=False,
        variant_strategy="handmade",
        dynamic_sampling=False,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda b: collate_mode_a(b, tokenizer=shared_tokenizer, max_seq_length=16),
    )

    # Huấn luyện với min_epochs=1, patience=1, max_epochs=5
    _trained_model, summary = train_biencoder(
        model=model,
        train_loader=train_loader,
        val_samples=val_samples,
        intent_bank=mock_intent_bank,
        tokenizer=shared_tokenizer,
        mode=TRAINING_MODE_A,
        learning_rate=1e-3,
        min_epochs=1,
        max_epochs=5,
        patience=1,
        warmup_ratio=0.1,
        device="cpu",
    )
    assert summary["epochs_trained"] <= 5
    assert summary["best_epoch"] >= 1
    assert "history" in summary


# =====================================================================
# 5. Kiểm thử Phòng vệ Quy tắc Trung thực #3 (Honesty Rule #3)
# =====================================================================


def test_honesty_rule_3_guard_test_csv() -> None:
    """
    Kiểm tra hệ thống phòng vệ Honesty Rule #3:
    Ngăn chặn tuyệt đối việc truyền tệp data/test.csv vào quy trình huấn luyện hoặc validation.
    """
    with pytest.raises(PermissionError, match="Honesty Rule #3"):
        run_training_pipeline(
            train_path=TEST_CSV_PATH,
            smoke_mode=True,
        )

    with pytest.raises(PermissionError, match="Honesty Rule #3"):
        run_training_pipeline(
            val_path=TEST_CSV_PATH,
            smoke_mode=True,
        )


# =====================================================================
# 6. Kiểm thử Lịch trình Tốc độ học (Linear Schedule with Warm-up)
# =====================================================================


def test_linear_schedule_with_warmup() -> None:
    """Kiểm tra bộ điều chỉnh tốc độ học: warm-up tuyến tính và suy giảm dần về 0."""
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = create_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
    )

    # Pha warm-up: tỷ lệ tăng tuyến tính
    scheduler.step()
    lr_step1 = scheduler.get_last_lr()[0]
    assert 0.0 < lr_step1 <= 0.2

    # Tại đỉnh warm-up (step 10): tỷ lệ đạt 1.0
    for _ in range(9):
        scheduler.step()
    lr_peak = scheduler.get_last_lr()[0]
    assert abs(lr_peak - 1.0) < 1e-4

    # Pha suy giảm (decay): suy giảm dần về 0
    for _ in range(90):
        scheduler.step()
    lr_end = scheduler.get_last_lr()[0]
    assert abs(lr_end - 0.0) < 1e-4


# =====================================================================
# 7. Kiểm thử Giao diện Dòng lệnh (CLI Argument Parser)
# =====================================================================


def test_cli_argument_parser() -> None:
    """Kiểm tra parser dòng lệnh với các tùy chọn cấu hình."""
    parser = build_arg_parser()

    # Mặc định
    args = parser.parse_args([])
    assert args.mode == DEFAULT_TRAINING_MODE
    assert args.shared_weights is True
    assert args.lr == BIENCODER_DEFAULT_LR
    assert args.batch_size == BIENCODER_DEFAULT_BATCH_SIZE
    assert args.smoke is False

    # Chuyển cờ dual-encoder
    args_dual = parser.parse_args(["--dual-encoder", "--mode", "mode_a2", "--smoke", "--batch-size", "32"])
    assert args_dual.shared_weights is False
    assert args_dual.mode == TRAINING_MODE_A2
    assert args_dual.smoke is True
    assert args_dual.batch_size == 32

    # Chuyển cờ mode_b
    args_b = parser.parse_args(["--mode", "mode_b", "--lr", "3e-5"])
    assert args_b.mode == TRAINING_MODE_B
    assert args_b.lr == 3e-5


# =====================================================================
# 8. Kiểm thử Tích hợp Toàn diện Chế độ Smoke Mode trên CPU
# =====================================================================


def test_smoke_pipeline_end_to_end_mode_a() -> None:
    """
    Kiểm thử tích hợp end-to-end Mode A (In-batch Contrastive) ở chế độ Smoke Mode trên CPU.
    Xác minh: chạy siêu tốc (< 2 giây), trả về kết quả hợp lệ, lưu checkpoint và kết quả JSON.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_dir = tmp_path / "ckpt"
        results_file = tmp_path / "results.json"

        results = run_training_pipeline(
            mode=TRAINING_MODE_A,
            shared_weights=True,
            smoke_mode=True,
            checkpoint_dir=ckpt_dir,
            results_path=results_file,
            train_path=TRAIN_CSV_PATH,
            val_path=VAL_CSV_PATH,
        )

        assert results["mode"] == TRAINING_MODE_A
        assert results["shared_weights"] is True
        assert results["smoke_mode"] is True
        assert results["epochs_trained"] == 1
        assert 0.0 <= results["best_val_top1"] <= 1.0
        assert (ckpt_dir / "pytorch_model.bin").exists()
        assert (ckpt_dir / "biencoder_config.json").exists()
        assert results_file.exists()


def test_smoke_pipeline_end_to_end_mode_a2_and_b() -> None:
    """Kiểm thử tích hợp end-to-end Mode A2 và Mode B ở chế độ Smoke Mode trên CPU."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Mode A2: Hard Negative
        results_a2 = run_training_pipeline(
            mode=TRAINING_MODE_A2,
            shared_weights=False,
            smoke_mode=True,
            checkpoint_dir=tmp_path / "ckpt_a2",
            results_path=tmp_path / "res_a2.json",
        )
        assert results_a2["mode"] == TRAINING_MODE_A2
        assert results_a2["shared_weights"] is False

        # Mode B: Triplet Loss
        results_b = run_training_pipeline(
            mode=TRAINING_MODE_B,
            shared_weights=True,
            smoke_mode=True,
            checkpoint_dir=tmp_path / "ckpt_b",
            results_path=tmp_path / "res_b.json",
        )
        assert results_b["mode"] == TRAINING_MODE_B
        assert results_b["shared_weights"] is True
