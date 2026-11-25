"""
Bộ kiểm thử đơn vị và xác minh tensor cho các hàm mất mát tương phản và Triplet Loss.

Tệp tin: tests/test_losses.py
Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Identifiers in English; docstrings và chú thích tiếng Việt giải thích lý do (WHY) và đối chuẩn công thức toán học.
- Xác minh trên các tensor tổng hợp nhỏ (small synthetic tensors) theo đúng yêu cầu kiểm định:
  1. Mode A in-batch loss khớp chính xác với phép tính thủ công negative log-softmax trên đường chéo chính.
  2. Cơ chế mặt nạ đồng nhất âm vô cực (Identity Masking) loại bỏ hoàn toàn các mẫu âm trùng lặp (duplicate negatives).
  3. Mode A2 hard negative denominator expansion mở rộng mẫu số với hard negatives và che mẫu âm khó trùng scenario index.
  4. Mode B Euclidean margin triplet loss triệt tiêu về 0 (loss = 0.0) khi thỏa mãn lề khoảng cách và dương khi bị vi phạm.
  5. Mean pooling cho biểu diễn chuỗi token 3D và rút gọn trung bình theo lô (batch mean reduction).
  6. Tính tương đương giữa các lớp nn.Module và hàm chức năng (functional API).
  7. Kiểm thử phòng vệ dữ liệu đầu vào (shape, invalid metrics, edge cases) và kiểm tra lan truyền ngược (backward pass).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from core.config import (
    set_seed,
)
from train.losses import (
    EuclideanTripletLoss,
    HardNegativeContrastiveLoss,
    InBatchContrastiveLoss,
    euclidean_triplet_loss,
    hard_negative_contrastive_loss,
    in_batch_contrastive_loss,
    mode_a2_contrastive_loss,
    mode_a_contrastive_loss,
    mode_b_triplet_loss,
)


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    """Thiết lập hạt giống ngẫu nhiên cố định trước mỗi ca kiểm thử."""
    set_seed(42)


# ==============================================================================
# 1. KIỂM THỬ MODE A: IN-BATCH CONTRASTIVE LOSS
# ==============================================================================


def test_in_batch_contrastive_loss_matches_manual_log_softmax_diagonal_dot() -> None:
    """
    Xác minh Mode A (tích vô hướng dot product) khớp chính xác với công thức thủ công
    negative log-softmax trên đường chéo chính khi không có scenario indices trùng lặp.

    Công thức InfoNCE (Oord et al., 2018; DPR Karpukhin et al., 2020):
        L = - (1/B) * sum_{i=0}^{B-1} log [ exp(S_{ii}) / sum_{j=0}^{B-1} exp(S_{ij}) ]
          = - (1/B) * sum_{i=0}^{B-1} log_softmax(S, dim=-1)[i, i]
    """
    # Khởi tạo tensor tổng hợp nhỏ (batch_size=4, embed_dim=8)
    queries = torch.tensor(
        [
            [1.0, 0.5, -0.2, 0.8, -1.0, 0.4, 0.2, -0.6],
            [-0.5, 1.2, 0.3, -0.7, 0.9, -0.1, 0.5, 0.2],
            [0.2, -0.4, 1.5, 0.1, -0.3, 0.8, -0.9, 0.6],
            [0.7, 0.3, -0.6, 1.1, 0.4, -0.5, 0.8, -0.2],
        ],
        dtype=torch.float32,
    )
    positives = torch.tensor(
        [
            [0.9, 0.4, -0.1, 0.7, -0.8, 0.5, 0.1, -0.5],
            [-0.4, 1.0, 0.2, -0.6, 0.8, 0.0, 0.4, 0.3],
            [0.3, -0.5, 1.3, 0.0, -0.2, 0.7, -0.8, 0.5],
            [0.6, 0.2, -0.5, 1.0, 0.3, -0.4, 0.7, -0.1],
        ],
        dtype=torch.float32,
    )
    # Tính toán thủ công:
    # Bước 1: Ma trận tương đồng tích vô hướng S = Q @ P^T
    sim_manual = torch.matmul(queries, positives.t())
    # Bước 2: Log softmax trên từng hàng
    log_probs_manual = F.log_softmax(sim_manual, dim=-1)
    # Bước 3: Lấy các giá trị trên đường chéo chính (mẫu dương) và lấy đối dấu (negative log-likelihood)
    diag_log_probs = torch.diagonal(log_probs_manual)
    expected_loss = -diag_log_probs.mean()

    # Gọi hàm mất mát triển khai trong train/losses.py
    loss = in_batch_contrastive_loss(queries, positives, metric="dot", reduction="mean")

    # Xác minh sai số tuyệt đối xấp xỉ 0 (sai số số thực máy tính < 1e-6)
    assert torch.allclose(loss, expected_loss, atol=1e-6), (
        f"In-batch contrastive loss ({loss.item()}) không khớp với giá trị tính tay "
        f"negative log-softmax đường chéo ({expected_loss.item()})."
    )


def test_in_batch_contrastive_loss_matches_manual_scaled_cosine() -> None:
    """
    Xác minh Mode A với scaled cosine (scale = 20.0 theo SimCSE - Gao et al., 2021)
    khớp chính xác với negative log-softmax trên đường chéo ma trận cosine nhân tỉ lệ.
    """
    queries = torch.randn(3, 16)
    positives = torch.randn(3, 16)
    scale = 20.0

    # Tính toán thủ công ma trận cosine scaled: scale * (q_norm @ p_norm^T)
    q_norm = F.normalize(queries, p=2, dim=-1)
    p_norm = F.normalize(positives, p=2, dim=-1)
    sim_manual = torch.matmul(q_norm, p_norm.t()) * scale

    log_probs_manual = F.log_softmax(sim_manual, dim=-1)
    expected_loss = -torch.diagonal(log_probs_manual).mean()

    loss = in_batch_contrastive_loss(
        queries, positives, metric="scaled_cosine", scale=scale, reduction="mean"
    )

    assert torch.allclose(loss, expected_loss, atol=1e-6)


def test_identity_masking_removes_duplicate_negatives() -> None:
    """
    Xác minh cơ chế Negative Infinity Identity Masking khi scenario indices trùng nhau trong lô:
    - Các mẫu âm ngoài đường chéo chính có cùng scenario index phải được gán -inf.
    - Trong mẫu số softmax, exp(-inf) = 0, loại bỏ hoàn toàn mẫu âm trùng lặp khỏi mẫu số.
    - Đường chéo chính (mẫu dương hợp lệ) KHÔNG bị gán -inf.
    """
    queries = torch.tensor(
        [
            [1.0, 0.0],  # Mẫu 0 (Scenario A)
            [0.0, 1.0],  # Mẫu 1 (Scenario B)
            [1.0, 0.1],  # Mẫu 2 (Scenario A - Trùng scenario với mẫu 0!)
            [0.0, -1.0],  # Mẫu 3 (Scenario C)
        ],
        dtype=torch.float32,
    )
    positives = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    # Scenario indices: mẫu 0 và mẫu 2 cùng thuộc scenario 101; mẫu 1 là 202; mẫu 3 là 303
    scenario_indices = torch.tensor([101, 202, 101, 303])

    # Ma trận tích vô hướng thô:
    # S[0, :] = [1.0, 0.0, 1.0, 0.0]
    # S[1, :] = [0.0, 1.0, 0.0, -1.0]
    # S[2, :] = [1.0, 0.1, 1.0, -0.1]
    # S[3, :] = [0.0, -1.0, 0.0, 1.0]
    # Với Identity Masking:
    # Tại dòng 0: Cột 2 có cùng scenario 101 -> S[0, 2] = -inf
    # Tại dòng 2: Cột 0 có cùng scenario 101 -> S[2, 0] = -inf
    # Đường chéo chính S[0, 0] và S[2, 2] KHÔNG bị che!

    loss = in_batch_contrastive_loss(
        queries, positives, scenario_indices=scenario_indices, metric="dot", reduction="none"
    )

    # Tính toán thủ công cho dòng 0 (chỉ còn cột 0, 1, 3; cột 2 bị loại bỏ hoàn toàn):
    expected_loss_row0 = -math.log(math.exp(1.0) / (math.exp(1.0) + math.exp(0.0) + math.exp(0.0)))
    assert abs(loss[0].item() - expected_loss_row0) < 1e-6

    # Tính toán thủ công cho dòng 2 (chỉ còn cột 1, 2 (dương), 3; cột 0 bị loại bỏ hoàn toàn):
    expected_loss_row2 = -math.log(
        math.exp(1.0) / (math.exp(0.1) + math.exp(1.0) + math.exp(-0.1))
    )
    assert abs(loss[2].item() - expected_loss_row2) < 1e-6

    # Xác minh rằng nếu KHÔNG áp dụng identity masking, hàm mất mát dòng 0 sẽ khác vì bao gồm cả cột 2
    loss_unmasked = in_batch_contrastive_loss(
        queries, positives, scenario_indices=None, metric="dot", reduction="none"
    )
    assert not torch.allclose(loss[0], loss_unmasked[0]), (
        "Hàm mất mát khi áp dụng identity masking phải khác so với khi không che mẫu âm trùng lặp."
    )


def test_identity_masking_all_identical_scenarios_evaluates_to_zero() -> None:
    """
    Xác minh trường hợp biên: Khi toàn bộ các mẫu trong lô đều có chung một scenario index,
    mọi mẫu âm ngoài đường chéo đều bị che (-inf), trong mẫu số chỉ còn duy nhất mẫu dương
    trên đường chéo chính: exp(S_ii) / exp(S_ii) = 1 => -log(1) = 0.0.
    """
    queries = torch.randn(4, 8)
    positives = torch.randn(4, 8)
    # Tất cả mẫu đều cùng scenario index 99
    scenario_indices = torch.tensor([99, 99, 99, 99])

    loss = in_batch_contrastive_loss(
        queries, positives, scenario_indices=scenario_indices, metric="dot", reduction="mean"
    )

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), (
        f"Khi toàn bộ lô có cùng scenario index, loss phải triệt tiêu về 0.0, nhận được {loss.item()}."
    )


# ==============================================================================
# 2. KIỂM THỬ MODE A2: HARD NEGATIVE DENOMINATOR EXPANSION
# ==============================================================================


def test_mode_a2_hard_negatives_expands_denominator() -> None:
    """
    Xác minh Mode A2 mở rộng mẫu số với hard negatives:
    - Khi bổ sung mẫu âm khó có điểm số hữu hạn vào mẫu số, tổng mẫu số tăng lên,
      do đó độ hợp lý log âm phải tăng lên (loss_a2 > loss_a).
    - Khi các hard negatives bị che bằng -inf, loss của Mode A2 phải giảm về bằng đúng Mode A.
    """
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    # Mỗi truy vấn có 1 hard negative: shape (2, 1, 2)
    hard_negatives = torch.tensor([[[0.8, 0.0]], [[0.0, 0.8]]], dtype=torch.float32)

    loss_mode_a = in_batch_contrastive_loss(queries, positives, metric="dot")
    loss_mode_a2 = hard_negative_contrastive_loss(queries, positives, hard_negatives, metric="dot")

    # Mẫu số có thêm exp(0.8) nên mất mát phải lớn hơn
    assert loss_mode_a2.item() > loss_mode_a.item(), (
        f"Mất mát Mode A2 ({loss_mode_a2.item()}) phải lớn hơn Mode A ({loss_mode_a.item()}) "
        f"khi có thêm hard negatives trong mẫu số."
    )

    # Tính tay đối chứng cho truy vấn 0:
    # S_ib[0] = [1.0, 0.0] => exp(1.0), exp(0.0) = e, 1
    # S_hard[0] = [0.8] => exp(0.8)
    # expected_loss_0 = -log(e / (e + 1 + e^0.8))
    expected_loss_0 = -math.log(math.exp(1.0) / (math.exp(1.0) + math.exp(0.0) + math.exp(0.8)))
    loss_a2_per_sample = hard_negative_contrastive_loss(
        queries, positives, hard_negatives, metric="dot", reduction="none"
    )
    assert abs(loss_a2_per_sample[0].item() - expected_loss_0) < 1e-6


def test_mode_a2_identity_masking_on_hard_negatives() -> None:
    """
    Xác minh cơ chế Identity Masking trên các hard negatives:
    Khi hard negative có scenario index trùng với truy vấn, nó bị gán -inf và loại bỏ khỏi mẫu số.
    Nếu toàn bộ hard negatives bị che, loss Mode A2 trở về đúng bằng loss Mode A.
    """
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    hard_negatives = torch.tensor([[[0.9, 0.0]], [[0.0, 0.9]]], dtype=torch.float32)

    scenario_indices = torch.tensor([1, 2])
    # Giả lập cả hai hard negatives đều bị gán nhầm scenario index trùng với truy vấn tương ứng
    hard_scenario_indices = torch.tensor([1, 2])

    loss_masked = hard_negative_contrastive_loss(
        queries,
        positives,
        hard_negatives,
        scenario_indices=scenario_indices,
        hard_scenario_indices=hard_scenario_indices,
        metric="dot",
    )
    loss_mode_a = in_batch_contrastive_loss(
        queries, positives, scenario_indices=scenario_indices, metric="dot"
    )

    # Vì toàn bộ hard negatives bị che thành -inf, exp(-inf)=0 nên kết quả phải bằng Mode A in-batch!
    assert torch.allclose(loss_masked, loss_mode_a, atol=1e-6), (
        f"Khi toàn bộ hard negatives bị identity mask, loss ({loss_masked.item()}) "
        f"phải bằng Mode A in-batch loss ({loss_mode_a.item()})."
    )


def test_mode_a2_shared_hard_negatives_2d() -> None:
    """
    Xác minh Mode A2 khi hard_negatives là kho chia sẻ 2D (num_negatives, embed_dim)
    với cờ shared_hard_negatives=True.
    """
    queries = torch.randn(3, 8)
    positives = torch.randn(3, 8)
    shared_pool = torch.randn(5, 8)  # 5 hard negatives dùng chung cho 3 queries

    loss = hard_negative_contrastive_loss(
        queries,
        positives,
        shared_pool,
        metric="dot",
        shared_hard_negatives=True,
        reduction="mean",
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


# ==============================================================================
# 3. KIỂM THỬ MODE B: EUCLIDEAN MARGIN TRIPLET LOSS
# ==============================================================================


def test_triplet_loss_evaluates_to_zero_when_distance_margin_satisfied() -> None:
    """
    Xác minh Mode B: Triplet Loss triệt tiêu về đúng 0.0 khi lề khoảng cách được thỏa mãn.

    Công thức FaceNet (Schroff et al., 2015, Eq.(3)):
        L = max(0, ||a - p||_2 - ||a - n||_2 + alpha)
    Khi ||a - n||_2 - ||a - p||_2 >= alpha (khoảng cách tới âm lớn hơn dương ít nhất 1 lề alpha):
        ||a - p||_2 - ||a - n||_2 + alpha <= 0 => L = 0.0.
    """
    # a = (0, 0), p = (0, 0) => d(a, p) = 0.0
    # n = (2, 0) => d(a, n) = 2.0
    # margin alpha = 1.0
    # d(a, p) - d(a, n) + alpha = 0.0 - 2.0 + 1.0 = -1.0 <= 0 => loss = 0.0
    anchor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    positive = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    negative = torch.tensor([[2.0, 0.0]], dtype=torch.float32)

    loss = euclidean_triplet_loss(
        anchor, positive, negative, margin=1.0, eps=0.0, reduction="mean"
    )

    assert loss.item() == 0.0, (
        f"Triplet loss phải bằng 0.0 khi lề khoảng cách được thỏa mãn, nhận được {loss.item()}."
    )


def test_triplet_loss_evaluates_positive_otherwise() -> None:
    """
    Xác minh Mode B: Triplet Loss nhận giá trị dương (> 0.0) khi lề khoảng cách bị vi phạm.

    Trường hợp vi phạm:
    - a = (0, 0), p = (0.8, 0) => d(a, p) = 0.8
    - n = (1.0, 0) => d(a, n) = 1.0
    - margin = 1.0
    - d(a, p) - d(a, n) + margin = 0.8 - 1.0 + 1.0 = 0.8 > 0 => loss = 0.8
    """
    anchor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    positive = torch.tensor([[0.8, 0.0]], dtype=torch.float32)
    negative = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss = euclidean_triplet_loss(
        anchor, positive, negative, margin=1.0, eps=0.0, reduction="mean"
    )

    expected_loss = 0.8
    assert abs(loss.item() - expected_loss) < 1e-6, (
        f"Triplet loss ({loss.item()}) không khớp với giá trị kỳ vọng ({expected_loss})."
    )


def test_triplet_loss_mean_pooling_reduction_on_batch() -> None:
    """
    Xác minh tính năng rút gọn trung bình theo lô (batch mean pooling reduction):
    Lô gồm 2 bộ ba: bộ ba 1 thỏa mãn lề (loss = 0.0), bộ ba 2 vi phạm (loss = 0.6).
    Trung bình toàn lô phải là (0.0 + 0.6) / 2 = 0.3.
    """
    anchor = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    positive = torch.tensor([[0.0, 0.0], [0.6, 0.0]], dtype=torch.float32)
    negative = torch.tensor([[3.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    # Bộ ba 0: d_ap = 0.0, d_an = 3.0 => 0 - 3 + 1 = -2 <= 0 => loss = 0.0
    # Bộ ba 1: d_ap = 0.6, d_an = 1.0 => 0.6 - 1.0 + 1.0 = 0.6 => loss = 0.6
    loss_mean = euclidean_triplet_loss(
        anchor, positive, negative, margin=1.0, eps=0.0, reduction="mean"
    )
    loss_none = euclidean_triplet_loss(
        anchor, positive, negative, margin=1.0, eps=0.0, reduction="none"
    )

    assert abs(loss_none[0].item() - 0.0) < 1e-6
    assert abs(loss_none[1].item() - 0.6) < 1e-6
    assert abs(loss_mean.item() - 0.3) < 1e-6


def test_triplet_loss_token_sequence_mean_pooling_3d() -> None:
    """
    Xác minh tính năng gom cụm trung bình dọc theo chiều chuỗi token (token mean pooling)
    khi đầu vào là tensor 3D (batch_size, seq_len, embed_dim).
    """
    # Tensor 3D kích thước (1, 2, 2): 1 mẫu, 2 tokens, chiều vector 2
    # a: token 1 = [0, 0], token 2 = [0, 0] => mean pooled = [0, 0]
    # p: token 1 = [0.4, 0], token 2 = [0.6, 0] => mean pooled = [0.5, 0] (d_ap = 0.5)
    # n: token 1 = [1.0, 0], token 2 = [1.0, 0] => mean pooled = [1.0, 0] (d_an = 1.0)
    # Margin = 1.0 => loss = 0.5 - 1.0 + 1.0 = 0.5
    anchor_3d = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    positive_3d = torch.tensor([[[0.4, 0.0], [0.6, 0.0]]], dtype=torch.float32)
    negative_3d = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)

    loss = euclidean_triplet_loss(
        anchor_3d, positive_3d, negative_3d, margin=1.0, eps=0.0, mean_pooling=True
    )

    assert abs(loss.item() - 0.5) < 1e-6


# ==============================================================================
# 4. KIỂM THỬ TƯƠNG ĐƯƠNG GIỮA NN.MODULE VÀ FUNCTIONAL API CÙNG CÁC ALIASES
# ==============================================================================


def test_nn_module_and_functional_equivalence() -> None:
    """
    Xác minh các lớp nn.Module (InBatchContrastiveLoss, HardNegativeContrastiveLoss, EuclideanTripletLoss)
    và các bí danh tương ứng cho kết quả đồng nhất với hàm functional.
    """
    q = torch.randn(4, 8)
    p = torch.randn(4, 8)
    hn = torch.randn(4, 2, 8)
    s_idx = torch.tensor([0, 1, 2, 3])

    # Mode A
    mod_a = InBatchContrastiveLoss(metric="dot")
    loss_mod_a = mod_a(q, p, scenario_indices=s_idx)
    loss_fn_a = in_batch_contrastive_loss(q, p, scenario_indices=s_idx, metric="dot")
    loss_alias_a = mode_a_contrastive_loss(q, p, scenario_indices=s_idx, metric="dot")
    assert torch.allclose(loss_mod_a, loss_fn_a)
    assert torch.allclose(loss_mod_a, loss_alias_a)

    # Mode A2
    mod_a2 = HardNegativeContrastiveLoss(metric="dot")
    loss_mod_a2 = mod_a2(q, p, hn, scenario_indices=s_idx)
    loss_fn_a2 = hard_negative_contrastive_loss(q, p, hn, scenario_indices=s_idx, metric="dot")
    loss_alias_a2 = mode_a2_contrastive_loss(q, p, hn, scenario_indices=s_idx, metric="dot")
    assert torch.allclose(loss_mod_a2, loss_fn_a2)
    assert torch.allclose(loss_mod_a2, loss_alias_a2)

    # Mode B
    mod_b = EuclideanTripletLoss(margin=1.0)
    loss_mod_b = mod_b(q, p, hn[:, 0, :])
    loss_fn_b = euclidean_triplet_loss(q, p, hn[:, 0, :], margin=1.0)
    loss_alias_b = mode_b_triplet_loss(q, p, hn[:, 0, :], margin=1.0)
    assert torch.allclose(loss_mod_b, loss_fn_b)
    assert torch.allclose(loss_mod_b, loss_alias_b)


# ==============================================================================
# 5. KIỂM THỬ LAN TRUYỀN NGƯỢC (BACKWARD PASS VÀ GRADIENTS)
# ==============================================================================


def test_contrastive_and_triplet_backward_gradients() -> None:
    """
    Xác minh tất cả các hàm mất mát cho phép lan truyền ngược gradient đầy đủ,
    không sinh NaN hoặc Inf trong tensor đạo hàm.
    """
    queries = torch.randn(4, 8, requires_grad=True)
    positives = torch.randn(4, 8, requires_grad=True)
    hard_negatives = torch.randn(4, 1, 8, requires_grad=True)
    s_idx = torch.tensor([1, 2, 1, 3])

    # Mode A backward
    loss_a = in_batch_contrastive_loss(queries, positives, scenario_indices=s_idx)
    loss_a.backward()
    assert queries.grad is not None
    assert not torch.isnan(queries.grad).any()
    assert not torch.isinf(queries.grad).any()

    # Reset grad
    queries.grad.zero_()
    positives.grad.zero_()

    # Mode A2 backward
    loss_a2 = hard_negative_contrastive_loss(
        queries, positives, hard_negatives, scenario_indices=s_idx
    )
    loss_a2.backward()
    assert hard_negatives.grad is not None
    assert not torch.isnan(hard_negatives.grad).any()

    # Mode B backward
    a = torch.randn(4, 8, requires_grad=True)
    p = torch.randn(4, 8, requires_grad=True)
    n = torch.randn(4, 8, requires_grad=True)
    loss_b = euclidean_triplet_loss(a, p, n, margin=1.0)
    loss_b.backward()
    assert a.grad is not None
    assert not torch.isnan(a.grad).any()


# ==============================================================================
# 6. KIỂM THỬ PHÒNG VỆ VÀ XỬ LÝ LỖI ĐẦU VÀO (INPUT VALIDATION)
# ==============================================================================


def test_input_validation_errors() -> None:
    """
    Xác minh các ngoại lệ ValueError được kích hoạt khi truyền tham số sai định dạng.
    """
    q = torch.randn(4, 8)
    p_wrong_dim = torch.randn(4, 10)
    p_wrong_batch = torch.randn(3, 8)

    # Lệch kích thước embedding
    with pytest.raises(ValueError, match="không khớp"):
        in_batch_contrastive_loss(q, p_wrong_dim)

    # Lệch batch_size
    with pytest.raises(ValueError, match="không khớp"):
        in_batch_contrastive_loss(q, p_wrong_batch)

    # Độ đo không hỗ trợ
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        in_batch_contrastive_loss(q, q, metric="unsupported_metric")

    # Reduction không hợp lệ
    with pytest.raises(ValueError, match="reduction"):
        in_batch_contrastive_loss(q, q, reduction="invalid")

    # Triplet margin âm
    with pytest.raises(ValueError, match="Margin phải >= 0"):
        euclidean_triplet_loss(q, q, q, margin=-0.5)

    # Scenario indices không khớp batch_size
    with pytest.raises(ValueError, match="scenario_indices phải là tensor 1D"):
        in_batch_contrastive_loss(q, q, scenario_indices=torch.tensor([1, 2]))
