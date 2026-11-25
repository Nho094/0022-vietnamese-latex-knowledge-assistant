"""
Module triển khai các hàm mất mát tương phản (Contrastive Loss) và Triplet Loss cho huấn luyện biểu diễn ngữ nghĩa.

Tệp tin: train/losses.py
Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Identifiers in English; docstrings và comments bằng tiếng Việt giải thích lý do (WHY) và trích dẫn công thức bài báo khoa học.
- Toàn bộ hằng số cấu hình được nạp từ core/config.py (không dùng số ma thuật).
- Kiểm tra tính hợp lệ dữ liệu đầu vào (shape, metric, reduction) và phòng vệ lỗi tensor.

Các chế độ huấn luyện hỗ trợ (Training Modes):
1. Chế độ Mode A (In-batch Contrastive Loss):
   - Dựa trên hàm mất mát InfoNCE (van den Oord et al., 2018 - arXiv:1807.03748) và Dense Passage Retrieval
     (DPR, Karpukhin et al., 2020 - arXiv:2004.04906):
     L(q_i, p_i) = -log [ exp(sim(q_i, p_i)) / sum_{j} exp(sim(q_i, p_j)) ]
   - Hỗ trợ độ đo tích vô hướng (dot product) và cosine có thang đo hệ số nhiệt độ (scaled cosine với scale = 20,
     tương đương nhiệt độ tau = 1/20 = 0.05 theo SimCSE: Gao et al., 2021 - arXiv:2104.08821).
   - Tối thiểu hóa độ hợp lý log âm (negative log-likelihood minimization).
   - Cơ chế mặt nạ đồng nhất âm vô cực (negative infinity identity masking) theo phong cách Supervised Contrastive
     Learning (Khosla et al., 2020 - arXiv:2004.11362): khi hai mẫu trong lô có cùng chỉ số kịch bản/ý định
     (scenario index / intent index), các mẫu âm giả (false negatives) ngoài đường chéo chính sẽ được gán giá trị
     âm vô cực (-inf) để exp(-inf) = 0, loại bỏ hoàn toàn chúng khỏi mẫu số softmax.

2. Chế độ Mode A2 (Hard Negative Denominator Expansion with Identity Masking):
   - Mở rộng mẫu số hàm mất mát tương phản với các mẫu âm khó (hard negatives) theo DPR Eq.(2) (Karpukhin et al., 2020):
     L(q_i, p_i^+, p_{i,1}^-, ..., p_{i,M}^-) = -log [ exp(sim(q_i, p_i^+)) /
       ( sum_{j, unmasked} exp(sim(q_i, p_j^+)) + sum_{m, unmasked} exp(sim(q_i, p_{i,m}^-)) ) ]
   - Cho phép mở rộng ma trận logits với các mẫu âm khó theo từng truy vấn hoặc chia sẻ toàn lô, kết hợp cơ chế mặt nạ
     đồng nhất âm vô cực (-inf) đối với các mẫu âm khó có chỉ số kịch bản trùng với truy vấn.

3. Chế độ Mode B (Euclidean Margin Triplet Loss):
   - Dựa trên mạng Siamese và FaceNet (Schroff, Kalenichenko, & Philbin, 2015 - arXiv:1503.03832, Eq.(3)):
     L(a, p, n) = max(0, ||a - p||_2 - ||a - n||_2 + alpha)
     với lề khoảng cách Euclid alpha = 1.0 (margin = 1.0).
   - Hỗ trợ gom cụm trung bình (mean pooling) dọc theo chiều độ dài chuỗi token khi nhận đầu vào tensor 3D,
     và tính trung bình trên toàn bộ lô (batch mean pooling reduction).
   - Hàm mất mát triệt tiêu về 0 (loss = 0.0) khi lề khoảng cách được thỏa mãn (||a - n||_2 - ||a - p||_2 >= margin),
     và nhận giá trị dương khi bị vi phạm.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from core.config import (
    DEFAULT_CONTRASTIVE_METRIC,
    DEFAULT_CONTRASTIVE_SCALE,
    DEFAULT_TRIPLET_EPS,
    DEFAULT_TRIPLET_MARGIN,
    DEFAULT_TRIPLET_P,
    IDENTITY_MASK_VALUE,
    SUPPORTED_CONTRASTIVE_METRICS,
)


def _compute_similarity_matrix(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    metric: str = DEFAULT_CONTRASTIVE_METRIC,
    scale: float = DEFAULT_CONTRASTIVE_SCALE,
) -> torch.Tensor:
    """
    Tính ma trận điểm tương đồng giữa các tensor truy vấn (queries) và ứng viên (candidates).

    Tham số:
        queries: Tensor biểu diễn truy vấn (B, D).
        candidates: Tensor biểu diễn ứng viên (K, D).
        metric: Độ đo tương đồng ('dot', 'cosine', hoặc 'scaled_cosine').
        scale: Hệ số phóng đại khi dùng scaled_cosine (mặc định 20.0).

    Trả về:
        Tensor ma trận tương đồng kích thước (B, K).
    """
    if metric == "dot":
        # Tích vô hướng chuẩn: sim(q, p) = q . p (Karpukhin et al., 2020)
        return torch.matmul(queries, candidates.transpose(-1, -2))
    if metric in ("cosine", "scaled_cosine"):
        # Cosine similarity chuẩn hóa L2 kèm hệ số tỷ lệ nhiệt độ scale (Gao et al., 2021)
        q_norm = F.normalize(queries, p=2, dim=-1)
        c_norm = F.normalize(candidates, p=2, dim=-1)
        return torch.matmul(q_norm, c_norm.transpose(-1, -2)) * scale

    raise ValueError(
        f"Độ đo tương đồng '{metric}' không được hỗ trợ. "
        f"Các giá trị hợp lệ: {SUPPORTED_CONTRASTIVE_METRICS}"
    )


def in_batch_contrastive_loss(
    queries: torch.Tensor,
    positives: torch.Tensor,
    scenario_indices: torch.Tensor | Sequence[int] | None = None,
    metric: str = DEFAULT_CONTRASTIVE_METRIC,
    scale: float = DEFAULT_CONTRASTIVE_SCALE,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Tính toán hàm mất mát tương phản trong lô (Mode A: In-batch Contrastive Loss).

    Công thức bài báo InfoNCE (van den Oord et al., 2018) và DPR (Karpukhin et al., 2020):
        L_i = -log [ exp(sim(q_i, p_i)) / sum_{j in U_i} exp(sim(q_i, p_j)) ]
    trong đó U_i là tập hợp các ứng viên hợp lệ (không bị gán mặt nạ âm vô cực).

    Cơ chế Identity Masking (Khosla et al., 2020):
    - Nếu hai mẫu i và j (i != j) trong cùng lô có scenario_indices[i] == scenario_indices[j],
      mẫu j là một mẫu dương bổ sung (duplicate positive), không được coi là mẫu âm trong mẫu số.
    - Vị trí (i, j) được gán giá trị IDENTITY_MASK_VALUE (-inf), khiến exp(-inf) = 0 trong mẫu số softmax.
    - Đường chéo chính (i == j) là nhãn mục tiêu dương nên luôn được bảo toàn, không bao giờ bị che.

    Tham số:
        queries: Tensor nhúng câu truy vấn, kích thước (batch_size, embed_dim).
        positives: Tensor nhúng kịch bản/đoạn văn tích cực tương ứng, kích thước (batch_size, embed_dim).
        scenario_indices: Chỉ số nhận diện kịch bản (1D tensor hoặc Sequence[int]), dùng để che mẫu âm trùng lặp.
        metric: Độ đo tương đồng ('dot', 'cosine', hoặc 'scaled_cosine').
        scale: Hệ số thang đo khi dùng scaled_cosine (mặc định 20.0).
        reduction: Phương thức rút gọn ('mean', 'sum', hoặc 'none').

    Trả về:
        Tensor vô hướng biểu thị giá trị hàm mất mát (hoặc tensor 1D nếu reduction='none').
    """
    if queries.dim() != 2 or positives.dim() != 2:
        raise ValueError(
            f"Queries và positives bắt buộc phải là tensor 2D (batch_size, embed_dim), "
            f"nhận được queries: {queries.shape}, positives: {positives.shape}."
        )
    if queries.shape != positives.shape:
        raise ValueError(
            f"Kích thước queries {queries.shape} và positives {positives.shape} không khớp."
        )
    batch_size = queries.size(0)
    if batch_size == 0:
        raise ValueError("Kích thước batch_size không được bằng 0.")
    if scale <= 0:
        raise ValueError(f"Hệ số thang đo scale phải > 0, nhận được {scale}.")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Chế độ reduction '{reduction}' không hợp lệ. Chọn 'mean', 'sum' hoặc 'none'.")

    # 1. Tính toán ma trận điểm tương đồng giữa tất cả các cặp trong lô: kích thước (B, B)
    similarity = _compute_similarity_matrix(queries, positives, metric=metric, scale=scale)

    # 2. Áp dụng mặt nạ đồng nhất âm vô cực (Identity Masking) khi các mẫu ngoài đường chéo trùng scenario index
    if scenario_indices is not None:
        idx_tensor = torch.as_tensor(scenario_indices, device=queries.device)
        if idx_tensor.dim() != 1 or idx_tensor.size(0) != batch_size:
            raise ValueError(
                f"scenario_indices phải là tensor 1D có độ dài bằng batch_size ({batch_size}), "
                f"nhận được shape {idx_tensor.shape}."
            )
        # Ma trận so khớp: True nếu cùng scenario index
        match_mask = idx_tensor.unsqueeze(0) == idx_tensor.unsqueeze(1)
        # Loại trừ đường chéo chính (i == j là mẫu dương thật sự của truy vấn i)
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=queries.device)
        duplicate_negatives_mask = match_mask & (~diag_mask)
        # Gán âm vô cực (-inf) cho các mẫu âm trùng lặp kịch bản để triệt tiêu khỏi mẫu số softmax
        similarity = similarity.masked_fill(duplicate_negatives_mask, IDENTITY_MASK_VALUE)

    # 3. Mục tiêu phân loại nhị thức đối xứng: Mỗi truy vấn i có mẫu dương tại vị trí i trên đường chéo chính
    targets = torch.arange(batch_size, device=queries.device)

    # 4. Tối thiểu hóa độ hợp lý log âm thông qua Cross Entropy Loss
    return F.cross_entropy(similarity, targets, reduction=reduction)


def hard_negative_contrastive_loss(
    queries: torch.Tensor,
    positives: torch.Tensor,
    hard_negatives: torch.Tensor,
    scenario_indices: torch.Tensor | Sequence[int] | None = None,
    hard_scenario_indices: torch.Tensor | Sequence[int] | None = None,
    metric: str = DEFAULT_CONTRASTIVE_METRIC,
    scale: float = DEFAULT_CONTRASTIVE_SCALE,
    reduction: str = "mean",
    shared_hard_negatives: bool = False,
) -> torch.Tensor:
    """
    Tính toán hàm mất mát tương phản mở rộng mẫu số với các mẫu âm khó (Mode A2: Hard Negative Expansion).

    Công thức DPR Eq.(2) (Karpukhin et al., 2020):
        L_i = -log [ exp(sim(q_i, p_i^+)) /
          ( exp(sim(q_i, p_i^+)) + sum_{j != i, unmasked} exp(sim(q_i, p_j^+)) + sum_{m, unmasked} exp(sim(q_i, p_{i,m}^-)) ) ]

    Cơ chế Identity Masking cho hard negatives:
    - Nếu một hard negative có scenario index trùng khớp với scenario index của truy vấn q_i,
      nó là một mẫu âm giả (false negative), điểm số tương đồng được gán -inf để exp(-inf) = 0.

    Tham số:
        queries: Tensor biểu diễn câu truy vấn (batch_size, embed_dim).
        positives: Tensor biểu diễn mẫu dương (batch_size, embed_dim).
        hard_negatives: Tensor mẫu âm khó. Hỗ trợ:
            - Tensor 3D (batch_size, num_hard_negatives, embed_dim): Mỗi truy vấn có danh sách hard negatives riêng.
            - Tensor 2D (batch_size, embed_dim) khi num_hard_negatives = 1 (mặc định shared_hard_negatives=False).
            - Tensor 2D (num_hard_negatives, embed_dim) khi dùng chung một kho hard negatives (shared_hard_negatives=True).
        scenario_indices: Chỉ số kịch bản cho queries và positives (batch_size,).
        hard_scenario_indices: Chỉ số kịch bản tương ứng của các mẫu âm khó.
        metric: Độ đo tương đồng ('dot', 'cosine', hoặc 'scaled_cosine').
        scale: Hệ số thang đo nhiệt độ (mặc định 20.0).
        reduction: Phương thức rút gọn ('mean', 'sum', hoặc 'none').
        shared_hard_negatives: Nếu True, toàn bộ hard negatives 2D được chia sẻ làm mẫu âm chung cho mọi truy vấn trong lô.

    Trả về:
        Tensor mất mát giá trị vô hướng hoặc tensor 1D.
    """
    if queries.dim() != 2 or positives.dim() != 2:
        raise ValueError("Queries và positives bắt buộc phải là tensor 2D (batch_size, embed_dim).")
    if queries.shape != positives.shape:
        raise ValueError(f"Shape queries {queries.shape} và positives {positives.shape} không khớp.")
    batch_size, embed_dim = queries.shape
    if batch_size == 0:
        raise ValueError("Kích thước batch_size không được bằng 0.")
    if scale <= 0:
        raise ValueError(f"Scale phải > 0, nhận được {scale}.")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Chế độ reduction '{reduction}' không hợp lệ.")

    # 1. Tính toán ma trận điểm tương đồng in-batch (B, B) kèm Identity Masking
    sim_in_batch = _compute_similarity_matrix(queries, positives, metric=metric, scale=scale)
    if scenario_indices is not None:
        idx_tensor = torch.as_tensor(scenario_indices, device=queries.device)
        match_mask = idx_tensor.unsqueeze(0) == idx_tensor.unsqueeze(1)
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=queries.device)
        sim_in_batch = sim_in_batch.masked_fill(match_mask & (~diag_mask), IDENTITY_MASK_VALUE)

    # 2. Tính toán điểm tương đồng với hard negatives
    if hard_negatives.dim() == 3:
        # Trường hợp mỗi truy vấn có danh sách hard negatives riêng: (B, M, D)
        if hard_negatives.size(0) != batch_size or hard_negatives.size(2) != embed_dim:
            raise ValueError(
                f"hard_negatives 3D phải có kích thước ({batch_size}, M, {embed_dim}), "
                f"nhận được {hard_negatives.shape}."
            )
        num_hn = hard_negatives.size(1)
        if metric == "dot":
            sim_hard = (queries.unsqueeze(1) * hard_negatives).sum(dim=-1)  # (B, M)
        elif metric in ("cosine", "scaled_cosine"):
            q_norm = F.normalize(queries, p=2, dim=-1).unsqueeze(1)
            hn_norm = F.normalize(hard_negatives, p=2, dim=-1)
            sim_hard = (q_norm * hn_norm).sum(dim=-1) * scale
        else:
            raise ValueError(f"Độ đo '{metric}' không được hỗ trợ.")

        # Áp dụng identity masking cho hard negatives nếu có thông tin scenario indices
        if scenario_indices is not None and hard_scenario_indices is not None:
            q_idx = torch.as_tensor(scenario_indices, device=queries.device).unsqueeze(1)  # (B, 1)
            hn_idx = torch.as_tensor(hard_scenario_indices, device=queries.device)
            if hn_idx.dim() == 1:
                if hn_idx.size(0) == num_hn:
                    hn_idx = hn_idx.unsqueeze(0)  # (1, M)
                elif hn_idx.size(0) == batch_size and num_hn == 1:
                    hn_idx = hn_idx.unsqueeze(1)  # (B, 1)
                else:
                    raise ValueError(
                        f"hard_scenario_indices 1D kích thước {hn_idx.shape} không tương thích "
                        f"với batch_size={batch_size}, num_hn={num_hn}."
                    )
            elif hn_idx.dim() == 2 and (hn_idx.size(0) != batch_size or hn_idx.size(1) != num_hn):
                raise ValueError(
                    f"hard_scenario_indices 2D kích thước {hn_idx.shape} phải là ({batch_size}, {num_hn})."
                )
            hard_mask = q_idx == hn_idx
            sim_hard = sim_hard.masked_fill(hard_mask, IDENTITY_MASK_VALUE)

    elif hard_negatives.dim() == 2:
        if not shared_hard_negatives and hard_negatives.size(0) == batch_size:
            # Xem mỗi dòng là 1 hard negative tương ứng cho mỗi truy vấn: (B, 1, D)
            hn_3d = hard_negatives.unsqueeze(1)
            if metric == "dot":
                sim_hard = (queries.unsqueeze(1) * hn_3d).sum(dim=-1)  # (B, 1)
            elif metric in ("cosine", "scaled_cosine"):
                q_norm = F.normalize(queries, p=2, dim=-1).unsqueeze(1)
                hn_norm = F.normalize(hn_3d, p=2, dim=-1)
                sim_hard = (q_norm * hn_norm).sum(dim=-1) * scale
            else:
                raise ValueError(f"Độ đo '{metric}' không được hỗ trợ.")

            if scenario_indices is not None and hard_scenario_indices is not None:
                q_idx = torch.as_tensor(scenario_indices, device=queries.device).view(-1, 1)
                hn_idx = torch.as_tensor(hard_scenario_indices, device=queries.device).view(-1, 1)
                hard_mask = q_idx == hn_idx
                sim_hard = sim_hard.masked_fill(hard_mask, IDENTITY_MASK_VALUE)
        else:
            # Kho hard negatives dùng chung: (M, D)
            if hard_negatives.size(1) != embed_dim:
                raise ValueError(
                    f"hard_negatives 2D phải có chiều embedding {embed_dim}, nhận được {hard_negatives.shape}."
                )
            sim_hard = _compute_similarity_matrix(queries, hard_negatives, metric=metric, scale=scale)  # (B, M)

            if scenario_indices is not None and hard_scenario_indices is not None:
                q_idx = torch.as_tensor(scenario_indices, device=queries.device).unsqueeze(1)  # (B, 1)
                hn_idx = torch.as_tensor(hard_scenario_indices, device=queries.device).unsqueeze(0)  # (1, M)
                hard_mask = q_idx == hn_idx
                sim_hard = sim_hard.masked_fill(hard_mask, IDENTITY_MASK_VALUE)
    else:
        raise ValueError(
            f"hard_negatives phải là tensor 2D hoặc 3D, nhận được shape {hard_negatives.shape}."
        )

    # 3. Mở rộng mẫu số: ghép ma trận tương đồng in-batch và ma trận tương đồng hard negatives
    # Logits kích thước (B, B + M), trong đó cột 0..B-1 là in-batch, cột B..B+M-1 là hard negatives
    all_logits = torch.cat([sim_in_batch, sim_hard], dim=-1)

    # Mục tiêu phân loại: vị trí đường chéo của in-batch (cột i) là mẫu dương của truy vấn i
    targets = torch.arange(batch_size, device=queries.device)

    # 4. Tối thiểu hóa độ hợp lý log âm
    return F.cross_entropy(all_logits, targets, reduction=reduction)


def euclidean_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = DEFAULT_TRIPLET_MARGIN,
    reduction: str = "mean",
    p: float = DEFAULT_TRIPLET_P,
    eps: float = DEFAULT_TRIPLET_EPS,
    mean_pooling: bool = True,
) -> torch.Tensor:
    """
    Tính toán hàm mất mát Triplet với khoảng cách Euclid và lề biên (Mode B: Euclidean Margin Triplet Loss).

    Công thức bài báo FaceNet (Schroff, Kalenichenko, & Philbin, 2015, Eq.(3)):
        L(a, p, n) = max(0, ||a - p||_2 - ||a - n||_2 + alpha)
    trong đó:
        - ||x - y||_2 là khoảng cách Euclid (L2 norm)
        - alpha là lề biên khoảng cách (margin, mặc định 1.0)
        - Khi ||a - n||_2 - ||a - p||_2 >= alpha: L = 0.0 (lề được thỏa mãn tuyệt đối)
        - Ngược lại: L > 0.0 (lề bị vi phạm).

    Tính năng gom cụm trung bình (Mean Pooling):
    - Khi nhận tensor 3D (batch_size, seq_length, embed_dim), tự động gom cụm trung bình dọc theo
      chiều chuỗi token để thu được tensor biểu diễn vector câu (batch_size, embed_dim).
    - Sau đó tính trung bình hàm mất mát trên toàn bộ lô (batch mean pooling reduction).

    Tham số:
        anchor: Tensor mỏ neo (a), kích thước (B, D) hoặc (B, L, D).
        positive: Tensor mẫu dương (p), kích thước tương ứng (B, D) hoặc (B, L, D).
        negative: Tensor mẫu âm (n), kích thước tương ứng (B, D) hoặc (B, L, D).
        margin: Lề khoảng cách Euclid alpha (mặc định 1.0).
        reduction: Phương thức rút gọn kết quả ('mean', 'sum', hoặc 'none').
        p: Chuẩn p-norm tính khoảng cách (mặc định 2.0 cho khoảng cách Euclid).
        eps: Số gia nhỏ chống lỗi chia cho 0 khi tính đạo hàm khoảng cách.
        mean_pooling: Nếu True và đầu vào là tensor 3D, thực hiện gom cụm trung bình qua token.

    Trả về:
        Tensor mất mát giá trị vô hướng hoặc tensor 1D.
    """
    if not (anchor.shape == positive.shape == negative.shape):
        raise ValueError(
            f"Kích thước anchor {anchor.shape}, positive {positive.shape}, "
            f"negative {negative.shape} bắt buộc phải giống nhau hoàn toàn."
        )
    if margin < 0:
        raise ValueError(f"Margin phải >= 0, nhận được {margin}.")
    if p <= 0:
        raise ValueError(f"Chuẩn p-norm phải > 0, nhận được {p}.")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Chế độ reduction '{reduction}' không hợp lệ. Chọn 'mean', 'sum' hoặc 'none'.")

    # 1. Thực hiện mean pooling dọc theo chiều chuỗi token nếu đầu vào là tensor 3D
    if anchor.dim() == 3:
        if mean_pooling:
            anchor = anchor.mean(dim=1)
            positive = positive.mean(dim=1)
            negative = negative.mean(dim=1)
        else:
            raise ValueError(
                "Đầu vào là tensor 3D nhưng mean_pooling=False. Vui lòng bật mean_pooling=True "
                "hoặc truyền tensor 2D (batch_size, embed_dim)."
            )
    elif anchor.dim() != 2:
        raise ValueError(f"Đầu vào phải là tensor 2D hoặc 3D, nhận được shape {anchor.shape}.")

    batch_size = anchor.size(0)
    if batch_size == 0:
        raise ValueError("Kích thước batch_size không được bằng 0.")

    # 2. Tính khoảng cách Euclid cặp đôi (Euclidean Pairwise Distance): d(a, p) và d(a, n)
    d_ap = F.pairwise_distance(anchor, positive, p=p, eps=eps)
    d_an = F.pairwise_distance(anchor, negative, p=p, eps=eps)

    # 3. Tính toán mất mát biên Triplet từng mẫu: max(0, d_ap - d_an + margin)
    losses = torch.clamp(d_ap - d_an + margin, min=0.0)

    # 4. Rút gọn kết quả theo phương thức quy định
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    return losses


class InBatchContrastiveLoss(nn.Module):
    """
    Lớp mô-đun PyTorch cho Mode A: In-batch Contrastive Loss.
    """

    def __init__(
        self,
        metric: str = DEFAULT_CONTRASTIVE_METRIC,
        scale: float = DEFAULT_CONTRASTIVE_SCALE,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.metric = metric
        self.scale = scale
        self.reduction = reduction

    def forward(
        self,
        queries: torch.Tensor,
        positives: torch.Tensor,
        scenario_indices: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        return in_batch_contrastive_loss(
            queries=queries,
            positives=positives,
            scenario_indices=scenario_indices,
            metric=self.metric,
            scale=self.scale,
            reduction=self.reduction,
        )


class HardNegativeContrastiveLoss(nn.Module):
    """
    Lớp mô-đun PyTorch cho Mode A2: Hard Negative Contrastive Loss.
    """

    def __init__(
        self,
        metric: str = DEFAULT_CONTRASTIVE_METRIC,
        scale: float = DEFAULT_CONTRASTIVE_SCALE,
        reduction: str = "mean",
        shared_hard_negatives: bool = False,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.scale = scale
        self.reduction = reduction
        self.shared_hard_negatives = shared_hard_negatives

    def forward(
        self,
        queries: torch.Tensor,
        positives: torch.Tensor,
        hard_negatives: torch.Tensor,
        scenario_indices: torch.Tensor | Sequence[int] | None = None,
        hard_scenario_indices: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        return hard_negative_contrastive_loss(
            queries=queries,
            positives=positives,
            hard_negatives=hard_negatives,
            scenario_indices=scenario_indices,
            hard_scenario_indices=hard_scenario_indices,
            metric=self.metric,
            scale=self.scale,
            reduction=self.reduction,
            shared_hard_negatives=self.shared_hard_negatives,
        )


class EuclideanTripletLoss(nn.Module):
    """
    Lớp mô-đun PyTorch cho Mode B: Euclidean Margin Triplet Loss.
    """

    def __init__(
        self,
        margin: float = DEFAULT_TRIPLET_MARGIN,
        reduction: str = "mean",
        p: float = DEFAULT_TRIPLET_P,
        eps: float = DEFAULT_TRIPLET_EPS,
        mean_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.p = p
        self.eps = eps
        self.mean_pooling = mean_pooling

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return euclidean_triplet_loss(
            anchor=anchor,
            positive=positive,
            negative=negative,
            margin=self.margin,
            reduction=self.reduction,
            p=self.p,
            eps=self.eps,
            mean_pooling=self.mean_pooling,
        )


# Các bí danh (aliases) hỗ trợ tra cứu linh hoạt theo mã chế độ huấn luyện
ModeAContrastiveLoss = InBatchContrastiveLoss
ModeA2ContrastiveLoss = HardNegativeContrastiveLoss
ModeBTripletLoss = EuclideanTripletLoss

mode_a_contrastive_loss = in_batch_contrastive_loss
contrastive_loss_mode_a = in_batch_contrastive_loss
mode_a2_contrastive_loss = hard_negative_contrastive_loss
contrastive_loss_mode_a2 = hard_negative_contrastive_loss
mode_b_triplet_loss = euclidean_triplet_loss
triplet_loss_mode_b = euclidean_triplet_loss

__all__ = [
    "IDENTITY_MASK_VALUE",
    "EuclideanTripletLoss",
    "HardNegativeContrastiveLoss",
    "InBatchContrastiveLoss",
    "ModeA2ContrastiveLoss",
    "ModeAContrastiveLoss",
    "ModeBTripletLoss",
    "contrastive_loss_mode_a",
    "contrastive_loss_mode_a2",
    "euclidean_triplet_loss",
    "hard_negative_contrastive_loss",
    "in_batch_contrastive_loss",
    "mode_a2_contrastive_loss",
    "mode_a_contrastive_loss",
    "mode_b_triplet_loss",
    "triplet_loss_mode_b",
]
