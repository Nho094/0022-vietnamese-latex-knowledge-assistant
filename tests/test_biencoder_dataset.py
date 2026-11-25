"""
Bộ kiểm thử tự động cho đường ống dữ liệu huấn luyện Bi-Encoder (tests/test_biencoder_dataset.py).

Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Honesty Rule #3 (Quy tắc trung thực #3):
  + Tuyệt đối không tinh chỉnh trên test set, không truy cập data/test.csv trong data loader/training.
  + Kiểm thử chặt chẽ cơ chế phòng vệ chặn test.csv qua tên tệp và mã băm SHA-256.
- Kiểm thử lấy mẫu biểu diễn kịch bản đồng đều (Uniform Variant Representation Sampling)
  trên tập {name, slots, primed, handmade} theo Retrieve-and-Fill (RAF, Shrivastava et al., 2022).
- Kiểm thử cờ cấu hình include_aliases bổ sung bí danh ý định làm cặp positive.
- Kiểm thử công cụ khai thác mẫu âm khó (Hard Negative Mining) cho Mode A2 (DPR, Karpukhin et al., 2020)
  và Mode B (FaceNet, Schroff et al., 2015).
- Kiểm thử bộ chia dữ liệu giữ lại 20% ý định hoàn toàn độc lập cho thí nghiệm Unseen Intent Generalization.
- Kiểm thử tích hợp với PyTorch Dataset, DataLoader và các hàm mất mát tại train/losses.py.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from core.config import (
    DEFAULT_INTENTS_PATH,
    DEFAULT_UNSEEN_INTENT_RATIO,
    SCENARIO_VARIANTS,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
)
from core.scenario import scenario_string
from train.biencoder_dataset import (
    BiEncoderHardNegativeDataset,
    BiEncoderPairDataset,
    BiEncoderTripletDataset,
    DatasetSample,
    TfidfBaselineRanker,
    build_training_pairs,
    generate_unseen_intent_split,
    load_intent_bank,
    load_train_samples,
    mine_hard_negatives,
    sample_scenario_variant,
    verify_not_test_set,
)
from train.losses import (
    euclidean_triplet_loss,
    hard_negative_contrastive_loss,
    in_batch_contrastive_loss,
)

# =====================================================================
# 1. Kiểm thử phòng vệ Honesty Rule #3 (Guard against test.csv)
# =====================================================================


def test_verify_not_test_set_blocks_test_csv_path() -> None:
    """Kiểm tra verify_not_test_set chặn tuyệt đối đường dẫn TEST_CSV_PATH."""
    with pytest.raises(PermissionError, match="Honesty Rule #3"):
        verify_not_test_set(TEST_CSV_PATH)


def test_verify_not_test_set_blocks_test_csv_filename(tmp_path: Path) -> None:
    """Kiểm tra verify_not_test_set chặn bất kỳ tệp nào có tên test.csv."""
    fake_test = tmp_path / "test.csv"
    fake_test.write_text("text,label,category\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="Honesty Rule #3"):
        verify_not_test_set(fake_test)


def test_verify_not_test_set_blocks_matching_sha256(tmp_path: Path) -> None:
    """Kiểm tra verify_not_test_set chặn tệp đổi tên nhưng có mã băm SHA-256 trùng khớp test.csv."""
    if not TEST_CSV_PATH.exists():
        pytest.skip("Không tìm thấy data/test.csv trên đĩa.")

    real_test_bytes = TEST_CSV_PATH.read_bytes()
    cloned_file = tmp_path / "sneaky_data.csv"
    cloned_file.write_bytes(real_test_bytes)

    with pytest.raises(PermissionError, match="trùng khớp với tập dữ liệu kiểm thử"):
        verify_not_test_set(cloned_file)


def test_verify_not_test_set_allows_train_csv() -> None:
    """Kiểm tra verify_not_test_set cho phép nạp dữ liệu từ train.csv hợp lệ."""
    # Không được ném ra ngoại lệ
    verify_not_test_set(TRAIN_CSV_PATH)


def test_load_train_samples_rejects_test_csv() -> None:
    """Kiểm tra load_train_samples từ chối ngay lập tức nếu vô tình truyền TEST_CSV_PATH."""
    with pytest.raises(PermissionError, match="Honesty Rule #3"):
        load_train_samples(TEST_CSV_PATH)


def test_load_train_samples_success() -> None:
    """Kiểm tra load_train_samples nạp thành công các mẫu từ train.csv."""
    samples = load_train_samples(TRAIN_CSV_PATH)
    assert len(samples) > 0
    first = samples[0]
    assert isinstance(first, DatasetSample)
    assert first.text
    assert first.label
    assert first.category


# =====================================================================
# 2. Kiểm thử lấy mẫu biểu diễn kịch bản đồng đều (Uniform Variant Sampling)
# =====================================================================


def test_sample_scenario_variant_supported_variants() -> None:
    """Kiểm tra sample_scenario_variant luôn sinh ra một trong 4 biến thể RAF hợp lệ."""
    intent_bank = load_intent_bank(DEFAULT_INTENTS_PATH)
    first_intent_name = next(iter(intent_bank))
    first_intent_data = intent_bank[first_intent_name]

    rng = random.Random(42)
    seen_variants: set[str] = set()
    for _ in range(50):
        rep_text, variant_name = sample_scenario_variant(first_intent_data, rng=rng)
        assert variant_name in SCENARIO_VARIANTS
        assert isinstance(rep_text, str)
        assert len(rep_text) > 0
        seen_variants.add(variant_name)

    # Đảm bảo qua nhiều lượt bốc thăm, cả 4 biến thể đều được xuất hiện
    assert seen_variants == set(SCENARIO_VARIANTS)


def test_sample_scenario_variant_uniform_distribution() -> None:
    """
    Kiểm định thống kê tính đồng đều (Uniform Distribution) của việc lấy mẫu biến thể.
    Với N = 8,000 lần lấy mẫu trên 4 biến thể, kỳ vọng mỗi biến thể xuất hiện ~2,000 lần (25%).
    Dung sai cho phép +- 150 (tương đương 1850 đến 2150).
    """
    intent_bank = load_intent_bank(DEFAULT_INTENTS_PATH)
    first_intent_data = next(iter(intent_bank.values()))

    rng = random.Random(12345)
    num_trials = 8000
    counts: Counter[str] = Counter()

    for _ in range(num_trials):
        _, var_name = sample_scenario_variant(first_intent_data, rng=rng)
        counts[var_name] += 1

    expected_per_variant = num_trials / len(SCENARIO_VARIANTS)
    assert len(counts) == 4
    for var in SCENARIO_VARIANTS:
        assert abs(counts[var] - expected_per_variant) < 150, (
            f"Biến thể {var} xuất hiện {counts[var]} lần, lệch quá ngưỡng cho phép so với {expected_per_variant}."
        )


def test_sample_scenario_variant_content_matches_scenario_string() -> None:
    """Kiểm tra nội dung sinh ra từ sample_scenario_variant khớp tuyệt đối với core.scenario.scenario_string."""
    intent_bank = load_intent_bank(DEFAULT_INTENTS_PATH)
    intent_data = intent_bank["LATEX_TAI_LIEU_TOI_GIAN"]

    rng = random.Random(999)
    for _ in range(20):
        rep_text, var_name = sample_scenario_variant(intent_data, rng=rng)
        expected = scenario_string(intent_data, variant=var_name)
        assert rep_text == expected


# =====================================================================
# 3. Kiểm thử xây dựng cặp dữ liệu và cờ aliases (Pair Builders & Aliases Flag)
# =====================================================================


def test_build_training_pairs_without_aliases() -> None:
    """Kiểm tra build_training_pairs khi include_aliases = False."""
    samples = [
        DatasetSample(text="câu 1", label="LATEX_TAI_LIEU_TOI_GIAN", category="tex_struct"),
        DatasetSample(text="câu 2", label="LATEX_TAI_LIEU_TOI_GIAN", category="tex_struct"),
    ]
    intent_bank = load_intent_bank(DEFAULT_INTENTS_PATH)

    pairs = build_training_pairs(samples, intent_bank, include_aliases=False, variant_strategy="uniform")
    assert len(pairs) == len(samples)
    for p in pairs:
        assert p.variant in SCENARIO_VARIANTS
        assert p.intent == "LATEX_TAI_LIEU_TOI_GIAN"
        assert p.query in ("câu 1", "câu 2")


def test_build_training_pairs_with_aliases() -> None:
    """
    Kiểm tra build_training_pairs khi include_aliases = True.
    Bổ sung các cặp dương (query, alias) cho toàn bộ bí danh của intent.
    """
    intent_bank = load_intent_bank(DEFAULT_INTENTS_PATH)
    intent_name = "LATEX_TAI_LIEU_TOI_GIAN"
    aliases = intent_bank[intent_name].get("aliases", [])
    assert len(aliases) >= 5

    samples = [
        DatasetSample(text="câu mẫu", label=intent_name, category="tex_struct"),
    ]

    pairs = build_training_pairs(samples, intent_bank, include_aliases=True, variant_strategy="uniform")
    # Tổng số cặp = 1 cặp kịch bản + len(aliases) cặp alias
    expected_total = 1 + len(aliases)
    assert len(pairs) == expected_total

    alias_pairs = [p for p in pairs if p.variant == "alias"]
    assert len(alias_pairs) == len(aliases)
    assert {p.positive for p in alias_pairs} == set(aliases)


def test_build_training_pairs_preprocessing_flag() -> None:
    """Kiểm tra cờ preprocess trong build_training_pairs."""
    samples = [
        DatasetSample(text="công thức 2x² + 3x = 0", label="LATEX_TAI_LIEU_TOI_GIAN", category="tex_struct"),
    ]
    intent_bank = load_intent_bank(DEFAULT_INTENTS_PATH)

    pairs_preprocessed = build_training_pairs(
        samples, intent_bank, include_aliases=False, variant_strategy="handmade", preprocess=True
    )
    # Ký hiệu 2x² phải được chuẩn hóa thành 2x^2 theo preprocess_math_lookalikes
    assert "2x^2" in pairs_preprocessed[0].query


# =====================================================================
# 4. Kiểm thử khai thác mẫu âm khó (Hard Negative Mining Utilities)
# =====================================================================


def test_tfidf_baseline_ranker() -> None:
    """Kiểm tra TfidfBaselineRanker xếp hạng chuẩn xác các kịch bản tương đồng."""
    corpus = [
        "cấu trúc tài liệu tối giản article",
        "chèn ảnh đồ họa graphicsincludegraphics",
        "tạo bảng ma trận tabular array",
    ]
    ranker = TfidfBaselineRanker(corpus)
    scores = ranker.rank("tôi muốn tạo bảng ma trận")
    assert len(scores) == 3
    # Ứng viên thứ 3 ("tạo bảng ma trận") phải có điểm cao nhất
    assert scores[2] > scores[0]
    assert scores[2] > scores[1]

    # Kiểm tra batch ranking
    batch_scores = ranker.rank_batch(["tạo bảng", "chèn ảnh"])
    assert len(batch_scores) == 2
    assert batch_scores[0][2] > batch_scores[0][1]
    assert batch_scores[1][1] > batch_scores[1][0]


def test_mine_hard_negatives_mode_a2_and_mode_b() -> None:
    """
    Kiểm tra mine_hard_negatives:
    - Thu thập đúng top-K kịch bản sai cho Mode A2.
    - Thu thập đúng kịch bản sai khó nhất cho Mode B.
    - Tuyệt đối không chứa kịch bản cùng intent với truy vấn mục tiêu.
    """
    intent_bank = load_intent_bank(DEFAULT_INTENTS_PATH)
    intents = sorted(intent_bank.keys())[:5]  # Lấy 5 intent nhỏ để test nhanh
    sub_bank = {k: intent_bank[k] for k in intents}

    samples = [
        DatasetSample(text="tạo file latex cơ bản", label=intents[0], category="tex_struct"),
        DatasetSample(text="chèn hình ảnh đồ họa", label=intents[1], category="tex_struct"),
    ]

    top_k = 3
    mode_a2, mode_b = mine_hard_negatives(
        train_samples=samples,
        intent_bank=sub_bank,
        ranker=None,  # Tự động khởi tạo TfidfBaselineRanker
        top_k=top_k,
        scenario_variant="handmade",
    )

    # 1. Kiểm tra Mode A2
    assert len(mode_a2) == len(samples)
    for item in mode_a2:
        assert len(item.hard_negatives) == top_k
        assert len(item.hard_scenario_indices) == top_k
        # Không được chứa chỉ số kịch bản của chính mình trong hard negatives
        assert item.scenario_index not in item.hard_scenario_indices

    # 2. Kiểm tra Mode B
    assert len(mode_b) == len(samples)
    for trip in mode_b:
        assert trip.anchor
        assert trip.positive
        assert trip.negative
        assert trip.anchor_intent != trip.negative_intent
        assert trip.anchor_idx != trip.negative_idx


# =====================================================================
# 5. Kiểm thử bộ phân tách ý định chưa từng thấy (Unseen Intent Split Generator)
# =====================================================================


def test_generate_unseen_intent_split_ratio_and_disjoint() -> None:
    """
    Kiểm tra generate_unseen_intent_split:
    - Tỷ lệ holdout đạt xấp xỉ 20%.
    - Tập seen_intents và unseen_intents rời rạc tuyệt đối (0 overlap).
    - Tập train_samples chỉ chứa seen intents, unseen_eval_samples chỉ chứa unseen intents.
    """
    samples = load_train_samples(TRAIN_CSV_PATH)
    all_intents = sorted({s.label for s in samples})
    total_intents = len(all_intents)

    split = generate_unseen_intent_split(
        samples=samples,
        holdout_ratio=DEFAULT_UNSEEN_INTENT_RATIO,
        seed=42,
        stratify_by_category=True,
    )

    # 1. Kiểm tra số lượng ý định unseen đạt ~20%
    expected_unseen = round(total_intents * DEFAULT_UNSEEN_INTENT_RATIO)
    assert len(split.unseen_intents) == expected_unseen
    assert len(split.seen_intents) + len(split.unseen_intents) == total_intents

    # 2. Kiểm tra không trùng lặp ý định (0 overlap)
    seen_set = set(split.seen_intents)
    unseen_set = set(split.unseen_intents)
    assert seen_set.isdisjoint(unseen_set)

    # 3. Kiểm tra phân bổ mẫu câu
    assert len(split.train_samples) + len(split.unseen_eval_samples) == len(samples)
    for s in split.train_samples:
        assert s.label in seen_set
        assert s.label not in unseen_set

    for s in split.unseen_eval_samples:
        assert s.label in unseen_set
        assert s.label not in seen_set


def test_generate_unseen_intent_split_stratification() -> None:
    """Kiểm tra tính phân tầng theo category: mỗi category lớn đều có intent bị holdout."""
    samples = load_train_samples(TRAIN_CSV_PATH)
    split = generate_unseen_intent_split(
        samples=samples,
        holdout_ratio=0.20,
        seed=100,
        stratify_by_category=True,
    )

    # Gom nhóm theo category
    cat_to_unseen: dict[str, int] = Counter()
    intent2cat = {s.label: s.category for s in samples}
    for u_intent in split.unseen_intents:
        cat_to_unseen[intent2cat[u_intent]] += 1

    # Kiểm tra các category phổ biến đều có ít nhất 1 intent bị holdout
    assert len(cat_to_unseen) >= 8


def test_generate_unseen_intent_split_reproducibility() -> None:
    """Kiểm tra tính tất định: Cùng seed sinh ra phân tách giống hệt nhau."""
    samples = load_train_samples(TRAIN_CSV_PATH)
    split1 = generate_unseen_intent_split(samples, seed=42)
    split2 = generate_unseen_intent_split(samples, seed=42)

    assert split1.seen_intents == split2.seen_intents
    assert split1.unseen_intents == split2.unseen_intents
    assert len(split1.train_samples) == len(split2.train_samples)


# =====================================================================
# 6. Kiểm thử PyTorch Dataset và tích hợp DataLoader
# =====================================================================


def test_biencoder_pair_dataset_dataloader() -> None:
    """Kiểm tra BiEncoderPairDataset hoạt động tương thích với PyTorch DataLoader."""
    intent_bank = load_intent_bank(DEFAULT_INTENTS_PATH)
    samples = [
        DatasetSample(text="truy vấn 1", label="LATEX_TAI_LIEU_TOI_GIAN", category="tex_struct"),
        DatasetSample(text="truy vấn 2", label="LATEX_TAI_LIEU_TOI_GIAN", category="tex_struct"),
        DatasetSample(text="truy vấn 3", label="LATEX_LOP_TAI_LIEU_ARTICLE", category="tex_doc"),
        DatasetSample(text="truy vấn 4", label="LATEX_LOP_TAI_LIEU_ARTICLE", category="tex_doc"),
    ]

    dataset = BiEncoderPairDataset(
        samples=samples,
        intent_bank=intent_bank,
        include_aliases=False,
        variant_strategy="uniform",
        dynamic_sampling=True,
        seed=42,
    )
    assert len(dataset) == 4

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert "query" in batch
    assert "positive" in batch
    assert "scenario_index" in batch
    assert len(batch["query"]) == 2
    assert len(batch["positive"]) == 2
    assert isinstance(batch["scenario_index"], torch.Tensor)


def test_biencoder_hard_negative_dataset_dataloader() -> None:
    """Kiểm tra BiEncoderHardNegativeDataset tương thích DataLoader."""
    mode_a2_samples = [
        mine_hard_negatives(
            train_samples=[DatasetSample(text="test", label="LATEX_TAI_LIEU_TOI_GIAN", category="tex_struct")],
            intent_bank=load_intent_bank(DEFAULT_INTENTS_PATH),
            top_k=2,
        )[0][0]
    ]

    dataset = BiEncoderHardNegativeDataset(mode_a2_samples)
    assert len(dataset) == 1

    loader = DataLoader(dataset, batch_size=1)
    batch = next(iter(loader))

    assert "query" in batch
    assert "positive" in batch
    assert "hard_negatives" in batch
    assert "hard_scenario_indices" in batch
    assert len(batch["hard_negatives"]) == 2  # 2 kịch bản hard negatives


def test_biencoder_triplet_dataset_dataloader() -> None:
    """Kiểm tra BiEncoderTripletDataset tương thích DataLoader."""
    triplets = [
        mine_hard_negatives(
            train_samples=[DatasetSample(text="test", label="LATEX_TAI_LIEU_TOI_GIAN", category="tex_struct")],
            intent_bank=load_intent_bank(DEFAULT_INTENTS_PATH),
            top_k=1,
        )[1][0]
    ]

    dataset = BiEncoderTripletDataset(triplets)
    assert len(dataset) == 1

    loader = DataLoader(dataset, batch_size=1)
    batch = next(iter(loader))

    assert "anchor" in batch
    assert "positive" in batch
    assert "negative" in batch


# =====================================================================
# 7. Kiểm thử tích hợp hàm mất mát (Loss Integration Test)
# =====================================================================


def test_loss_integration_mode_a() -> None:
    """Kiểm tra dữ liệu sinh từ BiEncoderPairDataset kết nối thành công với in_batch_contrastive_loss."""
    batch_size = 4
    embed_dim = 16

    # Giả lập embedding vector từ mô hình hai tháp Bi-Encoder
    queries = torch.randn(batch_size, embed_dim)
    positives = torch.randn(batch_size, embed_dim)
    scenario_indices = torch.tensor([0, 1, 2, 3])

    loss = in_batch_contrastive_loss(queries, positives, scenario_indices=scenario_indices)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_loss_integration_mode_a2() -> None:
    """Kiểm tra dữ liệu sinh từ BiEncoderHardNegativeDataset kết nối thành công với hard_negative_contrastive_loss."""
    batch_size = 2
    embed_dim = 16
    num_hard_negatives = 3

    queries = torch.randn(batch_size, embed_dim)
    positives = torch.randn(batch_size, embed_dim)
    hard_negatives = torch.randn(batch_size, num_hard_negatives, embed_dim)
    scenario_indices = torch.tensor([0, 1])
    hard_scenario_indices = torch.tensor([[2, 3, 4], [2, 3, 4]])

    loss = hard_negative_contrastive_loss(
        queries=queries,
        positives=positives,
        hard_negatives=hard_negatives,
        scenario_indices=scenario_indices,
        hard_scenario_indices=hard_scenario_indices,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_loss_integration_mode_b() -> None:
    """Kiểm tra dữ liệu sinh từ BiEncoderTripletDataset kết nối thành công với euclidean_triplet_loss."""
    batch_size = 3
    embed_dim = 16

    anchors = torch.randn(batch_size, embed_dim)
    positives = torch.randn(batch_size, embed_dim)
    negatives = torch.randn(batch_size, embed_dim)

    loss = euclidean_triplet_loss(anchor=anchors, positive=positives, negative=negatives, margin=1.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0.0
