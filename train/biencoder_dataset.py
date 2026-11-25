"""
Module xây dựng đường ống dữ liệu huấn luyện Bi-Encoder và lấy mẫu biểu diễn kịch bản (Bi-Encoder Dataset Pipeline).

Tệp tin: train/biencoder_dataset.py
Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Identifiers bằng tiếng Anh; docstrings và comments bằng tiếng Việt giải thích lý do (WHY) và trích dẫn bài báo khoa học.
- Toàn bộ hằng số cấu hình được nạp từ core/config.py (không dùng số ma thuật).
- Thiết lập seed ngẫu nhiên tất định thông qua core.config.set_seed.
- Quy tắc Trung thực #3 (Honesty Rule #3):
  + Tuyệt đối KHÔNG truy cập hoặc tải tệp data/test.csv trong quá trình huấn luyện hoặc nạp dữ liệu.
  + Mọi cặp dữ liệu (pairs) và bộ ba (triplets) được xây dựng ĐỘC QUYỀN từ train.csv.
- Retrieve-and-Fill (RAF, Shrivastava et al., EACL 2022 - arXiv:2202.00901):
  + Lấy mẫu đồng đều (uniform sampling) giữa 4 biến thể biểu diễn kịch bản:
    {name, slots, primed, handmade} thông qua hàm core.scenario.scenario_string.
  + Cờ cấu hình include_aliases cho phép mở rộng các bí danh ý định (aliases) làm cặp positive bổ sung.
- Khai thác mẫu âm khó (Hard Negative Mining):
  + Xếp hạng kịch bản ứng viên bằng mô hình đường cơ sở (baseline model).
  + Thu thập top-K kịch bản sai khó nhất cho Chế độ Mode A2 (DPR, Karpukhin et al., 2020).
  + Thu thập ý định âm khó nhất (hardest negative intent) cho Chế độ Mode B (FaceNet, Schroff et al., 2015).
- Bộ tạo phân tách tổng quát hóa ý định chưa từng thấy (Unseen Intent Split Generator):
  + Tách giữ lại chính xác 20% ý định hoàn toàn độc lập cho các thí nghiệm Zero-Shot / Unseen Intent Generalization.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from torch.utils.data import Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[misc, assignment]

from core.config import (
    DEFAULT_HARD_NEGATIVE_TOP_K,
    DEFAULT_INCLUDE_ALIASES,
    DEFAULT_INTENTS_PATH,
    DEFAULT_SCENARIO_VARIANT,
    DEFAULT_SEED,
    DEFAULT_UNSEEN_INTENT_RATIO,
    SCENARIO_VARIANTS,
    TEST_CSV_PATH,
    TEST_CSV_SHA256,
    TRAIN_CSV_PATH,
)
from core.preprocess import prepare_for_embedding
from core.scenario import scenario_string

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Phòng vệ Quy tắc Trung thực #3 (Honesty Rule #3 Guard)
# =====================================================================


def verify_not_test_set(path: Path | str) -> None:
    """
    Kiểm tra và ngăn chặn tuyệt đối hành vi truy cập tệp test.csv trong quá trình huấn luyện và nạp dữ liệu.

    Tuân thủ Honesty Rule #3 (AGENTS.md):
    - Không điều chỉnh trên tập kiểm thử (Never tune on the test set).
    - data/test.csv chỉ được đọc duy nhất một lần ở bước đánh giá cuối cùng.
    - Phát hiện vi phạm qua tên tệp, đường dẫn tuyệt đối hoặc mã băm SHA-256.

    Tham số:
        path: Đường dẫn tệp CSV cần kiểm tra.

    Ngoại lệ:
        PermissionError: Nếu đường dẫn trỏ tới data/test.csv hoặc có nội dung trùng khớp test set.
    """
    path_obj = Path(path).resolve()
    test_path_obj = TEST_CSV_PATH.resolve()

    # 1. Kiểm tra theo tên tệp hoặc đường dẫn trùng khớp
    if path_obj.name == "test.csv" or path_obj == test_path_obj:
        raise PermissionError(
            "Quy tắc Trung thực #3 (Honesty Rule #3) bị vi phạm: "
            f"Tệp '{path}' là tập kiểm thử (test.csv). "
            "Tuyệt đối không được truy cập test.csv trong quá trình huấn luyện hoặc chuẩn bị dữ liệu!"
        )

    # 2. Kiểm tra sâu theo mã băm SHA-256 nếu tệp tồn tại trên đĩa
    if path_obj.exists() and path_obj.is_file():
        hasher = hashlib.sha256()
        with path_obj.open("rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        current_hash = hasher.hexdigest()

        # Tính toán hash động của test.csv hiện tại trên đĩa nếu có
        blocked_hashes = {TEST_CSV_SHA256}
        if test_path_obj.exists():
            test_hasher = hashlib.sha256()
            with test_path_obj.open("rb") as f:
                while chunk := f.read(65536):
                    test_hasher.update(chunk)
            blocked_hashes.add(test_hasher.hexdigest())

        if current_hash in blocked_hashes:
            raise PermissionError(
                "Quy tắc Trung thực #3 (Honesty Rule #3) bị vi phạm: "
                f"Mã băm SHA-256 của '{path}' ({current_hash}) trùng khớp với tập dữ liệu kiểm thử test.csv. "
                "Tuyệt đối không được sử dụng dữ liệu kiểm thử để huấn luyện!"
            )


# =====================================================================
# 2. Cấu trúc dữ liệu mẫu (Data Structures)
# =====================================================================


@dataclass(frozen=True)
class DatasetSample:
    """Mẫu câu truy vấn huấn luyện từ tập dữ liệu CSV."""

    text: str
    label: str
    category: str


@dataclass(frozen=True)
class PairSample:
    """
    Mẫu cặp dữ liệu (query, positive) cho huấn luyện tương phản In-batch (Mode A).
    Theo InfoNCE (Oord et al., 2018) và DPR (Karpukhin et al., 2020).
    """

    query: str
    positive: str
    scenario_index: int
    intent: str
    variant: str  # 'name', 'slots', 'primed', 'handmade', hoặc 'alias'


@dataclass(frozen=True)
class ModeA2Sample:
    """
    Mẫu cặp dữ liệu mở rộng mẫu âm khó (Hard Negatives) cho Mode A2.
    Theo DPR Eq.(2) (Karpukhin et al., 2020).
    """

    query: str
    positive: str
    hard_negatives: list[str]
    scenario_index: int
    hard_scenario_indices: list[int]
    intent: str
    variant: str


@dataclass(frozen=True)
class TripletSample:
    """
    Mẫu bộ ba (anchor, positive, negative) cho huấn luyện lề khoảng cách Euclid (Mode B).
    Theo FaceNet Eq.(3) (Schroff, Kalenichenko, & Philbin, 2015).
    """

    anchor: str
    positive: str
    negative: str
    anchor_intent: str
    negative_intent: str
    anchor_idx: int
    negative_idx: int
    variant: str


@dataclass(frozen=True)
class UnseenIntentSplitResult:
    """Kết quả phân tách dữ liệu cho thí nghiệm đánh giá khái quát hóa ý định chưa từng thấy."""

    train_samples: list[DatasetSample]
    unseen_eval_samples: list[DatasetSample]
    seen_intents: list[str]
    unseen_intents: list[str]
    holdout_ratio: float


# =====================================================================
# 3. Trình nạp dữ liệu và Ngân hàng kịch bản (Data Loaders)
# =====================================================================


def load_train_samples(csv_path: Path | str = TRAIN_CSV_PATH) -> list[DatasetSample]:
    """
    Nạp danh sách các mẫu truy vấn độc quyền từ tập dữ liệu train.csv.

    Kiểm tra phòng vệ nghiêm ngặt: Tuyệt đối không cho phép đọc test.csv.

    Tham số:
        csv_path: Đường dẫn tới tệp CSV huấn luyện (mặc định data/train.csv).

    Trả về:
        Danh sách các đối tượng DatasetSample.
    """
    verify_not_test_set(csv_path)

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp huấn luyện tại '{csv_path}'.")

    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            category = (row.get("category") or "").strip()
            if text and label:
                samples.append(DatasetSample(text=text, label=label, category=category))

    return samples


def load_intent_bank(bank_path: Path | str = DEFAULT_INTENTS_PATH) -> dict[str, dict[str, Any]]:
    """
    Nạp ngân hàng kịch bản từ tệp intents.json thành từ điển tra cứu theo tên intent.

    Tham số:
        bank_path: Đường dẫn tới intents.json.

    Trả về:
        Từ điển ánh xạ từ tên intent -> thông tin chi tiết kịch bản.
    """
    path = Path(bank_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy ngân hàng kịch bản tại '{bank_path}'.")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {str(item["intent"]): item for item in data if isinstance(item, dict) and "intent" in item}
    elif isinstance(data, dict):
        return data
    return {}


# =====================================================================
# 4. Lấy mẫu biểu diễn biến thể đồng đều (Uniform Variant Sampling)
# =====================================================================


def sample_scenario_variant(
    intent_data: dict[str, Any] | str,
    rng: random.Random | None = None,
    allowed_variants: Sequence[str] = SCENARIO_VARIANTS,
    bank_path: Path | str = DEFAULT_INTENTS_PATH,
) -> tuple[str, str]:
    """
    Lấy mẫu ngẫu nhiên đồng đều một biến thể biểu diễn kịch bản từ tập hợp:
    {name, slots, primed, handmade} theo lý thuyết Retrieve-and-Fill (RAF, Mục 3.2, Table 10).

    Mỗi biến thể có xác suất được chọn chính xác là 1/len(allowed_variants) = 0.25 (với 4 biến thể).
    Biểu diễn chuỗi được sinh trực tiếp bằng hàm core.scenario.scenario_string.

    Tham số:
        intent_data: Dữ liệu kịch bản (dict) hoặc tên chuỗi định danh intent.
        rng: Bộ sinh số ngẫu nhiên random.Random để đảm bảo tính tất định (tùy chọn).
        allowed_variants: Danh sách các biến thể hợp lệ (mặc định SCENARIO_VARIANTS).
        bank_path: Đường dẫn intents.json nếu truyền intent_data dạng chuỗi tên.

    Trả về:
        Bộ (chuỗi biểu diễn kịch bản, tên biến thể đã chọn).
    """
    if not allowed_variants:
        raise ValueError("Danh sách biến thể allowed_variants không được rỗng.")

    chosen_variant = rng.choice(allowed_variants) if rng is not None else random.choice(allowed_variants)
    rep_string = scenario_string(intent_data, variant=chosen_variant, bank_path=bank_path)
    return rep_string, chosen_variant


# =====================================================================
# 5. Xây dựng tập cặp và bộ ba (Pair and Triplet Dataset Builders)
# =====================================================================


def build_training_pairs(
    samples: list[DatasetSample],
    intent_bank: dict[str, dict[str, Any]],
    include_aliases: bool = DEFAULT_INCLUDE_ALIASES,
    variant_strategy: str = "uniform",
    preprocess: bool = False,
    rng: random.Random | None = None,
    bank_path: Path | str = DEFAULT_INTENTS_PATH,
) -> list[PairSample]:
    """
    Xây dựng danh sách các cặp huấn luyện (query, positive) độc quyền từ train.csv cho Mode A.

    Phương pháp lấy mẫu biến thể (Variant Sampling):
    - Khi variant_strategy == 'uniform': Mỗi mẫu truy vấn được gán một biến thể kịch bản
      chọn ngẫu nhiên đồng đều từ {name, slots, primed, handmade}.
    - Khi variant_strategy cố định (ví dụ 'handmade'): Toàn bộ mẫu sử dụng biến thể đó.
    - Khi include_aliases == True: Bổ sung thêm các cặp dương (query, alias) cho toàn bộ
      các biệt danh ngữ nghĩa (aliases) của intent đó, giúp mở rộng độ phủ ngữ nghĩa.
    - Khi preprocess == True: Tiền xử lý văn bản bằng prepare_for_embedding.

    Tham số:
        samples: Danh sách mẫu DatasetSample từ train.csv.
        intent_bank: Ngân hàng kịch bản intents.json.
        include_aliases: Cờ cấu hình bổ sung các bí danh intent làm cặp positive.
        variant_strategy: Chiến lược biến thể ('uniform', 'handmade', 'slots', 'primed', 'name').
        preprocess: Cờ kích hoạt tiền xử lý prepare_for_embedding.
        rng: Bộ sinh số ngẫu nhiên phục vụ tính tất định.
        bank_path: Đường dẫn intents.json.

    Trả về:
        Danh sách các đối tượng PairSample.
    """
    intent_names = sorted(intent_bank.keys())
    intent2idx = {name: idx for idx, name in enumerate(intent_names)}

    pairs: list[PairSample] = []

    for s in samples:
        intent_name = s.label
        if intent_name not in intent_bank:
            continue
        intent_data = intent_bank[intent_name]
        intent_idx = intent2idx[intent_name]

        # 1. Sinh chuỗi positive từ biểu diễn kịch bản (Scenario Representation)
        if variant_strategy == "uniform":
            pos_text, var_name = sample_scenario_variant(
                intent_data, rng=rng, allowed_variants=SCENARIO_VARIANTS, bank_path=bank_path
            )
        else:
            var_name = variant_strategy
            pos_text = scenario_string(intent_data, variant=var_name, bank_path=bank_path)

        query_text = prepare_for_embedding(s.text) if preprocess else s.text
        pos_clean = prepare_for_embedding(pos_text) if preprocess else pos_text

        pairs.append(
            PairSample(
                query=query_text,
                positive=pos_clean,
                scenario_index=intent_idx,
                intent=intent_name,
                variant=var_name,
            )
        )

        # 2. Bổ sung các cặp positive từ intent aliases nếu include_aliases = True
        if include_aliases:
            aliases = intent_data.get("aliases", [])
            for alias in aliases:
                alias_clean = str(alias).strip()
                if alias_clean:
                    alias_final = prepare_for_embedding(alias_clean) if preprocess else alias_clean
                    pairs.append(
                        PairSample(
                            query=query_text,
                            positive=alias_final,
                            scenario_index=intent_idx,
                            intent=intent_name,
                            variant="alias",
                        )
                    )

    return pairs


# =====================================================================
# 6. Tiện ích khai thác mẫu âm khó (Hard Negative Mining Utilities)
# =====================================================================


class BaselineRanker:
    """Lớp cơ sở trừu tượng cho mô hình xếp hạng đường cơ sở."""

    def rank(self, query: str, candidates: Sequence[str]) -> list[float]:
        """
        Tính toán điểm số tương đồng giữa truy vấn và danh sách ứng viên.

        Trả về:
            Danh sách điểm số thực tương ứng từng ứng viên (điểm càng cao càng tương đồng).
        """
        raise NotImplementedError

    def rank_batch(self, queries: Sequence[str], candidates: Sequence[str] | None = None) -> list[list[float]]:
        """Xếp hạng theo lô (batch) cho nhiều câu truy vấn."""
        if candidates is None:
            raise ValueError("candidates không được để trống khi lớp cơ sở chưa có ngữ liệu mặc định.")
        return [self.rank(q, candidates) for q in queries]


class TfidfBaselineRanker(BaselineRanker):
    """
    Bộ xếp hạng đường cơ sở TF-IDF kết hợp Cosine Similarity.
    Hoạt động nhẹ, độc lập và chạy hoàn toàn ngoại tuyến không phụ thuộc mô hình ngoài.
    """

    def __init__(self, candidate_corpus: Sequence[str]) -> None:
        """
        Khởi tạo và huấn luyện ma trận TF-IDF trên kho ngữ liệu ứng viên.

        Tham số:
            candidate_corpus: Danh sách các chuỗi kịch bản/văn bản ứng viên.
        """
        self.candidate_corpus = list(candidate_corpus)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.candidate_matrix = self.vectorizer.fit_transform(self.candidate_corpus)

    def rank(self, query: str, candidates: Sequence[str] | None = None) -> list[float]:
        """Tính điểm cosine similarity giữa truy vấn và các ứng viên."""
        query_vec = self.vectorizer.transform([query])
        if candidates is None or list(candidates) == self.candidate_corpus:
            sims = cosine_similarity(query_vec, self.candidate_matrix)[0]
        else:
            c_matrix = self.vectorizer.transform(candidates)
            sims = cosine_similarity(query_vec, c_matrix)[0]
        return sims.tolist()

    def rank_batch(self, queries: Sequence[str], candidates: Sequence[str] | None = None) -> list[list[float]]:
        """Tính điểm cosine similarity theo lô ma trận hóa giữa nhiều truy vấn và các ứng viên."""
        query_vecs = self.vectorizer.transform(queries)
        if candidates is None or list(candidates) == self.candidate_corpus:
            c_matrix = self.candidate_matrix
        else:
            c_matrix = self.vectorizer.transform(candidates)
        sims = cosine_similarity(query_vecs, c_matrix)
        return sims.tolist()


def mine_hard_negatives(
    train_samples: list[DatasetSample],
    intent_bank: dict[str, dict[str, Any]],
    ranker: BaselineRanker | Callable[[str, Sequence[str]], Sequence[float]] | None = None,
    top_k: int = DEFAULT_HARD_NEGATIVE_TOP_K,
    scenario_variant: str = DEFAULT_SCENARIO_VARIANT,
    bank_path: Path | str = DEFAULT_INTENTS_PATH,
) -> tuple[list[ModeA2Sample], list[TripletSample]]:
    """
    Khai thác mẫu âm khó (Hard Negative Mining) từ kho dữ liệu train.csv bằng mô hình baseline.

    Phương pháp:
    - Với mỗi câu truy vấn q thuộc intent mục tiêu y:
      + Đánh giá điểm số tương đồng giữa q và toàn bộ các kịch bản ứng viên {s_j} với j != y.
      + Sắp xếp các ứng viên theo độ tương đồng giảm dần.
      + Mode A2: Thu thập top-K kịch bản sai có điểm số cao nhất (top-K hardest wrong scenarios)
        để đưa vào mẫu số của DPR Contrastive Loss (Karpukhin et al., 2020).
      + Mode B: Thu thập kịch bản sai khó nhất (top-1 hardest wrong scenario) tạo bộ ba
        (anchor, positive, negative) cho FaceNet Triplet Loss (Schroff et al., 2015).

    Tham số:
        train_samples: Danh sách mẫu huấn luyện độc quyền từ train.csv.
        intent_bank: Ngân hàng kịch bản intents.json.
        ranker: Đối tượng BaselineRanker hoặc hàm xếp hạng. Nếu None, tự động sử dụng TfidfBaselineRanker.
        top_k: Số lượng kịch bản âm khó cần khai thác cho Mode A2 (mặc định 3).
        scenario_variant: Biến thể kịch bản dùng làm ứng viên (mặc định 'handmade').
        bank_path: Đường dẫn intents.json.

    Trả về:
        Tuple (danh sách ModeA2Sample, danh sách TripletSample).
    """
    intent_names = sorted(intent_bank.keys())
    intent2idx = {name: idx for idx, name in enumerate(intent_names)}

    # Chuẩn bị danh sách kịch bản đại diện cho mỗi intent
    intent_scenarios: dict[str, str] = {}
    for name in intent_names:
        intent_scenarios[name] = scenario_string(intent_bank[name], variant=scenario_variant, bank_path=bank_path)

    # Danh sách ứng viên cố định
    candidate_intents = intent_names
    candidate_texts = [intent_scenarios[name] for name in candidate_intents]

    # Nếu không truyền ranker, sử dụng mặc định TfidfBaselineRanker
    active_ranker: BaselineRanker | Callable[[str, Sequence[str]], Sequence[float]]
    if ranker is None:
        active_ranker = TfidfBaselineRanker(candidate_texts)
    else:
        active_ranker = ranker

    mode_a2_samples: list[ModeA2Sample] = []
    triplet_samples: list[TripletSample] = []

    # Kiểm tra hỗ trợ xếp hạng theo lô
    has_batch = hasattr(active_ranker, "rank_batch") and callable(active_ranker.rank_batch)
    valid_samples = [s for s in train_samples if s.label in intent_bank]

    if has_batch:
        all_queries = [s.text for s in valid_samples]
        all_scores = active_ranker.rank_batch(all_queries, candidate_texts)  # type: ignore[union-attr]
    else:
        all_scores = [
            list(active_ranker(s.text, candidate_texts))  # type: ignore[operator]
            if callable(active_ranker) and not isinstance(active_ranker, BaselineRanker)
            else active_ranker.rank(s.text, candidate_texts)  # type: ignore[union-attr]
            for s in valid_samples
        ]

    for sample, scores in zip(valid_samples, all_scores, strict=False):
        anchor_intent = sample.label
        anchor_idx = intent2idx[anchor_intent]
        pos_scenario = intent_scenarios[anchor_intent]

        # Lọc bỏ các ứng viên thuộc cùng intent (chỉ giữ mẫu âm: candidate_intent != anchor_intent)
        negative_candidates: list[tuple[float, str, str, int]] = []
        for cand_intent, cand_text, score in zip(candidate_intents, candidate_texts, scores, strict=False):
            if cand_intent != anchor_intent:
                cand_idx = intent2idx[cand_intent]
                negative_candidates.append((score, cand_intent, cand_text, cand_idx))

        # Sắp xếp điểm số giảm dần (điểm cao nhất = mẫu âm khó nhất)
        negative_candidates.sort(key=lambda x: x[0], reverse=True)

        # 1. Trích xuất top-K cho Mode A2
        top_k_negatives = negative_candidates[:top_k]
        hard_neg_texts = [item[2] for item in top_k_negatives]
        hard_neg_indices = [item[3] for item in top_k_negatives]

        mode_a2_samples.append(
            ModeA2Sample(
                query=sample.text,
                positive=pos_scenario,
                hard_negatives=hard_neg_texts,
                scenario_index=anchor_idx,
                hard_scenario_indices=hard_neg_indices,
                intent=anchor_intent,
                variant=scenario_variant,
            )
        )

        # 2. Trích xuất top-1 khó nhất cho Mode B (Triplet)
        if negative_candidates:
            hardest_cand = negative_candidates[0]
            hardest_neg_intent = hardest_cand[1]
            hardest_neg_text = hardest_cand[2]
            hardest_neg_idx = hardest_cand[3]

            triplet_samples.append(
                TripletSample(
                    anchor=sample.text,
                    positive=pos_scenario,
                    negative=hardest_neg_text,
                    anchor_intent=anchor_intent,
                    negative_intent=hardest_neg_intent,
                    anchor_idx=anchor_idx,
                    negative_idx=hardest_neg_idx,
                    variant=scenario_variant,
                )
            )

    return mode_a2_samples, triplet_samples


# =====================================================================
# 7. Bộ chia tập ý định chưa từng thấy (Unseen Intent Split Generator)
# =====================================================================


def generate_unseen_intent_split(
    samples: list[DatasetSample],
    intent_bank: dict[str, dict[str, Any]] | None = None,
    holdout_ratio: float = DEFAULT_UNSEEN_INTENT_RATIO,
    seed: int = DEFAULT_SEED,
    stratify_by_category: bool = True,
) -> UnseenIntentSplitResult:
    """
    Tạo phân tách giữ lại (holdout) chính xác một tỷ lệ ý định (mặc định 20%) hoàn toàn độc lập
    nhằm phục vụ đánh giá tính khái quát hóa trên ý định chưa từng thấy (Unseen Intent Generalization).

    Nguyên lý:
    - Tập các ý định I được chia thành 2 tập rời rạc hoàn toàn:
      I_seen (80%) và I_unseen (20%), với I_seen giao I_unseen = rỗng.
    - Tập huấn luyện (train_samples) CHỈ chứa các câu truy vấn thuộc I_seen.
    - Tập đánh giá zero-shot (unseen_eval_samples) CHỈ chứa các câu truy vấn thuộc I_unseen.
    - Đảm bảo tính phân tầng (stratification): Nếu stratify_by_category=True, mỗi danh mục (category)
      sẽ được giữ lại tỷ lệ ý định tương ứng, giúp bảo đảm tính đại diện của mọi lĩnh vực LaTeX.

    Tham số:
        samples: Danh sách mẫu huấn luyện độc quyền từ train.csv.
        intent_bank: Ngân hàng kịch bản intents.json (tùy chọn, dùng để tra cứu danh mục nếu cần).
        holdout_ratio: Tỷ lệ ý định giữ lại làm unseen (mặc định 0.20 = 20%).
        seed: Hạt giống ngẫu nhiên đảm bảo tính tất định.
        stratify_by_category: Có phân tầng theo nhóm category hay không (mặc định True).

    Trả về:
        Đối tượng UnseenIntentSplitResult chứa mẫu chia và danh sách ý định seen/unseen.
    """
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError(f"holdout_ratio phải nằm trong khoảng (0.0, 1.0), nhận được {holdout_ratio}.")

    rng = random.Random(seed)

    # Thu thập tất cả các ý định duy nhất có trong samples
    all_intents = sorted({s.label for s in samples})
    total_intents = len(all_intents)
    target_unseen_count = round(total_intents * holdout_ratio)

    unseen_intents_set: set[str] = set()

    if stratify_by_category:
        # Gom nhóm intent theo category
        intent2category: dict[str, str] = {}
        for s in samples:
            if s.label not in intent2category:
                intent2category[s.label] = s.category

        category2intents: dict[str, list[str]] = defaultdict(list)
        for intent in all_intents:
            cat = intent2category.get(intent, "unknown")
            category2intents[cat].append(intent)

        # Lấy mẫu theo từng category
        for cat, cat_intents in sorted(category2intents.items()):
            cat_intents_shuffled = list(cat_intents)
            rng.shuffle(cat_intents_shuffled)
            n_holdout = round(len(cat_intents_shuffled) * holdout_ratio)
            # Đảm bảo mỗi category có ít nhất 1 intent holdout nếu category đủ lớn (>= 5 intents)
            if n_holdout == 0 and len(cat_intents_shuffled) >= 5:
                n_holdout = 1
            unseen_intents_set.update(cat_intents_shuffled[:n_holdout])

        # Điều chỉnh số lượng để sát với target_unseen_count nếu cần
        current_count = len(unseen_intents_set)
        if current_count < target_unseen_count:
            remaining = [i for i in all_intents if i not in unseen_intents_set]
            rng.shuffle(remaining)
            needed = target_unseen_count - current_count
            unseen_intents_set.update(remaining[:needed])
        elif current_count > target_unseen_count:
            excess = current_count - target_unseen_count
            unseen_list = sorted(unseen_intents_set)
            rng.shuffle(unseen_list)
            for item in unseen_list[:excess]:
                unseen_intents_set.remove(item)
    else:
        # Phân chia ngẫu nhiên thuần túy không phân tầng
        shuffled_intents = list(all_intents)
        rng.shuffle(shuffled_intents)
        unseen_intents_set = set(shuffled_intents[:target_unseen_count])

    seen_intents_set = set(all_intents) - unseen_intents_set

    # Phân bổ các mẫu dữ liệu vào 2 tập
    train_seen_samples = [s for s in samples if s.label in seen_intents_set]
    unseen_eval_samples = [s for s in samples if s.label in unseen_intents_set]

    seen_intents_sorted = sorted(seen_intents_set)
    unseen_intents_sorted = sorted(unseen_intents_set)

    return UnseenIntentSplitResult(
        train_samples=train_seen_samples,
        unseen_eval_samples=unseen_eval_samples,
        seen_intents=seen_intents_sorted,
        unseen_intents=unseen_intents_sorted,
        holdout_ratio=len(unseen_intents_sorted) / total_intents if total_intents > 0 else 0.0,
    )


# =====================================================================
# 8. Lớp Dataset PyTorch cho Bi-Encoder (PyTorch Dataset Classes)
# =====================================================================


class BiEncoderPairDataset(Dataset):
    """
    Lớp Dataset PyTorch cung cấp các cặp (query, positive) cho huấn luyện tương phản In-batch (Mode A).

    Hỗ trợ:
    - Lấy mẫu biểu diễn biến thể đồng đều (Uniform Variant Sampling) động theo từng epoch.
    - Cơ chế mở rộng biệt danh (intent aliases) làm cặp positive.
    - Cung cấp chỉ số kịch bản (scenario_index) để áp dụng mặt nạ đồng nhất (Identity Masking)
      trong in_batch_contrastive_loss.
    """

    def __init__(
        self,
        samples: list[DatasetSample],
        intent_bank: dict[str, dict[str, Any]],
        include_aliases: bool = DEFAULT_INCLUDE_ALIASES,
        variant_strategy: str = "uniform",
        dynamic_sampling: bool = True,
        seed: int = DEFAULT_SEED,
        bank_path: Path | str = DEFAULT_INTENTS_PATH,
    ) -> None:
        """
        Khởi tạo BiEncoderPairDataset.

        Tham số:
            samples: Danh sách mẫu từ train.csv.
            intent_bank: Ngân hàng kịch bản.
            include_aliases: Cờ cấu hình bổ sung các bí danh intent làm cặp positive.
            variant_strategy: Chiến lược biến thể ('uniform', 'handmade', v.v.).
            dynamic_sampling: Nếu True, lấy mẫu biến thể ngẫu nhiên khi gọi __getitem__.
            seed: Hạt giống ngẫu nhiên.
            bank_path: Đường dẫn intents.json.
        """
        self.samples = samples
        self.intent_bank = intent_bank
        self.include_aliases = include_aliases
        self.variant_strategy = variant_strategy
        self.dynamic_sampling = dynamic_sampling
        self.seed = seed
        self.bank_path = bank_path
        self.rng = random.Random(seed)

        self.intent_names = sorted(intent_bank.keys())
        self.intent2idx = {name: idx for idx, name in enumerate(self.intent_names)}

        # Nếu không lấy mẫu động, tạo danh sách cặp tĩnh sẵn
        if not dynamic_sampling:
            self.static_pairs = build_training_pairs(
                samples=samples,
                intent_bank=intent_bank,
                include_aliases=include_aliases,
                variant_strategy=variant_strategy,
                rng=self.rng,
                bank_path=bank_path,
            )
        else:
            self.static_pairs = []

    def __len__(self) -> int:
        if not self.dynamic_sampling:
            return len(self.static_pairs)
        # Khi dynamic_sampling và include_aliases, tính toán số lượng mẫu tương ứng
        if not self.include_aliases:
            return len(self.samples)
        # Nếu có include_aliases tĩnh, ta dựng bảng chỉ mục ánh xạ
        total = 0
        for s in self.samples:
            total += 1
            if self.include_aliases and s.label in self.intent_bank:
                total += len(self.intent_bank[s.label].get("aliases", []))
        return total

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.dynamic_sampling:
            pair = self.static_pairs[idx]
            return {
                "query": pair.query,
                "positive": pair.positive,
                "scenario_index": pair.scenario_index,
                "intent": pair.intent,
                "variant": pair.variant,
            }

        # Lấy mẫu động (Dynamic Uniform Variant Sampling)
        if not self.include_aliases:
            sample = self.samples[idx]
            intent_name = sample.label
            intent_data = self.intent_bank[intent_name]
            intent_idx = self.intent2idx[intent_name]

            if self.variant_strategy == "uniform":
                pos_text, var_name = sample_scenario_variant(
                    intent_data, rng=self.rng, allowed_variants=SCENARIO_VARIANTS, bank_path=self.bank_path
                )
            else:
                var_name = self.variant_strategy
                pos_text = scenario_string(intent_data, variant=var_name, bank_path=self.bank_path)

            return {
                "query": sample.text,
                "positive": pos_text,
                "scenario_index": intent_idx,
                "intent": intent_name,
                "variant": var_name,
            }

        # Nếu vừa dynamic vừa include_aliases, truy xuất qua danh sách đã lập chỉ mục
        if not self.static_pairs:
            self.static_pairs = build_training_pairs(
                samples=self.samples,
                intent_bank=self.intent_bank,
                include_aliases=self.include_aliases,
                variant_strategy=self.variant_strategy,
                rng=self.rng,
                bank_path=self.bank_path,
            )
        pair = self.static_pairs[idx]
        return {
            "query": pair.query,
            "positive": pair.positive,
            "scenario_index": pair.scenario_index,
            "intent": pair.intent,
            "variant": pair.variant,
        }


class BiEncoderHardNegativeDataset(Dataset):
    """
    Lớp Dataset PyTorch cung cấp các mẫu mở rộng mẫu âm khó (Hard Negatives) cho Mode A2.
    """

    def __init__(self, mode_a2_samples: list[ModeA2Sample]) -> None:
        """
        Khởi tạo BiEncoderHardNegativeDataset.

        Tham số:
            mode_a2_samples: Danh sách các mẫu ModeA2Sample đã được khai thác mẫu âm khó.
        """
        self.samples = mode_a2_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        return {
            "query": item.query,
            "positive": item.positive,
            "hard_negatives": item.hard_negatives,
            "scenario_index": item.scenario_index,
            "hard_scenario_indices": item.hard_scenario_indices,
            "intent": item.intent,
            "variant": item.variant,
        }


class BiEncoderTripletDataset(Dataset):
    """
    Lớp Dataset PyTorch cung cấp các bộ ba (anchor, positive, negative) cho Mode B (Triplet Loss).
    """

    def __init__(self, triplet_samples: list[TripletSample]) -> None:
        """
        Khởi tạo BiEncoderTripletDataset.

        Tham số:
            triplet_samples: Danh sách các mẫu TripletSample.
        """
        self.samples = triplet_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        return {
            "anchor": item.anchor,
            "positive": item.positive,
            "negative": item.negative,
            "anchor_intent": item.anchor_intent,
            "negative_intent": item.negative_intent,
            "anchor_idx": item.anchor_idx,
            "negative_idx": item.negative_idx,
            "variant": item.variant,
        }
