"""
Bộ kiểm thử đơn vị toàn diện cho module truy hồi ngữ nghĩa và lập chỉ mục động (tests/test_retrieval.py).

Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Kiến trúc xử lý:
    text -> preprocess (NFC, chuẩn hóa ký hiệu toán học, tách từ pyvi)
         -> PhoBERT classifier: p = softmax over intents
            - if max p >= tau: intent accepted
            - else retrieval top-3 over aliases + scenario strings -> ask the user to choose
              (if best score < theta: say not understood)
- Cơ sở lý thuyết kiểm thử:
    1. Sentence-BERT Cosine Similarity (Reimers & Gurevych, EMNLP 2019):
       Vector nhúng chuẩn hóa L2 ||u||_2 = 1, sim(u, v) = u . v.
       Kiểm chứng tính thứ tự giảm dần tuyệt đối của các ứng viên theo độ tương đồng cosine.
    2. Tổng hợp điểm theo intent (Per-intent score aggregation):
       - Chiến lược 'max': score(I) = max_{r in S_I} sim(r, q)
       - Chiến lược 'mean_top2': score(I) = (s_(1) + s_(2)) / 2 (với |S_I| >= 2) hoặc s_(1) (|S_I| == 1).
       Kiểm chứng hiện tượng đảo thứ hạng (rank reversal) khi chuyển đổi chiến lược.
    3. Định dạng đầu ra top_k và cấu trúc kết quả:
       Kiểm chứng RetrievalResult, RetrievedCandidate, cờ understood dựa trên ngưỡng theta.
    4. Tính toán độ chênh lệch nhập nhằng (Ambiguity Margin):
       Delta = score_top1 - score_top2, kiểm chứng phương thức is_ambiguous().
    5. Bộ nhớ đệm chỉ mục (Index Caching):
       Kiểm chứng lưu đệm .npz với mã băm SHA-256 tất định và không mã hóa lại khi cache hợp lệ.
    6. Lập chỉ mục động (Dynamic Indexing):
       Kiểm chứng add_intent chỉ mã hóa hàng mới, nối ma trận vector (vstack), không tính lại hàng cũ.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.config import (
    DEFAULT_AMBIGUITY_MARGIN_THRESHOLD,
    DEFAULT_RETRIEVAL_TOP_K,
    INDEX_CACHE_PREFIX,
    THETA_RETRIEVAL_THRESHOLD,
)
from core.preprocess import prepare_for_embedding, preprocess_fingerprint
from core.retrieval import (
    AggregationComparison,
    DuplicateIntentError,
    IndexRow,
    IntentNotFoundError,
    MockSentenceTransformer,
    RetrievalResult,
    RetrievedCandidate,
    Retriever,
    compute_index_hash,
)

# =====================================================================
# 1. Các lớp Fake / Mock Encoder phục vụ kiểm thử
# =====================================================================


class ControlledFakeEncoder:
    """
    Bộ mã hóa giả lập có kiểm soát (Controlled Fake Encoder).

    Cho phép gán trước các vector nhúng chính xác cho từng chuỗi văn bản cụ thể,
    giúp xác minh tính toán tích vô hướng cosine similarity và thứ tự xếp hạng
    mà không gặp bất kỳ sai số số học ngẫu nhiên nào.
    Đồng thời ghi nhận lịch sử các lượt gọi encode() để kiểm chứng cơ chế caching và dynamic indexing.
    """

    def __init__(
        self,
        embedding_dim: int = 4,
        model_name: str = "controlled-fake-encoder",
        default_vector: Sequence[float] | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.call_count: int = 0
        self.encoded_history: list[list[str]] = []
        self._text_to_vector: dict[str, np.ndarray] = {}

        if default_vector is not None:
            v = np.array(default_vector, dtype=np.float32)
            norm = np.linalg.norm(v)
            self._default_vector = (v / norm) if norm > 0 else v
        else:
            v = np.ones(embedding_dim, dtype=np.float32)
            self._default_vector = v / np.linalg.norm(v)

    def register_vector(self, text: str, vector: Sequence[float]) -> None:
        """Đăng ký vector chuẩn hóa cho một chuỗi văn bản cụ thể."""
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        normalized = (v / norm) if norm > 0 else v
        self._text_to_vector[text] = normalized

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim

    def encode(
        self,
        sentences: Sequence[str] | str,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        self.call_count += 1
        if isinstance(sentences, str):
            input_texts = [sentences]
        else:
            input_texts = list(sentences)

        self.encoded_history.append(list(input_texts))

        if not input_texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        vectors: list[np.ndarray] = []
        for text in input_texts:
            # Tra cứu vector đã đăng ký hoặc dùng vector mặc định
            if text in self._text_to_vector:
                vec = self._text_to_vector[text].copy()
            else:
                # Tạo vector ngẫu nhiên tất định dựa trên mã băm chuỗi
                val = sum(ord(c) for c in text) % 100
                vec = np.zeros(self.embedding_dim, dtype=np.float32)
                vec[0] = 1.0
                vec[1] = float(val) / 100.0

            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)


# =====================================================================
# 2. Fixtures dữ liệu kịch bản mẫu
# =====================================================================


@pytest.fixture
def sample_intents_data() -> list[dict[str, Any]]:
    """Tạo ngân hàng kịch bản mẫu phục vụ kiểm thử đơn vị."""
    return [
        {
            "intent": "MATH_PHAN_SO",
            "describe": "Chèn phân số toán học",
            "category": "math_expression",
            "aliases": ["phân số", "tạo phân thức", "a trên b"],
            "slots": [
                {"name": "tu_so", "describe": "Tử số", "required": True},
                {"name": "mau_so", "describe": "Mẫu số", "required": True},
            ],
            "template": {"latex": r"\frac{<<tu_so>>}{<<mau_so>>}", "python": "", "cpp": ""},
        },
        {
            "intent": "LATEX_TAO_BANG",
            "describe": "Tạo bảng tabular trong văn bản",
            "category": "tables",
            "aliases": ["tạo bảng", "chèn bảng biểu"],
            "slots": [
                {"name": "so_cot", "describe": "Số lượng cột", "required": True},
            ],
            "template": {"latex": r"\begin{tabular}{cc}\end{tabular}", "python": "", "cpp": ""},
        },
        {
            "intent": "LATEX_CHEN_ANH",
            "describe": "Chèn hình ảnh minh họa",
            "category": "figures",
            "aliases": ["chèn ảnh", "thêm hình ảnh", "chèn đồ thị"],
            "slots": [
                {"name": "duong_dan", "describe": "Đường dẫn tệp ảnh", "required": True},
            ],
            "template": {"latex": r"\includegraphics{<<duong_dan>>}", "python": "", "cpp": ""},
        },
    ]


# =====================================================================
# 3. Kiểm thử IndexRow và Hàm băm compute_index_hash
# =====================================================================


class TestIndexHashAndRow:
    """Kiểm tra cấu trúc IndexRow và tính tất định của hàm băm chỉ mục SHA-256."""

    def test_compute_index_hash_determinism(self) -> None:
        """Kiểm tra tính toán hash tất định cho cùng một đầu vào."""
        rows = [
            IndexRow(
                intent="MATH_PHAN_SO",
                text="phân_số",
                raw_text="phân số",
                source_type="alias",
            )
        ]
        fp = preprocess_fingerprint()
        h1 = compute_index_hash("model-a", rows, fp)
        h2 = compute_index_hash("model-a", rows, fp)

        assert h1 == h2
        assert len(h1) == 64
        # Kiểm tra tính nhạy: thay đổi model_name dẫn tới hash khác
        h3 = compute_index_hash("model-b", rows, fp)
        assert h1 != h3

    def test_index_row_immutability(self) -> None:
        """Kiểm tra IndexRow là dataclass đóng băng (frozen=True)."""
        row = IndexRow(
            intent="MATH_TEST",
            text="test",
            raw_text="test",
            source_type="alias",
        )
        assert row.intent == "MATH_TEST"
        with pytest.raises(FrozenInstanceError):
            row.intent = "MODIFIED"  # type: ignore[misc]


# =====================================================================
# 4. Kiểm thử MockSentenceTransformer và ControlledFakeEncoder
# =====================================================================


class TestEncoders:
    """Kiểm thử tính năng và tính tất định của các bộ mã hóa giả lập."""

    def test_mock_sentence_transformer_determinism(self) -> None:
        """Kiểm tra MockSentenceTransformer sinh vector tất định cho cùng một chuỗi."""
        mock_model = MockSentenceTransformer(embedding_dim=32, model_name="test-mock")
        assert mock_model.get_sentence_embedding_dimension() == 32

        vec1 = mock_model.encode("phân số toán học")
        vec2 = mock_model.encode("phân số toán học")
        np.testing.assert_allclose(vec1, vec2, rtol=1e-5, atol=1e-6)

        # Kiểm tra chuẩn hóa L2 ||v||_2 = 1.0
        norm = np.linalg.norm(vec1[0])
        assert abs(norm - 1.0) < 1e-5

    def test_mock_sentence_transformer_different_texts(self) -> None:
        """Kiểm tra các câu khác nhau sinh vector nhúng khác biệt."""
        mock_model = MockSentenceTransformer(embedding_dim=16)
        vec_a = mock_model.encode("tạo bảng")
        vec_b = mock_model.encode("chèn hình ảnh")
        # Không trùng khớp hoàn toàn
        assert not np.allclose(vec_a, vec_b)

    def test_mock_sentence_transformer_empty_input(self) -> None:
        """Kiểm tra xử lý danh sách đầu vào rỗng."""
        mock_model = MockSentenceTransformer(embedding_dim=16)
        res = mock_model.encode([])
        assert res.shape == (0, 16)

    def test_controlled_fake_encoder_vector_registration(self) -> None:
        """Kiểm tra ControlledFakeEncoder trả về chính xác vector đã đăng ký."""
        encoder = ControlledFakeEncoder(embedding_dim=3)
        encoder.register_vector("văn bản A", [1.0, 0.0, 0.0])
        encoder.register_vector("văn bản B", [0.0, 1.0, 0.0])

        embs = encoder.encode(["văn bản A", "văn bản B"])
        np.testing.assert_allclose(embs[0], [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(embs[1], [0.0, 1.0, 0.0], atol=1e-5)
        assert encoder.call_count == 1


# =====================================================================
# 5. Kiểm thử thứ tự tương đồng Cosine (Cosine Ordering Verification)
# =====================================================================


class TestCosineOrdering:
    """
    Kiểm tra cơ chế sắp xếp giảm dần theo tương đồng Cosine:
    sim(u, v) = u . v (khi ||u|| = ||v|| = 1).
    """

    def test_strict_cosine_ordering(self, tmp_path: Path) -> None:
        """
        Kiểm tra các ứng viên kịch bản được sắp xếp giảm dần theo điểm tương đồng cosine.
        Thiết lập hình học vector 3 chiều:
        - Query q = [1.0, 0.0, 0.0]
        - Intent 1 (MATH_PHAN_SO): sim = 0.95
        - Intent 2 (LATEX_TAO_BANG): sim = 0.70
        - Intent 3 (LATEX_CHEN_ANH): sim = 0.20
        Kết quả mong đợi: MATH_PHAN_SO > LATEX_TAO_BANG > LATEX_CHEN_ANH.
        """
        encoder = ControlledFakeEncoder(embedding_dim=3)
        # Chuẩn bị trước vector cho các chuỗi sau khi qua prepare_for_embedding
        prep_query = prepare_for_embedding("truy vấn kiểm thử", model_name=encoder.model_name)
        encoder.register_vector(prep_query, [1.0, 0.0, 0.0])

        intents = [
            {
                "intent": "MATH_PHAN_SO",
                "describe": "phân số",
                "category": "math",
                "aliases": ["alias_phan_so"],
            },
            {
                "intent": "LATEX_TAO_BANG",
                "describe": "bảng biểu",
                "category": "tables",
                "aliases": ["alias_tao_bang"],
            },
            {
                "intent": "LATEX_CHEN_ANH",
                "describe": "chèn ảnh",
                "category": "figures",
                "aliases": ["alias_chen_anh"],
            },
        ]

        # Khởi tạo Retriever với encoder kiểm soát
        retriever = Retriever(
            model=encoder,
            intents=intents,
            cache_dir=tmp_path,
            use_cache=False,
        )

        # Đăng ký vector cho từng hàng trong chỉ mục
        for row in retriever.rows:
            if row.intent == "MATH_PHAN_SO":
                # sim([1,0,0], [0.95, sqrt(1-0.95^2), 0]) = 0.95
                y = float(np.sqrt(1.0 - 0.95**2))
                encoder.register_vector(row.text, [0.95, y, 0.0])
            elif row.intent == "LATEX_TAO_BANG":
                # sim([1,0,0], [0.70, sqrt(1-0.70^2), 0]) = 0.70
                y = float(np.sqrt(1.0 - 0.70**2))
                encoder.register_vector(row.text, [0.70, y, 0.0])
            elif row.intent == "LATEX_CHEN_ANH":
                # sim([1,0,0], [0.20, sqrt(1-0.20^2), 0]) = 0.20
                y = float(np.sqrt(1.0 - 0.20**2))
                encoder.register_vector(row.text, [0.20, y, 0.0])

        # Cập nhật ma trận vector nhúng tương ứng trong retriever
        retriever._embeddings = retriever._encode_texts([r.text for r in retriever.rows])

        result = retriever.retrieve("truy vấn kiểm thử", top_k=3)
        assert len(result.candidates) == 3

        # Kiểm tra thứ tự nghiêm ngặt giảm dần
        scores = [c.score for c in result.candidates]
        assert scores[0] > scores[1] > scores[2]
        assert abs(scores[0] - 0.95) < 1e-4
        assert abs(scores[1] - 0.70) < 1e-4
        assert abs(scores[2] - 0.20) < 1e-4

        assert result.candidates[0].intent == "MATH_PHAN_SO"
        assert result.candidates[1].intent == "LATEX_TAO_BANG"
        assert result.candidates[2].intent == "LATEX_CHEN_ANH"

    def test_orthogonal_and_negative_similarity(self, tmp_path: Path) -> None:
        """Kiểm tra trường hợp vector trực giao (sim = 0) và ngược hướng (sim = -1)."""
        encoder = ControlledFakeEncoder(embedding_dim=2)
        prep_q = prepare_for_embedding("câu hỏi", model_name=encoder.model_name)
        encoder.register_vector(prep_q, [1.0, 0.0])

        intents = [
            {"intent": "INTENT_ORTHOGONAL", "describe": "trực giao", "aliases": []},
            {"intent": "INTENT_OPPOSITE", "describe": "đối lập", "aliases": []},
        ]
        retriever = Retriever(
            model=encoder,
            intents=intents,
            cache_dir=tmp_path,
            use_cache=False,
            threshold_theta=0.3,
        )

        for row in retriever.rows:
            if row.intent == "INTENT_ORTHOGONAL":
                encoder.register_vector(row.text, [0.0, 1.0])
            else:
                encoder.register_vector(row.text, [-1.0, 0.0])

        retriever._embeddings = retriever._encode_texts([r.text for r in retriever.rows])
        res = retriever.retrieve("câu hỏi", threshold_theta=0.3)

        assert len(res.candidates) == 2
        assert abs(res.candidates[0].score - 0.0) < 1e-4
        assert abs(res.candidates[1].score - (-1.0)) < 1e-4
        # Vì điểm cao nhất (0.0) < theta (0.3), hệ thống không hiểu câu lệnh
        assert res.understood is False


# =====================================================================
# 6. Kiểm thử tổng hợp điểm theo từng Intent (Per-Intent Score Aggregation)
# =====================================================================


class TestScoreAggregation:
    """
    Kiểm thử cơ chế tổng hợp điểm số theo từng kịch bản:
    1. 'max': Lấy điểm số cao nhất trong các hàng biểu diễn.
    2. 'mean_top2': Lấy trung bình cộng của 2 hàng cao nhất.
    """

    def test_max_aggregation_formula(self, tmp_path: Path) -> None:
        """
        Kiểm tra chiến lược 'max' chọn chính xác điểm cực đại trong các hàng của intent:
        score(I) = max(s_1, s_2, ..., s_m).
        """
        encoder = ControlledFakeEncoder(embedding_dim=4)
        prep_q = prepare_for_embedding("tìm kiếm phân số", model_name=encoder.model_name)
        encoder.register_vector(prep_q, [1.0, 0.0, 0.0, 0.0])

        intent = {
            "intent": "MATH_PHAN_SO",
            "describe": "phân số",
            "aliases": ["alias_1", "alias_2"],
        }
        retriever = Retriever(model=encoder, intents=[intent], cache_dir=tmp_path, use_cache=False)

        # Giả lập điểm số các hàng: row 0: 0.90, row 1: 0.60, row 2: 0.30
        sim_scores = [0.90, 0.60, 0.30]
        for row, target_sim in zip(retriever.rows, sim_scores):
            y = float(np.sqrt(1.0 - target_sim**2))
            encoder.register_vector(row.text, [target_sim, y, 0.0, 0.0])

        retriever._embeddings = retriever._encode_texts([r.text for r in retriever.rows])

        res = retriever.retrieve("tìm kiếm phân số", aggregation="max")
        candidate = res.candidates[0]
        assert abs(candidate.score - 0.90) < 1e-4
        assert abs(candidate.max_score - 0.90) < 1e-4
        assert abs(candidate.mean_top2_score - (0.90 + 0.60) / 2.0) < 1e-4
        assert candidate.all_row_scores == pytest.approx([0.90, 0.60, 0.30], abs=1e-4)

    def test_mean_top2_aggregation_formula(self, tmp_path: Path) -> None:
        """
        Kiểm tra chiến lược 'mean_top2' tính trung bình cộng chính xác 2 hàng cao nhất:
        score(I) = (s_(1) + s_(2)) / 2.
        """
        encoder = ControlledFakeEncoder(embedding_dim=4)
        prep_q = prepare_for_embedding("tìm kiếm bảng", model_name=encoder.model_name)
        encoder.register_vector(prep_q, [1.0, 0.0, 0.0, 0.0])

        intent = {
            "intent": "LATEX_TAO_BANG",
            "describe": "bảng biểu",
            "aliases": ["alias_1", "alias_2", "alias_3"],
        }
        retriever = Retriever(model=encoder, intents=[intent], cache_dir=tmp_path, use_cache=False)

        sim_scores = [0.80, 0.60, 0.40, 0.20]
        for row, target_sim in zip(retriever.rows, sim_scores):
            y = float(np.sqrt(1.0 - target_sim**2))
            encoder.register_vector(row.text, [target_sim, y, 0.0, 0.0])

        retriever._embeddings = retriever._encode_texts([r.text for r in retriever.rows])

        res = retriever.retrieve("tìm kiếm bảng", aggregation="mean_top2")
        candidate = res.candidates[0]
        expected_mean = (0.80 + 0.60) / 2.0  # 0.70
        assert abs(candidate.score - expected_mean) < 1e-4
        assert abs(candidate.max_score - 0.80) < 1e-4

    def test_single_row_intent_mean_top2(self, tmp_path: Path) -> None:
        """
        Kiểm tra intent chỉ có đúng 1 hàng (không có aliases):
        mean_top2 rơi vào trường hợp đặc biệt lấy chính điểm số của hàng duy nhất đó.
        """
        encoder = ControlledFakeEncoder(embedding_dim=2)
        prep_q = prepare_for_embedding("chỉ một hàng", model_name=encoder.model_name)
        encoder.register_vector(prep_q, [1.0, 0.0])

        intent = {"intent": "SINGLE_ROW_INTENT", "describe": "mô tả", "aliases": []}
        retriever = Retriever(model=encoder, intents=[intent], cache_dir=tmp_path, use_cache=False)

        assert retriever.num_rows == 1
        encoder.register_vector(retriever.rows[0].text, [0.85, float(np.sqrt(1.0 - 0.85**2))])
        retriever._embeddings = retriever._encode_texts([r.text for r in retriever.rows])

        res = retriever.retrieve("chỉ một hàng", aggregation="mean_top2")
        assert abs(res.candidates[0].score - 0.85) < 1e-4
        assert abs(res.candidates[0].mean_top2_score - 0.85) < 1e-4

    def test_rank_reversal_between_max_and_mean_top2(self, tmp_path: Path) -> None:
        """
        Kiểm chứng hiện tượng đảo thứ hạng (Rank Reversal) giữa hai chiến lược tổng hợp:
        - Intent A (Spiky): 1 hàng rất cao (0.95), các hàng còn lại rất thấp (0.45).
          -> max(A) = 0.95; mean_top2(A) = (0.95 + 0.45) / 2 = 0.70.
        - Intent B (Consistent): 2 hàng đều có điểm khá cao (0.85, 0.85).
          -> max(B) = 0.85; mean_top2(B) = (0.85 + 0.85) / 2 = 0.85.

        Dưới chiến lược 'max': A (0.95) > B (0.85).
        Dưới chiến lược 'mean_top2': B (0.85) > A (0.70).
        """
        encoder = ControlledFakeEncoder(embedding_dim=4)
        prep_q = prepare_for_embedding("câu hỏi đảo hạng", model_name=encoder.model_name)
        encoder.register_vector(prep_q, [1.0, 0.0, 0.0, 0.0])

        intents = [
            {"intent": "INTENT_A", "describe": "A", "aliases": ["alias_A"]},
            {"intent": "INTENT_B", "describe": "B", "aliases": ["alias_B"]},
        ]
        retriever = Retriever(model=encoder, intents=intents, cache_dir=tmp_path, use_cache=False)

        for row in retriever.rows:
            if row.intent == "INTENT_A":
                target = 0.95 if row.source_type == "scenario_string" else 0.45
            else:
                target = 0.85
            y = float(np.sqrt(1.0 - target**2))
            encoder.register_vector(row.text, [target, y, 0.0, 0.0])

        retriever._embeddings = retriever._encode_texts([r.text for r in retriever.rows])

        # 1. Truy hồi với aggregation='max'
        res_max = retriever.retrieve("câu hỏi đảo hạng", aggregation="max")
        assert res_max.candidates[0].intent == "INTENT_A"
        assert res_max.candidates[1].intent == "INTENT_B"
        assert res_max.candidates[0].score > res_max.candidates[1].score

        # 2. Truy hồi với aggregation='mean_top2'
        res_mean = retriever.retrieve("câu hỏi đảo hạng", aggregation="mean_top2")
        assert res_mean.candidates[0].intent == "INTENT_B"
        assert res_mean.candidates[1].intent == "INTENT_A"
        assert res_mean.candidates[0].score > res_mean.candidates[1].score

    def test_compare_aggregations_method(self, tmp_path: Path) -> None:
        """Kiểm tra phương thức compare_aggregations tạo bảng đối chuẩn chênh lệch điểm số."""
        encoder = ControlledFakeEncoder(embedding_dim=4)
        prep_q = prepare_for_embedding("so sánh chiến lược", model_name=encoder.model_name)
        encoder.register_vector(prep_q, [1.0, 0.0, 0.0, 0.0])

        intents = [{"intent": "INTENT_1", "describe": "1", "aliases": ["al1", "al2"]}]
        retriever = Retriever(model=encoder, intents=intents, cache_dir=tmp_path, use_cache=False)

        scores = [0.90, 0.70, 0.50]
        for row, s in zip(retriever.rows, scores):
            y = float(np.sqrt(1.0 - s**2))
            encoder.register_vector(row.text, [s, y, 0.0, 0.0])
        retriever._embeddings = retriever._encode_texts([r.text for r in retriever.rows])

        comp = retriever.compare_aggregations("so sánh chiến lược")
        assert isinstance(comp, AggregationComparison)
        assert "INTENT_1" in comp.differences
        diff = comp.differences["INTENT_1"]
        assert diff["max_score"] == pytest.approx(0.90, abs=1e-4)
        assert diff["mean_top2_score"] == pytest.approx(0.80, abs=1e-4)
        assert diff["score_difference"] == pytest.approx(0.10, abs=1e-4)

    def test_invalid_aggregation_raises_error(self, tmp_path: Path) -> None:
        """Kiểm tra truyền tên phương thức tổng hợp không hợp lệ sẽ ném ValueError."""
        encoder = MockSentenceTransformer(embedding_dim=8)
        with pytest.raises(ValueError, match="không được hỗ trợ"):
            Retriever(
                model=encoder,
                intents=[],
                cache_dir=tmp_path,
                aggregation="unsupported_agg",
                use_cache=False,
            )


# =====================================================================
# 7. Kiểm thử định dạng đầu ra top_k và cấu trúc kết quả
# =====================================================================


class TestTopKOutputFormatting:
    """Kiểm tra định dạng dữ liệu đầu ra và các trường của RetrievalResult/RetrievedCandidate."""

    def test_top_k_bounding(self, sample_intents_data: list[dict[str, Any]], tmp_path: Path) -> None:
        """Kiểm tra số lượng ứng viên trả về tuân thủ tham số top_k."""
        mock_model = MockSentenceTransformer(embedding_dim=16)
        retriever = Retriever(
            model=mock_model,
            intents=sample_intents_data,
            cache_dir=tmp_path,
            top_k=DEFAULT_RETRIEVAL_TOP_K,
            use_cache=False,
        )
        assert retriever.num_intents == 3
        assert retriever.top_k == DEFAULT_RETRIEVAL_TOP_K

        # Yêu cầu top_k = 1 -> trả về đúng 1
        res1 = retriever.retrieve("tạo phân số", top_k=1)
        assert len(res1.candidates) == 1
        assert res1.top_k == 1

        # Yêu cầu top_k = 2 -> trả về đúng 2
        res2 = retriever.retrieve("tạo phân số", top_k=2)
        assert len(res2.candidates) == 2

        # Yêu cầu top_k = 10 (vượt quá số intent thực tế 3) -> trả về tối đa 3
        res3 = retriever.retrieve("tạo phân số", top_k=10)
        assert len(res3.candidates) == 3

    def test_result_fields_and_properties(
        self, sample_intents_data: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Kiểm tra đầy đủ các thuộc tính và trường dữ liệu của kết quả truy hồi."""
        mock_model = MockSentenceTransformer(embedding_dim=16)
        retriever = Retriever(
            model=mock_model,
            intents=sample_intents_data,
            cache_dir=tmp_path,
            use_cache=False,
        )

        res = retriever.retrieve("viết phân số toán", top_k=2)
        assert isinstance(res, RetrievalResult)
        assert res.query == "viết phân số toán"
        assert isinstance(res.preprocessed_query, str)
        assert isinstance(res.ambiguity_margin, float)
        assert res.margin == res.ambiguity_margin
        assert isinstance(res.understood, bool)
        assert res.aggregation == "max"

        # Kiểm tra helper properties
        assert res.top_candidate is not None
        assert res.top_candidate == res.candidates[0]
        assert res.top_intent == res.candidates[0].intent
        assert res.top_score == res.candidates[0].score

        # Kiểm tra từng trường của ứng viên RetrievedCandidate
        candidate = res.candidates[0]
        assert isinstance(candidate, RetrievedCandidate)
        assert candidate.intent in ["MATH_PHAN_SO", "LATEX_TAO_BANG", "LATEX_CHEN_ANH"]
        assert -1.0 <= candidate.score <= 1.0
        assert -1.0 <= candidate.max_score <= 1.0
        assert -1.0 <= candidate.mean_top2_score <= 1.0
        assert candidate.best_source_type in ["scenario_string", "alias"]
        assert len(candidate.all_row_scores) > 0
        # Danh sách điểm các hàng phải sắp xếp giảm dần
        assert candidate.all_row_scores == sorted(candidate.all_row_scores, reverse=True)

    def test_empty_and_whitespace_query(
        self, sample_intents_data: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Kiểm tra xử lý truy vấn rỗng hoặc chỉ chứa khoảng trắng an toàn."""
        mock_model = MockSentenceTransformer(embedding_dim=16)
        retriever = Retriever(
            model=mock_model,
            intents=sample_intents_data,
            cache_dir=tmp_path,
            use_cache=False,
        )

        res_empty = retriever.retrieve("")
        assert res_empty.candidates == []
        assert res_empty.understood is False
        assert res_empty.ambiguity_margin == 0.0
        assert res_empty.top_candidate is None
        assert res_empty.top_intent is None
        assert res_empty.top_score == 0.0

        res_spaces = retriever.retrieve("     \t\n   ")
        assert res_spaces.candidates == []
        assert res_spaces.understood is False

    def test_empty_index_retrieval(self, tmp_path: Path) -> None:
        """Kiểm tra truy hồi trên kho kịch bản rỗng hoàn toàn."""
        mock_model = MockSentenceTransformer(embedding_dim=16)
        retriever = Retriever(
            model=mock_model,
            intents=[],
            cache_dir=tmp_path,
            use_cache=False,
        )
        assert retriever.num_rows == 0
        assert retriever.num_intents == 0

        res = retriever.retrieve("truy vấn trên kho rỗng")
        assert res.candidates == []
        assert res.understood is False
        assert res.ambiguity_margin == 0.0

    def test_understood_threshold_theta(self, tmp_path: Path) -> None:
        """Kiểm tra cờ understood kích hoạt chính xác theo ngưỡng THETA_RETRIEVAL_THRESHOLD."""
        encoder = ControlledFakeEncoder(embedding_dim=2)
        prep_q = prepare_for_embedding("câu hỏi", model_name=encoder.model_name)
        encoder.register_vector(prep_q, [1.0, 0.0])

        intents = [{"intent": "TEST_INTENT", "describe": "mô tả", "aliases": []}]
        retriever = Retriever(
            model=encoder,
            intents=intents,
            cache_dir=tmp_path,
            use_cache=False,
            threshold_theta=THETA_RETRIEVAL_THRESHOLD,
        )
        assert retriever.threshold_theta == THETA_RETRIEVAL_THRESHOLD

        # 1. Điểm số = 0.35 >= 0.30 -> understood = True
        encoder.register_vector(retriever.rows[0].text, [0.35, float(np.sqrt(1.0 - 0.35**2))])
        retriever._embeddings = retriever._encode_texts([r.text for r in retriever.rows])
        res_pass = retriever.retrieve("câu hỏi", threshold_theta=0.30)
        assert res_pass.understood is True

        # 2. Điểm số = 0.25 < 0.30 -> understood = False ('say not understood')
        encoder.register_vector(retriever.rows[0].text, [0.25, float(np.sqrt(1.0 - 0.25**2))])
        retriever._embeddings = retriever._encode_texts([r.text for r in retriever.rows])
        res_fail = retriever.retrieve("câu hỏi", threshold_theta=0.30)
        assert res_fail.understood is False


# =====================================================================
# 8. Kiểm thử tính toán độ chênh lệch nhập nhằng (Ambiguity Margin)
# =====================================================================


class TestAmbiguityMargin:
    """
    Kiểm tra độ chênh lệch nhập nhằng:
    Delta = score_top1 - score_top2.
    Nếu Delta < margin_threshold -> is_ambiguous() trả về True.
    """

    def test_margin_calculation_two_candidates(self, tmp_path: Path) -> None:
        """Kiểm tra công thức Delta = score_top1 - score_top2 với 2 ứng viên."""
        encoder = ControlledFakeEncoder(embedding_dim=3)
        prep_q = prepare_for_embedding("truy vấn nhập nhằng", model_name=encoder.model_name)
        encoder.register_vector(prep_q, [1.0, 0.0, 0.0])

        intents = [
            {"intent": "INTENT_1", "describe": "1", "aliases": []},
            {"intent": "INTENT_2", "describe": "2", "aliases": []},
        ]
        retriever = Retriever(model=encoder, intents=intents, cache_dir=tmp_path, use_cache=False)

        # Thiết lập điểm top1 = 0.85, top2 = 0.82 -> chênh lệch Delta = 0.03
        for row in retriever.rows:
            target = 0.85 if row.intent == "INTENT_1" else 0.82
            y = float(np.sqrt(1.0 - target**2))
            encoder.register_vector(row.text, [target, y, 0.0])

        retriever._embeddings = retriever._encode_texts([r.text for r in retriever.rows])
        res = retriever.retrieve("truy vấn nhập nhằng", top_k=2)

        expected_margin = 0.85 - 0.82  # 0.03
        assert abs(res.ambiguity_margin - expected_margin) < 1e-4

        # Với ngưỡng mặc định 0.10: 0.03 < 0.10 -> Nhập nhằng (is_ambiguous = True)
        assert res.is_ambiguous(threshold=DEFAULT_AMBIGUITY_MARGIN_THRESHOLD) is True
        # Với ngưỡng 0.02: 0.03 >= 0.02 -> Không nhập nhằng (is_ambiguous = False)
        assert res.is_ambiguous(threshold=0.02) is False

    def test_margin_single_candidate(self, tmp_path: Path) -> None:
        """Khi chỉ có 1 ứng viên, margin bằng chính score của ứng viên đó và không nhập nhằng."""
        encoder = ControlledFakeEncoder(embedding_dim=2)
        prep_q = prepare_for_embedding("truy vấn đơn", model_name=encoder.model_name)
        encoder.register_vector(prep_q, [1.0, 0.0])

        intents = [{"intent": "SOLO_INTENT", "describe": "đơn lẻ", "aliases": []}]
        retriever = Retriever(model=encoder, intents=intents, cache_dir=tmp_path, use_cache=False)

        encoder.register_vector(retriever.rows[0].text, [0.75, float(np.sqrt(1.0 - 0.75**2))])
        retriever._embeddings = retriever._encode_texts([r.text for r in retriever.rows])

        res = retriever.retrieve("truy vấn đơn")
        assert abs(res.ambiguity_margin - 0.75) < 1e-4
        # Vì chỉ có 1 ứng viên nên không thể xảy ra nhập nhằng giữa các intent
        assert res.is_ambiguous(threshold=0.9) is False


# =====================================================================
# 9. Kiểm thử bộ nhớ đệm chỉ mục (Index Caching Verification)
# =====================================================================


class TestIndexCaching:
    """
    Kiểm tra cơ chế lưu và nạp bộ nhớ đệm chỉ mục:
    - Lưu tệp nén models/index_<hash>.npz.
    - Không mã hóa lại mô hình khi mã băm và vân tay tiền xử lý khớp.
    - Tự động mã hóa lại khi bộ nhớ đệm bị sai lệch hoặc hỏng.
    """

    def test_index_cache_save_and_reload(
        self, sample_intents_data: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """
        Kiểm tra quy trình tạo cache lần đầu và tải tức thì ở lần khởi tạo thứ hai.
        Xác minh encoder.encode() chỉ được gọi một lần duy nhất cho toàn bộ kho kịch bản.
        """
        encoder = ControlledFakeEncoder(embedding_dim=8, model_name="cached-encoder")

        # 1. Khởi tạo Retriever lần thứ nhất -> Tính toán và ghi cache
        retriever_1 = Retriever(
            model=encoder,
            intents=sample_intents_data,
            cache_dir=tmp_path,
            use_cache=True,
        )
        assert retriever_1.cache_path is not None
        assert retriever_1.cache_path.exists()
        assert retriever_1.cache_path.name.startswith(INDEX_CACHE_PREFIX)
        # Số lần gọi encode() lần đầu: 1 lần cho danh sách các hàng
        initial_call_count = encoder.call_count
        assert initial_call_count >= 1

        # 2. Khởi tạo Retriever lần thứ hai với cùng tham số -> Nạp từ cache không encode lại
        retriever_2 = Retriever(
            model=encoder,
            intents=sample_intents_data,
            cache_dir=tmp_path,
            use_cache=True,
        )

        # Số lần gọi encode() không tăng lên vì đã đọc từ tệp .npz
        assert encoder.call_count == initial_call_count
        assert retriever_2.num_rows == retriever_1.num_rows
        assert retriever_2.num_intents == retriever_1.num_intents
        np.testing.assert_allclose(retriever_2.embeddings, retriever_1.embeddings, atol=1e-6)

        # Kiểm tra nội dung các hàng khớp hoàn toàn
        for r1, r2 in zip(retriever_1.rows, retriever_2.rows):
            assert r1.intent == r2.intent
            assert r1.text == r2.text
            assert r1.source_type == r2.source_type

    def test_cache_corrupted_fallback(
        self, sample_intents_data: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Kiểm tra khi tệp cache bị hỏng hoặc metadata sai, hệ thống tự động tính toán lại an toàn."""
        encoder = ControlledFakeEncoder(embedding_dim=8, model_name="cached-encoder-2")

        # Khởi tạo lần 1 để tạo tệp cache
        retriever = Retriever(
            model=encoder,
            intents=sample_intents_data,
            cache_dir=tmp_path,
            use_cache=True,
        )
        cache_file = retriever.cache_path
        assert cache_file is not None and cache_file.exists()

        # Ghi đè tệp cache bằng dữ liệu rác không hợp lệ
        with open(cache_file, "wb") as f:
            f.write(b"corrupted binary content")

        call_count_before = encoder.call_count
        # Khởi tạo lại -> Không sập mà tự tính toán lại
        retriever_rebuilt = Retriever(
            model=encoder,
            intents=sample_intents_data,
            cache_dir=tmp_path,
            use_cache=True,
        )
        assert encoder.call_count > call_count_before
        assert retriever_rebuilt.num_rows == retriever.num_rows

    def test_save_index_explicit_path(
        self, sample_intents_data: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Kiểm tra phương thức save_index lưu vào đường dẫn chỉ định rõ ràng."""
        mock_model = MockSentenceTransformer(embedding_dim=16)
        retriever = Retriever(
            model=mock_model,
            intents=sample_intents_data,
            cache_dir=tmp_path,
            use_cache=False,
        )

        custom_path = tmp_path / "custom_dir" / "saved_index.npz"
        saved = retriever.save_index(custom_path)
        assert saved == custom_path
        assert custom_path.exists()

        # Kiểm tra nội dung tệp .npz
        with np.load(custom_path, allow_pickle=True) as data:
            assert "embeddings" in data
            assert "intents" in data
            assert "metadata" in data


# =====================================================================
# 10. Kiểm thử lập chỉ mục động (Dynamic Indexing & Zero-Shot Append)
# =====================================================================


class TestDynamicIndexing:
    """
    Kiểm tra cơ chế thêm/xóa ý định động trong bộ nhớ RAM không cần huấn luyện lại:
    - add_intent: Chỉ mã hóa các hàng của intent mới, giữ nguyên ma trận hàng cũ.
    - remove_intent: Loại bỏ các hàng và cập nhật ma trận vector.
    - Xử lý trùng lặp và ghi đè.
    """

    def test_add_intent_does_not_recompute_existing_rows(self, tmp_path: Path) -> None:
        """
        Kiểm chứng tính tối ưu của add_intent:
        Chỉ mã hóa các hàng của ý định mới được thêm vào, KHÔNG tính toán lại bất kỳ hàng cũ nào.
        """
        encoder = ControlledFakeEncoder(embedding_dim=4, model_name="dynamic-encoder")

        initial_intents = [
            {
                "intent": "MATH_PHAN_SO",
                "describe": "phân số",
                "category": "math",
                "aliases": ["alias_1", "alias_2"],
            }
        ]
        retriever = Retriever(model=encoder, intents=initial_intents, cache_dir=tmp_path, use_cache=False)
        initial_rows_count = retriever.num_rows
        initial_intents_count = retriever.num_intents
        initial_embeddings = retriever.embeddings.copy()
        calls_before = encoder.call_count

        # Định nghĩa intent mới cần bổ sung động
        new_intent = {
            "intent": "LATEX_TAO_BANG",
            "describe": "bảng biểu",
            "category": "tables",
            "aliases": ["alias_bang_1"],
        }

        # Gọi add_intent
        added_count = retriever.add_intent(new_intent, overwrite=False)
        assert added_count > 0
        assert retriever.num_intents == initial_intents_count + 1
        assert retriever.num_rows == initial_rows_count + added_count

        # Xác minh encoder chỉ được gọi thêm đúng 1 lần cho các hàng mới
        assert encoder.call_count == calls_before + 1
        last_encoded_batch = encoder.encoded_history[-1]
        assert len(last_encoded_batch) == added_count

        # Xác minh ma trận vector ban đầu được giữ nguyên ở các hàng đầu (không bị thay đổi)
        np.testing.assert_allclose(
            retriever.embeddings[:initial_rows_count],
            initial_embeddings,
            atol=1e-6,
        )

        # Xác minh intent mới có thể truy hồi được ngay lập tức
        assert "LATEX_TAO_BANG" in retriever
        res = retriever.retrieve("bảng biểu")
        assert any(c.intent == "LATEX_TAO_BANG" for c in res.candidates)

    def test_add_duplicate_intent_raises_error(self, tmp_path: Path) -> None:
        """Kiểm tra thêm intent trùng lặp khi overwrite=False sẽ ném ngoại lệ DuplicateIntentError."""
        encoder = MockSentenceTransformer(embedding_dim=8)
        intents = [{"intent": "EXISTING_INTENT", "describe": "hiện có", "aliases": []}]
        retriever = Retriever(model=encoder, intents=intents, cache_dir=tmp_path, use_cache=False)

        with pytest.raises(DuplicateIntentError, match="đã tồn tại trong chỉ mục"):
            retriever.add_intent(
                {"intent": "EXISTING_INTENT", "describe": "trùng lặp"},
                overwrite=False,
            )

    def test_add_duplicate_intent_with_overwrite(self, tmp_path: Path) -> None:
        """Kiểm tra thêm intent trùng lặp khi overwrite=True sẽ thay thế thành công."""
        encoder = MockSentenceTransformer(embedding_dim=8)
        intents = [{"intent": "REPLACE_INTENT", "describe": "cũ", "aliases": ["alias_cu"]}]
        retriever = Retriever(model=encoder, intents=intents, cache_dir=tmp_path, use_cache=False)
        assert retriever.num_intents == 1

        updated_intent = {
            "intent": "REPLACE_INTENT",
            "describe": "mới cập nhật",
            "aliases": ["alias_moi_1", "alias_moi_2"],
        }
        added = retriever.add_intent(updated_intent, overwrite=True)
        assert added > 0
        assert retriever.num_intents == 1
        # Hàng mới đã thay thế hàng cũ
        assert any("alias_moi_1" in r.raw_text for r in retriever.rows)
        assert not any("alias_cu" in r.raw_text for r in retriever.rows)

    def test_remove_intent_success_and_missing(self, tmp_path: Path) -> None:
        """Kiểm tra loại bỏ ý định khỏi chỉ mục và ném ngoại lệ nếu không tồn tại."""
        encoder = MockSentenceTransformer(embedding_dim=8)
        intents = [
            {"intent": "INTENT_KEEP", "describe": "giữ lại", "aliases": []},
            {"intent": "INTENT_DROP", "describe": "loại bỏ", "aliases": ["xóa"]},
        ]
        retriever = Retriever(model=encoder, intents=intents, cache_dir=tmp_path, use_cache=False)
        assert retriever.num_intents == 2

        # 1. Xóa intent tồn tại
        success = retriever.remove_intent("INTENT_DROP")
        assert success is True
        assert retriever.num_intents == 1
        assert "INTENT_DROP" not in retriever
        assert "INTENT_KEEP" in retriever
        assert len(retriever.rows) == retriever.embeddings.shape[0]

        # 2. Xóa intent không tồn tại -> ném IntentNotFoundError
        with pytest.raises(IntentNotFoundError, match="không tồn tại trong chỉ mục"):
            retriever.remove_intent("NON_EXISTING_INTENT")


# =====================================================================
# 11. Kiểm thử Monkeypatching và tích hợp không cần mạng ngoài
# =====================================================================


class TestMonkeypatchedSentenceTransformer:
    """
    Kiểm thử cơ chế monkeypatching lớp SentenceTransformer thực tế,
    đảm bảo Retriever có thể khởi tạo trực tiếp từ định danh mô hình chuỗi
    mà không cần tải bất kỳ trọng số nào qua Internet.
    """

    def test_monkeypatched_encoder_initialization(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """
        Monkeypatch SentenceTransformer bằng MockSentenceTransformer để kiểm tra
        Retriever(model='bkai-foundation-models/vietnamese-bi-encoder') chạy hoàn toàn offline.
        """
        import core.retrieval as ret_module

        # Mock constructor SentenceTransformer
        def fake_st_init(model_name: str, device: str = "cpu", **kwargs: Any) -> Any:
            return MockSentenceTransformer(embedding_dim=32, model_name=str(model_name))

        monkeypatch.setattr(ret_module, "SentenceTransformer", fake_st_init, raising=False)
        monkeypatch.setattr(ret_module, "_SENTENCE_TRANSFORMERS_AVAILABLE", True)

        intents = [
            {"intent": "OFFLINE_INTENT_1", "describe": "offline 1", "aliases": ["off1"]},
            {"intent": "OFFLINE_INTENT_2", "describe": "offline 2", "aliases": ["off2"]},
        ]

        # Khởi tạo với chuỗi định danh mô hình mà không cần truyền model object
        retriever = Retriever(
            model="bkai-foundation-models/vietnamese-bi-encoder",
            intents=intents,
            cache_dir=tmp_path,
            use_cache=False,
        )

        assert retriever.embedding_dim == 32
        assert retriever.num_intents == 2
        res = retriever.retrieve("kiểm thử offline hoàn toàn")
        assert isinstance(res, RetrievalResult)
        assert len(res.candidates) == 2
