"""
Module truy hồi ngữ nghĩa ngữ cảnh (SBERT Semantic Retriever with Dynamic Indexing and Caching).

Tệp tin: core/retrieval.py
Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Kiến trúc xử lý:
    text -> preprocess (NFC, chuẩn hóa ký hiệu toán học, tách từ pyvi cho PhoBERT)
         -> PhoBERT classifier: p = softmax over intents
            - if max p >= tau: intent accepted
            - else retrieval top-3 over aliases + scenario strings -> ask the user to choose (if best score < theta: say not understood)
- Cơ sở lý thuyết và công thức tham chiếu:
    1. Sentence-BERT (Reimers & Gurevych, 2019 - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019):
       Biểu diễn ngữ nghĩa câu thành vector nhúng mật độ cao e in R^d.
       Với vector đã chuẩn hóa L2 (||u||_2 = 1, ||v||_2 = 1):
           sim(u, v) = (u . v) / (||u||_2 * ||v||_2) = u . v (Cosine Similarity)
    2. Retrieve-and-Fill (RAF) đơn giản hóa (Shrivastava et al., EACL 2022 - arXiv:2202.00901):
       Lập chỉ mục từng ý định (intent) thông qua các chuỗi biểu diễn:
       - Biến thể kịch bản chuẩn 'handmade': scenario_string(intent, 'handmade')
       - Danh sách biệt danh ngữ nghĩa (aliases) của kịch bản
       Toàn bộ chuỗi văn bản được tiền xử lý nhất quán qua core.preprocess.prepare_for_embedding.
    3. Bộ nhớ đệm chỉ mục đã biên dịch (Precomputed Index Caching):
       Lưu trữ ma trận vector nhúng và siêu dữ liệu tại models/index_<hash>.npz.
       Mã băm SHA-256 được tính toán tất định dựa trên:
           hash = SHA256(model_name || preprocessing_fingerprint || rows)
       giúp tải chỉ mục tức thì (O(1) I/O) mà không cần tính toán lại mô hình nặng.
    4. Tổng hợp điểm số theo từng ý định (Per-intent score aggregation):
       Mỗi intent có nhiều hàng biểu diễn {r_1, r_2, ..., r_m} với các điểm số {s_1, s_2, ..., s_m}.
       Sắp xếp điểm giảm dần: s_(1) >= s_(2) >= ...
       - max: score(I) = s_(1) = max(S_I)
       - mean_top2: score(I) = (s_(1) + s_(2)) / 2 (nếu m >= 2) hoặc s_(1) (nếu m == 1)
    5. Độ chênh lệch nhập nhằng (Ambiguity margin calculation):
       Margin Delta = score_top1 - score_top2. Khi Delta nhỏ (Delta < margin_threshold),
       yêu cầu của người dùng có tính nhập nhằng cao giữa hai kịch bản hàng đầu.
    6. Cập nhật động không cần huấn luyện lại (Dynamic add_intent & remove_intent):
       Bổ sung hoặc loại bỏ ý định trực tiếp trên ma trận vector nhúng bộ nhớ trong (zero-shot dynamic indexing).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

from core.config import (
    DEFAULT_AMBIGUITY_MARGIN_THRESHOLD,
    DEFAULT_INDEXING_MODE,
    DEFAULT_INTENTS_PATH,
    DEFAULT_RETRIEVAL_AGGREGATION,
    DEFAULT_RETRIEVAL_TOP_K,
    INDEX_CACHE_PREFIX,
    INDEX_CACHE_DIR,
    INDEXING_MODE_ALIAS_ONLY,
    INDEXING_MODE_COMBINED,
    INDEXING_MODE_SCENARIO_ONLY,
    SBERT_MODEL_NAME,
    resolve_sbert_model_dir,
    SUPPORTED_INDEXING_MODES,
    SUPPORTED_RETRIEVAL_AGGREGATIONS,
    THETA_RETRIEVAL_THRESHOLD,
    get_default_device,
)
from core.preprocess import prepare_for_embedding, preprocess_fingerprint
from core.scenario import scenario_string

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Các ngoại lệ tùy biến (Custom Exceptions)
# =====================================================================


class IndexCacheError(RuntimeError):
    """Ngoại lệ khi tải, lưu hoặc xác thực tệp chỉ mục bộ nhớ đệm .npz."""


class IntentNotFoundError(KeyError):
    """Ngoại lệ khi không tìm thấy mã ý định trong chỉ mục truy hồi."""


class DuplicateIntentError(ValueError):
    """Ngoại lệ khi ý định đã tồn tại trong chỉ mục và không kích hoạt ghi đè."""


# =====================================================================
# 2. Cấu trúc dữ liệu chỉ mục và kết quả truy hồi (Data Structures)
# =====================================================================


@dataclass(frozen=True)
class IndexRow:
    """
    Biểu diễn một hàng văn bản được lập chỉ mục trong kho truy hồi ngữ nghĩa.

    Thuộc tính:
        intent: Mã định danh kịch bản (ví dụ: 'MATHJAX_PHAN_SO').
        text: Chuỗi văn bản đã qua tiền xử lý chuẩn bằng prepare_for_embedding.
        raw_text: Chuỗi văn bản gốc ban đầu (chưa tiền xử lý).
        source_type: Loại nguồn trích xuất ('scenario_string' hoặc 'alias').
        describe: Mô tả tác vụ bằng tiếng Việt.
        category: Nhóm danh mục kịch bản.
    """

    intent: str
    text: str
    raw_text: str
    source_type: str
    describe: str = ""
    category: str = ""


@dataclass
class RetrievedCandidate:
    """
    Ứng viên kịch bản trong kết quả truy hồi kèm thông tin đối chiếu điểm số chi tiết.

    Thuộc tính:
        intent: Mã định danh ý định kịch bản.
        score: Điểm tương đồng đã tổng hợp theo phương thức được chọn (max hoặc mean_top2).
        max_score: Điểm cosine similarity lớn nhất trong số các hàng của intent này.
        mean_top2_score: Điểm trung bình cộng của 2 hàng cao nhất (hoặc max_score nếu chỉ có 1 hàng).
        best_text: Chuỗi văn bản đạt điểm cao nhất của intent này.
        best_source_type: Loại nguồn của hàng đạt điểm cao nhất ('scenario_string' hoặc 'alias').
        all_row_scores: Danh sách toàn bộ điểm tương đồng các hàng của intent sắp xếp giảm dần.
        describe: Mô tả kịch bản.
        category: Danh mục kịch bản.
    """

    intent: str
    score: float
    max_score: float
    mean_top2_score: float
    best_text: str
    best_source_type: str
    all_row_scores: list[float] = field(default_factory=list)
    describe: str = ""
    category: str = ""


@dataclass
class RetrievalResult:
    """
    Kết quả tổng hợp từ bước truy hồi ngữ nghĩa Sentence-BERT.

    Thuộc tính:
        query: Chuỗi câu hỏi/yêu cầu ban đầu từ người dùng.
        preprocessed_query: Chuỗi truy vấn sau khi qua prepare_for_embedding.
        candidates: Danh sách tối đa top_k ứng viên kịch bản sắp xếp giảm dần theo score.
        ambiguity_margin: Độ chênh lệch điểm số giữa top-1 và top-2 (score_1 - score_2).
        understood: Hệ thống có hiểu truy vấn hay không (True nếu top-1 score >= threshold_theta).
        aggregation: Phương thức tổng hợp điểm đã áp dụng ('max' hoặc 'mean_top2').
        top_k: Số lượng ứng viên được yêu cầu lấy về.
    """

    query: str
    preprocessed_query: str
    candidates: list[RetrievedCandidate]
    ambiguity_margin: float
    understood: bool
    aggregation: str
    top_k: int

    @property
    def margin(self) -> float:
        """Định danh ngắn gọn cho độ chênh lệch nhập nhằng (ambiguity margin)."""
        return self.ambiguity_margin

    @property
    def top_candidate(self) -> RetrievedCandidate | None:
        """Ứng viên đứng đầu danh sách nếu có."""
        return self.candidates[0] if self.candidates else None

    @property
    def top_intent(self) -> str | None:
        """Mã ý định đứng đầu danh sách nếu có."""
        return self.candidates[0].intent if self.candidates else None

    @property
    def top_score(self) -> float:
        """Điểm số của ứng viên đứng đầu (mặc định 0.0 nếu rỗng)."""
        return self.candidates[0].score if self.candidates else 0.0

    def is_ambiguous(
        self, threshold: float = DEFAULT_AMBIGUITY_MARGIN_THRESHOLD
    ) -> bool:
        """
        Kiểm tra yêu cầu truy vấn có đang bị nhập nhằng giữa các kịch bản hay không.
        Trả về True nếu có ít nhất 2 ứng viên và khoảng cách điểm số nhỏ hơn ngưỡng threshold.
        """
        return len(self.candidates) >= 2 and self.ambiguity_margin < threshold


@dataclass
class AggregationComparison:
    """
    Kết quả đối chuẩn so sánh giữa hai chiến lược tổng hợp điểm số (max và mean_top2).
    """

    query: str
    max_result: RetrievalResult
    mean_top2_result: RetrievalResult
    differences: dict[str, dict[str, Any]]


# =====================================================================
# 3. Hàm băm chỉ mục và Mock Encoder phục vụ kiểm thử
# =====================================================================


def compute_index_hash(
    model_name: str,
    rows: Sequence[IndexRow | tuple[str, str]],
    fingerprint: str,
    indexing_mode: str = DEFAULT_INDEXING_MODE,
) -> str:
    """
    Tính toán mã băm SHA-256 tất định cho chỉ mục dựa trên các yếu tố:
    1. Tên định danh mô hình embedding (model_name).
    2. Toàn bộ các hàng biểu diễn được lập chỉ mục (intent, preprocessed_text).
    3. Mã vân tay cấu hình tiền xử lý (preprocessing fingerprint).
    4. Chế độ lập chỉ mục (indexing_mode).

    Công thức:
        hash = SHA256(model_name || fingerprint || indexing_mode || row_1 || row_2 || ... || row_N)

    Tham số:
        model_name: Tên mô hình hoặc đường dẫn checkpoint.
        rows: Danh sách các hàng IndexRow hoặc cặp (intent, text).
        fingerprint: Chuỗi vân tay tiền xử lý 64 ký tự SHA-256.
        indexing_mode: Chế độ lập chỉ mục ('combined', 'alias_only', 'scenario_only').

    Trả về:
        Chuỗi thập lục phân SHA-256 độ dài 64 ký tự.
    """
    hasher = hashlib.sha256()
    hasher.update(str(model_name).strip().encode("utf-8"))
    hasher.update(b"\x1f")  # Ký tự phân tách trường (Unit Separator)
    hasher.update(str(fingerprint).strip().encode("utf-8"))
    hasher.update(b"\x1f")
    hasher.update(str(indexing_mode).strip().encode("utf-8"))
    hasher.update(b"\x1f")

    for r in rows:
        intent = r.intent if hasattr(r, "intent") else r[0]
        text = r.text if hasattr(r, "text") else r[1]
        hasher.update(intent.encode("utf-8"))
        hasher.update(b"\x1e")  # Ký tự phân tách bản ghi (Record Separator)
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\n")

    return hasher.hexdigest()


class MockSentenceTransformer:
    """
    Bộ mã hóa giả lập SentenceTransformer tất định và siêu nhẹ phục vụ kiểm thử đơn vị.
    Sinh vector nhúng chuẩn hóa L2 (||v||_2 = 1.0) dựa trên mã băm nội dung văn bản,
    đảm bảo tính tái lập 100% không phụ thuộc trọng số nạp qua internet.
    """

    def __init__(self, embedding_dim: int = 64, model_name: str = "mock-sbert") -> None:
        self.embedding_dim = embedding_dim
        self.model_name = model_name

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

    def get_sentence_embedding_dimension(self) -> int:
        """Tương thích ngược với test/đối tượng cũ; đường chạy chính dùng API mới."""
        return self.get_embedding_dimension()

    def encode(
        self,
        sentences: Sequence[str] | str,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            input_texts = [sentences]
        else:
            input_texts = list(sentences)

        if not input_texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        vectors: list[np.ndarray] = []
        for text in input_texts:
            # Tạo seed tất định từ mã băm văn bản
            h = hashlib.sha256(text.encode("utf-8")).digest()
            seed_val = int.from_bytes(h[:4], "big")
            rng = np.random.RandomState(seed_val)
            vec = rng.randn(self.embedding_dim).astype(np.float32)

            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0.0:
                    vec = vec / norm
            vectors.append(vec)

        res = np.array(vectors, dtype=np.float32)
        return res


# =====================================================================
# 4. Lớp Retriever chính
# =====================================================================


class Retriever:
    """
    Bộ truy hồi ngữ nghĩa Sentence-BERT với cơ chế lập chỉ mục động và lưu đệm chỉ mục.

    Nguyên lý hoạt động:
    1. Lập chỉ mục kịch bản từ aliases và biến thể 'handmade' của scenario_string.
    2. Toàn bộ văn bản được tiền xử lý chuẩn hóa qua core.preprocess.prepare_for_embedding.
    3. Mã hóa bằng vector nhúng đã chuẩn hóa L2 (L2-normalized embeddings) để tích vô hướng
       u . v tương đương chính xác với Cosine Similarity.
    4. Lưu đệm chỉ mục vào models/index_<hash>.npz dựa trên model_name, rows và fingerprint.
    5. Cung cấp truy hồi top_k với 2 cơ chế tổng hợp điểm số (max và mean_top2) cùng tính toán
       độ chênh lệch nhập nhằng (ambiguity margin).
    6. Hỗ trợ thêm/xóa ý định động (add_intent, remove_intent) trực tiếp trên RAM không cần huấn luyện lại.
    """

    def __init__(
        self,
        model: str | Path | Any | None = None,
        intents: Sequence[dict[str, Any] | Any] | Path | str | None = None,
        cache_dir: Path | str | None = None,
        threshold_theta: float = THETA_RETRIEVAL_THRESHOLD,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        aggregation: str = DEFAULT_RETRIEVAL_AGGREGATION,
        use_cache: bool = True,
        device: str | None = None,
        model_name: str | None = None,
        embedding_dim: int = 768,
        indexing_mode: str = DEFAULT_INDEXING_MODE,
    ) -> None:
        """
        Khởi tạo bộ truy hồi ngữ nghĩa Retriever.

        Tham số:
            model: Đối tượng mô hình (SentenceTransformer / Mock), đường dẫn tệp hoặc định danh Hub.
            intents: Danh sách kịch bản (list dict/Scenario), đường dẫn tệp JSON, hoặc None (tải intents.json mặc định).
            cache_dir: Thư mục lưu đệm các tệp index_<hash>.npz (mặc định MODELS_DIR).
            threshold_theta: Ngưỡng tương đồng cosine tối thiểu để xác nhận hiểu truy vấn (mặc định 0.3).
            top_k: Số lượng ứng viên kịch bản trả về mặc định trong danh sách lựa chọn.
            aggregation: Phương thức tổng hợp điểm mặc định ('max' hoặc 'mean_top2').
            use_cache: Cờ bật/tắt đọc và ghi bộ nhớ đệm index_<hash>.npz.
            device: Thiết bị tính toán ('cpu', 'cuda').
            model_name: Tên định danh mô hình embedding phục vụ tính toán mã băm.
            embedding_dim: Số chiều vector nhúng dự phòng khi dùng mock encoder.
            indexing_mode: Chế độ biểu diễn lập chỉ mục ('combined', 'alias_only', 'scenario_only').
        """
        self.device = device if device is not None else get_default_device()
        self.threshold_theta = float(threshold_theta)
        self.top_k = int(top_k)

        norm_agg = self._normalize_aggregation(aggregation)
        self.aggregation = norm_agg

        norm_mode = str(indexing_mode).strip().lower()
        if norm_mode not in SUPPORTED_INDEXING_MODES:
            raise ValueError(
                f"Chế độ lập chỉ mục '{indexing_mode}' không được hỗ trợ. "
                f"Các chế độ hợp lệ: {list(SUPPORTED_INDEXING_MODES)}"
            )
        self.indexing_mode = norm_mode

        self.use_cache = bool(use_cache)
        self._index_lock = RLock()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else INDEX_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Xác định và nạp mô hình Sentence-BERT
        self._model: Any = None
        self._model_name: str = ""
        self._embedding_dim: int = embedding_dim
        self._init_model(model=model, model_name=model_name)

        # 2. Khởi tạo kho lưu trữ chỉ mục trong bộ nhớ RAM
        self._rows: list[IndexRow] = []
        self._embeddings: np.ndarray = np.empty((0, self._embedding_dim), dtype=np.float32)
        self._intent_data: dict[str, Any] = {}
        self._cache_path: Path | None = None

        # 3. Lập chỉ mục kịch bản (từ danh sách, JSON hoặc cache)
        raw_intents = self._load_intents_input(intents)
        if raw_intents:
            self.build_index(raw_intents)

    # -----------------------------------------------------------------
    # Thuộc tính công khai (Public Properties)
    # -----------------------------------------------------------------

    @property
    def model(self) -> Any:
        """Đối tượng mô hình embedding đang được sử dụng."""
        return self._model

    @property
    def model_name(self) -> str:
        """Tên định danh mô hình embedding."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Số chiều của vector nhúng (embedding dimension)."""
        return self._embedding_dim

    @property
    def rows(self) -> list[IndexRow]:
        """Danh sách các hàng văn bản đã lập chỉ mục."""
        with self._index_lock:
            return list(self._rows)

    @property
    def embeddings(self) -> np.ndarray:
        """Ma trận vector nhúng đã chuẩn hóa kích thước (N, D)."""
        with self._index_lock:
            return self._embeddings.copy()

    @property
    def num_rows(self) -> int:
        """Tổng số hàng biểu diễn đã lập chỉ mục."""
        return len(self._rows)

    @property
    def num_intents(self) -> int:
        """Tổng số ý định (intents) độc nhất trong chỉ mục."""
        return len(self.intents)

    @property
    def intents(self) -> list[str]:
        """Danh sách tên các ý định độc nhất đã lập chỉ mục (đã sắp xếp)."""
        with self._index_lock:
            seen: set[str] = set()
            res: list[str] = []
            for r in self._rows:
                if r.intent not in seen:
                    seen.add(r.intent)
                    res.append(r.intent)
            return sorted(res)

    @property
    def index_lock(self) -> RLock:
        """Lock chia sẻ để thao tác registry không lộ index trạng thái dở."""
        return self._index_lock

    @property
    def cache_path(self) -> Path | None:
        """Đường dẫn tệp bộ nhớ đệm .npz của chỉ mục hiện tại (nếu có)."""
        return self._cache_path

    def __contains__(self, intent_name: str) -> bool:
        """Kiểm tra một ý định có tồn tại trong chỉ mục hay không (in operator)."""
        with self._index_lock:
            return intent_name in self._intent_data or any(
                r.intent == intent_name for r in self._rows
            )

    # -----------------------------------------------------------------
    # Khởi tạo mô hình và tải dữ liệu kịch bản
    # -----------------------------------------------------------------

    def _init_model(self, model: Any, model_name: str | None) -> None:
        """Khởi tạo bộ mã hóa embedding (hỗ trợ SentenceTransformer, Mock, hoặc nạp từ đĩa)."""
        if (
            model is not None
            and not isinstance(model, (str, Path))
            and hasattr(model, "encode")
            and callable(model.encode)
        ):
            self._model = model
            if model_name:
                self._model_name = str(model_name)
            elif hasattr(model, "model_name"):
                self._model_name = str(model.model_name)
            else:
                self._model_name = "custom-encoder"

            dimension_getter = getattr(model, "get_embedding_dimension", None) or getattr(
                model, "get_sentence_embedding_dimension", None
            )
            if callable(dimension_getter):
                self._embedding_dim = dimension_getter()
            return

        target_name_or_path: str | Path
        if model is not None:
            target_name_or_path = model
            self._model_name = str(model_name or model)
        elif resolve_sbert_model_dir() is not None:
            target_name_or_path = resolve_sbert_model_dir()
            self._model_name = str(model_name or SBERT_MODEL_NAME)
        else:
            target_name_or_path = SBERT_MODEL_NAME
            self._model_name = str(model_name or SBERT_MODEL_NAME)

        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Thư viện 'sentence-transformers' chưa được cài đặt. "
                "Vui lòng cài đặt qua pip hoặc truyền đối tượng encoder tùy biến có hàm encode()."
            )

        try:
            self._model = SentenceTransformer(
                str(target_name_or_path),
                device=self.device,
            )
            dimension_getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
                self._model, "get_sentence_embedding_dimension", None
            )
            if callable(dimension_getter):
                self._embedding_dim = dimension_getter()
        except (OSError, RuntimeError, ValueError) as e:
            raise RuntimeError(
                f"Không thể nạp mô hình Sentence-BERT '{target_name_or_path}'. "
                f"Lỗi: {e}. Vui lòng kiểm tra checkpoint cục bộ tại models/hf_cache hoặc truyền model."
            ) from e

    def _load_intents_input(
        self, intents_input: Sequence[dict[str, Any] | Any] | Path | str | None
    ) -> list[dict[str, Any] | Any]:
        """Chuẩn hóa dữ liệu đầu vào kịch bản thành danh sách."""
        if intents_input is None:
            if DEFAULT_INTENTS_PATH.exists():
                with DEFAULT_INTENTS_PATH.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else list(data.values())
            return []

        if isinstance(intents_input, (str, Path)):
            p = Path(intents_input)
            if not p.exists():
                raise FileNotFoundError(f"Tệp ngân hàng kịch bản không tồn tại: {p}")
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else list(data.values())

        return list(intents_input)

    @staticmethod
    def _normalize_aggregation(aggregation: str) -> str:
        """Chuẩn hóa và kiểm tra tên phương thức tổng hợp điểm số."""
        agg = str(aggregation).strip().lower()
        if agg == "mean_top_2":
            agg = "mean_top2"
        if agg not in SUPPORTED_RETRIEVAL_AGGREGATIONS:
            raise ValueError(
                f"Phương thức tổng hợp điểm '{aggregation}' không được hỗ trợ. "
                f"Các phương thức hợp lệ: {list(SUPPORTED_RETRIEVAL_AGGREGATIONS)}"
            )
        return agg

    # -----------------------------------------------------------------
    # Mã hóa vector nhúng (Normalized Embeddings)
    # -----------------------------------------------------------------

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Mã hóa danh sách chuỗi văn bản thành ma trận vector nhúng chuẩn hóa L2 (||v||_2 = 1.0).

        Tuân thủ công thức Reimers & Gurevych (2019):
            e_norm = e / ||e||_2
            sim(u, v) = u_norm . v_norm
        """
        if not texts:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        raw_emb = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if not isinstance(raw_emb, np.ndarray):
            emb = np.array(raw_emb, dtype=np.float32)
        else:
            emb = raw_emb.astype(np.float32)

        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        # Cập nhật số chiều vector thực tế nếu chưa được gán chính xác
        if emb.shape[1] != self._embedding_dim and emb.shape[1] > 0:
            self._embedding_dim = emb.shape[1]

        # Đảm bảo chuẩn hóa L2 tuyệt đối (phòng ngừa sai số làm tròn số học)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = emb / norms
        return normalized.astype(np.float32)

    # -----------------------------------------------------------------
    # Xây dựng chỉ mục và cơ chế lưu đệm (Index Building & Caching)
    # -----------------------------------------------------------------

    def _build_intent_rows(
        self, intent_item: dict[str, Any] | Any
    ) -> tuple[str, list[IndexRow]]:
        """
        Trích xuất các hàng biểu diễn cho một ý định duy nhất:
        - Chuỗi kịch bản tự nhiên RAF: scenario_string(intent, 'handmade') (khi mode combined hoặc scenario_only)
        - Danh sách các biệt danh ngữ nghĩa (aliases) (khi mode combined hoặc alias_only)
        Tất cả đều được xử lý qua prepare_for_embedding.
        """
        if isinstance(intent_item, dict):
            intent_name = str(intent_item.get("intent", "")).strip()
            describe = str(intent_item.get("describe", "")).strip()
            category = str(intent_item.get("category", "")).strip()
            raw_aliases = list(intent_item.get("aliases", []) or [])
            learned_aliases = intent_item.get("learned_aliases", [])
            if isinstance(learned_aliases, (list, tuple, set)):
                raw_aliases.extend(learned_aliases)
        else:
            intent_name = str(getattr(intent_item, "intent", "")).strip()
            describe = str(getattr(intent_item, "describe", "")).strip()
            category = str(getattr(intent_item, "category", "")).strip()
            raw_aliases = list(getattr(intent_item, "aliases", []) or [])
            learned_aliases = getattr(intent_item, "learned_aliases", [])
            if isinstance(learned_aliases, (list, tuple, set)):
                raw_aliases.extend(learned_aliases)

        if not intent_name:
            raise ValueError(f"Kịch bản thiếu trường định danh 'intent': {intent_item}")

        rows: list[IndexRow] = []
        seen_texts: set[str] = set()

        # 1. Hàng biểu diễn tự nhiên chuẩn RAF 'handmade'
        if self.indexing_mode in (INDEXING_MODE_COMBINED, INDEXING_MODE_SCENARIO_ONLY):
            raw_scenario = scenario_string(intent_item, variant="handmade")
            prep_scenario = prepare_for_embedding(raw_scenario, model_name=self.model_name)
            rows.append(
                IndexRow(
                    intent=intent_name,
                    text=prep_scenario,
                    raw_text=raw_scenario,
                    source_type="scenario_string",
                    describe=describe,
                    category=category,
                )
            )
            seen_texts.add(prep_scenario)

        # 2. Các hàng biểu diễn từ biệt danh (aliases)
        if self.indexing_mode in (INDEXING_MODE_COMBINED, INDEXING_MODE_ALIAS_ONLY):
            if isinstance(raw_aliases, (list, tuple, set)):
                for alias in raw_aliases:
                    alias_str = str(alias).strip()
                    if not alias_str:
                        continue
                    try:
                        prep_alias = prepare_for_embedding(
                            alias_str, model_name=self.model_name
                        )
                    except ValueError:
                        continue

                    if prep_alias not in seen_texts:
                        seen_texts.add(prep_alias)
                        rows.append(
                            IndexRow(
                                intent=intent_name,
                                text=prep_alias,
                                raw_text=alias_str,
                                source_type="alias",
                                describe=describe,
                                category=category,
                            )
                        )

        # Dự phòng trường hợp intent không có alias khi chạy ở chế độ alias_only
        if not rows and self.indexing_mode == INDEXING_MODE_ALIAS_ONLY:
            raw_scenario = scenario_string(intent_item, variant="handmade")
            prep_scenario = prepare_for_embedding(raw_scenario, model_name=self.model_name)
            rows.append(
                IndexRow(
                    intent=intent_name,
                    text=prep_scenario,
                    raw_text=raw_scenario,
                    source_type="scenario_string",
                    describe=describe,
                    category=category,
                )
            )

        return intent_name, rows

    def _build_rows(
        self, intents: Sequence[dict[str, Any] | Any]
    ) -> list[IndexRow]:
        """Xây dựng danh sách hàng IndexRow cho toàn bộ ngân hàng kịch bản."""
        all_rows: list[IndexRow] = []
        self._intent_data.clear()

        for item in intents:
            name, item_rows = self._build_intent_rows(item)
            self._intent_data[name] = item
            all_rows.extend(item_rows)

        return all_rows

    def build_index(
        self,
        intents: Sequence[dict[str, Any] | Any],
        indexing_mode: str | None = None,
    ) -> None:
        """Rebuild index while blocking readers from seeing partial state."""
        with self._index_lock:
            self._build_index_unlocked(intents, indexing_mode)

    def _build_index_unlocked(
        self,
        intents: Sequence[dict[str, Any] | Any],
        indexing_mode: str | None = None,
    ) -> None:
        """
        Lập chỉ mục ngữ nghĩa cho tập kịch bản:
        Kiểm tra bộ nhớ đệm models/index_<hash>.npz:
        - Nếu mã băm khớp với model_name, rows, fingerprint và indexing_mode: Tải trực tiếp O(1).
        - Nếu chưa có hoặc lệch: Mã hóa vector nhúng và lưu vào bộ nhớ đệm.
        """
        if indexing_mode is not None:
            norm_mode = str(indexing_mode).strip().lower()
            if norm_mode not in SUPPORTED_INDEXING_MODES:
                raise ValueError(
                    f"Chế độ lập chỉ mục '{indexing_mode}' không được hỗ trợ. "
                    f"Các chế độ hợp lệ: {list(SUPPORTED_INDEXING_MODES)}"
                )
            self.indexing_mode = norm_mode

        rows = self._build_rows(intents)
        if not rows:
            self._rows = []
            self._embeddings = np.empty((0, self._embedding_dim), dtype=np.float32)
            self._cache_path = None
            return

        fp = preprocess_fingerprint()
        index_hash = compute_index_hash(
            self.model_name, rows, fp, indexing_mode=self.indexing_mode
        )
        cache_file = self.cache_dir / f"{INDEX_CACHE_PREFIX}{index_hash}.npz"
        self._cache_path = cache_file

        # Kiểm tra nạp từ bộ nhớ đệm nếu use_cache=True
        if self.use_cache and cache_file.exists():
            loaded = self._load_cache(cache_file, expected_hash=index_hash, expected_fp=fp)
            if loaded is not None:
                self._rows, self._embeddings = loaded
                return

        # Tính toán mã hóa mới
        texts = [r.text for r in rows]
        embeddings = self._encode_texts(texts)

        self._rows = rows
        self._embeddings = embeddings

        # Lưu đệm xuống đĩa nếu được bật
        if self.use_cache:
            self._save_cache(cache_file, index_hash=index_hash, fingerprint=fp)

    def _save_cache(
        self,
        cache_path: Path,
        index_hash: str,
        fingerprint: str,
    ) -> None:
        """Lưu trữ chỉ mục và ma trận vector nhúng xuống tệp nén .npz."""
        metadata = {
            "model_name": self.model_name,
            "fingerprint": fingerprint,
            "index_hash": index_hash,
            "indexing_mode": self.indexing_mode,
            "num_rows": len(self._rows),
            "num_intents": len(self.intents),
        }

        np.savez_compressed(
            cache_path,
            embeddings=self._embeddings,
            intents=np.array([r.intent for r in self._rows], dtype=object),
            texts=np.array([r.text for r in self._rows], dtype=object),
            raw_texts=np.array([r.raw_text for r in self._rows], dtype=object),
            source_types=np.array([r.source_type for r in self._rows], dtype=object),
            describes=np.array([r.describe for r in self._rows], dtype=object),
            categories=np.array([r.category for r in self._rows], dtype=object),
            metadata=np.array(json.dumps(metadata, ensure_ascii=False), dtype=object),
        )

    def _load_cache(
        self,
        cache_path: Path,
        expected_hash: str,
        expected_fp: str,
    ) -> tuple[list[IndexRow], np.ndarray] | None:
        """Nạp chỉ mục và ma trận vector từ tệp .npz, kiểm thực tính toàn vẹn metadata."""
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                if "metadata" not in data or "embeddings" not in data:
                    return None
                meta_str = str(data["metadata"])
                meta = json.loads(meta_str)

                if meta.get("index_hash") != expected_hash:
                    return None
                if meta.get("fingerprint") != expected_fp:
                    return None
                if meta.get("model_name") != self.model_name:
                    return None
                if meta.get("indexing_mode", DEFAULT_INDEXING_MODE) != self.indexing_mode:
                    return None

                intents = list(data["intents"])
                texts = list(data["texts"])
                raw_texts = list(data["raw_texts"])
                source_types = list(data["source_types"])
                describes = list(data["describes"])
                categories = list(data["categories"])
                emb = data["embeddings"].astype(np.float32)

                rows: list[IndexRow] = []
                for i, intent in enumerate(intents):
                    rows.append(
                        IndexRow(
                            intent=str(intent),
                            text=str(texts[i]),
                            raw_text=str(raw_texts[i]),
                            source_type=str(source_types[i]),
                            describe=str(describes[i]),
                            category=str(categories[i]),
                        )
                    )

                if emb.shape[1] > 0:
                    self._embedding_dim = emb.shape[1]

                return rows, emb
        except (
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            pickle.UnpicklingError,
            zipfile.BadZipFile,
            EOFError,
            AttributeError,
        ) as e:
            logger.warning(
                "Không thể nạp bộ nhớ đệm chỉ mục từ %s: %s. Sẽ tiến hành tính toán lại.",
                cache_path,
                e,
            )
            return None

    def save_index(self, path: Path | str | None = None) -> Path:
        """
        Lưu trạng thái chỉ mục hiện tại vào tệp .npz chỉ định hoặc tệp theo mã băm hiện hành.
        """
        fp = preprocess_fingerprint()
        index_hash = compute_index_hash(
            self.model_name, self._rows, fp, indexing_mode=self.indexing_mode
        )

        if path is not None:
            target_path = Path(path)
        else:
            target_path = self.cache_dir / f"{INDEX_CACHE_PREFIX}{index_hash}.npz"

        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_cache(target_path, index_hash=index_hash, fingerprint=fp)
        self._cache_path = target_path
        return target_path

    # -----------------------------------------------------------------
    # Cập nhật động không cần huấn luyện lại (Dynamic Indexing)
    # -----------------------------------------------------------------

    def add_intent(
        self,
        intent: dict[str, Any] | Any,
        overwrite: bool = False,
    ) -> int:
        """Add or replace an intent under the shared index lock."""
        with self._index_lock:
            return self._add_intent_unlocked(intent, overwrite)

    def _add_intent_unlocked(
        self,
        intent: dict[str, Any] | Any,
        overwrite: bool = False,
    ) -> int:
        """
        Bổ sung một ý định mới vào chỉ mục ngữ nghĩa trực tiếp trong bộ nhớ RAM (zero-shot)
        mà không cần huấn luyện lại toàn bộ mô hình.

        Tham số:
            intent: Cấu trúc kịch bản mới (dict hoặc Scenario).
            overwrite: Ghi đè kịch bản nếu đã tồn tại trong chỉ mục.

        Trả về:
            Số lượng hàng biểu diễn mới được thêm vào chỉ mục.

        Ngoại lệ:
            DuplicateIntentError: Nếu intent đã tồn tại và overwrite=False.
        """
        intent_name, new_rows = self._build_intent_rows(intent)

        if intent_name in self:
            if not overwrite:
                raise DuplicateIntentError(
                    f"Ý định '{intent_name}' đã tồn tại trong chỉ mục. "
                    f"Sử dụng overwrite=True để thay thế."
                )
            self.remove_intent(intent_name)

        if not new_rows:
            self._intent_data[intent_name] = intent
            return 0

        # Mã hóa các hàng mới bằng mô hình
        new_texts = [r.text for r in new_rows]
        new_embeddings = self._encode_texts(new_texts)

        # Cập nhật danh sách hàng và ma trận vector nhúng
        if len(self._rows) == 0 or self._embeddings.shape[0] == 0:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        self._rows.extend(new_rows)
        self._intent_data[intent_name] = intent
        return len(new_rows)

    def remove_intent(self, intent_name: str) -> bool:
        """Remove an intent under the shared index lock."""
        with self._index_lock:
            return self._remove_intent_unlocked(intent_name)

    def _remove_intent_unlocked(self, intent_name: str) -> bool:
        """
        Loại bỏ một ý định và toàn bộ các hàng biểu diễn liên quan khỏi chỉ mục.

        Tham số:
            intent_name: Mã định danh kịch bản cần loại bỏ.

        Trả về:
            True nếu loại bỏ thành công.

        Ngoại lệ:
            IntentNotFoundError: Nếu không tìm thấy intent_name trong chỉ mục.
        """
        if intent_name not in self:
            raise IntentNotFoundError(
                f"Ý định '{intent_name}' không tồn tại trong chỉ mục."
            )

        keep_mask = [r.intent != intent_name for r in self._rows]
        if not all(keep_mask):
            self._rows = [r for r, keep in zip(self._rows, keep_mask) if keep]
            if len(self._rows) > 0:
                self._embeddings = self._embeddings[np.array(keep_mask)]
            else:
                self._embeddings = np.empty((0, self._embedding_dim), dtype=np.float32)

        self._intent_data.pop(intent_name, None)
        return True

    # -----------------------------------------------------------------
    # Truy hồi ngữ nghĩa và tổng hợp điểm số (Retrieval & Scoring)
    # -----------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        aggregation: str | None = None,
        threshold_theta: float | None = None,
    ) -> RetrievalResult:
        """Score against one consistent row and embedding snapshot."""
        with self._index_lock:
            return self._retrieve_unlocked(query, top_k, aggregation, threshold_theta)

    def _retrieve_unlocked(
        self,
        query: str,
        top_k: int | None = None,
        aggregation: str | None = None,
        threshold_theta: float | None = None,
    ) -> RetrievalResult:
        """
        Thực hiện truy hồi ngữ nghĩa tìm kiếm các kịch bản tương đồng nhất với câu truy vấn.

        Quy trình xử lý:
        1. Tiền xử lý câu truy vấn qua prepare_for_embedding(query, model_name=self.model_name).
        2. Mã hóa truy vấn thành vector chuẩn hóa L2 q_norm.
        3. Tính toán tương đồng Cosine qua tích vô hướng: sim = Embeddings . q_norm.
        4. Gom nhóm điểm số theo từng intent và áp dụng tổng hợp (max hoặc mean_top2).
        5. Sắp xếp danh sách kịch bản theo điểm giảm dần và tính độ chênh lệch nhập nhằng
           Delta = score_top1 - score_top2.
        6. Kiểm tra điều kiện hiểu: understood = (top_score >= threshold_theta).

        Tham số:
            query: Chuỗi văn bản truy vấn từ người dùng.
            top_k: Số lượng ứng viên cần lấy (nếu None dùng self.top_k).
            aggregation: Phương thức tổng hợp điểm ('max' hoặc 'mean_top2').
            threshold_theta: Ngưỡng cosine tối thiểu (nếu None dùng self.threshold_theta).

        Trả về:
            Đối tượng RetrievalResult chứa danh sách ứng viên, điểm số và độ chênh lệch nhập nhằng.
        """
        k = int(top_k if top_k is not None else self.top_k)
        agg = self._normalize_aggregation(aggregation if aggregation is not None else self.aggregation)
        theta = float(threshold_theta if threshold_theta is not None else self.threshold_theta)

        # Xử lý chuỗi rỗng an toàn
        if not isinstance(query, str) or not query.strip():
            return RetrievalResult(
                query=query if isinstance(query, str) else "",
                preprocessed_query="",
                candidates=[],
                ambiguity_margin=0.0,
                understood=False,
                aggregation=agg,
                top_k=k,
            )

        try:
            prep_query = prepare_for_embedding(query, model_name=self.model_name)
        except ValueError:
            return RetrievalResult(
                query=query,
                preprocessed_query="",
                candidates=[],
                ambiguity_margin=0.0,
                understood=False,
                aggregation=agg,
                top_k=k,
            )

        # Nếu chỉ mục rỗng, trả về kết quả rỗng
        if len(self._rows) == 0 or self._embeddings.shape[0] == 0:
            return RetrievalResult(
                query=query,
                preprocessed_query=prep_query,
                candidates=[],
                ambiguity_margin=0.0,
                understood=False,
                aggregation=agg,
                top_k=k,
            )

        # Mã hóa truy vấn (vector chuẩn hóa 1D)
        query_emb = self._encode_texts([prep_query])[0]

        # Tích vô hướng tính tương đồng cosine với toàn bộ các hàng trong kho
        row_scores = np.dot(self._embeddings, query_emb)

        # Gom nhóm kết quả theo từng ý định (intent)
        intent_groups: dict[str, list[tuple[float, IndexRow]]] = {}
        for score, row in zip(row_scores, self._rows):
            s_val = max(-1.0, min(1.0, float(score)))
            intent_groups.setdefault(row.intent, []).append((s_val, row))

        # Tính toán điểm tổng hợp (max và mean_top2) cho từng intent
        scored_candidates: list[RetrievedCandidate] = []
        for intent_id, item_list in intent_groups.items():
            # Sắp xếp điểm số của các hàng thuộc intent này theo thứ tự giảm dần
            item_list.sort(key=lambda x: x[0], reverse=True)
            sorted_scores = [x[0] for x in item_list]
            best_score, best_row = item_list[0]

            max_score = best_score
            if len(sorted_scores) >= 2:
                mean_top2_score = (sorted_scores[0] + sorted_scores[1]) / 2.0
            else:
                mean_top2_score = sorted_scores[0]

            chosen_score = max_score if agg == "max" else mean_top2_score

            candidate = RetrievedCandidate(
                intent=intent_id,
                score=float(chosen_score),
                max_score=float(max_score),
                mean_top2_score=float(mean_top2_score),
                best_text=best_row.text,
                best_source_type=best_row.source_type,
                all_row_scores=sorted_scores,
                describe=best_row.describe,
                category=best_row.category,
            )
            scored_candidates.append(candidate)

        # Sắp xếp danh sách ứng viên theo điểm số tổng hợp giảm dần
        scored_candidates.sort(key=lambda c: c.score, reverse=True)

        # Tính toán độ chênh lệch nhập nhằng (Ambiguity Margin)
        if len(scored_candidates) >= 2:
            ambiguity_margin = max(
                0.0, float(scored_candidates[0].score - scored_candidates[1].score)
            )
        elif len(scored_candidates) == 1:
            ambiguity_margin = float(scored_candidates[0].score)
        else:
            ambiguity_margin = 0.0

        # Kiểm tra ngưỡng hiểu (understood)
        top_score = scored_candidates[0].score if scored_candidates else 0.0
        understood = (len(scored_candidates) > 0) and (top_score >= theta)

        return RetrievalResult(
            query=query,
            preprocessed_query=prep_query,
            candidates=scored_candidates[:k],
            ambiguity_margin=float(ambiguity_margin),
            understood=understood,
            aggregation=agg,
            top_k=k,
        )

    def compare_aggregations(
        self,
        query: str,
        top_k: int | None = None,
    ) -> AggregationComparison:
        """
        Đối chuẩn so sánh kết quả truy hồi giữa hai chiến lược tổng hợp:
        1. 'max': Tối đa hóa điểm tương đồng của hàng tốt nhất (nhạy bén với alias trùng khớp cao).
        2. 'mean_top2': Trung bình cộng 2 hàng đầu tiên (ưu tiên kịch bản có độ bao phủ ổn định).

        Trả về:
            Đối tượng AggregationComparison bao gồm kết quả cả hai phương pháp và chênh lệch điểm.
        """
        k = int(top_k if top_k is not None else self.top_k)
        res_max = self.retrieve(query, top_k=k, aggregation="max")
        res_mean2 = self.retrieve(query, top_k=k, aggregation="mean_top2")

        max_ranks = {c.intent: rank for rank, c in enumerate(res_max.candidates)}
        mean_ranks = {c.intent: rank for rank, c in enumerate(res_mean2.candidates)}
        all_intents = set(max_ranks.keys()).union(set(mean_ranks.keys()))

        diffs: dict[str, dict[str, Any]] = {}
        for intent_id in all_intents:
            c_max = next((c for c in res_max.candidates if c.intent == intent_id), None)
            c_m2 = next((c for c in res_mean2.candidates if c.intent == intent_id), None)

            s_max = c_max.score if c_max else 0.0
            s_m2 = c_m2.score if c_m2 else 0.0
            diffs[intent_id] = {
                "max_score": s_max,
                "mean_top2_score": s_m2,
                "score_difference": s_max - s_m2,
                "rank_max": max_ranks.get(intent_id),
                "rank_mean_top2": mean_ranks.get(intent_id),
            }

        return AggregationComparison(
            query=query,
            max_result=res_max,
            mean_top2_result=res_mean2,
            differences=diffs,
        )


def inspect_index_cache(
    cache_dir: Path | str,
    intents: Sequence[dict[str, Any]],
    model_name: str = SBERT_MODEL_NAME,
    indexing_mode: str = DEFAULT_INDEXING_MODE,
) -> tuple[bool, str, Path | None]:
    """Kiểm tra cache theo hash, metadata và shape mà Retriever thật sử dụng."""
    probe = Retriever.__new__(Retriever)
    probe._model_name = model_name
    probe.indexing_mode = indexing_mode
    probe._intent_data = {}
    rows = probe._build_rows(intents)
    if not rows:
        return False, "Ngân hàng kiến thức chưa có hàng để lập chỉ mục.", None
    fingerprint = preprocess_fingerprint()
    index_hash = compute_index_hash(model_name, rows, fingerprint, indexing_mode)
    cache_path = Path(cache_dir) / f"{INDEX_CACHE_PREFIX}{index_hash}.npz"
    if not cache_path.is_file():
        return False, "Model retrieval có sẵn; index chưa được tạo và sẽ build ở lần hỏi đầu tiên.", cache_path
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            metadata = json.loads(str(data["metadata"]))
            embeddings = data["embeddings"]
            cached_intents = data["intents"]
            count = len(cached_intents)
            expected_intents = {str(item.get("intent", "")) for item in intents if isinstance(item, dict)}
            aligned_rows = all(
                len(data[key]) == len(rows)
                for key in ("texts", "raw_texts", "source_types", "describes", "categories")
            )
            valid = (
                metadata.get("index_hash") == index_hash
                and metadata.get("fingerprint") == fingerprint
                and metadata.get("model_name") == model_name
                and metadata.get("indexing_mode", DEFAULT_INDEXING_MODE) == indexing_mode
                and metadata.get("num_rows") == len(rows)
                and metadata.get("num_intents") == len(expected_intents)
                and embeddings.ndim == 2
                and embeddings.shape[0] == count == len(rows)
                and aligned_rows
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile, EOFError):
        valid = False
    if not valid:
        return False, "Cache index hiện có không hợp lệ hoặc đã cũ; Retriever sẽ build lại.", cache_path
    return True, f"Index cache hợp lệ: {cache_path}", cache_path
