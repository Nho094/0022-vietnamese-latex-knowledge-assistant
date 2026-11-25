"""
Module truy hồi ngữ nghĩa dự phòng (Semantic Retrieval Fallback) theo phong cách Sentence-BERT.

Tuân thủ kiến trúc và quy tắc AGENTS.md:
- Kiến trúc xử lý:
  text -> preprocess -> PhoBERT classifier: p = softmax over intents
     - if max p >= tau: intent accepted
     - else retrieval top-3 over aliases + scenario strings -> ask the user to choose (if best score < theta: say not understood)
- Cảm hứng thiết kế từ Sentence-BERT (Reimers & Gurevych, 2019 - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks)
  và hệ thống Retrieve-and-Fill (RAF) đơn giản hóa theo Honesty Rule #1:
  Cosine similarity:
  sim(u, v) = (u . v) / (||u||_2 * ||v||_2)
- Xử lý các ý định mới (novel) hoặc chưa đăng ký (unregistered) mà không cần huấn luyện lại mô hình (zero-shot/few-shot retrieval).
- Khi điểm tương đồng cao nhất < theta (với theta = THETA_RETRIEVAL_THRESHOLD = 0.3): báo không hiểu ('say not understood').
- Ngược lại: trả về top-3 ứng viên và hỏi người dùng chọn ('ask the user to choose').
- Hỗ trợ tải ngoại tuyến từ thư mục models/hf_cache không cần internet, hỗ trợ CPU và GPU ('cuda').
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Optional, Sequence, Union

from core.config import (
    DEFAULT_INTENTS_PATH,
    DEFAULT_RETRIEVAL_TOP_K,
    SBERT_DIR,
    SBERT_MODEL_NAME,
    THETA_RETRIEVAL_THRESHOLD,
    get_default_device,
)
from src.scenarios import Scenario, ScenarioBank, load_scenario_bank


@dataclass
class CandidateMatch:
    """
    Thông tin một ứng viên kịch bản trong kết quả truy hồi.

    Thuộc tính:
        intent: Mã định danh kịch bản (ví dụ: MATHJAX_PHAN_SO).
        score: Điểm số tương đồng ngữ nghĩa (cosine similarity, từ 0.0 đến 1.0).
        describe: Mô tả tác vụ bằng tiếng Việt.
        matched_text: Chuỗi văn bản kịch bản khớp nhất (alias, câu mẫu hoặc mô tả).
        category: Danh mục của kịch bản.
    """
    intent: str
    score: float
    describe: str
    matched_text: str
    category: str = ""


@dataclass
class FallbackResult:
    """
    Kết quả tổng hợp từ bước truy hồi ngữ nghĩa dự phòng.

    Thuộc tính:
        query: Chuỗi văn bản truy vấn gốc từ người dùng.
        understood: Trạng thái hệ thống có hiểu yêu cầu hay không (True nếu top score >= theta).
        top_candidates: Danh sách tối đa top_k ứng viên kịch bản sắp xếp giảm dần theo điểm.
        prompt_message: Thông điệp phản hồi người dùng (gợi ý lựa chọn hoặc báo không hiểu).
    """
    query: str
    understood: bool
    top_candidates: list[CandidateMatch] = field(default_factory=list)
    prompt_message: str = ""


class SbertRetrievalFallback:
    """
    Bộ truy hồi ngữ nghĩa SBERT hỗ trợ fallback khi độ tin cậy phân loại PhoBERT dưới ngưỡng tau.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        threshold_theta: float = THETA_RETRIEVAL_THRESHOLD,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        scenario_bank: Optional[ScenarioBank] = None,
    ) -> None:
        """
        Khởi tạo bộ truy hồi ngữ nghĩa dự phòng.

        Tham số:
            model_path: Đường dẫn thư mục mô hình SBERT hoặc bộ nhớ đệm cục bộ models/hf_cache.
            device: Thiết bị tính toán ('cpu', 'cuda'). Mặc định lấy từ get_default_device.
            threshold_theta: Ngưỡng tương đồng cosine tối thiểu để chấp nhận hiểu (mặc định 0.3).
            top_k: Số lượng kịch bản trả về trong danh sách lựa chọn (mặc định 3).
            scenario_bank: Ngân hàng kịch bản cần lập chỉ mục ngữ nghĩa.
        """
        self.model_path = Path(model_path) if model_path is not None else SBERT_DIR
        self.device = device if device is not None else get_default_device()
        self.threshold_theta = threshold_theta
        self.top_k = top_k

        if scenario_bank is None:
            if DEFAULT_INTENTS_PATH.exists():
                self.scenario_bank = load_scenario_bank(DEFAULT_INTENTS_PATH)
            else:
                self.scenario_bank = ScenarioBank([])
        else:
            self.scenario_bank = scenario_bank

        self._sbert_model: Any = None
        self._is_sbert_loaded: bool = False

        # Chỉ mục kịch bản phục vụ tìm kiếm ngữ nghĩa
        self._scenario_index: list[tuple[Scenario, str, str]] = []  # (scenario, text_type, text_content)
        self._build_scenario_index()

        # Thử nạp mô hình SentenceTransformer offline
        self._try_load_sbert_model()

    @property
    def is_sbert_loaded(self) -> bool:
        """Kiểm tra mô hình SBERT thực tế đã được nạp hay chưa."""
        return self._is_sbert_loaded

    def _build_scenario_index(self) -> None:
        """
        Xây dựng kho văn bản kịch bản từ aliases, câu mẫu (sample utterances),
        mô tả (describe) và tên intent để phục vụ so khớp ngữ nghĩa.
        """
        self._scenario_index.clear()
        for sc in self.scenario_bank.all():
            # Thêm các alias
            for alias in sc.aliases:
                self._scenario_index.append((sc, "alias", alias))
            # Thêm câu mẫu
            for utt in sc.sample_utterances:
                self._scenario_index.append((sc, "sample_utterance", utt))
            # Thêm mô tả tiếng Việt
            if sc.describe:
                self._scenario_index.append((sc, "describe", sc.describe))
            # Thêm dạng chuẩn của tên intent
            intent_readable = sc.intent.lower().replace("_", " ")
            self._scenario_index.append((sc, "intent_name", intent_readable))

    def _try_load_sbert_model(self) -> bool:
        """
        Thử nạp mô hình SentenceTransformer từ thư mục cục bộ models/hf_cache.
        Sử dụng local_files_only để không kết nối internet.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            target_path = str(self.model_path)
            if self.model_path.exists():
                self._sbert_model = SentenceTransformer(target_path, device=self.device)
                self._is_sbert_loaded = True
                return True
        except (ImportError, Exception):
            # shortcut: khi môi trường chưa cài sentence-transformers hoặc checkpoint chưa có trong cache,
            # sử dụng TF-IDF vectorizer + cosine similarity làm bộ ước lượng ngữ nghĩa offline.
            pass

        self._is_sbert_loaded = False
        return False

    @staticmethod
    def _cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
        """
        Tính toán độ tương đồng Cosine giữa 2 vector thưa:
        CosineSim(u, v) = (u . v) / (||u||_2 * ||v||_2)
        """
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(vec1[k] * vec2[k] for k in vec1 if k in vec2)
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return max(0.0, min(1.0, dot_product / (norm1 * norm2)))

    @staticmethod
    def _text_to_vector(text: str) -> dict[str, float]:
        """
        Biến đổi chuỗi văn bản thành vector đặc trưng n-gram (từ và ký tự kết hợp).
        Giúp so khớp tương đồng ngữ nghĩa chính xác cho văn bản tiếng Việt ngắn.
        """
        cleaned = text.lower().strip()
        words = re.findall(r"[\w\+]+", cleaned)
        vector: dict[str, float] = {}

        # 1-gram từ
        for w in words:
            vector[f"w_{w}"] = vector.get(f"w_{w}", 0.0) + 1.0

        # 2-gram từ liên tiếp
        for i in range(len(words) - 1):
            bg = f"bg_{words[i]}_{words[i+1]}"
            vector[bg] = vector.get(bg, 0.0) + 1.5

        # 3-gram ký tự con để bắt các biến thể chính tả tiếng Việt
        for i in range(len(cleaned) - 2):
            cg = f"c_{cleaned[i:i+3]}"
            vector[cg] = vector.get(cg, 0.0) + 0.5

        return vector

    def retrieve(self, query: str) -> FallbackResult:
        """
        Truy hồi top-k kịch bản tương đồng nhất từ câu người dùng.

        Quy trình xử lý:
        1. Biến đổi câu truy vấn và kho văn bản thành biểu diễn vector.
        2. Tính độ tương đồng cosine giữa truy vấn và từng bản ghi kịch bản.
        3. Tổng hợp điểm số cao nhất cho mỗi intent.
        4. Sắp xếp giảm dần và lấy top_k kịch bản.
        5. Nếu điểm cao nhất < threshold_theta:
           Đánh dấu understood = False và trả về thông điệp không hiểu ('say not understood').
           Nếu điểm >= threshold_theta:
           Đánh dấu understood = True và tạo câu hỏi người dùng xác nhận ('ask the user to choose').

        Tham số:
            query: Chuỗi văn bản truy vấn cần tìm kịch bản phù hợp.

        Trả về:
            FallbackResult chứa các ứng viên và thông điệp tương ứng.
        """
        query_clean = query.strip()
        if not query_clean or not self._scenario_index:
            return FallbackResult(
                query=query,
                understood=False,
                top_candidates=[],
                prompt_message="Xin lỗi, tôi không hiểu yêu cầu của bạn. Vui lòng thử diễn đạt lại bằng câu khác.",
            )

        # Lưu điểm cao nhất theo intent: intent -> (best_score, scenario, matched_text)
        best_per_intent: dict[str, tuple[float, Scenario, str]] = {}

        if self._is_sbert_loaded and self._sbert_model is not None:
            # SBERT embedding inference
            query_emb = self._sbert_model.encode(query_clean, normalize_embeddings=True)
            for sc, text_type, text_content in self._scenario_index:
                doc_emb = self._sbert_model.encode(text_content, normalize_embeddings=True)
                score = float(sum(q * d for q, d in zip(query_emb, doc_emb)))
                if sc.intent not in best_per_intent or score > best_per_intent[sc.intent][0]:
                    best_per_intent[sc.intent] = (score, sc, text_content)
        else:
            # Heuristic vector Cosine Similarity
            query_vec = self._text_to_vector(query_clean)
            for sc, text_type, text_content in self._scenario_index:
                doc_vec = self._text_to_vector(text_content)
                score = self._cosine_similarity(query_vec, doc_vec)

                # Tăng cường điểm nếu chuỗi con xuất hiện trực tiếp
                if text_content.lower() in query_clean.lower() or query_clean.lower() in text_content.lower():
                    score = min(1.0, score + 0.25)

                if sc.intent not in best_per_intent or score > best_per_intent[sc.intent][0]:
                    best_per_intent[sc.intent] = (score, sc, text_content)

        # Sắp xếp các intent theo điểm số giảm dần
        sorted_intents = sorted(best_per_intent.items(), key=lambda item: item[1][0], reverse=True)
        top_items = sorted_intents[: self.top_k]

        candidates: list[CandidateMatch] = [
            CandidateMatch(
                intent=intent,
                score=round(score, 4),
                describe=sc.describe,
                matched_text=matched_text,
                category=sc.category,
            )
            for intent, (score, sc, matched_text) in top_items
        ]

        # Kiểm tra ngưỡng tương đồng theta
        best_score = candidates[0].score if candidates else 0.0
        if best_score < self.threshold_theta:
            # Theo AGENTS.md: 'if best score < theta: say not understood'
            understood = False
            message = "Xin lỗi, tôi không hiểu yêu cầu này. Vui lòng kiểm tra lại câu lệnh hoặc nhập lại cụ thể hơn."
        else:
            # Theo AGENTS.md: 'ask the user to choose'
            understood = True
            lines = ["Có phải bạn muốn thực hiện một trong các tác vụ sau không? Vui lòng chọn số tương ứng:"]
            for idx, c in enumerate(candidates, start=1):
                desc = c.describe if c.describe else c.intent
                lines.append(f"{idx}. {desc} (độ phù hợp: {int(c.score * 100)}%)")
            message = "\n".join(lines)

        return FallbackResult(
            query=query,
            understood=understood,
            top_candidates=candidates,
            prompt_message=message,
        )
