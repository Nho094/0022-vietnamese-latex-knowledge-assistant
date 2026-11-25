"""
Module so khớp mờ dự phòng từ vựng (Fuzzy String Matching Lexical Fallback Module).

Tệp tin: core/matcher.py
Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Kiến trúc xử lý:
    text -> preprocess (NFC, chuẩn hóa ký hiệu toán học, tách từ pyvi cho PhoBERT)
         -> PhoBERT classifier: p = softmax over intents
            - if max p >= tau: intent accepted
            - else retrieval top-3 over aliases + scenario strings -> ask the user to choose
              (if best score < theta: say not understood)
- Vai trò của mô-đun (WHY):
    Khi điểm tin cậy truy hồi ngữ nghĩa (Sentence-BERT cosine similarity) rơi xuống dưới ngưỡng theta
    (THETA_RETRIEVAL_THRESHOLD = 0.3), hệ thống kích hoạt cơ chế fallback từ vựng nhanh (fast lexical fallback)
    dựa trên thuật toán so khớp mờ (Fuzzy String Matching).
    Cơ chế này đặc biệt hiệu quả khi người dùng nhập câu lệnh chứa lỗi chính tả nhẹ, ký hiệu viết tắt,
    từ ghép toán học chưa có trong kho từ vựng nhúng, hoặc văn phong đàm thoại khẩu ngữ.
- Cơ sở lý thuyết và công thức tham chiếu:
    1. Token Set Ratio (rapidfuzz / fuzzywuzzy):
       Tách chuỗi truy vấn q và mẫu kịch bản s thành các tập hợp từ vựng riêng biệt:
           T_{inter} = S_q \\cap S_s
           T_q = S_q \\setminus S_s
           T_s = S_s \\setminus S_q
       Tạo các chuỗi kết hợp:
           u_0 = T_{inter}
           u_1 = T_{inter} + T_q
           u_2 = T_{inter} + T_s
       Điểm số chuẩn hóa (thang 0.0 đến 1.0):
           score = \\max( \\text{ratio}(u_0, u_1), \\text{ratio}(u_0, u_2), \\text{ratio}(u_1, u_2) ) / 100.0
       Thuật toán này cực kỳ bền vững trước các từ đệm mở đầu câu tiếng Việt (ví dụ: 'tôi muốn tạo',
       'làm thế nào để viết') khi đối chiếu với các alias ngắn gọn của kịch bản.
    2. Deterministic Tie-Breaking (Giải quyết thế hòa tất định):
       Khi nhiều mẫu kịch bản đạt cùng điểm primary score (thường gặp khi alias là tập con hoàn hảo của câu),
       sử dụng tỷ lệ Levenshtein cơ bản (fuzz.ratio * 1e-4) làm trọng số phụ để ưu tiên mẫu có độ tương đồng
       chiều dài và cấu trúc sát nhất với câu truy vấn.
    3. Per-Intent Aggregation (Tổng hợp điểm theo kịch bản):
       Mỗi kịch bản có nhiều mẫu ứng viên (aliases, scenario string, describe, sample utterances).
       Điểm kịch bản là điểm cao nhất trong tất cả các mẫu của kịch bản đó (max pooling).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import (
    DEFAULT_FUZZY_SCORER,
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_FUZZY_TOP_K,
    DEFAULT_INTENTS_PATH,
    SUPPORTED_FUZZY_SCORERS,
    THETA_RETRIEVAL_THRESHOLD,
)
from core.preprocess import normalize
from core.scenario import scenario_string

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz

    _HAS_RAPIDFUZZ: bool = True
except ImportError:  # pragma: no cover
    _HAS_RAPIDFUZZ = False


# =====================================================================
# 1. Cấu trúc dữ liệu mẫu ứng viên và kết quả so khớp
# =====================================================================


@dataclass(frozen=True)
class CandidatePattern:
    """
    Biểu diễn một mẫu hình thức (pattern) của kịch bản cần so khớp từ vựng.

    Thuộc tính:
        intent: Mã định danh kịch bản (ví dụ: 'MATHJAX_PHAN_SO').
        pattern: Chuỗi ký tự của mẫu hình (alias, chuỗi kịch bản, câu mẫu, v.v.).
        pattern_type: Phân loại nguồn gốc của mẫu ('alias', 'scenario_string', 'sample_utterance', 'describe', 'intent_name').
        category: Danh mục của kịch bản.
        describe: Mô tả tác vụ bằng tiếng Việt.
    """

    intent: str
    pattern: str
    pattern_type: str = "alias"
    category: str = ""
    describe: str = ""


@dataclass
class FuzzyMatch:
    """
    Kết quả khớp mờ cho một kịch bản ứng viên.

    Thuộc tính:
        intent: Mã định danh kịch bản.
        score: Điểm tương đồng chuẩn hóa từ 0.0 đến 1.0.
        matched_pattern: Chuỗi mẫu khớp đạt điểm cao nhất của kịch bản này.
        pattern_type: Phân loại của mẫu khớp ('alias', 'scenario_string', v.v.).
        category: Danh mục của kịch bản.
        describe: Mô tả tác vụ tiếng Việt.
    """

    intent: str
    score: float
    matched_pattern: str
    pattern_type: str = "alias"
    category: str = ""
    describe: str = ""


@dataclass
class FuzzyFallbackResult:
    """
    Kết quả tổng hợp từ bước xử lý dự phòng so khớp mờ từ vựng.

    Thuộc tính:
        query: Chuỗi văn bản truy vấn gốc từ người dùng.
        semantic_score: Điểm tin cậy truy hồi ngữ nghĩa ban đầu.
        fallback_triggered: Cờ đánh dấu có kích hoạt fallback từ vựng hay không (True khi semantic_score < theta).
        understood: Hệ thống có hiểu truy vấn hay không (True khi tìm thấy ứng viên có điểm >= threshold).
        matches: Danh sách tối đa top_k ứng viên khớp mờ sắp xếp giảm dần theo điểm.
        prompt_message: Thông điệp phản hồi người dùng bằng tiếng Việt.
    """

    query: str
    semantic_score: float
    fallback_triggered: bool
    understood: bool
    matches: list[FuzzyMatch] = field(default_factory=list)
    prompt_message: str = ""

    @property
    def top_candidate(self) -> FuzzyMatch | None:
        """Ứng viên đứng đầu danh sách nếu có."""
        return self.matches[0] if self.matches else None

    @property
    def top_intent(self) -> str | None:
        """Mã ý định đứng đầu danh sách nếu có."""
        return self.matches[0].intent if self.matches else None

    @property
    def top_score(self) -> float:
        """Điểm số của ứng viên đứng đầu (mặc định 0.0 nếu rỗng)."""
        return self.matches[0].score if self.matches else 0.0


# =====================================================================
# 2. Hàm tính điểm tương đồng mờ (Scoring Helpers)
# =====================================================================


def compute_fuzzy_score(
    text1: str,
    text2: str,
    scorer: str = DEFAULT_FUZZY_SCORER,
) -> float:
    """
    Tính toán điểm tương đồng mờ giữa hai chuỗi văn bản, chuẩn hóa về đoạn [0.0, 1.0].

    Lý do chọn Token Set Ratio (WHY):
    Đối với câu hỏi đàm thoại tiếng Việt, người dùng thường thêm các từ ngữ đệm mở đầu
    như 'cho tôi hỏi cách tạo', 'làm sao để chèn', 'hướng dẫn tạo'...
    Token Set Ratio tách hai chuỗi thành tập hợp các từ (tokens), tìm phần giao và tính tỷ lệ
    Levenshtein trên các tập hợp, giúp bỏ qua các từ đệm dư thừa mà vẫn nhận diện chính xác
    các từ khóa cốt lõi của tác vụ.

    Tham số:
        text1: Chuỗi văn bản thứ nhất (đã chuẩn hóa).
        text2: Chuỗi văn bản thứ hai (đã chuẩn hóa).
        scorer: Thuật toán so khớp ('token_set_ratio', 'token_sort_ratio', 'partial_ratio', 'ratio').

    Trả về:
        Điểm tương đồng số thực trong đoạn [0.0, 1.0].
    """
    if not text1 or not text2:
        return 0.0

    if _HAS_RAPIDFUZZ:
        if scorer == "token_set_ratio":
            primary = float(fuzz.token_set_ratio(text1, text2))
        elif scorer == "token_sort_ratio":
            primary = float(fuzz.token_sort_ratio(text1, text2))
        elif scorer == "partial_ratio":
            primary = float(fuzz.partial_ratio(text1, text2))
        elif scorer == "ratio":
            primary = float(fuzz.ratio(text1, text2))
        else:
            raise ValueError(
                f"Thuật toán scorer không được hỗ trợ: '{scorer}'. "
                f"Các scorer hợp lệ: {list(SUPPORTED_FUZZY_SCORERS)}"
            )
        # rapidfuzz trả về thang điểm [0, 100], chuẩn hóa về [0.0, 1.0]
        return max(0.0, min(1.0, primary / 100.0))

    # shortcut: pure Python fallback khi môi trường thiếu rapidfuzz, tính Jaccard trên tập token, nâng cấp khi cài đặt rapidfuzz
    tokens1 = set(text1.split())
    tokens2 = set(text2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return float(intersection / union) if union > 0 else 0.0


# =====================================================================
# 3. Lớp quản lý so khớp mờ dự phòng (FuzzyMatcher)
# =====================================================================


class FuzzyMatcher:
    """
    Bộ so khớp chuỗi mờ (Fuzzy String Matcher) phục vụ cơ chế dự phòng từ vựng nhanh
    khi độ tin cậy truy hồi ngữ nghĩa dưới ngưỡng theta.

    Các tính năng chính:
    1. Lập chỉ mục đa mẫu kịch bản: nạp từ intents.json gồm bí danh (aliases),
       chuỗi kịch bản chuẩn (scenario strings), câu mẫu (sample utterances), và mô tả.
    2. So khớp nhanh dựa trên RapidFuzz Token Set Ratio.
    3. Giải quyết thế hòa tất định (Deterministic tie-breaking).
    4. Tổng hợp điểm theo kịch bản (Per-intent max pooling).
    5. Cơ chế fallback kích hoạt có điều kiện khi semantic_confidence < theta.
    6. Hỗ trợ giao diện phân loại cơ sở (baseline prediction) predict / predict_single.
    """

    def __init__(
        self,
        intents_path: Path | str | None = None,
        patterns: Sequence[CandidatePattern] | None = None,
        threshold: float = DEFAULT_FUZZY_THRESHOLD,
        theta: float = THETA_RETRIEVAL_THRESHOLD,
        top_k: int = DEFAULT_FUZZY_TOP_K,
        scorer: str = DEFAULT_FUZZY_SCORER,
    ) -> None:
        """
        Khởi tạo bộ so khớp mờ.

        Tham số:
            intents_path: Đường dẫn tệp cấu hình intents.json (nếu None sẽ dùng DEFAULT_INTENTS_PATH).
            patterns: Danh sách mẫu ứng viên tùy biến (nếu cung cấp sẽ dùng trực tiếp).
            threshold: Ngưỡng điểm số mờ tối thiểu để chấp nhận kết quả (mặc định 0.5).
            theta: Ngưỡng truy hồi ngữ nghĩa kích hoạt fallback (mặc định 0.3).
            top_k: Số lượng ứng viên trả về mặc định (mặc định 3).
            scorer: Thuật toán so khớp mờ (mặc định 'token_set_ratio').
        """
        self.threshold = threshold
        self.theta = theta
        self.top_k = top_k
        self.scorer = scorer

        if self.scorer not in SUPPORTED_FUZZY_SCORERS:
            raise ValueError(
                f"Thuật toán scorer không được hỗ trợ: '{self.scorer}'. "
                f"Các scorer hợp lệ: {list(SUPPORTED_FUZZY_SCORERS)}"
            )

        self._patterns: list[CandidatePattern] = []

        if patterns is not None:
            self._patterns.extend(patterns)
        else:
            target_path = Path(intents_path) if intents_path is not None else DEFAULT_INTENTS_PATH
            if target_path.exists():
                self.load_intents_from_json(target_path)

    @property
    def patterns(self) -> list[CandidatePattern]:
        """Danh sách toàn bộ các mẫu hình thức đã được lập chỉ mục."""
        return list(self._patterns)

    @property
    def total_patterns(self) -> int:
        """Tổng số lượng mẫu hình thức được lập chỉ mục."""
        return len(self._patterns)

    def add_pattern(
        self,
        intent: str,
        pattern: str,
        pattern_type: str = "custom",
        category: str = "",
        describe: str = "",
    ) -> None:
        """
        Bổ sung một mẫu ứng viên vào ngân hàng so khớp mờ.

        Tham số:
            intent: Mã định danh kịch bản.
            pattern: Chuỗi mẫu cần khớp.
            pattern_type: Phân loại mẫu ('alias', 'scenario_string', 'custom', v.v.).
            category: Danh mục kịch bản.
            describe: Mô tả tác vụ tiếng Việt.
        """
        norm_p = normalize(pattern, lowercase=True)
        if norm_p:
            self._patterns.append(
                CandidatePattern(
                    intent=intent.strip(),
                    pattern=norm_p,
                    pattern_type=pattern_type,
                    category=category.strip(),
                    describe=describe.strip(),
                )
            )

    def load_intents_from_json(self, intents_path: Path | str) -> int:
        """
        Nạp dữ liệu kịch bản từ tệp intents.json và trích xuất toàn diện các mẫu ứng viên:
        - Bí danh (aliases)
        - Chuỗi kịch bản tự nhiên (scenario_string theo biến thể 'handmade')
        - Câu đàm thoại mẫu (sample_utterances nếu có)
        - Mô tả tiếng Việt (describe)
        - Tên kịch bản chuẩn hóa (intent_name)

        Tham số:
            intents_path: Đường dẫn tệp JSON kịch bản.

        Trả về:
            Số lượng mẫu đã được lập chỉ mục thành công.
        """
        path = Path(intents_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp ngân hàng kịch bản: {path}")

        with open(path, mode="r", encoding="utf-8") as f:
            intents_data: list[dict[str, Any]] = json.load(f)

        count_before = len(self._patterns)

        for item in intents_data:
            intent = str(item.get("intent", "")).strip()
            category = str(item.get("category", "")).strip()
            describe = str(item.get("describe", "")).strip()
            aliases = item.get("aliases", [])
            sample_utterances = item.get("sample_utterances", [])

            if not intent:
                continue

            # 1. Lập chỉ mục các bí danh (aliases)
            for alias in aliases:
                self.add_pattern(
                    intent=intent,
                    pattern=str(alias),
                    pattern_type="alias",
                    category=category,
                    describe=describe,
                )

            # 2. Lập chỉ mục chuỗi kịch bản tự nhiên (scenario_string 'handmade')
            try:
                sc_str = scenario_string(item, variant="handmade")
                if sc_str:
                    self.add_pattern(
                        intent=intent,
                        pattern=sc_str,
                        pattern_type="scenario_string",
                        category=category,
                        describe=describe,
                    )
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                # shortcut: bỏ qua khi item không đủ cấu trúc kịch bản RAF, nâng cấp khi toàn bộ intents có schema đầy đủ
                logger.debug("Bỏ qua lỗi tạo chuỗi kịch bản RAF cho %s: %s", intent, exc)

            # 3. Lập chỉ mục các câu đàm thoại mẫu (sample_utterances)
            for utt in sample_utterances:
                self.add_pattern(
                    intent=intent,
                    pattern=str(utt),
                    pattern_type="sample_utterance",
                    category=category,
                    describe=describe,
                )

            # 4. Lập chỉ mục chuỗi mô tả (describe)
            if describe:
                self.add_pattern(
                    intent=intent,
                    pattern=describe,
                    pattern_type="describe",
                    category=category,
                    describe=describe,
                )

            # 5. Lập chỉ mục tên ý định dạng từ tự nhiên (intent_name)
            readable_intent = intent.lower().replace("_", " ")
            self.add_pattern(
                intent=intent,
                pattern=readable_intent,
                pattern_type="intent_name",
                category=category,
                describe=describe,
            )

        return len(self._patterns) - count_before

    def match(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        per_intent: bool = True,
    ) -> list[FuzzyMatch]:
        """
        So khớp câu truy vấn đầu vào với toàn bộ các mẫu hình thức đã lập chỉ mục.

        Quy trình xử lý:
        1. Tiền xử lý truy vấn: Chuẩn hóa NFC, chuyển chữ thường và ký hiệu toán học qua core.preprocess.normalize.
        2. Tính điểm tương đồng mờ đối với từng mẫu bằng compute_fuzzy_score.
        3. Phá vỡ thế hòa tất định (Deterministic Tie-Breaking):
           Khi điểm chính bằng nhau, dùng Levenshtein ratio cơ bản làm điểm phụ (secondary score: 1e-4 * ratio)
           và sắp xếp ổn định theo tên intent.
        4. Tổng hợp điểm theo kịch bản (per_intent=True):
           Nếu một kịch bản có nhiều mẫu khớp, chỉ giữ lại mẫu có điểm số cao nhất (max pooling).
        5. Lọc theo ngưỡng threshold và lấy top_k ứng viên có điểm cao nhất.

        Tham số:
            query: Chuỗi văn bản truy vấn từ người dùng.
            top_k: Số lượng kết quả tối đa cần lấy (mặc định lấy self.top_k).
            threshold: Ngưỡng điểm tối thiểu (mặc định lấy self.threshold).
            per_intent: Gom nhóm kết quả theo intent (mỗi intent chỉ xuất hiện 1 lần).

        Trả về:
            Danh sách FuzzyMatch sắp xếp giảm dần theo điểm số.
        """
        k = top_k if top_k is not None else self.top_k
        min_thresh = threshold if threshold is not None else self.threshold

        norm_query = normalize(query, lowercase=True)
        if not norm_query or not self._patterns:
            return []

        # Lưu trữ điểm số thô: (primary_score, secondary_score, pattern_record)
        scored_candidates: list[tuple[float, float, CandidatePattern]] = []

        for pat in self._patterns:
            primary_score = compute_fuzzy_score(norm_query, pat.pattern, scorer=self.scorer)
            if primary_score < min_thresh:
                continue

            # Điểm phụ Levenshtein ratio giúp phân biệt khi token_set_ratio bằng nhau
            if _HAS_RAPIDFUZZ:
                secondary_score = float(fuzz.ratio(norm_query, pat.pattern)) / 100.0
            else:
                secondary_score = 0.0

            scored_candidates.append((primary_score, secondary_score, pat))

        if not scored_candidates:
            return []

        # Sắp xếp giảm dần theo primary_score, sau đó secondary_score, sau đó thứ tự chuỗi intent
        scored_candidates.sort(
            key=lambda x: (x[0], x[1], -len(x[2].pattern), x[2].intent),
            reverse=True,
        )

        if not per_intent:
            results: list[FuzzyMatch] = []
            for pri_score, _, pat in scored_candidates[:k]:
                results.append(
                    FuzzyMatch(
                        intent=pat.intent,
                        score=round(pri_score, 4),
                        matched_pattern=pat.pattern,
                        pattern_type=pat.pattern_type,
                        category=pat.category,
                        describe=pat.describe,
                    )
                )
            return results

        # Gom nhóm theo kịch bản (Per-Intent Max Pooling)
        seen_intents: dict[str, FuzzyMatch] = {}
        for pri_score, _, pat in scored_candidates:
            if pat.intent not in seen_intents:
                seen_intents[pat.intent] = FuzzyMatch(
                    intent=pat.intent,
                    score=round(pri_score, 4),
                    matched_pattern=pat.pattern,
                    pattern_type=pat.pattern_type,
                    category=pat.category,
                    describe=pat.describe,
                )
            if len(seen_intents) >= k:
                break

        return list(seen_intents.values())

    def best_match(
        self,
        query: str,
        threshold: float | None = None,
    ) -> FuzzyMatch | None:
        """
        Tìm kiếm kịch bản khớp mờ tốt nhất cho một câu truy vấn.

        Tham số:
            query: Chuỗi văn bản truy vấn.
            threshold: Ngưỡng điểm tối thiểu.

        Trả về:
            Đối tượng FuzzyMatch tốt nhất hoặc None nếu không có mẫu nào vượt qua ngưỡng.
        """
        matches = self.match(query, top_k=1, threshold=threshold, per_intent=True)
        return matches[0] if matches else None

    def fallback_match(
        self,
        query: str,
        semantic_score: float,
        theta: float | None = None,
        top_k: int | None = None,
    ) -> FuzzyFallbackResult:
        """
        Thực hiện so khớp mờ dự phòng khi độ tin cậy truy hồi ngữ nghĩa dưới ngưỡng theta.

        Quy tắc xử lý kiến trúc:
        - Nếu semantic_score >= theta:
          Độ tin cậy truy hồi ngữ nghĩa đã đủ cao, không kích hoạt fallback từ vựng.
          Trả về fallback_triggered=False, understood=True.
        - Nếu semantic_score < theta:
          Kích hoạt fallback từ vựng (fallback_triggered=True).
          Tìm kiếm các kịch bản tương đồng thông qua so khớp mờ.
          + Nếu tìm thấy ít nhất 1 ứng viên đạt score >= threshold:
            understood=True, prompt_message gợi ý người dùng lựa chọn ('ask the user to choose').
          + Nếu không tìm thấy ứng viên nào đạt threshold:
            understood=False, prompt_message thông báo không hiểu ('say not understood').

        Tham số:
            query: Chuỗi câu hỏi hoặc yêu cầu từ người dùng.
            semantic_score: Điểm cosine similarity từ mô hình SBERT hoặc phân loại.
            theta: Ngưỡng kích hoạt fallback (mặc định lấy self.theta = 0.3).
            top_k: Số lượng ứng viên trả về (mặc định lấy self.top_k = 3).

        Trả về:
            Đối tượng FuzzyFallbackResult chứa đầy đủ thông tin xử lý.
        """
        effective_theta = theta if theta is not None else self.theta
        effective_k = top_k if top_k is not None else self.top_k

        # 1. Trường hợp truy hồi ngữ nghĩa đã tự tin (semantic_score >= theta)
        if semantic_score >= effective_theta:
            return FuzzyFallbackResult(
                query=query,
                semantic_score=semantic_score,
                fallback_triggered=False,
                understood=True,
                matches=[],
                prompt_message="Truy hồi ngữ nghĩa tự tin (semantic retrieval confident).",
            )

        # 2. Kích hoạt bước so khớp mờ từ vựng fallback
        matches = self.match(query, top_k=effective_k, threshold=self.threshold, per_intent=True)

        if not matches:
            return FuzzyFallbackResult(
                query=query,
                semantic_score=semantic_score,
                fallback_triggered=True,
                understood=False,
                matches=[],
                prompt_message="Xin lỗi, tôi không hiểu yêu cầu này.",
            )

        # Định dạng thông điệp gợi ý người dùng lựa chọn theo AGENTS.md ('ask the user to choose')
        cand_descriptions = []
        for i, m in enumerate(matches, start=1):
            desc = m.describe if m.describe else m.intent
            cand_descriptions.append(f"{i}. {desc} (khớp từ khóa: '{m.matched_pattern}')")
        choices_text = "; ".join(cand_descriptions)
        prompt_message = (
            f"Có phải bạn muốn thực hiện một trong các thao tác sau: {choices_text}? "
            f"Vui lòng chọn hoặc làm rõ yêu cầu."
        )

        return FuzzyFallbackResult(
            query=query,
            semantic_score=semantic_score,
            fallback_triggered=True,
            understood=True,
            matches=matches,
            prompt_message=prompt_message,
        )

    def predict_single(self, text: str, target: str = "intent") -> str:
        """
        Dự đoán nhãn (intent hoặc category) cho một câu truy vấn theo giao diện baseline classifier.

        Tham số:
            text: Chuỗi câu văn bản tiếng Việt.
            target: Mục tiêu phân loại ('intent' hoặc 'category').

        Trả về:
            Tên nhãn dự đoán tốt nhất hoặc chuỗi rỗng nếu không tìm thấy.
        """
        match = self.best_match(text, threshold=0.0)
        if match is None:
            return ""
        return match.intent if target == "intent" else match.category

    def predict(self, texts: Sequence[str], target: str = "intent") -> list[str]:
        """
        Dự đoán hàng loạt cho danh sách câu văn bản.

        Tham số:
            texts: Danh sách văn bản truy vấn.
            target: 'intent' hoặc 'category'.

        Trả về:
            Danh sách nhãn dự đoán tương ứng.
        """
        return [self.predict_single(t, target=target) for t in texts]


# =====================================================================
# 4. Hàm tiện ích so khớp nhanh (Standalone Utility)
# =====================================================================


def fuzzy_match(
    query: str,
    candidates: Sequence[str | tuple[str, str] | CandidatePattern],
    top_k: int = DEFAULT_FUZZY_TOP_K,
    threshold: float = 0.0,
    scorer: str = DEFAULT_FUZZY_SCORER,
) -> list[FuzzyMatch]:
    """
    Hàm tiện ích so khớp mờ nhanh cho danh sách ứng viên tùy ý mà không cần khởi tạo lớp.

    Tham số:
        query: Chuỗi văn bản truy vấn.
        candidates: Danh sách ứng viên, có thể là:
            - Danh sách chuỗi ký tự (str)
            - Danh sách tuple (intent, pattern)
            - Danh sách CandidatePattern
        top_k: Số lượng ứng viên trả về tối đa.
        threshold: Ngưỡng điểm tối thiểu.
        scorer: Thuật toán so khớp.

    Trả về:
        Danh sách FuzzyMatch sắp xếp giảm dần theo điểm số.
    """
    patterns: list[CandidatePattern] = []
    for c in candidates:
        if isinstance(c, CandidatePattern):
            patterns.append(c)
        elif isinstance(c, tuple) and len(c) >= 2:
            patterns.append(CandidatePattern(intent=str(c[0]), pattern=str(c[1])))
        elif isinstance(c, str):
            patterns.append(CandidatePattern(intent=c, pattern=c))

    matcher = FuzzyMatcher(patterns=patterns, threshold=threshold, top_k=top_k, scorer=scorer)
    return matcher.match(query, top_k=top_k, threshold=threshold, per_intent=False)
