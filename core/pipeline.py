"""
Module điều phối đường ống suy luận và định tuyến trung tâm (Inference Pipeline and Routing Orchestrator).

Tệp tin: core/pipeline.py
Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Kiến trúc luồng xử lý trung tâm:
    text -> preprocess (NFC, chuẩn hóa ký hiệu toán học, tách từ pyvi cho PhoBERT)
      -> PhoBERT classifier: p = softmax over intents (hoặc over categories trong chế độ 2 tầng - two-tier)
         - if max p >= tau: intent accepted
         - else retrieval top-3 over aliases + scenario strings -> ask the user to choose (if best score < theta: say not understood)
      -> slot extraction (regex per intent) -> if a required slot is missing: ask for exactly the missing slots
      -> render latex template (text slots escaped) -> output
- Quy tắc trung thực (Honesty Rule #1):
    Dự án là một hệ thống kịch bản đơn giản hóa lấy cảm hứng từ Retrieve-and-Fill (RAF, Shrivastava et al., EACL 2022).
    Bước trích xuất tham số dựa trên biểu thức chính quy (regex per intent) thay vì span-pointer decoder nguyên bản của RAF.
- Quy tắc an toàn (Security Rules):
    Không bao giờ biên dịch mã ngoài ngân hàng mẫu; luôn kiểm tra cú pháp và an toàn tĩnh (static check)
    với danh sách từ cấm bảo mật (input, include, write, openout, v.v.).
- Năm trạng thái dự đoán tường minh (Five Explicit Statuses):
    1. 'ok': Đã xác định kịch bản, trích xuất đủ tham số bắt buộc và sinh mã LaTeX thành công.
    2. 'need_choice': Độ tin cậy dưới ngưỡng hoặc có nhập nhằng, trả về danh sách ứng viên yêu cầu người dùng chọn.
    3. 'need_slots': Đã xác định kịch bản nhưng thiếu một hoặc nhiều tham số bắt buộc.
    4. 'out_of_scope': Câu truy vấn được xác định nằm ngoài phạm vi hỗ trợ (OUT_OF_SCOPE hoặc tex_misc).
    5. 'unknown': Không nhận diện được ý định (confidence < tau và fallback retrieval/fuzzy < theta).
- Các chế độ cắt bỏ (Ablation Modes):
    + 'phobert_only': Chỉ phân loại qua PhoBERT, không sử dụng truy hồi hoặc khớp mờ.
    + 'phobert_retrieval': Phân loại PhoBERT kết hợp truy hồi ngữ nghĩa SBERT fallback.
    + 'phobert_retrieval_fuzzy': Đường ống đầy đủ: PhoBERT -> SBERT retrieval -> RapidFuzz matching.
    + 'retrieval_only': Chỉ sử dụng truy hồi ngữ nghĩa SBERT trực tiếp.
    + 'tfidf_only': Chỉ sử dụng mô hình xếp hạng tương đồng từ vựng TF-IDF n-gram.
- Định tuyến phân cấp hai tầng (Two-Tier Hierarchical Routing):
    + Tầng 1: Dự đoán nhóm danh mục ngữ nghĩa (Category) của truy vấn.
    + Tầng 2: Giới hạn không gian tìm kiếm và phân giải ý định (Intent) bên trong danh mục đã chọn.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

# Đảm bảo đường dẫn gốc dự án có trong sys.path khi chạy module trực tiếp
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_INTENTS_PATH,
    DEFAULT_OUT_OF_SCOPE_CATEGORY,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_THRESHOLDS_PATH,
    INDEX_CACHE_DIR,
    MODE_PHOBERT_ONLY,
    MODE_PHOBERT_RETRIEVAL,
    MODE_PHOBERT_RETRIEVAL_FUZZY,
    MODE_RETRIEVAL_ONLY,
    MODE_TFIDF_ONLY,
    OUT_OF_SCOPE_INTENT,
    STATUS_NEED_CHOICE,
    STATUS_NEED_SLOTS,
    STATUS_OK,
    STATUS_OUT_OF_SCOPE,
    STATUS_UNKNOWN,
    STATUS_UNSUPPORTED_TARGET,
    STATUS_PROCESSING_ERROR,
    ENGINE_VERSION_GENERIC,
    ENGINE_VERSION_LEGACY,
    SUPPORTED_ENGINE_VERSIONS,
    SUPPORTED_TARGETS,
    SUPPORTED_PIPELINE_MODES,
    TARGET_CANDIDATE_OVERSAMPLE_FACTOR,
    TAU_INTENT_THRESHOLD,
    THETA_RETRIEVAL_THRESHOLD,
    checkpoint_is_smoke,
    checkpoint_sha256,
    classifier_artifact_status,
    load_calibrated_thresholds,
    resolve_sbert_model_dir,
    retrieval_model_status,
    SBERT_MODEL_NAME,
)
from core.latex_tools import POST_ACTIONS, escape_text, resolve_template, static_check, wrap_snippet
from core.preprocess import canonicalize_retrieval_numbers, normalize, preprocess_fingerprint
from core.scenario import scenario_string
from core.slots import extract_slots, normalize_slot_value

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Định nghĩa Kiểu Liệt Kê và Hằng số (Enums and Constants)
# =====================================================================


class PipelineStatus(str, Enum):
    """Trạng thái kết quả pipeline, gồm trường hợp target không được hỗ trợ."""

    OK = STATUS_OK
    NEED_CHOICE = STATUS_NEED_CHOICE
    NEED_SLOTS = STATUS_NEED_SLOTS
    OUT_OF_SCOPE = STATUS_OUT_OF_SCOPE
    UNKNOWN = STATUS_UNKNOWN
    UNSUPPORTED_TARGET = STATUS_UNSUPPORTED_TARGET
    PROCESSING_ERROR = STATUS_PROCESSING_ERROR


class PipelineMode(str, Enum):
    """Các chế độ thực thi và cắt bỏ (ablation modes) được hỗ trợ."""

    PHOBERT_ONLY = MODE_PHOBERT_ONLY
    PHOBERT_RETRIEVAL = MODE_PHOBERT_RETRIEVAL
    PHOBERT_RETRIEVAL_FUZZY = MODE_PHOBERT_RETRIEVAL_FUZZY
    RETRIEVAL_ONLY = MODE_RETRIEVAL_ONLY
    TFIDF_ONLY = MODE_TFIDF_ONLY


# =====================================================================
# 2. Cấu trúc Dữ liệu Kết quả Đường ống (Pipeline Data Structures)
# =====================================================================


@dataclass
class CandidateChoice:
    """
    Thông tin kịch bản ứng viên được đề xuất trong trường hợp nhập nhằng hoặc truy hồi fallback.

    Thuộc tính:
        intent: Mã định danh kịch bản (ví dụ: 'LATEX_NAP_PACKAGE').
        score: Điểm tin cậy tương đồng (chuẩn hóa 0.0 - 1.0).
        describe: Mô tả ngắn gọn tác vụ kịch bản bằng tiếng Việt.
        category: Danh mục ngữ nghĩa của kịch bản.
        source: Nguồn gốc gợi ý ('phobert', 'retrieval', 'fuzzy', 'tfidf').
    """

    intent: str
    score: float
    describe: str = ""
    category: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi thành từ điển chuẩn hóa."""
        return {
            "intent": self.intent,
            "score": round(float(self.score), 4),
            "describe": self.describe,
            "category": self.category,
            "source": self.source,
        }


@dataclass
class PipelineResult:
    """
    Kết quả suy luận toàn diện từ đường ống Pipeline.

    Thuộc tính:
        status: Trạng thái tường minh ('ok', 'need_choice', 'need_slots', 'out_of_scope', 'unknown').
        query: Chuỗi văn bản truy vấn gốc của người dùng.
        intent: Mã ý định được phân giải (None nếu chưa xác định hoặc nhập nhằng).
        category: Danh mục ngữ nghĩa tương ứng (nếu có).
        confidence: Điểm tin cậy tối đa của bước phân giải (0.0 - 1.0).
        mode: Chế độ cắt bỏ đường ống đã sử dụng.
        two_tier: Cờ định tuyến phân cấp hai tầng (True nếu kích hoạt).
        slots: Bảng ánh xạ các tham số đã trích xuất {tên_slot: giá_trị}.
        missing_slots: Danh sách tên các tham số bắt buộc còn thiếu.
        slot_errors: Danh sách thông báo lỗi kiểm thực tham số (nếu có).
        latex_code: Đoạn mã LaTeX hoàn chỉnh được sinh từ template.
        candidates: Danh sách các ứng viên gợi ý khi trạng thái là 'need_choice'.
        message: Thông điệp giải thích kết quả bằng tiếng Việt.
        static_check_passed: Kết quả kiểm tra tĩnh cú pháp và an toàn mã LaTeX.
        execution_time_ms: Thời gian thực thi đo bằng mili-giây.
    """

    status: str
    query: str
    intent: str | None = None
    category: str | None = None
    confidence: float = 0.0
    mode: str = PipelineMode.PHOBERT_RETRIEVAL_FUZZY.value
    two_tier: bool = False
    slots: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    slot_errors: list[str] = field(default_factory=list)
    latex_code: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""
    static_check_passed: bool = False
    execution_time_ms: float = 0.0
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Xuất kết quả thành từ điển Python có cấu trúc chuẩn."""
        result = {
            "status": self.status,
            "query": self.query,
            "intent": self.intent,
            "category": self.category,
            "confidence": round(float(self.confidence), 4),
            "mode": self.mode,
            "two_tier": self.two_tier,
            "slots": self.slots,
            "missing_slots": self.missing_slots,
            "slot_errors": self.slot_errors,
            "latex_code": self.latex_code,
            "candidates": self.candidates,
            "message": self.message,
            "static_check_passed": self.static_check_passed,
            "execution_time_ms": round(float(self.execution_time_ms), 2),
        }
        # Trường mới chỉ xuất hiện ở generic engine để giữ nguyên JSON legacy.
        if self.stages:
            result["stages"] = self.stages
        if self.outputs:
            result["outputs"] = self.outputs
        return result

    def to_json(self, indent: int = 2) -> str:
        """Xuất kết quả dạng chuỗi JSON có định dạng đẹp phục vụ CLI và API."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# =====================================================================
# 3. Tiện ích Sinh Mã LaTeX và Nội suy Tham số (Template Rendering)
# =====================================================================


def render_latex_template(
    intent_data: dict[str, Any],
    slots: dict[str, str],
    target: str = "latex",
    escape_text_slots: bool = True,
    wrap_document: bool = False,
) -> tuple[str, bool, list[str]]:
    """
    Nội suy các giá trị tham số vào template kịch bản và thực hiện kiểm tra an toàn tĩnh.

    Tuân thủ quy tắc bảo mật AGENTS.md:
    - Thoát các ký tự đặc biệt LaTeX cho các tham số dạng văn bản (text slots) bằng escape_text.
    - Không thoát ký tự đối với tham số toán học hoặc số nguyên/thực.
    - Thay thế linh hoạt các placeholder dạng <<slot>>, {{slot}}, {slot}.
    - Kiểm tra an toàn tĩnh (static check): chặn các từ điều khiển cấm và kiểm tra cân bằng ngoặc.

    Tham số:
        intent_data: Dữ liệu kịch bản từ ngân hàng kịch bản intents.json.
        slots: Bảng ánh xạ tham số đã trích xuất và chuẩn hóa.
        target: Ngôn ngữ đích cần sinh mã (mặc định 'latex').
        escape_text_slots: Có áp dụng thoát ký tự an toàn cho slot text hay không.
        wrap_document: Tùy chọn đóng gói thành tài liệu LaTeX hoàn chỉnh.

    Trả về:
        Tuple gồm: (mã kết quả, cờ kiểm tra tĩnh hợp lệ, danh sách cảnh báo kiểm tra tĩnh).
    """
    template_str = resolve_template(intent_data, target)
    if template_str is None:
        return "", False, [f"Intent không hỗ trợ target '{target}'."]

    slots_def = intent_data.get("slots", {})

    # Tổng hợp giá trị mặc định cho các slot không được cung cấp
    merged_slots: dict[str, str] = {}
    for s_name, s_def in slots_def.items():
        if isinstance(s_def, dict):
            default_val = s_def.get("default")
        else:
            default_val = getattr(s_def, "default", None)
        if default_val is not None:
            merged_slots[s_name] = str(default_val)

    merged_slots.update(slots)

    rendered = str(template_str)

    # Thay thế các placeholder
    for key, val in merged_slots.items():
        sub_val = str(val)

        # Xác định loại slot để thực hiện thoát ký tự an toàn
        s_def = slots_def.get(key)
        s_type = "text"
        if isinstance(s_def, dict):
            s_type = str(s_def.get("type", "text"))
        elif s_def is not None:
            s_type = str(getattr(s_def, "type", "text"))

        if escape_text_slots and s_type == "text" and target in {"latex", "mathjax"}:
            sub_val = escape_text(sub_val)

        rendered = rendered.replace(f"<<{key}>>", sub_val)
        if target in {"latex", "mathjax"}:
            rendered = rendered.replace(f"{{{{{key}}}}}", sub_val)
            rendered = rendered.replace(f"{{{key}}}", sub_val)

    # Đóng gói tài liệu nếu được yêu cầu
    if wrap_document:
        kind = str(intent_data.get("kind", "snippet"))
        requires = intent_data.get("requires", [])
        rendered = wrap_snippet(rendered, requires=requires, kind=kind)

    post_name = intent_data.get("post")
    if post_name:
        action = POST_ACTIONS.get(str(post_name))
        if action is None:
            return rendered, False, [f"Post action '{post_name}' chưa được đăng ký."]
        rendered = action(rendered)

    unresolved = re.findall(r"<<[a-zA-Z][a-zA-Z0-9_]*>>", rendered)
    if unresolved:
        return rendered, False, [f"Còn placeholder chưa được điền: {sorted(set(unresolved))}."]

    # Kiểm tra tĩnh chỉ có ý nghĩa cho LaTeX/MathJax, không áp lên code target khác.
    check_errors = static_check(rendered) if target in {"latex", "mathjax"} else []
    check_ok = len(check_errors) == 0

    return rendered, check_ok, check_errors


# =====================================================================
# 4. Bộ Xếp hạng Đường cơ sở TF-IDF (TfidfRanker Baseline Component)
# =====================================================================


class TfidfRanker:
    """
    Bộ xếp hạng đường cơ sở dựa trên ma trận TF-IDF n-gram và độ tương đồng Cosine.
    Cung cấp khả năng xếp hạng ngoại tuyến siêu nhẹ phục vụ chế độ ablation 'tfidf_only'.
    """

    def __init__(
        self,
        intents_bank: dict[str, dict[str, Any]],
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        """
        Khởi tạo và lập chỉ mục kho ngữ liệu từ ngân hàng kịch bản.

        Tham số:
            intents_bank: Ngân hàng kịch bản {tên_intent: dữ_liệu_kịch_bản}.
            ngram_range: Khoảng n-gram cho TF-IDF (mặc định (1, 2)).
        """
        self.intents_bank = intents_bank
        self.corpus_texts: list[str] = []
        self.corpus_intents: list[str] = []
        self.corpus_categories: list[str] = []
        self.corpus_describes: list[str] = []

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._cosine_similarity = cosine_similarity

        # Lập chỉ mục các biểu diễn kịch bản
        for intent_name, data in intents_bank.items():
            category = str(data.get("category", "")).strip()
            describe = str(data.get("describe", "")).strip()

            # 1. Alias kịch bản
            for alias in data.get("aliases", []):
                alias_clean = normalize(str(alias), lowercase=True)
                if alias_clean:
                    self.corpus_texts.append(alias_clean)
                    self.corpus_intents.append(intent_name)
                    self.corpus_categories.append(category)
                    self.corpus_describes.append(describe)

            # 2. Chuỗi mô tả
            if describe:
                desc_clean = normalize(describe, lowercase=True)
                self.corpus_texts.append(desc_clean)
                self.corpus_intents.append(intent_name)
                self.corpus_categories.append(category)
                self.corpus_describes.append(describe)

            # 3. Chuỗi biến thể kịch bản
            s_str = scenario_string(data, variant="handmade")
            if s_str:
                s_clean = normalize(s_str, lowercase=True)
                self.corpus_texts.append(s_clean)
                self.corpus_intents.append(intent_name)
                self.corpus_categories.append(category)
                self.corpus_describes.append(describe)

        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, sublinear_tf=True)
        if self.corpus_texts:
            self.matrix = self.vectorizer.fit_transform(self.corpus_texts)
        else:
            self.matrix = None

    def rank(
        self,
        query: str,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        category: str | None = None,
    ) -> list[CandidateChoice]:
        """
        Tính điểm tương đồng TF-IDF và gom nhóm điểm tối đa (max pooling) theo từng intent.

        Tham số:
            query: Chuỗi văn bản truy vấn người dùng.
            top_k: Số lượng ứng viên trả về tối đa.
            category: Tùy chọn lọc chỉ lấy các kịch bản trong danh mục này (phục vụ two-tier).

        Trả về:
            Danh sách CandidateChoice sắp xếp giảm dần theo điểm tương đồng.
        """
        norm_q = canonicalize_retrieval_numbers(query)
        if not norm_q or self.matrix is None:
            return []

        q_vec = self.vectorizer.transform([norm_q])
        sims = self._cosine_similarity(q_vec, self.matrix)[0]

        intent_best_score: dict[str, float] = {}
        intent_category: dict[str, str] = {}
        intent_describe: dict[str, str] = {}

        for idx, score in enumerate(sims):
            val = float(score)
            intent = self.corpus_intents[idx]
            cat = self.corpus_categories[idx]
            desc = self.corpus_describes[idx]

            # Lọc theo category nếu chế độ two-tier yêu cầu
            if category and cat != category:
                continue

            if intent not in intent_best_score or val > intent_best_score[intent]:
                intent_best_score[intent] = val
                intent_category[intent] = cat
                intent_describe[intent] = desc

        sorted_items = sorted(intent_best_score.items(), key=lambda x: x[1], reverse=True)
        results: list[CandidateChoice] = []
        for it_name, score in sorted_items[:top_k]:
            results.append(
                CandidateChoice(
                    intent=it_name,
                    score=score,
                    describe=intent_describe.get(it_name, ""),
                    category=intent_category.get(it_name, ""),
                    source="tfidf",
                )
            )

        return results


# =====================================================================
# 5. Lớp Điều phối Đường ống Chính (Pipeline Class)
# =====================================================================


class Pipeline:
    """
    Bộ điều phối đường ống suy luận trung tâm (Central Inference Pipeline & Routing Orchestrator).

    Chức năng chính:
    1. Hỗ trợ 5 chế độ thực thi cắt bỏ (Ablation Modes):
       - phobert_only, phobert_retrieval, phobert_retrieval_fuzzy, retrieval_only, tfidf_only.
    2. Hỗ trợ định tuyến phân cấp 2 tầng (Two-Tier Hierarchical Routing):
       - Tầng 1: Phân loại hoặc định tuyến nhóm danh mục (Category Routing).
       - Tầng 2: Thu hẹp không gian giải quyết ý định (Intent Routing) theo danh mục.
    3. Phân luồng kết quả thành 5 trạng thái tường minh (5 Explicit Statuses):
       - ok, need_choice, need_slots, out_of_scope, unknown.
    4. Cung cấp phương thức tương tác chọn lựa kịch bản (interactive disambiguation `choose`).
    5. Điểm vào thực thi trực tiếp module (CLI entry point) trả về JSON có cấu trúc.
    """

    def __init__(
        self,
        mode: PipelineMode | str = PipelineMode.PHOBERT_RETRIEVAL_FUZZY,
        two_tier: bool = False,
        classifier: Any | None = None,
        retriever: Any | None = None,
        matcher: Any | None = None,
        category_classifier: Any | None = None,
        tfidf_ranker: Any | None = None,
        intents_path: Path | str | None = None,
        tau: float | None = None,
        theta: float | None = None,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        auto_accept_retrieval: bool = False,
        lazy_load: bool = True,
        checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
        thresholds_path: Path | str = DEFAULT_THRESHOLDS_PATH,
        target: str = "latex",
        engine_version: str = ENGINE_VERSION_LEGACY,
    ) -> None:
        """
        Khởi tạo Pipeline với các thành phần và cấu hình tương ứng.

        Tham số:
            mode: Chế độ cắt bỏ đường ống (mặc định 'phobert_retrieval_fuzzy').
            two_tier: Cờ kích hoạt định tuyến phân cấp hai tầng (mặc định False).
            classifier: Thể hiện bộ phân loại ý định IntentClassifier (tùy chọn tiêm phụ thuộc).
            retriever: Thể hiện bộ truy hồi ngữ nghĩa Retriever (tùy chọn tiêm phụ thuộc).
            matcher: Thể hiện bộ so khớp mờ FuzzyMatcher (tùy chọn tiêm phụ thuộc).
            category_classifier: Bộ phân loại danh mục cho tầng 1 (tùy chọn tiêm phụ thuộc).
            tfidf_ranker: Bộ xếp hạng TF-IDF cho chế độ tfidf_only (tùy chọn).
            intents_path: Đường dẫn tệp intents.json (mặc định DEFAULT_INTENTS_PATH).
            tau: Ngưỡng xác suất softmax chấp nhận intent (mặc định TAU_INTENT_THRESHOLD=0.8).
            theta: Ngưỡng cosine tối thiểu trong truy hồi fallback (mặc định THETA_RETRIEVAL_THRESHOLD=0.3).
            fuzzy_threshold: Ngưỡng so khớp mờ tối thiểu (mặc định DEFAULT_FUZZY_THRESHOLD=0.5).
            top_k: Số lượng ứng viên trả về trong bước gợi ý lựa chọn (mặc định 3).
            auto_accept_retrieval: Tự động chấp nhận ứng viên đầu nếu đạt ngưỡng theta (mặc định False).
            lazy_load: Trì hoãn nạp trọng số mô hình nặng cho đến khi thực sự cần (mặc định True).
        """
        # Chuẩn hóa mode
        if isinstance(mode, PipelineMode):
            self.mode = mode.value
        else:
            self.mode = str(mode).strip().lower()

        if self.mode not in SUPPORTED_PIPELINE_MODES:
            raise ValueError(
                f"Chế độ pipeline '{self.mode}' không hợp lệ. "
                f"Các chế độ được hỗ trợ: {list(SUPPORTED_PIPELINE_MODES)}"
            )

        self.two_tier = bool(two_tier)
        self.engine_version = str(engine_version).strip().lower()
        if self.engine_version not in SUPPORTED_ENGINE_VERSIONS:
            raise ValueError(
                f"Engine version '{engine_version}' không được hỗ trợ: {tuple(SUPPORTED_ENGINE_VERSIONS)}"
            )
        self.target = str(target).strip()
        if self.engine_version == ENGINE_VERSION_LEGACY:
            self.target = self.target.lower()
        if self.engine_version == ENGINE_VERSION_LEGACY and self.target not in SUPPORTED_TARGETS:
            raise ValueError(f"Target '{target}' không được hỗ trợ: {tuple(SUPPORTED_TARGETS)}")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.thresholds_path = Path(thresholds_path)
        threshold_values = {
            "tau": TAU_INTENT_THRESHOLD,
            "theta": THETA_RETRIEVAL_THRESHOLD,
            "temperature": DEFAULT_TEMPERATURE,
        }
        if self.checkpoint_dir.exists() and not checkpoint_is_smoke(self.checkpoint_dir):
            try:
                threshold_values = load_calibrated_thresholds(
                    self.thresholds_path,
                    expected_checkpoint_sha256=checkpoint_sha256(self.checkpoint_dir),
                    expected_preprocessing_fingerprint=preprocess_fingerprint(),
                )
            except (OSError, ValueError) as error:
                logger.warning("Threshold calibration bị bỏ qua: %s. Dùng default.", error)
        else:
            logger.warning("PhoBERT chưa có checkpoint thật; dùng threshold mặc định.")
        self.tau = float(tau if tau is not None else threshold_values["tau"])
        self.theta = float(theta if theta is not None else threshold_values["theta"])
        self.temperature = float(threshold_values["temperature"])
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.top_k = int(top_k)
        self.auto_accept_retrieval = bool(auto_accept_retrieval)
        self.lazy_load = bool(lazy_load)

        self.intents_path = Path(intents_path) if intents_path is not None else DEFAULT_INTENTS_PATH
        self.intents_bank: dict[str, dict[str, Any]] = self._load_intents_bank(self.intents_path)
        self.generic_engine: Any | None = None
        self.generic_definitions: dict[str, dict[str, Any]] = {}
        self._dynamic_generic_intents: set[str] = set()
        if self.engine_version == ENGINE_VERSION_GENERIC:
            from core.definitions import DefinitionLoader
            from core.engine import GenericKnowledgeEngine

            self.generic_definitions = {
                loaded.intent: loaded.definition
                for loaded in DefinitionLoader().load(self.intents_path)
            }
            self.generic_engine = GenericKnowledgeEngine()
            if not any(
                self.target in item.get("render", {}).get("templates", {})
                for item in self.generic_definitions.values()
            ):
                raise ValueError(f"Không có definition v2 nào khai báo target '{self.target}'.")

        # Xây dựng bảng ánh xạ danh mục: {intent_name: category}
        self.category_map: dict[str, str] = {
            name: str(data.get("category", "")).strip()
            for name, data in self.intents_bank.items()
        }

        # Lưu trữ các thành phần con
        self.classifier = classifier
        self.retriever = retriever
        self.component_errors: dict[str, str] = {}
        self.matcher = matcher
        self.category_classifier = category_classifier
        self.tfidf_ranker = tfidf_ranker

        # Bộ nhớ tạm lưu trữ phiên tương tác gần nhất phục vụ phương thức choose()
        self._last_query: str = ""
        self._last_candidates: list[CandidateChoice] = []

        if not self.lazy_load:
            self._ensure_components_loaded()

    def sync_definition_snapshot(
        self,
        definitions: dict[str, dict[str, Any]],
        dynamic_intents: set[str] | None = None,
    ) -> None:
        """Refresh definitions without touching PhoBERT labels or checkpoint hashes.

        New/changed IDs are routed through the v2 processor after SBERT proposes
        them. The existing classifier remains frozen and cannot claim those IDs.
        """
        from core.engine import GenericKnowledgeEngine

        self.intents_bank = {name: dict(item) for name, item in definitions.items()}
        self.category_map = {
            name: str(item.get("category", "")).strip()
            for name, item in self.intents_bank.items()
        }
        self.generic_definitions = {
            name: dict(item)
            for name, item in self.intents_bank.items()
            if item.get("definition_version") == "2.0"
        }
        self._dynamic_generic_intents = set(dynamic_intents or ())
        if self.generic_engine is None:
            self.generic_engine = GenericKnowledgeEngine()

    @staticmethod
    def _load_intents_bank(path: Path) -> dict[str, dict[str, Any]]:
        """Nạp ngân hàng kịch bản từ tệp tin JSON."""
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {
                str(item["intent"]): item
                for item in data
                if isinstance(item, dict) and "intent" in item
            }
        elif isinstance(data, dict):
            return data
        return {}

    def _ensure_components_loaded(self) -> None:
        """Đảm bảo các thành phần mô hình cần thiết cho chế độ hiện tại đã được nạp."""
        # 1. Nạp bộ phân loại PhoBERT nếu cần
        if self.mode in (
            PipelineMode.PHOBERT_ONLY.value,
            PipelineMode.PHOBERT_RETRIEVAL.value,
            PipelineMode.PHOBERT_RETRIEVAL_FUZZY.value,
        ) and self.classifier is None:
            self._init_default_classifier()

        # 2. Nạp bộ truy hồi ngữ nghĩa SBERT nếu cần
        if self.mode in (
            PipelineMode.PHOBERT_RETRIEVAL.value,
            PipelineMode.PHOBERT_RETRIEVAL_FUZZY.value,
            PipelineMode.RETRIEVAL_ONLY.value,
        ) and self.retriever is None:
            self._init_default_retriever()

        # 3. Nạp bộ so khớp mờ FuzzyMatcher nếu cần
        if self.mode in (PipelineMode.PHOBERT_RETRIEVAL_FUZZY.value,) and self.matcher is None:
            self._init_default_matcher()

        # 4. Nạp bộ xếp hạng TF-IDF nếu cần
        if self.mode in (PipelineMode.TFIDF_ONLY.value,) and self.tfidf_ranker is None:
            self._init_default_tfidf_ranker()

    def _init_default_classifier(self) -> None:
        """Khởi tạo IntentClassifier từ checkpoint mặc định nếu khả dụng."""
        from core.classifier import IntentClassifier

        available, reason = classifier_artifact_status(self.checkpoint_dir)
        if not available:
            self.component_errors["classifier"] = reason
            logger.warning("PhoBERT chưa khả dụng: %s", reason)
            return
        try:
            self.classifier = IntentClassifier.from_pretrained(
                self.checkpoint_dir,
                threshold_tau=self.tau,
                temperature=self.temperature,
                category_map=self.category_map,
            )
            self.component_errors.pop("classifier", None)
        except Exception as error:
            self.component_errors["classifier"] = str(error)
            logger.warning("Không thể nạp checkpoint PhoBERT từ %s: %s", self.checkpoint_dir, error)

    def _init_default_retriever(self) -> None:
        """Khởi tạo Retriever ngữ nghĩa SBERT từ cấu hình mặc định."""
        from core.retrieval import Retriever

        try:
            model_dir = resolve_sbert_model_dir()
            if model_dir is None and os.environ.get("HF_HUB_OFFLINE") == "1":
                _, reason, _ = retrieval_model_status()
                self.component_errors["retrieval"] = reason
                logger.warning("Retrieval chưa khả dụng: %s", reason)
                return
            self.retriever = Retriever(
                model=model_dir if model_dir is not None else None,
                model_name=SBERT_MODEL_NAME,
                intents=self.intents_path,
                cache_dir=INDEX_CACHE_DIR,
                top_k=self.top_k,
                threshold_theta=self.theta,
            )
            self.component_errors.pop("retrieval", None)
        except Exception as error:
            self.component_errors["retrieval"] = str(error)
            logger.warning("Không thể khởi tạo Retriever: %s", error)

    def _init_default_matcher(self) -> None:
        """Khởi tạo FuzzyMatcher từ ngân hàng kịch bản."""
        from core.matcher import FuzzyMatcher

        try:
            self.matcher = FuzzyMatcher(
                intents_path=self.intents_path,
                threshold=self.fuzzy_threshold,
                theta=self.theta,
                top_k=self.top_k,
            )
        except Exception as e:
            logger.warning("Không thể khởi tạo FuzzyMatcher: %s", e)

    def _init_default_tfidf_ranker(self) -> None:
        """Khởi tạo TfidfRanker từ ngân hàng kịch bản."""
        try:
            self.tfidf_ranker = TfidfRanker(self.intents_bank)
        except Exception as e:
            logger.warning("Không thể khởi tạo TfidfRanker: %s", e)

    # -----------------------------------------------------------------
    # Điểm vào chính: Dự đoán và Định tuyến (Predict & Route)
    # -----------------------------------------------------------------

    def predict(self, query: str) -> PipelineResult:
        """
        Thực hiện toàn bộ quy trình suy luận và định tuyến câu truy vấn theo cấu hình.

        Các bước xử lý:
        1. Tiền xử lý chuỗi truy vấn (Unicode NFC, chuẩn hóa ký hiệu toán học).
        2. Nếu rỗng -> Trả về trạng thái 'unknown'.
        3. Kiểm tra định tuyến 2 tầng (Two-Tier Hierarchical Routing nếu two_tier=True):
           - Tầng 1: Dự đoán Category. Nếu là ngoài phạm vi ('tex_misc' / 'OUT_OF_SCOPE') -> 'out_of_scope'.
           - Tầng 2: Giới hạn giải quyết Intent trong Category đã chọn.
        4. Định tuyến theo chế độ cắt bỏ (Ablation Mode):
           - phobert_only: Chỉ tin cậy PhoBERT classifier (p >= tau -> intent, else -> 'unknown').
           - phobert_retrieval: PhoBERT (p >= tau) -> nếu thấp rơi vào SBERT retrieval top-3 (score >= theta -> 'need_choice', else -> 'unknown').
           - phobert_retrieval_fuzzy: PhoBERT -> SBERT retrieval -> RapidFuzz matching nếu retrieval < theta.
           - retrieval_only: Bỏ qua PhoBERT, trực tiếp truy hồi ngữ nghĩa.
           - tfidf_only: Bỏ qua PhoBERT, trực tiếp xếp hạng TF-IDF.
        5. Khi Intent được chấp nhận:
           - Trích xuất slot qua core.slots.extract_slots.
           - Nếu thiếu slot bắt buộc -> 'need_slots'.
           - Nếu đủ slot -> Render template LaTeX -> Kiểm tra an toàn tĩnh -> 'ok'.

        Tham số:
            query: Chuỗi văn bản tiếng Việt từ người dùng.

        Trả về:
            Đối tượng PipelineResult chứa trạng thái và dữ liệu tương ứng.
        """
        start_time = time.perf_counter()
        raw_query = query if isinstance(query, str) else ""
        norm_query = normalize(raw_query)

        # Xử lý trường hợp truy vấn rỗng
        if not norm_query.strip():
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return PipelineResult(
                status=PipelineStatus.UNKNOWN.value,
                query=raw_query,
                mode=self.mode,
                two_tier=self.two_tier,
                message="Truy vấn đầu vào rỗng hoặc không hợp lệ.",
                execution_time_ms=elapsed,
            )

        self._ensure_components_loaded()
        self._last_query = raw_query
        self._last_candidates = []

        if self.classifier is None and self.retriever is None and self.mode != PipelineMode.TFIDF_ONLY.value:
            details = " ".join(self.component_errors.values())
            result = PipelineResult(
                status=PipelineStatus.UNKNOWN.value,
                query=raw_query,
                mode=self.mode,
                two_tier=self.two_tier,
                message=("Chưa có model PhoBERT hoặc retrieval khả dụng. Knowledge Manager vẫn hoạt động. "
                         f"{details}".strip()),
            )
            result.execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            return result

        # Xử lý theo định tuyến 2 tầng hoặc định tuyến phẳng
        if self.two_tier:
            result = self._predict_two_tier(raw_query, norm_query)
        else:
            result = self._predict_flat(raw_query, norm_query)

        if result.status == PipelineStatus.UNKNOWN.value and self.component_errors:
            unavailable = " ".join(self.component_errors.values())
            result.message = f"{result.message} Thành phần NLP chưa khả dụng: {unavailable}".strip()

        result.execution_time_ms = (time.perf_counter() - start_time) * 1000.0
        return result

    # Cho phép gọi Pipeline(query) tương đương Pipeline.predict(query)
    __call__ = predict

    def _predict_two_tier(self, raw_query: str, norm_query: str) -> PipelineResult:
        """
        Định tuyến phân cấp hai tầng (Two-Tier Hierarchical Routing):
        - Tầng 1: Phân loại danh mục (Category).
        - Tầng 2: Phân loại hoặc truy hồi ý định (Intent) bên trong danh mục đó.
        """
        predicted_category: str | None = None
        category_confidence: float = 0.0

        # Tầng 1: Xác định danh mục
        if self.category_classifier is not None:
            try:
                cat_pred = self.category_classifier.predict(norm_query)
                predicted_category = getattr(cat_pred, "category", None) or getattr(cat_pred, "intent", None)
                category_confidence = float(getattr(cat_pred, "confidence", 1.0))
            except Exception as e:
                logger.warning("Lỗi phân loại danh mục tầng 1: %s", e)

        if predicted_category is None and self.classifier is not None:
            try:
                clf_pred = self.classifier.predict(norm_query)
                predicted_category = clf_pred.category or self.category_map.get(clf_pred.intent)
                category_confidence = clf_pred.confidence
            except Exception as e:
                logger.warning("Lỗi trích xuất danh mục từ classifier: %s", e)

        # Nếu danh mục là ngoài phạm vi (tex_misc hoặc OUT_OF_SCOPE)
        if predicted_category in (DEFAULT_OUT_OF_SCOPE_CATEGORY, "tex_misc", OUT_OF_SCOPE_INTENT):
            return PipelineResult(
                status=PipelineStatus.OUT_OF_SCOPE.value,
                query=raw_query,
                intent=OUT_OF_SCOPE_INTENT,
                category=predicted_category,
                confidence=category_confidence,
                mode=self.mode,
                two_tier=True,
                message="Yêu cầu nằm ngoài phạm vi hỗ trợ của hệ thống (danh mục ngoài phạm vi).",
            )

        # Tầng 2: Phân giải ý định bên trong danh mục predicted_category
        return self._predict_flat(raw_query, norm_query, restrict_category=predicted_category)

    def _predict_flat(
        self,
        raw_query: str,
        norm_query: str,
        restrict_category: str | None = None,
    ) -> PipelineResult:
        """
        Thực thi phân giải ý định theo các chế độ cắt bỏ (Ablation Modes).
        """
        # Chế độ 1: TF-IDF Only
        if self.mode == PipelineMode.TFIDF_ONLY.value:
            return self._route_tfidf_only(raw_query, norm_query, restrict_category)

        # Chế độ 2: Retrieval Only
        if self.mode == PipelineMode.RETRIEVAL_ONLY.value:
            return self._route_retrieval_only(raw_query, norm_query, restrict_category)

        # Chế độ 3, 4, 5: Các chế độ dựa trên PhoBERT (phobert_only, phobert_retrieval, phobert_retrieval_fuzzy)
        clf_result = None
        if self.classifier is not None:
            try:
                clf_result = self.classifier.predict(norm_query)
            except Exception as e:
                logger.warning("Lỗi dự đoán từ PhoBERT classifier: %s", e)

        if clf_result is not None:
            pred_intent = clf_result.intent
            pred_conf = clf_result.confidence
            pred_cat = clf_result.category or self.category_map.get(pred_intent, "")

            # Kiểm tra nếu bộ phân loại trực tiếp nhận diện ngoài phạm vi
            if pred_intent == OUT_OF_SCOPE_INTENT or pred_cat == DEFAULT_OUT_OF_SCOPE_CATEGORY:
                return PipelineResult(
                    status=PipelineStatus.OUT_OF_SCOPE.value,
                    query=raw_query,
                    intent=OUT_OF_SCOPE_INTENT,
                    category=pred_cat,
                    confidence=pred_conf,
                    mode=self.mode,
                    two_tier=self.two_tier,
                    message="Yêu cầu được xác định nằm ngoài phạm vi hỗ trợ.",
                )

            if not self._supports_target(pred_intent):
                probabilities = getattr(clf_result, "all_probabilities", {})
                supported_scores = [
                    (name, float(score))
                    for name, score in probabilities.items()
                    if self._supports_target(str(name))
                ]
                if supported_scores:
                    pred_intent, pred_conf = max(supported_scores, key=lambda pair: pair[1])
                    pred_cat = self.category_map.get(pred_intent, "")
                else:
                    pred_intent = ""
                    pred_conf = 0.0
                    pred_cat = ""

            if pred_intent == OUT_OF_SCOPE_INTENT or pred_cat == DEFAULT_OUT_OF_SCOPE_CATEGORY:
                return PipelineResult(
                    status=PipelineStatus.OUT_OF_SCOPE.value,
                    query=raw_query,
                    intent=OUT_OF_SCOPE_INTENT,
                    category=pred_cat,
                    confidence=pred_conf,
                    mode=self.mode,
                    two_tier=self.two_tier,
                    message="Yêu cầu được xác định nằm ngoài phạm vi hỗ trợ.",
                )

            # Nếu giới hạn danh mục (two-tier), kiểm tra tính phù hợp
            category_match = True
            if restrict_category is not None and pred_cat != restrict_category:
                category_match = False

            # if max p >= tau: intent accepted
            if pred_conf >= self.tau and category_match:
                return self._accept_intent_and_extract_slots(
                    intent=pred_intent,
                    query=raw_query,
                    confidence=pred_conf,
                    category=pred_cat,
                )

        # Nếu p < tau:
        # Nếu chế độ phobert_only -> Không fallback, trả về 'unknown'
        if self.mode == PipelineMode.PHOBERT_ONLY.value:
            conf = clf_result.confidence if clf_result is not None else 0.0
            return PipelineResult(
                status=PipelineStatus.UNKNOWN.value,
                query=raw_query,
                confidence=conf,
                mode=self.mode,
                two_tier=self.two_tier,
                message=f"Độ tin cậy phân loại ({conf:.2f}) dưới ngưỡng tau ({self.tau:.2f}).",
            )

        # Fallback 1: SBERT Retrieval
        retrieval_candidates = self._run_retrieval(norm_query, restrict_category)
        if retrieval_candidates:
            top_cand = retrieval_candidates[0]
            if top_cand.score >= self.theta:
                if top_cand.intent == OUT_OF_SCOPE_INTENT or top_cand.category == DEFAULT_OUT_OF_SCOPE_CATEGORY:
                    return PipelineResult(
                        status=PipelineStatus.OUT_OF_SCOPE.value,
                        query=raw_query,
                        intent=OUT_OF_SCOPE_INTENT,
                        category=top_cand.category,
                        confidence=top_cand.score,
                        mode=self.mode,
                        two_tier=self.two_tier,
                        message="Truy hồi ngữ nghĩa xác định câu hỏi nằm ngoài phạm vi hỗ trợ.",
                    )

                if self.auto_accept_retrieval:
                    return self._accept_intent_and_extract_slots(
                        intent=top_cand.intent,
                        query=raw_query,
                        confidence=top_cand.score,
                        category=top_cand.category,
                    )

                # Theo AGENTS.md: retrieval top-3 -> ask the user to choose
                self._last_candidates = retrieval_candidates
                return PipelineResult(
                    status=PipelineStatus.NEED_CHOICE.value,
                    query=raw_query,
                    confidence=top_cand.score,
                    category=top_cand.category,
                    mode=self.mode,
                    two_tier=self.two_tier,
                    candidates=[c.to_dict() for c in retrieval_candidates],
                    message="Yêu cầu người dùng lựa chọn một trong các kịch bản tương đồng từ truy hồi ngữ nghĩa.",
                )

        # Fallback 2: RapidFuzz Matching (chỉ áp dụng trong chế độ phobert_retrieval_fuzzy)
        if self.mode == PipelineMode.PHOBERT_RETRIEVAL_FUZZY.value:
            fuzzy_candidates = self._run_fuzzy(norm_query, restrict_category)
            if fuzzy_candidates:
                top_fuz = fuzzy_candidates[0]
                if top_fuz.score >= self.fuzzy_threshold:
                    if top_fuz.intent == OUT_OF_SCOPE_INTENT or top_fuz.category == DEFAULT_OUT_OF_SCOPE_CATEGORY:
                        return PipelineResult(
                            status=PipelineStatus.OUT_OF_SCOPE.value,
                            query=raw_query,
                            intent=OUT_OF_SCOPE_INTENT,
                            category=top_fuz.category,
                            confidence=top_fuz.score,
                            mode=self.mode,
                            two_tier=self.two_tier,
                            message="So khớp mờ từ vựng xác định yêu cầu nằm ngoài phạm vi hỗ trợ.",
                        )

                    self._last_candidates = fuzzy_candidates
                    return PipelineResult(
                        status=PipelineStatus.NEED_CHOICE.value,
                        query=raw_query,
                        confidence=top_fuz.score,
                        category=top_fuz.category,
                        mode=self.mode,
                        two_tier=self.two_tier,
                        candidates=[c.to_dict() for c in fuzzy_candidates],
                        message="Yêu cầu người dùng lựa chọn một trong các kịch bản tương đồng từ so khớp mờ.",
                    )

        # Nếu không có phương pháp nào đạt ngưỡng -> unknown
        best_conf = 0.0
        if clf_result is not None:
            best_conf = max(best_conf, clf_result.confidence)
        if retrieval_candidates:
            best_conf = max(best_conf, retrieval_candidates[0].score)

        return PipelineResult(
            status=PipelineStatus.UNKNOWN.value,
            query=raw_query,
            confidence=best_conf,
            mode=self.mode,
            two_tier=self.two_tier,
            message="Không nhận diện được ý định (điểm tin cậy dưới ngưỡng quy định).",
        )

    def _route_retrieval_only(
        self,
        raw_query: str,
        norm_query: str,
        restrict_category: str | None = None,
    ) -> PipelineResult:
        """Thực thi định tuyến cho chế độ cắt bỏ retrieval_only."""
        candidates = self._run_retrieval(norm_query, restrict_category)
        if not candidates or candidates[0].score < self.theta:
            top_score = candidates[0].score if candidates else 0.0
            return PipelineResult(
                status=PipelineStatus.UNKNOWN.value,
                query=raw_query,
                confidence=top_score,
                mode=self.mode,
                two_tier=self.two_tier,
                message="Truy hồi ngữ nghĩa không tìm thấy kịch bản phù hợp đạt ngưỡng theta.",
            )

        top_cand = candidates[0]
        if top_cand.intent == OUT_OF_SCOPE_INTENT or top_cand.category == DEFAULT_OUT_OF_SCOPE_CATEGORY:
            return PipelineResult(
                status=PipelineStatus.OUT_OF_SCOPE.value,
                query=raw_query,
                intent=OUT_OF_SCOPE_INTENT,
                category=top_cand.category,
                confidence=top_cand.score,
                mode=self.mode,
                two_tier=self.two_tier,
                message="Truy hồi ngữ nghĩa xác định câu hỏi nằm ngoài phạm vi hỗ trợ.",
            )

        if self.auto_accept_retrieval:
            return self._accept_intent_and_extract_slots(
                intent=top_cand.intent,
                query=raw_query,
                confidence=top_cand.score,
                category=top_cand.category,
            )

        self._last_candidates = candidates
        return PipelineResult(
            status=PipelineStatus.NEED_CHOICE.value,
            query=raw_query,
            confidence=top_cand.score,
            category=top_cand.category,
            mode=self.mode,
            two_tier=self.two_tier,
            candidates=[c.to_dict() for c in candidates],
            message="Yêu cầu người dùng lựa chọn một trong các kịch bản tương đồng từ truy hồi ngữ nghĩa.",
        )

    def _route_tfidf_only(
        self,
        raw_query: str,
        norm_query: str,
        restrict_category: str | None = None,
    ) -> PipelineResult:
        """Thực thi định tuyến cho chế độ cắt bỏ tfidf_only."""
        candidates = self._run_tfidf(norm_query, restrict_category)
        if not candidates or candidates[0].score < self.theta:
            top_score = candidates[0].score if candidates else 0.0
            return PipelineResult(
                status=PipelineStatus.UNKNOWN.value,
                query=raw_query,
                confidence=top_score,
                mode=self.mode,
                two_tier=self.two_tier,
                message="Xếp hạng TF-IDF không tìm thấy kịch bản phù hợp đạt ngưỡng tương đồng.",
            )

        top_cand = candidates[0]
        if top_cand.intent == OUT_OF_SCOPE_INTENT or top_cand.category == DEFAULT_OUT_OF_SCOPE_CATEGORY:
            return PipelineResult(
                status=PipelineStatus.OUT_OF_SCOPE.value,
                query=raw_query,
                intent=OUT_OF_SCOPE_INTENT,
                category=top_cand.category,
                confidence=top_cand.score,
                mode=self.mode,
                two_tier=self.two_tier,
                message="Xếp hạng TF-IDF xác định câu hỏi nằm ngoài phạm vi hỗ trợ.",
            )

        if self.auto_accept_retrieval:
            return self._accept_intent_and_extract_slots(
                intent=top_cand.intent,
                query=raw_query,
                confidence=top_cand.score,
                category=top_cand.category,
            )

        self._last_candidates = candidates
        return PipelineResult(
            status=PipelineStatus.NEED_CHOICE.value,
            query=raw_query,
            confidence=top_cand.score,
            category=top_cand.category,
            mode=self.mode,
            two_tier=self.two_tier,
            candidates=[c.to_dict() for c in candidates],
            message="Yêu cầu người dùng lựa chọn một trong các kịch bản tương đồng từ xếp hạng TF-IDF.",
        )

    # -----------------------------------------------------------------
    # Bước bóc tách tham số và sinh mã (Slot Extraction & Rendering)
    # -----------------------------------------------------------------

    def _accept_intent_and_extract_slots(
        self,
        intent: str,
        query: str,
        confidence: float,
        category: str,
        custom_slots: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Xử lý khi một Intent đã được chấp nhận:
        - Bóc tách slot theo regex.
        - Nếu thiếu slot bắt buộc -> 'need_slots'.
        - Nếu đủ slot -> Render template LaTeX -> 'ok'.
        """
        if (
            self.engine_version == ENGINE_VERSION_GENERIC
            or intent in self._dynamic_generic_intents
        ):
            return self._accept_generic_definition(
                intent=intent,
                query=query,
                confidence=confidence,
                category=category,
                custom_slots=custom_slots,
            )

        intent_data = self.intents_bank.get(intent, {})
        if intent != OUT_OF_SCOPE_INTENT and resolve_template(intent_data, self.target) is None:
            return PipelineResult(
                status=PipelineStatus.UNSUPPORTED_TARGET.value,
                query=query,
                intent=intent,
                category=category,
                confidence=confidence,
                mode=self.mode,
                two_tier=self.two_tier,
                message=f"Intent '{intent}' chưa có template cho target '{self.target}'.",
            )

        # Trích xuất slot bằng core.slots.extract_slots
        slot_res = extract_slots(query, intent, intents_bank=self.intents_bank)
        merged_slots = dict(slot_res.slots)
        slots_def = intent_data.get("slots", {})
        custom_names = set(custom_slots or {})
        custom_errors: list[str] = []
        invalid_custom_required: set[str] = set()
        for slot_name, raw_value in (custom_slots or {}).items():
            slot_spec = slots_def.get(slot_name)
            if not isinstance(slot_spec, dict):
                custom_errors.append(f"Slot nhập tay '{slot_name}' không được khai báo.")
                continue
            slot_type = str(slot_spec.get("type", "text"))
            try:
                normalized_value = normalize_slot_value(raw_value, slot_type)
                not_equal = slot_spec.get("not_equal")
                if slot_type == "number" and not_equal is not None:
                    if float(normalized_value) == float(not_equal):
                        raise ValueError(
                            str(slot_spec.get("constraint_error") or "Giá trị slot vi phạm ràng buộc.")
                        )
            except (TypeError, ValueError) as error:
                custom_errors.append(f"{slot_name}: {error}")
                if slot_spec.get("required"):
                    invalid_custom_required.add(slot_name)
                continue
            merged_slots[slot_name] = normalized_value

        # Tính lại field còn thiếu sau khi đã kiểm thực cả giá trị người dùng nhập tay.
        missing_slots = []
        for s_name, s_def in slots_def.items():
            req = s_def.get("required", False) if isinstance(s_def, dict) else getattr(s_def, "required", False)
            if req and (
                s_name in invalid_custom_required
                or s_name not in merged_slots
                or not str(merged_slots[s_name]).strip()
            ):
                missing_slots.append(s_name)

        # Lỗi parser cần gắn với field; giá trị nhập tay hợp lệ sẽ thay thế lỗi cùng slot.
        unresolved_error_indexes = [
            index
            for index, slot_name in enumerate(slot_res.error_slots)
            if slot_name not in custom_names
        ]
        active_slot_errors = [slot_res.errors[index] for index in unresolved_error_indexes]
        active_slot_errors.extend(custom_errors)
        for slot_name in slot_res.missing:
            if slot_name in slots_def and slot_name not in custom_names and slot_name not in missing_slots:
                missing_slots.append(slot_name)

        # Nếu có lỗi kiểm thực ràng buộc (ví dụ: a == 0 cho phương trình bậc hai)
        for index in unresolved_error_indexes:
            slot_name = slot_res.error_slots[index]
            slot_spec = slots_def.get(slot_name, {})
            required = slot_spec.get("required", False) if isinstance(slot_spec, dict) else False
            if required and slot_name not in missing_slots:
                missing_slots.append(slot_name)

        # Nếu thiếu tham số bắt buộc -> Trả về status 'need_slots'
        if missing_slots:
            return PipelineResult(
                status=PipelineStatus.NEED_SLOTS.value,
                query=query,
                intent=intent,
                category=category,
                confidence=confidence,
                mode=self.mode,
                two_tier=self.two_tier,
                slots=merged_slots,
                missing_slots=missing_slots,
                slot_errors=active_slot_errors,
                message=(
                    f"Kịch bản '{intent}' yêu cầu bổ sung các tham số còn thiếu: {', '.join(missing_slots)}."
                    if missing_slots
                    else "Giá trị slot nhập tay không hợp lệ: " + "; ".join(custom_errors)
                ),
            )

        if custom_errors:
            return PipelineResult(
                status=PipelineStatus.NEED_SLOTS.value,
                query=query,
                intent=intent,
                category=category,
                confidence=confidence,
                mode=self.mode,
                two_tier=self.two_tier,
                slots=merged_slots,
                slot_errors=active_slot_errors,
                message="Giá trị slot nhập tay không hợp lệ: " + "; ".join(custom_errors),
            )

        # Đã đủ tham số -> Render template LaTeX
        rendered_code, check_ok, check_errs = render_latex_template(
            intent_data=intent_data,
            slots=merged_slots,
            target=self.target,
            escape_text_slots=True,
            wrap_document=False,
        )
        return PipelineResult(
            status=(
                PipelineStatus.OK.value if rendered_code.strip()
                else PipelineStatus.UNSUPPORTED_TARGET.value
            ),
            query=query,
            intent=intent,
            category=category,
            confidence=confidence,
            mode=self.mode,
            two_tier=self.two_tier,
            slots=merged_slots,
            missing_slots=[],
            slot_errors=active_slot_errors,
            latex_code=rendered_code,
            static_check_passed=check_ok,
            message=(
                "Đã trích xuất đầy đủ tham số và kết xuất mã thành công."
                if rendered_code.strip()
                else f"Intent '{intent}' chưa sinh nội dung cho target '{self.target}'."
            ),
        )

    def _accept_generic_definition(
        self,
        intent: str,
        query: str,
        confidence: float,
        category: str,
        custom_slots: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Chuyển NLU đã chọn intent sang Contract v2 mà không đổi legacy route."""
        definition = self.generic_definitions.get(intent)
        if definition is None:
            return PipelineResult(
                status=PipelineStatus.UNKNOWN.value,
                query=query,
                intent=intent,
                category=category,
                confidence=confidence,
                mode=self.mode,
                two_tier=self.two_tier,
                message=f"Intent '{intent}' không có trong definition registry v2.",
            )
        templates = definition.get("render", {}).get("templates", {})
        if self.target not in templates or not templates[self.target]:
            return PipelineResult(
                status=PipelineStatus.UNSUPPORTED_TARGET.value,
                query=query,
                intent=intent,
                category=category,
                confidence=confidence,
                mode=self.mode,
                two_tier=self.two_tier,
                message=f"Intent '{intent}' chưa có template v2 cho target '{self.target}'.",
            )
        result = self.generic_engine.process(
            normalize(query),
            definition,
            target=self.target,
            manual_values=custom_slots,
        )
        stages = dict(result.stages)
        stages["intent_detection"] = {
            "status": "accepted",
            "intent": intent,
            "confidence": confidence,
            "source": "classifier_or_retrieval",
        }
        code = result.outputs.get(self.target, "")
        slot_errors = [
            f"{error.path}: {error.message}" for error in result.errors
        ]
        if result.status == PipelineStatus.NEED_SLOTS.value:
            message = "Cần bổ sung hoặc sửa các trường được báo theo JSONPath."
        elif result.status == STATUS_PROCESSING_ERROR:
            for stage in ("transform", "render"):
                error = stages.get(stage, {}).get("error")
                if error:
                    slot_errors.append(str(error))
            message = "Generic Knowledge Engine không thể xử lý definition này."
        else:
            message = "Đã xử lý definition v2 qua các stage tổng quát."
        return PipelineResult(
            status=result.status,
            query=query,
            intent=intent,
            category=category,
            confidence=confidence,
            mode=self.mode,
            two_tier=self.two_tier,
            slots=result.values,
            missing_slots=result.missing,
            slot_errors=slot_errors,
            latex_code=code,
            message=message,
            static_check_passed=(self.target.casefold() == "latex" and result.status == STATUS_OK),
            stages=stages,
            outputs=result.outputs,
        )

    # -----------------------------------------------------------------
    # Phương thức Tương tác Chọn Kịch bản (Interactive Disambiguation)
    # -----------------------------------------------------------------

    def choose(
        self,
        candidate: str | int,
        query: str | None = None,
        slots: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Giải quyết nhập nhằng (disambiguation) khi người dùng chủ động chọn một kịch bản
        từ danh sách ứng viên (khi trạng thái trước đó là 'need_choice').

        Tham số:
            candidate: Mã định danh kịch bản (str, ví dụ: 'LATEX_NAP_PACKAGE')
                       HOẶC chỉ số số nguyên (int, 0 đến len(candidates)-1) từ danh sách gợi ý trước đó.
            query: Chuỗi truy vấn cần trích xuất tham số (nếu None sẽ dùng query gần nhất).
            slots: Từ điển các tham số bổ sung do người dùng cung cấp trực tiếp (tùy chọn).

        Trả về:
            PipelineResult mới tương ứng với kịch bản được chọn ('ok', 'need_slots', hoặc 'out_of_scope').
        """
        start_time = time.perf_counter()
        target_query = query if query is not None else self._last_query

        # Phân giải candidate thành tên intent
        chosen_intent: str = ""
        if isinstance(candidate, int):
            if not self._last_candidates or candidate < 0 or candidate >= len(self._last_candidates):
                return PipelineResult(
                    status=PipelineStatus.UNKNOWN.value,
                    query=target_query,
                    message=f"Chỉ số lựa chọn {candidate} không hợp lệ hoặc danh sách ứng viên rỗng.",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                )
            chosen_intent = self._last_candidates[candidate].intent
        else:
            chosen_intent = str(candidate).strip()

        if not chosen_intent:
            return PipelineResult(
                status=PipelineStatus.UNKNOWN.value,
                query=target_query,
                message="Tên kịch bản lựa chọn không được để trống.",
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # Xử lý trường hợp chọn kịch bản ngoài phạm vi
        if chosen_intent == OUT_OF_SCOPE_INTENT:
            return PipelineResult(
                status=PipelineStatus.OUT_OF_SCOPE.value,
                query=target_query,
                intent=OUT_OF_SCOPE_INTENT,
                category=DEFAULT_OUT_OF_SCOPE_CATEGORY,
                confidence=1.0,
                mode=self.mode,
                two_tier=self.two_tier,
                message="Kịch bản được chọn là ngoài phạm vi hỗ trợ.",
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        if chosen_intent not in self.intents_bank:
            return PipelineResult(
                status=PipelineStatus.UNKNOWN.value,
                query=target_query,
                intent=chosen_intent,
                message=f"Kịch bản '{chosen_intent}' không tồn tại trong ngân hàng kịch bản.",
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        category = self.category_map.get(chosen_intent, "")
        result = self._accept_intent_and_extract_slots(
            intent=chosen_intent,
            query=target_query,
            confidence=1.0,
            category=category,
            custom_slots=slots,
        )
        result.execution_time_ms = (time.perf_counter() - start_time) * 1000.0
        return result

    # -----------------------------------------------------------------
    # Các hàm thực thi mô hình phụ trợ (Helper Dispatchers)
    # -----------------------------------------------------------------

    def _supports_target(self, intent: str) -> bool:
        """Giữ OOS để pipeline có thể phát hiện; ứng viên thường cần template đích."""
        if (
            self.engine_version == ENGINE_VERSION_GENERIC
            or intent in self._dynamic_generic_intents
        ):
            definition = self.generic_definitions.get(intent, {})
            templates = definition.get("render", {}).get("templates", {})
            return intent == OUT_OF_SCOPE_INTENT or (
                isinstance(templates, dict) and self.target in templates
            )
        return intent == OUT_OF_SCOPE_INTENT or resolve_template(
            self.intents_bank.get(intent, {}), self.target
        ) is not None

    def _candidate_fetch_k(self) -> int:
        return self.top_k * TARGET_CANDIDATE_OVERSAMPLE_FACTOR

    def _run_retrieval(
        self,
        norm_query: str,
        restrict_category: str | None = None,
    ) -> list[CandidateChoice]:
        """Thực thi truy hồi ngữ nghĩa SBERT và chuyển đổi định dạng ứng viên."""
        if self.retriever is None:
            return []

        try:
            res = self.retriever.retrieve(norm_query, top_k=self._candidate_fetch_k())
            candidates: list[CandidateChoice] = []
            for c in res.candidates:
                cat = c.category or self.category_map.get(c.intent, "")
                if not self._supports_target(c.intent):
                    continue
                if restrict_category is not None and cat != restrict_category:
                    continue
                candidates.append(
                    CandidateChoice(
                        intent=c.intent,
                        score=c.score,
                        describe=c.describe or self.intents_bank.get(c.intent, {}).get("describe", ""),
                        category=cat,
                        source="retrieval",
                    )
                )
            return candidates[: self.top_k]
        except Exception as e:
            logger.warning("Lỗi trong quá trình truy hồi ngữ nghĩa: %s", e)
            return []

    def _run_fuzzy(
        self,
        norm_query: str,
        restrict_category: str | None = None,
    ) -> list[CandidateChoice]:
        """Thực thi so khớp mờ RapidFuzz và chuyển đổi định dạng ứng viên."""
        if self.matcher is None:
            return []

        try:
            matches = self.matcher.match(
                norm_query, top_k=self._candidate_fetch_k(), threshold=self.fuzzy_threshold
            )
            candidates: list[CandidateChoice] = []
            for m in matches:
                cat = m.category or self.category_map.get(m.intent, "")
                if not self._supports_target(m.intent):
                    continue
                if restrict_category is not None and cat != restrict_category:
                    continue
                candidates.append(
                    CandidateChoice(
                        intent=m.intent,
                        score=m.score,
                        describe=m.describe or self.intents_bank.get(m.intent, {}).get("describe", ""),
                        category=cat,
                        source="fuzzy",
                    )
                )
            return candidates[: self.top_k]
        except Exception as e:
            logger.warning("Lỗi trong quá trình so khớp mờ: %s", e)
            return []

    def _run_tfidf(
        self,
        norm_query: str,
        restrict_category: str | None = None,
    ) -> list[CandidateChoice]:
        """Thực thi xếp hạng TF-IDF."""
        if self.tfidf_ranker is None:
            return []

        try:
            ranked = self.tfidf_ranker.rank(
                norm_query, top_k=self._candidate_fetch_k(), category=restrict_category
            )
            return [candidate for candidate in ranked if self._supports_target(candidate.intent)][
                : self.top_k
            ]
        except Exception as e:
            logger.warning("Lỗi trong quá trình xếp hạng TF-IDF: %s", e)
            return []


# =====================================================================
# 6. Điểm vào thực thi Module trực tiếp (CLI Entry Point)
# =====================================================================


def main() -> None:
    """
    Điểm vào dòng lệnh thực thi trực tiếp module core.pipeline.
    Nhận tham số dòng lệnh và trả về kết quả JSON có cấu trúc.
    """
    parser = argparse.ArgumentParser(
        description="Đường ống suy luận và điều phối định tuyến Vietnamese LaTeX Knowledge Assistant."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Câu truy vấn tiếng Việt từ người dùng.",
    )
    parser.add_argument(
        "-q",
        "--query-text",
        dest="query_opt",
        default=None,
        help="Câu truy vấn tiếng Việt (tùy chọn cờ).",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default=MODE_PHOBERT_RETRIEVAL_FUZZY,
        choices=list(SUPPORTED_PIPELINE_MODES),
        help=f"Chế độ thực thi cắt bỏ của pipeline (mặc định: {MODE_PHOBERT_RETRIEVAL_FUZZY}).",
    )
    parser.add_argument(
        "--two-tier",
        action="store_true",
        help="Kích hoạt cơ chế định tuyến phân cấp hai tầng (Two-Tier Hierarchical Routing).",
    )
    parser.add_argument(
        "--choose",
        default=None,
        help="Lựa chọn kịch bản mục tiêu để giải quyết nhập nhằng (tên intent hoặc chỉ số).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=TAU_INTENT_THRESHOLD,
        help=f"Ngưỡng chấp nhận intent trực tiếp từ PhoBERT (mặc định: {TAU_INTENT_THRESHOLD}).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=THETA_RETRIEVAL_THRESHOLD,
        help=f"Ngưỡng chấp nhận truy hồi ngữ nghĩa (mặc định: {THETA_RETRIEVAL_THRESHOLD}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
        help=f"Số lượng ứng viên trả về tối đa (mặc định: {DEFAULT_RETRIEVAL_TOP_K}).",
    )
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="Tự động chấp nhận ứng viên truy hồi đứng đầu nếu vượt ngưỡng theta.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Độ thụt lề định dạng JSON đầu ra (mặc định: 2).",
    )

    args = parser.parse_args()
    query_str = args.query_opt if args.query_opt is not None else args.query

    pipeline = Pipeline(
        mode=args.mode,
        two_tier=args.two_tier,
        tau=args.tau,
        theta=args.theta,
        top_k=args.top_k,
        auto_accept_retrieval=args.auto_accept,
    )

    if args.choose is not None:
        # Nếu choose là số nguyên
        try:
            cand_idx = int(args.choose)
            result = pipeline.choose(cand_idx, query=query_str)
        except ValueError:
            result = pipeline.choose(args.choose, query=query_str)
    else:
        result = pipeline.predict(query_str)

    print(result.to_json(indent=args.indent))


if __name__ == "__main__":
    main()
