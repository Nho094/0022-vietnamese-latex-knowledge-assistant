"""
Kiểm thử đơn vị cho bộ truy hồi ngữ nghĩa dự phòng SbertRetrievalFallback (src/fallback.py).

Tuân thủ kiến trúc AGENTS.md:
- Đảm bảo tính toán tương đồng cosine, top-3 candidates.
- Kiểm tra ngưỡng theta (THETA_RETRIEVAL_THRESHOLD = 0.3):
  + Nếu score >= theta: understood = True và 'ask the user to choose'.
  + Nếu score < theta: understood = False và 'say not understood'.
- Kiểm tra hoạt động offline không cần internet.
"""

from __future__ import annotations

import pytest

from core.config import DEFAULT_INTENTS_PATH, THETA_RETRIEVAL_THRESHOLD
from src.fallback import CandidateMatch, FallbackResult, SbertRetrievalFallback
from src.scenarios import Scenario, ScenarioBank, load_scenario_bank


@pytest.fixture
def sample_bank() -> ScenarioBank:
    """Fixture ngân hàng kịch bản mẫu cho kiểm thử truy hồi."""
    scenarios = [
        Scenario(
            intent="MATHJAX_PHAN_SO",
            category="math_expression",
            describe="Chèn ký hiệu hoặc cấu trúc phân số toán học",
            aliases=["phân số", "phân thức", "a trên b"],
            sample_utterances=["tạo phân số với tử và mẫu", "chèn phân thức"],
        ),
        Scenario(
            intent="LATEX_TAO_BANG",
            category="tables",
            describe="Tạo bảng tabular trong LaTeX",
            aliases=["bảng", "tạo bảng", "table"],
            sample_utterances=["tạo một bảng trong tài liệu", "chèn bảng 3 cột"],
        ),
        Scenario(
            intent="LATEX_CHEN_ANH",
            category="figures",
            describe="Chèn hình ảnhincludegraphics vào văn bản",
            aliases=["ảnh", "chèn ảnh", "hình ảnh", "figure"],
            sample_utterances=["chèn một bức ảnh vào bài viết", "thêm hình ảnh"],
        ),
    ]
    return ScenarioBank(scenarios)


def test_fallback_initialization(sample_bank: ScenarioBank) -> None:
    """Kiểm tra khởi tạo SbertRetrievalFallback và xây dựng chỉ mục."""
    fallback = SbertRetrievalFallback(scenario_bank=sample_bank, threshold_theta=0.3, top_k=3)
    assert fallback.threshold_theta == 0.3
    assert fallback.top_k == 3
    assert len(fallback._scenario_index) > 0


def test_cosine_similarity_basic() -> None:
    """Kiểm tra tính đúng đắn của hàm cosine similarity."""
    vec1 = {"a": 1.0, "b": 2.0}
    vec2 = {"a": 1.0, "b": 2.0}
    vec3 = {"c": 3.0}

    # Vector giống hệt nhau -> sim = 1.0
    sim_identical = SbertRetrievalFallback._cosine_similarity(vec1, vec2)
    assert abs(sim_identical - 1.0) < 1e-4

    # Vector trực giao không chia sẻ chiều nào -> sim = 0.0
    sim_orthogonal = SbertRetrievalFallback._cosine_similarity(vec1, vec3)
    assert sim_orthogonal == 0.0

    # Vector rỗng -> sim = 0.0
    sim_empty = SbertRetrievalFallback._cosine_similarity({}, vec1)
    assert sim_empty == 0.0


def test_fallback_retrieve_understood(sample_bank: ScenarioBank) -> None:
    """Kiểm tra truy hồi thành công khi câu chứa từ khóa ngữ nghĩa liên quan (score >= theta)."""
    fallback = SbertRetrievalFallback(scenario_bank=sample_bank, threshold_theta=0.3, top_k=3)
    res = fallback.retrieve("tôi muốn tạo phân số toán học")

    assert isinstance(res, FallbackResult)
    assert res.understood is True
    assert len(res.top_candidates) > 0
    assert res.top_candidates[0].intent == "MATHJAX_PHAN_SO"
    assert res.top_candidates[0].score >= 0.3
    # Xác nhận câu hỏi lựa chọn được hiển thị theo AGENTS.md ('ask the user to choose')
    assert "Có phải bạn muốn thực hiện" in res.prompt_message
    assert "phân số" in res.prompt_message


def test_fallback_retrieve_not_understood(sample_bank: ScenarioBank) -> None:
    """Kiểm tra câu hoàn toàn vô nghĩa trả về score < theta, understood = False ('say not understood')."""
    fallback = SbertRetrievalFallback(scenario_bank=sample_bank, threshold_theta=0.4, top_k=3)
    res = fallback.retrieve("xyzqwerty 123456 asdfgh")

    assert isinstance(res, FallbackResult)
    assert res.understood is False
    # Xác nhận thông báo không hiểu ('say not understood')
    assert "không hiểu" in res.prompt_message.lower()


def test_fallback_retrieve_empty_query(sample_bank: ScenarioBank) -> None:
    """Kiểm tra truy vấn rỗng."""
    fallback = SbertRetrievalFallback(scenario_bank=sample_bank)
    res = fallback.retrieve("   ")

    assert res.understood is False
    assert res.top_candidates == []
    assert "không hiểu" in res.prompt_message.lower()


def test_fallback_top_k_ordering(sample_bank: ScenarioBank) -> None:
    """Kiểm tra danh sách ứng viên trả về được sắp xếp giảm dần theo điểm và tối đa top_k phần tử."""
    fallback = SbertRetrievalFallback(scenario_bank=sample_bank, threshold_theta=0.1, top_k=2)
    res = fallback.retrieve("chèn ảnh hoặc bảng vào tài liệu")

    assert len(res.top_candidates) <= 2
    if len(res.top_candidates) >= 2:
        assert res.top_candidates[0].score >= res.top_candidates[1].score


def test_fallback_with_official_bank() -> None:
    """Kiểm tra fallback với ngân hàng kịch bản 162 intent đầy đủ."""
    if not DEFAULT_INTENTS_PATH.exists():
        pytest.skip("File intents.json chưa sẵn sàng.")

    bank = load_scenario_bank(DEFAULT_INTENTS_PATH)
    fallback = SbertRetrievalFallback(scenario_bank=bank, threshold_theta=THETA_RETRIEVAL_THRESHOLD, top_k=3)

    # Truy vấn ý định tạo ma trận
    res = fallback.retrieve("muốn gõ ma trận latex")
    assert res.understood is True
    assert len(res.top_candidates) > 0
    intents = [c.intent for c in res.top_candidates]
    assert any("MA_TRAN" in it for it in intents)


def test_fallback_mock_sbert_model(sample_bank: ScenarioBank) -> None:
    """Kiểm tra đường dẫn suy luận khi SentenceTransformer thực tế được nạp."""
    fallback = SbertRetrievalFallback(scenario_bank=sample_bank, threshold_theta=0.3, top_k=2)

    # Giả lập SentenceTransformer model với embedding vector đơn vị
    from unittest.mock import MagicMock
    mock_sbert = MagicMock()
    # Query vector: [1.0, 0.0]
    # Doc vector: [1.0, 0.0] cho kịch bản đầu tiên, [0.0, 1.0] cho kịch bản khác
    def mock_encode(text: str, normalize_embeddings: bool = True) -> list[float]:
        if "phân số" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    mock_sbert.encode.side_effect = mock_encode
    fallback._sbert_model = mock_sbert
    fallback._is_sbert_loaded = True

    res = fallback.retrieve("muốn làm phân số")
    assert res.understood is True
    assert res.top_candidates[0].intent == "MATHJAX_PHAN_SO"
    assert res.top_candidates[0].score == 1.0

