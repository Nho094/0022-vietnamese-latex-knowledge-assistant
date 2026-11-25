"""
Bộ kiểm thử đơn vị và tích hợp toàn diện cho bộ phân loại ý định (tests/test_classifier.py).

Tuân thủ nghiêm ngặt các quy tắc trong AGENTS.md:
- Kiểm tra tính toán softmax và chuẩn hóa xác suất: sum(p) == 1.0 (Chen et al., 2019 - JointBERT).
- Kiểm tra co giãn nhiệt độ (Temperature Scaling theo Guo et al., ICML 2017): thay đổi phân phối
  (làm mượt/làm sắc, entropy tăng theo T) nhưng bảo toàn tuyệt đối thứ tự xếp hạng (rank order invariance).
- Kiểm tra cơ chế kiểm thực vân tay tiền xử lý (Preprocessing Fingerprint Verification):
  phát hiện sai lệch vân tay tại thời điểm chạy và ném ngoại lệ FingerprintMismatchError phù hợp.
- Kiểm tra tính nhất quán giữa suy luận đơn lẻ predict và suy luận theo lô predict_batch.
- Kiểm tra quy trình huấn luyện thử nghiệm siêu tốc (Smoke Training) hoàn thành thành công trên CPU.
- Duy trì tính tương thích ngược với bộ kiểm thử ban đầu của PhoBertClassifier (src/classifier.py).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import RobertaConfig

from core.classifier import (
    ClassificationResult as CoreClassificationResult,
)
from core.classifier import (
    FingerprintMismatchError,
    IntentClassifier,
    save_artifact_fingerprint,
    verify_preprocessing_fingerprint,
)
from core.config import (
    DEFAULT_INTENTS_PATH,
    TAU_INTENT_THRESHOLD,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
)
from core.preprocess import preprocess_fingerprint
from src.classifier import ClassificationResult, PhoBertClassifier
from src.scenarios import Scenario, ScenarioBank, load_scenario_bank
from train.finetune_phobert import run_experiment_pipeline

# =====================================================================
# Fixtures kiểm thử chung
# =====================================================================


@pytest.fixture
def sample_bank() -> ScenarioBank:
    """Fixture ngân hàng kịch bản mẫu cho kiểm thử src/classifier.py."""
    scenarios = [
        Scenario(
            intent="MATHJAX_PHAN_SO",
            category="math_expression",
            describe="Chèn phân số toán học",
            aliases=["phân số", "phân thức"],
            sample_utterances=["tạo phân số với tử và mẫu", "chèn phân thức"],
        ),
        Scenario(
            intent="LATEX_TAO_BANG",
            category="tables",
            describe="Tạo bảng trong LaTeX",
            aliases=["bảng", "tạo bảng", "table"],
            sample_utterances=["tạo một bảng trong tài liệu", "chèn bảng 3 cột"],
        ),
        Scenario(
            intent="LATEX_CHEN_ANH",
            category="figures",
            describe="Chèn hình ảnh vào văn bản",
            aliases=["ảnh", "chèn ảnh", "hình ảnh"],
            sample_utterances=["chèn một bức ảnh vào bài viết", "thêm hình ảnh"],
        ),
    ]
    return ScenarioBank(scenarios)


@pytest.fixture
def tiny_roberta_config() -> RobertaConfig:
    """Cấu hình RobertaConfig siêu nhẹ phục vụ kiểm thử nhanh không cần tải trọng số nặng."""
    return RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        classifier_dropout=0.1,
    )


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Mock bộ tách từ AutoTokenizer tương thích chuẩn PyTorch Transformers."""
    tokenizer = MagicMock()

    def fake_tokenize(
        texts: list[str] | str,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        seq_len = 8
        return {
            "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }

    tokenizer.side_effect = fake_tokenize
    return tokenizer


# =====================================================================
# Phần 1: Kiểm thử ban đầu cho PhoBertClassifier (src/classifier.py)
# =====================================================================


def test_classifier_initialization(sample_bank: ScenarioBank) -> None:
    """Kiểm tra khởi tạo PhoBertClassifier với tham số chuẩn."""
    clf = PhoBertClassifier(scenario_bank=sample_bank, threshold_tau=0.8)
    assert len(clf.labels) == 3
    assert "MATHJAX_PHAN_SO" in clf.labels
    assert clf.label2id["MATHJAX_PHAN_SO"] == 0
    assert clf.id2label[0] == "MATHJAX_PHAN_SO"
    assert clf.device in ("cpu", "cuda")
    assert clf.threshold_tau == 0.8


def test_classifier_predict_exact_match(sample_bank: ScenarioBank) -> None:
    """Kiểm tra dự đoán với câu khớp câu mẫu, độ tin cậy >= tau và accepted = True."""
    clf = PhoBertClassifier(scenario_bank=sample_bank, threshold_tau=0.8)
    res = clf.predict("tạo phân số với tử và mẫu")

    assert isinstance(res, ClassificationResult)
    assert res.intent == "MATHJAX_PHAN_SO"
    assert res.confidence >= 0.8
    assert res.accepted is True
    assert res.category == "math_expression"
    assert res.intent in res.all_probabilities


def test_classifier_predict_alias_match(sample_bank: ScenarioBank) -> None:
    """Kiểm tra dự đoán với câu chứa alias, độ tin cậy >= tau."""
    clf = PhoBertClassifier(scenario_bank=sample_bank, threshold_tau=0.8)
    res = clf.predict("hướng dẫn tạo bảng")

    assert res.intent == "LATEX_TAO_BANG"
    assert res.confidence >= 0.8
    assert res.accepted is True
    assert res.category == "tables"


def test_classifier_predict_low_confidence(sample_bank: ScenarioBank) -> None:
    """Kiểm tra câu lạ / không khớp làm giảm confidence < tau và accepted = False."""
    clf = PhoBertClassifier(scenario_bank=sample_bank, threshold_tau=0.85)
    res = clf.predict("làm thế nào để xuất file pdf nhanh")

    assert isinstance(res, ClassificationResult)
    assert res.accepted is False


def test_classifier_predict_empty_text(sample_bank: ScenarioBank) -> None:
    """Kiểm tra xử lý văn bản rỗng."""
    clf = PhoBertClassifier(scenario_bank=sample_bank)
    res = clf.predict("   ")
    assert res.intent == "UNKNOWN"
    assert res.confidence == 0.0
    assert res.accepted is False


def test_classifier_predict_batch(sample_bank: ScenarioBank) -> None:
    """Kiểm tra dự đoán theo lô (batch)."""
    clf = PhoBertClassifier(scenario_bank=sample_bank)
    texts = [
        "tạo phân số với tử và mẫu",
        "chèn một bức ảnh vào bài viết",
        "chuỗi ngẫu nhiên không có nghĩa",
    ]
    results = clf.predict_batch(texts, batch_size=2)
    assert len(results) == 3
    assert results[0].intent == "MATHJAX_PHAN_SO"
    assert results[1].intent == "LATEX_CHEN_ANH"
    assert results[2].accepted is False


def test_classifier_to_device(sample_bank: ScenarioBank) -> None:
    """Kiểm tra phương thức chuyển đổi thiết bị .to('cpu')."""
    clf = PhoBertClassifier(scenario_bank=sample_bank)
    ret = clf.to("cpu")
    assert ret is clf
    assert clf.device == "cpu"


def test_classifier_with_official_bank() -> None:
    """Kiểm tra khởi tạo bộ phân loại với toàn bộ intent trong ngân hàng kịch bản."""
    if not DEFAULT_INTENTS_PATH.exists():
        pytest.skip("File intents.json chưa sẵn sàng.")

    bank = load_scenario_bank(DEFAULT_INTENTS_PATH)
    clf = PhoBertClassifier(scenario_bank=bank, threshold_tau=TAU_INTENT_THRESHOLD)
    assert len(clf.labels) == len(bank)

    res = clf.predict("tạo phân số với tử và mẫu")
    assert res.intent == "MATH_MATH_BASIC_PHAN_SO"
    assert res.accepted is True


def test_classifier_mock_hf_inference(sample_bank: ScenarioBank) -> None:
    """Kiểm tra đường dẫn suy luận PyTorch/Transformers khi mô hình được nạp đầy đủ."""
    clf = PhoBertClassifier(scenario_bank=sample_bank, threshold_tau=0.8)

    mock_tokenizer_inst = MagicMock()
    mock_input_ids = MagicMock()
    mock_input_ids.to.return_value = mock_input_ids
    mock_tokenizer_inst.return_value = {"input_ids": mock_input_ids}

    mock_model = MagicMock()
    mock_outputs = MagicMock()

    class FakeTensor:
        def __init__(self, data: list[float]) -> None:
            self.data = data

        def squeeze(self, dim: int) -> FakeTensor:
            return self

        def item(self) -> float:
            return self.data[0]

    class FakeTorch:
        @staticmethod
        def softmax(logits: Any, dim: int = -1) -> FakeTensor:
            return FakeTensor([0.95, 0.03, 0.02])

        @staticmethod
        def max(tensor: Any, dim: int = -1) -> tuple[FakeTensor, FakeTensor]:
            return FakeTensor([0.95]), FakeTensor([0])

        @staticmethod
        def topk(tensor: Any, k: int) -> tuple[list[FakeTensor], list[FakeTensor]]:
            return [FakeTensor([0.95]), FakeTensor([0.03])], [
                FakeTensor([0]),
                FakeTensor([1]),
            ]

        class no_grad:
            def __enter__(self) -> None:
                pass

            def __exit__(self, *args: object) -> None:
                pass

    mock_outputs.logits = "fake_logits"
    mock_model.return_value = mock_outputs

    clf._tokenizer = mock_tokenizer_inst
    clf._model = mock_model
    clf._is_hf_loaded = True

    with patch.dict("sys.modules", {"torch": FakeTorch}):
        res = clf.predict("tạo phân số")
        assert res.intent == "MATHJAX_PHAN_SO"
        assert res.confidence == 0.95
        assert res.accepted is True
        assert res.category == "math_expression"


# =====================================================================
# Phần 2: Kiểm thử chuẩn hóa xác suất: sum(p) == 1.0
# =====================================================================


def test_prediction_probabilities_sum_to_one(
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """
    Kiểm tra tổng xác suất dự đoán luôn bằng chính xác 1.0 (sai số số học < 1e-5).

    Cơ sở lý thuyết (Chen et al., 2019 - JointBERT Eq. (1)):
        p^i = softmax(z^i) => sum_k p^i_k = 1.0 và 0.0 <= p^i_k <= 1.0 với mọi k.
    Kiểm tra trên cả suy luận câu đơn lẻ (predict) và suy luận theo lô (predict_batch).
    """
    labels = {0: "MATH_PHAN_SO", 1: "TEX_BANG", 2: "TEX_ANH", 3: "TEX_DOC"}
    clf = IntentClassifier.from_config(
        tiny_roberta_config,
        num_labels=4,
        tokenizer=mock_tokenizer,
        id2label=labels,
        label2id={v: k for k, v in labels.items()},
    )

    # 1. Kiểm thử trên suy luận câu đơn lẻ
    single_inputs = [
        "hướng dẫn tạo phân số trong LaTeX",
        "tạo bảng 3 cột có đường kẻ viền",
        "công thức tích phân $\\int_0^1 x dx$",
    ]
    for text in single_inputs:
        res = clf.predict(text)
        assert isinstance(res, CoreClassificationResult)
        assert 0.0 <= res.confidence <= 1.0

        total_prob = sum(res.all_probabilities.values())
        assert total_prob == pytest.approx(1.0, abs=1e-5), (
            f"Tổng xác suất đơn lẻ phải bằng 1.0, thực tế nhận được: {total_prob}"
        )
        for label, prob in res.all_probabilities.items():
            assert 0.0 <= prob <= 1.0, (
                f"Xác suất của nhãn '{label}' nằm ngoài [0, 1]: {prob}"
            )
            assert label in labels.values()

        # Giá trị confidence phải khớp với xác suất lớn nhất
        max_prob = max(res.all_probabilities.values())
        assert res.confidence == pytest.approx(max_prob, abs=1e-5)

    # 2. Kiểm thử trên suy luận theo lô (batch inference)
    batch_results = clf.predict_batch(single_inputs, batch_size=2)
    assert len(batch_results) == len(single_inputs)
    for b_res in batch_results:
        b_total = sum(b_res.all_probabilities.values())
        assert b_total == pytest.approx(1.0, abs=1e-5), (
            f"Tổng xác suất lô phải bằng 1.0, thực tế: {b_total}"
        )
        for prob in b_res.all_probabilities.values():
            assert 0.0 <= prob <= 1.0


# =====================================================================
# Phần 3: Kiểm thử co giãn nhiệt độ (Temperature Scaling) và bảo toàn thứ hạng
# =====================================================================


def test_temperature_scaling_alters_distribution_without_changing_rank_order(
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """
    Kiểm tra co giãn nhiệt độ (Guo et al., ICML 2017):
    p_i(T) = exp(z_i / T) / sum_j exp(z_j / T) với T > 0.

    Các tính chất cần xác thực:
    1. Thay đổi phân phối:
       - Khi T tăng (T > 1.0): phân phối bị làm mượt (entropy tăng, xác suất cực đại giảm).
       - Khi T giảm (T < 1.0): phân phối trở nên sắc nhọn hơn (xác suất cực đại tăng).
    2. Bảo toàn tuyệt đối thứ tự xếp hạng (Rank Order Invariance):
       Hàm số f(z) = z / T là đơn điệu tăng ngặt với mọi T > 0, do đó:
       z_i > z_j <=> z_i / T > z_j / T <=> p_i(T) > p_j(T).
       Thứ tự sắp xếp của toàn bộ các nhãn không hề thay đổi với bất kỳ nhiệt độ T nào.
    """
    id2label = {
        0: "INTENT_ALPHA",
        1: "INTENT_BETA",
        2: "INTENT_GAMMA",
        3: "INTENT_DELTA",
    }
    clf = IntentClassifier.from_config(
        tiny_roberta_config,
        num_labels=4,
        tokenizer=mock_tokenizer,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )

    # Cố định logits đầu ra với các giá trị phân biệt rõ rệt
    # Logits: ALPHA (4.0) > BETA (2.0) > GAMMA (0.5) > DELTA (-1.0)
    fixed_logits = torch.tensor([[4.0, 2.0, 0.5, -1.0]])
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = fixed_logits
    mock_model.return_value = mock_output
    clf.model = mock_model

    temperatures = [0.5, 0.8, 1.0, 2.0, 5.0]
    results: dict[float, CoreClassificationResult] = {}
    entropies: dict[float, float] = {}

    for temp in temperatures:
        res = clf.predict("câu kiểm tra nhiệt độ", temperature=temp)
        results[temp] = res

        # Tính Shannon entropy: H(p) = -sum(p * ln(p))
        entropy = -sum(
            p * math.log(p) for p in res.all_probabilities.values() if p > 0.0
        )
        entropies[temp] = entropy

    # 1. Xác thực phân phối bị thay đổi (Distribution is altered):
    # Confidence giảm đơn điệu theo nhiệt độ tăng
    assert (
        results[0.5].confidence
        > results[0.8].confidence
        > results[1.0].confidence
        > results[2.0].confidence
        > results[5.0].confidence
    ), "Xác suất cực đại phải giảm dần khi nhiệt độ T tăng."

    # Entropy tăng đơn điệu theo nhiệt độ tăng (phân phối làm mượt đều hơn)
    assert (
        entropies[0.5]
        < entropies[0.8]
        < entropies[1.0]
        < entropies[2.0]
        < entropies[5.0]
    ), "Entropy phải tăng dần khi nhiệt độ T tăng (phân phối mượt hơn)."

    # 2. Xác thực thứ tự xếp hạng được bảo toàn tuyệt đối (Rank order invariance):
    expected_rank = ["INTENT_ALPHA", "INTENT_BETA", "INTENT_GAMMA", "INTENT_DELTA"]
    for temp in temperatures:
        res = results[temp]
        # Nhãn dự đoán cao nhất (argmax) luôn giữ nguyên
        assert res.intent == "INTENT_ALPHA", (
            f"Nhãn dự đoán tại T={temp} bị thay đổi thành '{res.intent}'!"
        )

        # Sắp xếp toàn bộ các lớp theo xác suất giảm dần
        sorted_labels = sorted(
            res.all_probabilities.keys(),
            key=lambda k: res.all_probabilities[k],
            reverse=True,
        )
        assert sorted_labels == expected_rank, (
            f"Thứ tự xếp hạng tại T={temp} bị thay đổi! Nhận được: {sorted_labels}"
        )


# =====================================================================
# Phần 4: Kiểm thử lệch vân tay tiền xử lý ném ngoại lệ FingerprintMismatchError
# =====================================================================


def test_preprocessing_fingerprint_mismatch_raises_exception(
    tmp_path: Path,
    tiny_roberta_config: RobertaConfig,
) -> None:
    """
    Kiểm tra cơ chế phát hiện sai lệch vân tay tiền xử lý tại thời điểm chạy:
    - Khi mã băm SHA-256 lưu trong metadata artifact không khớp với môi trường chạy hiện tại,
      hệ thống bắt buộc phải ném ngoại lệ FingerprintMismatchError (ngăn chặn segmenter drift).
    - Khi vân tay khớp chính xác, xác thực thành công và không ném lỗi.
    - Khi chế độ nghiêm ngặt tắt (strict=False), trả về False an toàn mà không ném lỗi.
    """
    artifact_dir = tmp_path / "mock_model_checkpoint"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ghi vân tay bị làm sai lệch (Mismatched Fingerprint)
    tampered_hash = "deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678"
    save_artifact_fingerprint(artifact_dir, fingerprint_hash=tampered_hash)

    # Kiểm thực nghiêm ngặt bắt buộc ném FingerprintMismatchError
    with pytest.raises(FingerprintMismatchError, match="Lệch vân tay tiền xử lý"):
        verify_preprocessing_fingerprint(artifact_dir, strict=True)

    # Kiểm thực không nghiêm ngặt trả về False
    assert verify_preprocessing_fingerprint(artifact_dir, strict=False) is False

    # Kiểm tra phương thức tiện ích verify_artifact_fingerprint trên IntentClassifier
    mock_tok = MagicMock()
    clf = IntentClassifier.from_config(
        tiny_roberta_config, num_labels=2, tokenizer=mock_tok
    )

    with pytest.raises(FingerprintMismatchError, match="Lệch vân tay tiền xử lý"):
        clf.verify_artifact_fingerprint(artifact_dir, strict=True)
    assert clf.verify_artifact_fingerprint(artifact_dir, strict=False) is False

    # 2. Ghi vân tay chuẩn khớp với môi trường runtime hiện tại
    runtime_hash = preprocess_fingerprint()
    save_artifact_fingerprint(artifact_dir, fingerprint_hash=runtime_hash)

    # Kiểm thực thành công khi vân tay trùng khớp
    assert verify_preprocessing_fingerprint(artifact_dir, strict=True) is True
    assert clf.verify_artifact_fingerprint(artifact_dir, strict=True) is True


# =====================================================================
# Phần 5: Kiểm thử tính nhất quán giữa predict và predict_batch
# =====================================================================


def test_predict_and_predict_batch_yield_consistent_outputs(
    tiny_roberta_config: RobertaConfig,
    mock_tokenizer: MagicMock,
) -> None:
    """
    Kiểm tra suy luận đơn lẻ (predict) và suy luận theo lô (predict_batch) cho kết quả đồng nhất:
    - Nhãn dự đoán (intent) giống hệt nhau.
    - Điểm tin cậy (confidence) bằng nhau (dung sai số học < 1e-5).
    - Trạng thái chấp nhận (accepted) giống hệt nhau.
    - Nhóm danh mục (category) giống hệt nhau.
    - Chuỗi tiền xử lý (preprocessed_text) giống hệt nhau.
    - Phân phối toàn bộ xác suất (all_probabilities) khớp nhau từng giá trị.
    - Tính nhất quán được kiểm chứng trên nhiều kích thước batch_size khác nhau (1, 2, 5).
    """
    id2label = {0: "MATH_PHAN_SO", 1: "TEX_BANG", 2: "TEX_ANH"}
    category_map = {
        "MATH_PHAN_SO": "math_expression",
        "TEX_BANG": "tables",
        "TEX_ANH": "figures",
    }
    clf = IntentClassifier.from_config(
        tiny_roberta_config,
        num_labels=3,
        tokenizer=mock_tokenizer,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        category_map=category_map,
        threshold_tau=0.5,
    )

    test_sentences = [
        "hướng dẫn tạo phân số 1/2 trong LaTeX",
        "chèn một bảng số liệu 3 cột",
        "",  # Mẫu văn bản rỗng
        "thêm hình ảnh minh họa bài viết",
        "   \t\n  ",  # Mẫu khoảng trắng
        "công thức toán ma trận phức tạp",
    ]

    # Suy luận tuần tự từng câu bằng predict
    single_results = [clf.predict(text) for text in test_sentences]

    # Suy luận bằng predict_batch với nhiều kích thước batch_size khác nhau
    for batch_size in (1, 2, 5):
        batch_results = clf.predict_batch(test_sentences, batch_size=batch_size)
        assert len(batch_results) == len(test_sentences)

        for i, (single_res, batch_res) in enumerate(
            zip(single_results, batch_results, strict=True)
        ):
            assert single_res.intent == batch_res.intent, (
                f"Lệch intent tại vị trí {i} với batch_size={batch_size}: "
                f"{single_res.intent} vs {batch_res.intent}"
            )
            assert single_res.confidence == pytest.approx(
                batch_res.confidence, abs=1e-5
            ), (
                f"Lệch confidence tại vị trí {i}: {single_res.confidence} vs {batch_res.confidence}"
            )
            assert single_res.accepted == batch_res.accepted, (
                f"Lệch accepted tại vị trí {i}: {single_res.accepted} vs {batch_res.accepted}"
            )
            assert single_res.category == batch_res.category, (
                f"Lệch category tại vị trí {i}: {single_res.category} vs {batch_res.category}"
            )
            assert single_res.preprocessed_text == batch_res.preprocessed_text, (
                f"Lệch preprocessed_text tại vị trí {i}: "
                f"'{single_res.preprocessed_text}' vs '{batch_res.preprocessed_text}'"
            )

            # Kiểm tra khớp toàn bộ từ điển xác suất
            assert set(single_res.all_probabilities.keys()) == set(
                batch_res.all_probabilities.keys()
            )
            for label, prob in single_res.all_probabilities.items():
                batch_prob = batch_res.all_probabilities[label]
                assert prob == pytest.approx(batch_prob, abs=1e-5), (
                    f"Lệch xác suất nhãn '{label}' tại vị trí {i}: {prob} vs {batch_prob}"
                )


# =====================================================================
# Phần 6: Kiểm thử tích hợp huấn luyện thử nghiệm siêu tốc trên CPU (Smoke Training)
# =====================================================================


def test_smoke_training_completes_successfully_on_cpu(tmp_path: Path) -> None:
    """
    Kiểm thử tích hợp đầu-cuối: Quy trình huấn luyện thử nghiệm (Smoke Training)
    chạy thành công trên CPU và tạo ra mô hình suy luận hoàn chỉnh có thể nạp lại được.

    Các bước xác thực:
    1. Huấn luyện thử nghiệm siêu tốc với smoke_mode=True trên thiết bị device='cpu'.
    2. Xác thực tính toàn vẹn của mã băm SHA-256 tập test (Honesty Rule #3).
    3. Kiểm tra tính hợp lệ của các chỉ số đánh giá (accuracy, macro-F1, ECE).
    4. Kiểm tra sự tồn tại của các tệp artifact xuất xưởng:
       - pytorch_model.bin: Trọng số checkpoint.
       - config.json: Cấu hình kiến trúc.
       - label2id.json, id2label.json: Bảng ánh xạ nhãn xác định.
       - preprocessing_fingerprint.json: Metadata vân tay tiền xử lý.
       - phobert_results.json: Báo cáo kết quả huấn luyện.
    5. Kiểm thử tích hợp suy luận: Nạp checkpoint vừa xuất bằng IntentClassifier.from_pretrained()
       với strict_fingerprint=True và thực hiện suy luận dự đoán thành công.
    """
    checkpoint_dir = tmp_path / "smoke_checkpoints"
    results_path = tmp_path / "smoke_phobert_results.json"

    # Chạy quy trình thực nghiệm ở chế độ smoke run trên CPU
    results = run_experiment_pipeline(
        target="category",
        smoke_mode=True,
        device="cpu",
        train_path=TRAIN_CSV_PATH,
        val_path=VAL_CSV_PATH,
        test_path=TEST_CSV_PATH,
        checkpoint_dir=checkpoint_dir,
        results_path=results_path,
    )

    # 1. Xác thực thông số chế độ chạy
    assert results["target"] == "category"
    assert results["smoke_mode"] is True
    assert "test_csv_sha256" in results
    assert len(results["test_csv_sha256"]) == 64, (
        "Mã băm test_csv_sha256 phải là SHA-256 (64 ký tự hex)."
    )
    assert "preprocessing_fingerprint" in results

    # 2. Xác thực các chỉ số đánh giá hợp lệ trong miền [0.0, 1.0]
    assert "aggregate_test_metrics" in results
    agg_metrics = results["aggregate_test_metrics"]
    for metric_name in ("accuracy_mean", "macro_f1_mean", "ece_mean"):
        assert metric_name in agg_metrics, (
            f"Thiếu chỉ số '{metric_name}' trong báo cáo kết quả."
        )
        val = agg_metrics[metric_name]
        assert 0.0 <= val <= 1.0, (
            f"Chỉ số '{metric_name}' nằm ngoài khoảng [0, 1]: {val}"
        )

    # 3. Xác thực các tệp tin artifact được ghi đầy đủ
    assert results_path.exists(), "Tệp phobert_results.json không được tạo."
    assert (checkpoint_dir / "pytorch_model.bin").exists(), (
        "Thiếu tệp trọng số pytorch_model.bin."
    )
    assert (checkpoint_dir / "config.json").exists(), "Thiếu tệp cấu hình config.json."
    assert (checkpoint_dir / "label2id.json").exists(), (
        "Thiếu tệp ánh xạ label2id.json."
    )
    assert (checkpoint_dir / "id2label.json").exists(), (
        "Thiếu tệp ánh xạ id2label.json."
    )
    assert (checkpoint_dir / "preprocessing_fingerprint.json").exists(), (
        "Thiếu tệp preprocessing_fingerprint.json."
    )

    # 4. Tích hợp suy luận: Nạp lại checkpoint vừa huấn luyện và thực hiện dự đoán
    loaded_clf = IntentClassifier.from_pretrained(
        checkpoint_dir,
        device="cpu",
        verify_fingerprint=True,
        strict_fingerprint=True,
    )
    assert loaded_clf.num_labels > 0
    assert loaded_clf.device == "cpu"

    # Kiểm tra dự đoán với văn bản tiếng Việt
    sample_prediction = loaded_clf.predict("hướng dẫn tạo bảng số liệu trong LaTeX")
    assert isinstance(sample_prediction, CoreClassificationResult)
    assert sample_prediction.intent in loaded_clf.label2id
    assert 0.0 <= sample_prediction.confidence <= 1.0
    assert sum(sample_prediction.all_probabilities.values()) == pytest.approx(
        1.0, abs=1e-5
    )
