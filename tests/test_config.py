"""
Kiểm thử đơn vị và xác minh tính toàn vẹn cho module cấu hình core/config.py.

Tuân thủ kiến trúc và các quy tắc AGENTS.md:
- Kiểm tra tính tồn tại của các thư mục và tệp chính yếu đã cấu hình.
- Xác thực tính tất định (deterministic reproducibility) của hàm set_seed khi sinh tensor giả ngẫu nhiên.
- Kiểm tra các siêu tham số, cấu hình biên dịch LaTeX an toàn (-no-shell-escape), và định danh mô hình.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config import (
    BASE_DIR,
    CLI_BANNER_TITLE,
    CLI_EXIT_COMMANDS,
    CLI_PROMPT,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_TIMING_RUNS,
    DEFAULT_EVAL_WARMUP_RUNS,
    DEFAULT_EVALUATION_REPORT_PATH,
    DEFAULT_EVALUATION_SUMMARY_PATH,
    DEFAULT_INTENTS_PATH,
    DEFAULT_LATEX_ENGINE,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_SEED,
    EMBED_MODEL_NAME,
    EMBED_MODEL_REQUIRES_WORD_SEGMENTATION,
    HF_CACHE_DIR,
    LATEX_COMMAND_FLAGS,
    LATEX_COMPILE_TIMEOUT_SECONDS,
    LATEX_STATIC_CHECK_ONLY,
    MAX_SEQ_LENGTH,
    MODELS_DIR,
    PHOBERT_DIR,
    PHOBERT_MAX_SEQUENCE_LIMIT,
    PHOBERT_MODEL_NAME,
    RAW_SENTENCES_PATH,
    REPORT_DIR,
    REQUIRED_TEMPLATE_KEYS,
    SBERT_DIR,
    SBERT_MODEL_NAME,
    SOURCE_COVERAGE_PATH,
    SUPPORTED_LATEX_ENGINES,
    SUPPORTED_TARGETS,
    TAU_INTENT_THRESHOLD,
    TEST_CSV_PATH,
    TEST_CSV_SHA256,
    THETA_RETRIEVAL_THRESHOLD,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    VERIFIED_HF_MODELS,
    get_default_device,
    resolve_phobert_tokenizer_dir,
    resolve_sbert_model_dir,
    set_seed,
    runtime_model_paths,
    verify_hf_model_on_hub,
)


def test_primary_directories_exist() -> None:
    """
    Xác minh tất cả các thư mục chính được cấu hình trong core/config.py đều tồn tại trên đĩa.
    Bao gồm thư mục gốc, thư mục dữ liệu, báo cáo, và mô hình offline.
    """
    primary_directories = [
        BASE_DIR,
        DATA_DIR,
        REPORT_DIR,
        MODELS_DIR,
        HF_CACHE_DIR,
    ]
    for directory in primary_directories:
        assert isinstance(directory, Path), f"Đường dẫn không phải kiểu pathlib.Path: {directory}"
        assert directory.exists(), f"Thư mục chính không tồn tại trên hệ thống tệp: {directory}"
        assert directory.is_dir(), f"Đường dẫn phải là thư mục: {directory}"


def test_primary_paths_exist() -> None:
    """
    Xác minh tất cả các đường dẫn tệp dữ liệu huấn luyện và báo cáo số liệu chính tồn tại trên đĩa.
    Tuân thủ Honesty Rule #2 và #3: tệp dữ liệu và báo cáo phải thực sự tồn tại.
    """
    primary_file_paths = [
        DEFAULT_INTENTS_PATH,
        TRAIN_CSV_PATH,
        VAL_CSV_PATH,
        TEST_CSV_PATH,
        RAW_SENTENCES_PATH,
        SOURCE_COVERAGE_PATH,
        DEFAULT_EVALUATION_REPORT_PATH,
        DEFAULT_EVALUATION_SUMMARY_PATH,
    ]
    for file_path in primary_file_paths:
        assert isinstance(file_path, Path), f"Đường dẫn không phải kiểu pathlib.Path: {file_path}"
        assert file_path.exists(), f"Tệp cấu hình chính không tồn tại trên hệ thống tệp: {file_path}"
        assert file_path.is_file(), f"Đường dẫn phải là tệp: {file_path}"


def test_path_configurations() -> None:
    """Kiểm tra cấu trúc quan hệ phân cấp của toàn bộ đường dẫn pathlib.Path trong dự án."""
    assert isinstance(BASE_DIR, Path)
    assert BASE_DIR.is_dir()

    for p in (DATA_DIR, REPORT_DIR, MODELS_DIR, HF_CACHE_DIR):
        assert isinstance(p, Path)
        assert p.parent == BASE_DIR or p.parent == MODELS_DIR

    for file_path in (
        DEFAULT_INTENTS_PATH,
        TRAIN_CSV_PATH,
        VAL_CSV_PATH,
        TEST_CSV_PATH,
        RAW_SENTENCES_PATH,
        SOURCE_COVERAGE_PATH,
        DEFAULT_EVALUATION_REPORT_PATH,
        DEFAULT_EVALUATION_SUMMARY_PATH,
        PHOBERT_DIR,
        SBERT_DIR,
    ):
        assert isinstance(file_path, Path)
        assert file_path.is_relative_to(BASE_DIR)


def test_runtime_model_path_overrides_are_independent_of_vscode(tmp_path: Path) -> None:
    paths = runtime_model_paths(
        {
            "KA_MODEL_ROOT": "workspace-models",
            "KA_HF_CACHE_DIR": "cache/hf",
            "KA_CHECKPOINT_DIR": "custom/intent-checkpoint",
            "KA_SBERT_DIR": "custom/retrieval-model",
            "KA_INDEX_CACHE_DIR": "runtime-index",
        },
        base_dir=tmp_path,
    )

    assert paths.root == (tmp_path / "workspace-models").resolve()
    assert paths.hf_cache == (tmp_path / "cache/hf").resolve()
    assert paths.phobert == (tmp_path / "cache/hf/phobert-base-v2").resolve()
    assert paths.checkpoint == (tmp_path / "custom/intent-checkpoint").resolve()
    assert paths.sbert == (tmp_path / "custom/retrieval-model").resolve()
    assert paths.index_cache == (tmp_path / "runtime-index").resolve()


def test_local_huggingface_snapshots_are_discovered_from_runtime_cache(tmp_path: Path, monkeypatch) -> None:
    import core.config as config

    cache = tmp_path / "hf_cache"
    phobert = cache / "hub" / "models--vinai--phobert-base-v2" / "snapshots" / "abc123"
    phobert.mkdir(parents=True)
    (phobert / "vocab.txt").write_text("<s>\n</s>\n", encoding="utf-8")
    (phobert / "bpe.codes").write_text("#version: 0.2\n", encoding="utf-8")
    sbert = cache / "hub" / "models--bkai-foundation-models--vietnamese-bi-encoder" / "snapshots" / "def456"
    sbert.mkdir(parents=True)
    (sbert / "config.json").write_text('{"model_type":"bert"}', encoding="utf-8")
    (sbert / "pytorch_model.bin").write_bytes(b"weights")
    (sbert / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config, "HF_CACHE_DIR", cache)
    monkeypatch.setattr(config, "PHOBERT_DIR", tmp_path / "missing-phobert")
    monkeypatch.setattr(config, "SBERT_DIR", tmp_path / "missing-sbert")

    assert resolve_phobert_tokenizer_dir() == phobert
    assert resolve_sbert_model_dir() == sbert


def test_hyperparameters_and_thresholds() -> None:
    """Kiểm tra các siêu tham số và ngưỡng phân loại/truy hồi theo kiến trúc AGENTS.md."""
    assert 0.0 < TAU_INTENT_THRESHOLD <= 1.0
    assert 0.0 < THETA_RETRIEVAL_THRESHOLD <= 1.0
    assert THETA_RETRIEVAL_THRESHOLD < TAU_INTENT_THRESHOLD
    assert MAX_SEQ_LENGTH > 0
    assert PHOBERT_MAX_SEQUENCE_LIMIT == 64
    assert DEFAULT_BATCH_SIZE > 0
    assert DEFAULT_RETRIEVAL_TOP_K >= 1
    assert DEFAULT_SEED == 42
    assert DEFAULT_EVAL_WARMUP_RUNS > 0
    assert DEFAULT_EVAL_TIMING_RUNS > 0
    assert len(TEST_CSV_SHA256) == 64


def test_template_and_cli_configurations() -> None:
    """Kiểm tra cấu hình target template và giao diện dòng lệnh."""
    assert "latex" in SUPPORTED_TARGETS
    assert "latex" in REQUIRED_TEMPLATE_KEYS
    assert set(REQUIRED_TEMPLATE_KEYS).issubset(set(SUPPORTED_TARGETS))

    assert len(CLI_BANNER_TITLE) > 0
    assert len(CLI_PROMPT) > 0
    assert "exit" in CLI_EXIT_COMMANDS


def test_latex_engine_configurations() -> None:
    """Kiểm tra cấu hình engine LaTeX tuân thủ quy tắc bảo mật không dùng shell escape."""
    assert DEFAULT_LATEX_ENGINE in SUPPORTED_LATEX_ENGINES
    assert "pdflatex" in SUPPORTED_LATEX_ENGINES
    assert "-no-shell-escape" in LATEX_COMMAND_FLAGS
    assert LATEX_COMPILE_TIMEOUT_SECONDS > 0
    assert LATEX_STATIC_CHECK_ONLY == "static check only"


def test_huggingface_model_definitions() -> None:
    """Kiểm tra danh sách mô hình HuggingFace đã kiểm chứng trên Hub."""
    assert PHOBERT_MODEL_NAME == "vinai/phobert-base-v2"
    assert SBERT_MODEL_NAME == "bkai-foundation-models/vietnamese-bi-encoder"
    assert EMBED_MODEL_NAME == SBERT_MODEL_NAME
    assert EMBED_MODEL_REQUIRES_WORD_SEGMENTATION is True
    assert PHOBERT_MODEL_NAME in VERIFIED_HF_MODELS
    assert SBERT_MODEL_NAME in VERIFIED_HF_MODELS


def test_set_seed_deterministic_tensor_generation() -> None:
    """
    Xác minh hàm set_seed đảm bảo tính tất định khi sinh tensor giả ngẫu nhiên qua các lần gọi lặp lại.
    Kiểm tra cả torch.randn, torch.rand, và torch.randint với cùng hạt giống seed.
    """
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch chưa được cài đặt trong môi trường này")

    test_seed = 42
    shape = (4, 8)

    # Lần gọi 1: Thiết lập seed và sinh tensor
    set_seed(test_seed)
    randn_first = torch.randn(*shape)
    rand_first = torch.rand(*shape)
    randint_first = torch.randint(low=0, high=1000, size=shape)

    # Lần gọi 2: Thiết lập lại cùng seed và sinh lại tensor
    set_seed(test_seed)
    randn_second = torch.randn(*shape)
    rand_second = torch.rand(*shape)
    randint_second = torch.randint(low=0, high=1000, size=shape)

    # Khẳng định tính tất định tuyệt đối giữa các lần gọi
    assert torch.equal(randn_first, randn_second), "torch.randn không tất định qua các lần gọi lặp lại"
    assert torch.equal(rand_first, rand_second), "torch.rand không tất định qua các lần gọi lặp lại"
    assert torch.equal(randint_first, randint_second), "torch.randint không tất định qua các lần gọi lặp lại"

    # Lần gọi 3: Hạt giống khác phải tạo ra kết quả khác nhau
    set_seed(test_seed + 1)
    randn_different_seed = torch.randn(*shape)
    assert not torch.equal(randn_first, randn_different_seed), "Hai seed khác nhau không được sinh cùng tensor"

    # Kiểm tra tính tất định trên GPU nếu CUDA khả dụng
    if torch.cuda.is_available():
        set_seed(test_seed)
        cuda_first = torch.randn(*shape, device="cuda")
        set_seed(test_seed)
        cuda_second = torch.randn(*shape, device="cuda")
        assert torch.equal(cuda_first, cuda_second), "Tensor sinh trên CUDA không tất định sau khi gán seed"


def test_set_seed_deterministic_numpy_and_random() -> None:
    """Kiểm tra tính tất định của set_seed đối với thư viện chuẩn random và numpy."""
    # Kiểm tra random chuẩn
    set_seed(123)
    val_1 = [random.random() for _ in range(5)]
    set_seed(123)
    val_2 = [random.random() for _ in range(5)]
    assert val_1 == val_2

    # Kiểm tra numpy nếu có
    try:
        import numpy as np

        set_seed(123)
        np_arr1 = np.random.randn(3, 3)
        set_seed(123)
        np_arr2 = np.random.randn(3, 3)
        assert np.array_equal(np_arr1, np_arr2)
    except ImportError:
        pass


def test_set_seed_with_mocked_libraries() -> None:
    """Kiểm tra các lệnh gọi thiết lập deterministic backend khi thư viện numpy và torch được nạp."""
    mock_np = MagicMock()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with patch.dict(sys.modules, {"numpy": mock_np, "torch": mock_torch}):
        res = set_seed(999)
        assert res == 999
        mock_np.random.seed.assert_called_once_with(999)
        mock_torch.manual_seed.assert_called_once_with(999)
        mock_torch.cuda.manual_seed_all.assert_called_once_with(999)
        assert mock_torch.backends.cudnn.deterministic is True
        assert mock_torch.backends.cudnn.benchmark is False


def test_get_default_device() -> None:
    """Kiểm tra hàm get_default_device khi có hoặc không có GPU."""
    # Khi torch chưa cài đặt
    with patch.dict(sys.modules, {"torch": None}):
        device = get_default_device()
        assert device == "cpu"

    # Khi torch có cuda
    mock_torch_cuda = MagicMock()
    mock_torch_cuda.cuda.is_available.return_value = True
    with patch.dict(sys.modules, {"torch": mock_torch_cuda}):
        assert get_default_device() == "cuda"

    # Khi torch chỉ có cpu
    mock_torch_cpu = MagicMock()
    mock_torch_cpu.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": mock_torch_cpu}):
        assert get_default_device() == "cpu"


def test_verify_hf_model_on_hub_mocked() -> None:
    """Kiểm tra hàm verify_hf_model_on_hub với phản hồi HTTP giả lập."""
    # Kịch bản mô hình tồn tại
    fake_response = MagicMock()
    fake_response.status = 200
    fake_response.read.return_value = b'{"id": "vinai/phobert-base-v2", "modelId": "vinai/phobert-base-v2"}'
    fake_response.__enter__.return_value = fake_response

    with patch("urllib.request.urlopen", return_value=fake_response):
        assert verify_hf_model_on_hub("vinai/phobert-base-v2") is True

    # Kịch bản mô hình không tìm thấy hoặc lỗi mạng
    with patch("urllib.request.urlopen", side_effect=OSError("Network error")):
        assert verify_hf_model_on_hub("nonexistent/model-test") is False


def test_biencoder_fine_tuning_configurations() -> None:
    """Kiểm tra các hằng số cấu hình tinh chỉnh Bi-Encoder (KA-17)."""
    from core.config import (
        BIENCODER_CANDIDATE_BATCH_SIZES,
        BIENCODER_DEFAULT_BATCH_SIZE,
        BIENCODER_DEFAULT_LR,
        BIENCODER_DEFAULT_PATIENCE,
        BIENCODER_MAX_EPOCHS,
        BIENCODER_MIN_EPOCHS,
        BIENCODER_WARMUP_RATIO,
        DEFAULT_BIENCODER_CHECKPOINT_DIR,
        DEFAULT_BIENCODER_RESULTS_PATH,
        DEFAULT_SHARED_WEIGHTS,
        DEFAULT_TRAINING_MODE,
        SMOKE_RUN_BIENCODER_BATCH_SIZE,
        SMOKE_RUN_BIENCODER_EPOCHS,
        SMOKE_RUN_BIENCODER_MAX_SAMPLES,
        SUPPORTED_TRAINING_MODES,
        TRAINING_MODE_A,
        TRAINING_MODE_A2,
        TRAINING_MODE_B,
    )

    assert TRAINING_MODE_A in SUPPORTED_TRAINING_MODES
    assert TRAINING_MODE_A2 in SUPPORTED_TRAINING_MODES
    assert TRAINING_MODE_B in SUPPORTED_TRAINING_MODES
    assert DEFAULT_TRAINING_MODE == "mode_a"
    assert DEFAULT_SHARED_WEIGHTS is True
    assert BIENCODER_DEFAULT_LR == 2e-5
    assert 16 in BIENCODER_CANDIDATE_BATCH_SIZES
    assert 32 in BIENCODER_CANDIDATE_BATCH_SIZES
    assert BIENCODER_DEFAULT_BATCH_SIZE == 16
    assert BIENCODER_MIN_EPOCHS == 3
    assert BIENCODER_MAX_EPOCHS == 10
    assert BIENCODER_DEFAULT_PATIENCE == 3
    assert BIENCODER_WARMUP_RATIO == 0.1
    assert isinstance(DEFAULT_BIENCODER_CHECKPOINT_DIR, Path)
    assert isinstance(DEFAULT_BIENCODER_RESULTS_PATH, Path)
    assert SMOKE_RUN_BIENCODER_EPOCHS >= 1
    assert SMOKE_RUN_BIENCODER_BATCH_SIZE >= 1
    assert SMOKE_RUN_BIENCODER_MAX_SAMPLES >= 4
