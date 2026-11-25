"""
Module quản lý cấu hình tập trung và hạt giống ngẫu nhiên của toàn bộ dự án.

Tuân thủ quy tắc kiến trúc AGENTS.md:
- Tất cả hằng số cấu hình được đặt tại core/config.py, không dùng số ma thuật (magic numbers).
- Hạt giống ngẫu nhiên (seed) được thiết lập thông qua hàm set_seed tập trung (random, numpy, torch).
- An toàn biên dịch LaTeX: Luôn dùng cờ -no-shell-escape và không biên dịch văn bản ngoài ngân hàng mẫu.
- Các mô hình HuggingFace được kiểm chứng tính khả dụng trên HuggingFace Hub:
  + vinai/phobert-base-v2 (PhoBERT tiếng Việt, kiến trúc RoBERTa-base pre-trained)
  + bkai-foundation-models/vietnamese-bi-encoder (Sentence-BERT tiếng Việt cho retrieval)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

# Đường dẫn thư mục gốc dự án (sử dụng pathlib.Path theo quy chuẩn)
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
PROGRESS_PATH: Final[Path] = BASE_DIR / "PROGRESS.md"

# Đường dẫn dữ liệu
DATA_DIR: Final[Path] = BASE_DIR / "data"
DEFAULT_INTENTS_PATH: Final[Path] = DATA_DIR / "intents.json"
DATA_STANDARDS_DIR: Final[Path] = DATA_DIR / "standards"
DATA_CONTRACT_MANIFEST_PATH: Final[Path] = DATA_STANDARDS_DIR / "manifest.json"
INTENT_DEFINITION_SCHEMA_PATH: Final[Path] = DATA_STANDARDS_DIR / "intent-definition.schema.json"
KNOWLEDGE_CATALOG_PATH: Final[Path] = DATA_DIR / "knowledge" / "catalog.json"
GUI_STATIC_DIR: Final[Path] = BASE_DIR / "gui" / "static"
GUI_MOUNT_PATH: Final[str] = "/ui"

# Guardrails cho JSONata expressions do người dùng cấu hình trong knowledge definition.
DEFAULT_JSONATA_TIMEOUT_MS: Final[int] = 1000
DEFAULT_JSONATA_STACK_LIMIT: Final[int] = 500
DEFAULT_JSONATA_SEQUENCE_LIMIT: Final[int] = 100_000
DEFINITION_BACKUP_SUFFIX: Final[str] = ".bak"
ENGINE_VERSION_LEGACY: Final[str] = "legacy"
ENGINE_VERSION_GENERIC: Final[str] = "v2"
SUPPORTED_ENGINE_VERSIONS: Final[Sequence[str]] = (
    ENGINE_VERSION_LEGACY,
    ENGINE_VERSION_GENERIC,
)
TRAIN_CSV_PATH: Final[Path] = DATA_DIR / "train.csv"
VAL_CSV_PATH: Final[Path] = DATA_DIR / "val.csv"
TEST_CSV_PATH: Final[Path] = DATA_DIR / "test.csv"
TEST_OOS_PATH: Final[Path] = DATA_DIR / "test_oos.csv"
TEST_OWNER_PATH: Final[Path] = DATA_DIR / "test_owner.csv"
TEST_OWNER_TEMPLATE_PATH: Final[Path] = DATA_DIR / "test_owner_template.csv"
RAW_SENTENCES_PATH: Final[Path] = DATA_DIR / "raw_sentences.csv"
SOURCE_COVERAGE_PATH: Final[Path] = DATA_DIR / "source_coverage.csv"
EXTERNAL_DATA_DIR: Final[Path] = DATA_DIR / "external"
PHOATIS_DIR: Final[Path] = EXTERNAL_DATA_DIR / "phoatis"
MINED_CANDIDATES_PATH: Final[Path] = DATA_DIR / "mined_candidates.csv"
SYNTH_SENTENCES_PATH: Final[Path] = DATA_DIR / "synth_sentences.py"
TRAIN_METADATA_PATH: Final[Path] = DATA_DIR / "train_metadata.csv"

# Cấu hình sinh dữ liệu tổng hợp (Synthetic Data Generation)
SYNTH_TRAIN_SAMPLES_PER_INTENT: Final[int] = 10
DEFAULT_INTEGER_MIN: Final[int] = -9
DEFAULT_INTEGER_MAX: Final[int] = 9
MIN_FRAMES_PER_INTENT: Final[int] = 8
MAX_FRAMES_PER_INTENT: Final[int] = 10

# Cấu hình tổng hợp và kiểm định tập dữ liệu (KA-11 Dataset Compilation Pipeline)
BUILD_DATASET_PATH: Final[Path] = DATA_DIR / "build_dataset.py"
LEAKAGE_REPORT_PATH: Final[Path] = DATA_DIR / "leakage_report.py"
MIN_TRAIN_SAMPLES_PER_INTENT: Final[int] = 10
MIN_TEST_SAMPLES_PER_INTENT: Final[int] = 5
OUT_OF_SCOPE_INTENT: Final[str] = "OUT_OF_SCOPE"
DEFAULT_OUT_OF_SCOPE_CATEGORY: Final[str] = "tex_misc"

# Cấu hình nguồn dữ liệu ngoại bộ PhoATIS (VinAI Research JointIDSF)
PHOATIS_BASE_URL: Final[str] = (
    "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level"
)
PHOATIS_TRAIN_URL: Final[str] = f"{PHOATIS_BASE_URL}/train/seq.in"
PHOATIS_DEV_URL: Final[str] = f"{PHOATIS_BASE_URL}/dev/seq.in"
PHOATIS_TEST_URL: Final[str] = f"{PHOATIS_BASE_URL}/test/seq.in"

# Đường dẫn báo cáo số liệu thực nghiệm
REPORT_DIR: Final[Path] = BASE_DIR / "report"
FIGURES_DIR: Final[Path] = REPORT_DIR / "figures"
DEFAULT_CONFUSION_MATRIX_PATH: Final[Path] = FIGURES_DIR / "confusion_matrix.png"
DEFAULT_EVALUATION_REPORT_PATH: Final[Path] = REPORT_DIR / "baseline_results.json"
DEFAULT_EVALUATION_SUMMARY_PATH: Final[Path] = REPORT_DIR / "REPORT.md"
DEFAULT_DATA_REPORT_PATH: Final[Path] = REPORT_DIR / "data_report.txt"
DEFAULT_BASELINE_RESULTS_PATH: Final[Path] = REPORT_DIR / "baseline_results.json"
DEFAULT_BASELINE_SVM_CM_PATH: Final[Path] = FIGURES_DIR / "baseline_svm_confusion_matrix.png"
DEFAULT_BASELINE_FUZZY_CM_PATH: Final[Path] = FIGURES_DIR / "baseline_fuzzy_confusion_matrix.png"
DEFAULT_RETRIEVAL_RESULTS_PATH: Final[Path] = REPORT_DIR / "retrieval_results.json"
DEFAULT_RETRIEVAL_SUMMARY_PATH: Final[Path] = REPORT_DIR / "retrieval_evaluation_report.md"
DEFAULT_COMPARISON_RESULTS_PATH: Final[Path] = REPORT_DIR / "model_comparison_results.json"
DEFAULT_COMPARISON_REPORT_PATH: Final[Path] = REPORT_DIR / "model_comparison_report.md"
DEFAULT_COMPARISON_FIGURE_PATH: Final[Path] = FIGURES_DIR / "model_comparison_deltas.png"
DEFAULT_RISK_COVERAGE_PATH: Final[Path] = FIGURES_DIR / "risk_coverage.png"
DEFAULT_CALIBRATION_RESULTS_PATH: Final[Path] = REPORT_DIR / "calibration_results.json"
CALIBRATE_TAU_SCRIPT_PATH: Final[Path] = BASE_DIR / "train" / "calibrate_tau.py"

# Cấu hình so sánh mô hình và kiểm định ý nghĩa thống kê (KA-15)
DEFAULT_COMPARISON_BASELINE_TYPE: Final[str] = "ka04_retrieval"
SUPPORTED_COMPARISON_BASELINES: Final[Sequence[str]] = (
    "ka04_retrieval",
    "sbert",
    "tfidf_svm",
    "fuzzy",
)
DEFAULT_SIGNIFICANCE_ALPHA: Final[float] = 0.05

# Cấu hình mô hình baseline TF-IDF LinearSVC và Fuzzy Matching (KA-12)
BASELINE_C_CANDIDATES: Final[Sequence[float]] = (0.1, 1.0, 10.0)
DEFAULT_BASELINE_C: Final[float] = 1.0
DEFAULT_TFIDF_NGRAM_RANGE: Final[tuple[int, int]] = (1, 2)
DEFAULT_TFIDF_SUBLINEAR_TF: Final[bool] = True
DEFAULT_SVM_MAX_ITER: Final[int] = 2000

# Cấu hình phát hiện rò rỉ dữ liệu và tương đồng n-gram (KA-12 Data Leakage Reporter)
TRIGRAM_SIMILARITY_THRESHOLD: Final[float] = 0.8
TRIGRAM_N: Final[int] = 3
SPLIT_SHIFT_TRIGRAM_WARNING_THRESHOLD: Final[float] = 0.8
SPLIT_SHIFT_BOILERPLATE_WARNING_THRESHOLD: Final[float] = 0.5
SPLIT_SHIFT_TRIGRAM_SAMPLE_SIZE: Final[int] = 300

# Cấu hình thống kê và đo lường đánh giá học thuật (KA-09, KA-17)
DEFAULT_BOOTSTRAP_ROUNDS: Final[int] = 1000
DEFAULT_CONFIDENCE_LEVEL: Final[float] = 0.95
DEFAULT_ECE_BINS: Final[int] = 10
DEFAULT_ZERO_DIVISION: Final[float] = 0.0
DEFAULT_RANKING_TOP_K: Final[int] = 3
DEFAULT_MCNEMAR_ALPHA: Final[float] = 0.05
TEST_CSV_SHA256_REGEX: Final[str] = r"sha256\s+test\.csv:\s*([a-fA-F0-9]{64})"
DEFAULT_ABLATION_METRICS: Final[Sequence[str]] = ("accuracy", "macro_f1")

# Mã băm SHA-256 cố định của data/test.csv theo Honesty Rule #3 (AGENTS.md)
TEST_CSV_SHA256: Final[str] = "ef8781ba7d231aceaf9043b394acd343a23e6beac8454d168bfe4a19ad3886fd"
LEGACY_TEST_CSV_SHA256: Final[str] = "637f54fdae2d1be7d39ca960ce91003adb2f50a09ee446b87fbb092b1b9c6066"

class RuntimeModelPaths(NamedTuple):
    """Các vị trí model theo cấu hình môi trường của một tiến trình Python."""

    root: Path
    hf_cache: Path
    phobert: Path
    sbert: Path
    checkpoint: Path
    checkpoints: Path
    index_cache: Path


def runtime_model_paths(
    environ: dict[str, str] | None = None,
    base_dir: Path | str = BASE_DIR,
) -> RuntimeModelPaths:
    """Tính đường dẫn runtime từ override mà không phụ thuộc vào VS Code."""
    environment = os.environ if environ is None else environ
    base = Path(base_dir).expanduser().resolve()

    def resolve(env_key: str, default: Path) -> Path:
        candidate = Path(environment.get(env_key, str(default)))
        candidate = candidate.expanduser()
        return (candidate if candidate.is_absolute() else base / candidate).resolve()

    root = resolve("KA_MODEL_ROOT", base / "models")
    hf_cache = resolve("KA_HF_CACHE_DIR", root / "hf_cache")
    checkpoints = resolve("KA_CHECKPOINTS_DIR", root / "checkpoints")
    return RuntimeModelPaths(
        root=root,
        hf_cache=hf_cache,
        phobert=resolve("KA_PHOBERT_DIR", hf_cache / "phobert-base-v2"),
        sbert=resolve("KA_SBERT_DIR", hf_cache / "vietnamese-bi-encoder"),
        checkpoint=resolve("KA_CHECKPOINT_DIR", checkpoints / "phobert_classifier"),
        checkpoints=checkpoints,
        index_cache=resolve("KA_INDEX_CACHE_DIR", root),
    )


# Không có override thì mọi lệnh CLI/training vẫn giữ cây BASE_DIR/models cũ.
_RUNTIME_MODEL_PATHS = runtime_model_paths()
MODEL_ROOT: Final[Path] = _RUNTIME_MODEL_PATHS.root
MODELS_DIR: Final[Path] = MODEL_ROOT
HF_CACHE_DIR: Final[Path] = _RUNTIME_MODEL_PATHS.hf_cache
PHOBERT_DIR: Final[Path] = _RUNTIME_MODEL_PATHS.phobert
PHOBERT_MODEL_NAME: Final[str] = "vinai/phobert-base-v2"
SBERT_DIR: Final[Path] = _RUNTIME_MODEL_PATHS.sbert
SBERT_MODEL_NAME: Final[str] = "bkai-foundation-models/vietnamese-bi-encoder"
EMBED_MODEL_NAME: Final[str] = SBERT_MODEL_NAME
EMBED_MODEL_REQUIRES_WORD_SEGMENTATION: Final[bool] = True

# Danh sách định danh mô hình HuggingFace chính thức đã được xác thực trên Hub
VERIFIED_HF_MODELS: Final[Sequence[str]] = (
    PHOBERT_MODEL_NAME,
    SBERT_MODEL_NAME,
)

# Cấu hình kiến trúc bộ phân loại ý định JointBERT-Style PhoBERT (KA-11, JointBERT Eq.(1))
DEFAULT_CLASSIFIER_DROPOUT: Final[float] = 0.1
PHOBERT_HIDDEN_DIM: Final[int] = 768

# Cấu hình tinh chỉnh (Fine-Tuning) PhoBERT và tìm kiếm siêu tham số (KA-11)
DEFAULT_FINETUNE_TARGET: Final[str] = "intent"
SUPPORTED_CLASSIFICATION_TARGETS: Final[Sequence[str]] = ("intent", "category")
PHOBERT_CANDIDATE_LRS: Final[Sequence[float]] = (1e-5, 2e-5, 3e-5, 5e-5)
DEFAULT_LEARNING_RATE: Final[float] = 2e-5
DEFAULT_FINETUNE_SEEDS: Final[Sequence[int]] = (42, 43, 44)
DEFAULT_EARLY_STOPPING_PATIENCE: Final[int] = 5
DEFAULT_WARMUP_RATIO: Final[float] = 0.1
DEFAULT_MAX_EPOCHS: Final[int] = 10
DEFAULT_WEIGHT_DECAY: Final[float] = 0.01

# Cấu hình chế độ smoke run kiểm thử nhanh trên CPU
SMOKE_RUN_MAX_SAMPLES: Final[int] = 16
SMOKE_RUN_EPOCHS: Final[int] = 1
SMOKE_RUN_SEEDS: Final[Sequence[int]] = (42,)
SMOKE_RUN_CANDIDATE_LRS: Final[Sequence[float]] = (2e-5,)

# Đường dẫn lưu trữ checkpoint và kết quả thực nghiệm fine-tuning PhoBERT
CHECKPOINTS_DIR: Final[Path] = _RUNTIME_MODEL_PATHS.checkpoints
DEFAULT_CHECKPOINT_DIR: Final[Path] = _RUNTIME_MODEL_PATHS.checkpoint
DEFAULT_PHOBERT_RESULTS_PATH: Final[Path] = REPORT_DIR / "phobert_results.json"

# Chiều dài chuỗi tối đa và kích thước batch mặc định cho suy luận
MAX_SEQ_LENGTH: Final[int] = 128
PHOBERT_MAX_SEQUENCE_LIMIT: Final[int] = 64
DEFAULT_BATCH_SIZE: Final[int] = 16
DEFAULT_TEMPERATURE: Final[float] = 1.0
FINGERPRINT_METADATA_FILENAME: Final[str] = "preprocessing_fingerprint.json"

# Ngưỡng phân loại và truy hồi (theo kiến trúc AGENTS.md)
# tau: ngưỡng xác suất softmax chấp nhận intent trực tiếp từ bộ phân loại PhoBERT
TAU_INTENT_THRESHOLD: Final[float] = 0.8

# theta: ngưỡng điểm tương đồng tối thiểu trong bước truy hồi fallback top-3 (dưới theta -> không hiểu)
THETA_RETRIEVAL_THRESHOLD: Final[float] = 0.3

# Cấu hình hiệu chuẩn ngưỡng và co giãn nhiệt độ (KA-17 Threshold Calibration & Temperature Scaling)
DEFAULT_THRESHOLDS_PATH: Final[Path] = MODELS_DIR / "thresholds.json"
SMOKE_THRESHOLDS_PATH: Final[Path] = MODELS_DIR / "smoke" / "thresholds_smoke.json"
DEFAULT_TARGET_SELECTIVE_ACCURACY: Final[float] = 0.95
DEFAULT_CALIBRATION_TAU_STEPS: Final[int] = 101
DEFAULT_CALIBRATION_THETA_STEPS: Final[int] = 81
DEFAULT_TEMPERATURE_BOUNDS: Final[tuple[float, float]] = (0.05, 10.0)

# Số lượng kịch bản trả về mặc định trong bước truy hồi top-3
DEFAULT_RETRIEVAL_TOP_K: Final[int] = 3
DEFAULT_RETRIEVAL_AGGREGATION: Final[str] = "max"
SUPPORTED_RETRIEVAL_AGGREGATIONS: Final[Sequence[str]] = ("max", "mean_top2")
DEFAULT_AMBIGUITY_MARGIN_THRESHOLD: Final[float] = 0.1
INDEX_CACHE_PREFIX: Final[str] = "index_"
INDEX_CACHE_DIR: Final[Path] = _RUNTIME_MODEL_PATHS.index_cache

# Cấu hình so khớp mờ dự phòng từ vựng (KA-16 Fuzzy Matching Fallback Module)
DEFAULT_FUZZY_THRESHOLD: Final[float] = 0.5
DEFAULT_FUZZY_TOP_K: Final[int] = 3
DEFAULT_FUZZY_SCORER: Final[str] = "token_set_ratio"
SUPPORTED_FUZZY_SCORERS: Final[Sequence[str]] = (
    "token_set_ratio",
    "token_sort_ratio",
    "partial_ratio",
    "ratio",
)

# Cấu hình biểu diễn lập chỉ mục truy hồi và cắt bỏ (KA-15 Retrieval Indexing & Ablation)
INDEXING_MODE_COMBINED: Final[str] = "combined"
INDEXING_MODE_ALIAS_ONLY: Final[str] = "alias_only"
INDEXING_MODE_SCENARIO_ONLY: Final[str] = "scenario_only"
SUPPORTED_INDEXING_MODES: Final[Sequence[str]] = (
    INDEXING_MODE_COMBINED,
    INDEXING_MODE_ALIAS_ONLY,
    INDEXING_MODE_SCENARIO_ONLY,
)
DEFAULT_INDEXING_MODE: Final[str] = INDEXING_MODE_COMBINED

# Cấu hình đường ống suy luận và điều phối định tuyến (KA-17 Inference Pipeline & Routing Orchestrator)
MODE_PHOBERT_ONLY: Final[str] = "phobert_only"
MODE_PHOBERT_RETRIEVAL: Final[str] = "phobert_retrieval"
MODE_PHOBERT_RETRIEVAL_FUZZY: Final[str] = "phobert_retrieval_fuzzy"
MODE_RETRIEVAL_ONLY: Final[str] = "retrieval_only"
MODE_TFIDF_ONLY: Final[str] = "tfidf_only"
SUPPORTED_PIPELINE_MODES: Final[Sequence[str]] = (
    MODE_PHOBERT_ONLY,
    MODE_PHOBERT_RETRIEVAL,
    MODE_PHOBERT_RETRIEVAL_FUZZY,
    MODE_RETRIEVAL_ONLY,
    MODE_TFIDF_ONLY,
)
DEFAULT_PIPELINE_MODE: Final[str] = MODE_PHOBERT_RETRIEVAL_FUZZY

STATUS_OK: Final[str] = "ok"
STATUS_NEED_CHOICE: Final[str] = "need_choice"
STATUS_NEED_SLOTS: Final[str] = "need_slots"
STATUS_OUT_OF_SCOPE: Final[str] = "out_of_scope"
STATUS_UNKNOWN: Final[str] = "unknown"
STATUS_UNSUPPORTED_TARGET: Final[str] = "unsupported_target"
STATUS_PROCESSING_ERROR: Final[str] = "processing_error"
SUPPORTED_PIPELINE_STATUSES: Final[Sequence[str]] = (
    STATUS_OK,
    STATUS_NEED_CHOICE,
    STATUS_NEED_SLOTS,
    STATUS_OUT_OF_SCOPE,
    STATUS_UNKNOWN,
    STATUS_UNSUPPORTED_TARGET,
)

# Cấu hình hàm mất mát Contrastive và Triplet Loss (KA-14: Mode A, Mode A2, Mode B)
# Mode A & Mode A2: In-batch contrastive loss theo InfoNCE (Oord et al., 2018) và DPR (Karpukhin et al., 2020)
DEFAULT_CONTRASTIVE_SCALE: Final[float] = 20.0
DEFAULT_CONTRASTIVE_METRIC: Final[str] = "dot"
SUPPORTED_CONTRASTIVE_METRICS: Final[Sequence[str]] = ("dot", "cosine", "scaled_cosine")
IDENTITY_MASK_VALUE: Final[float] = float("-inf")

# Mode B: Euclidean margin triplet loss theo FaceNet (Schroff et al., 2015, Eq. (3))
DEFAULT_TRIPLET_MARGIN: Final[float] = 1.0
DEFAULT_TRIPLET_P: Final[float] = 2.0
DEFAULT_TRIPLET_EPS: Final[float] = 1e-6

# Cấu hình biểu diễn chuỗi kịch bản tự nhiên theo Retrieve-and-Fill (RAF - arXiv:2202.00901)
# Hỗ trợ 4 biến thể: 'name', 'slots', 'primed', 'handmade'
SCENARIO_VARIANTS: Final[Sequence[str]] = ("name", "slots", "primed", "handmade")
DEFAULT_SCENARIO_VARIANT: Final[str] = "handmade"
MAX_SLOT_EXAMPLES: Final[int] = 2

# Cấu hình đường ống dữ liệu huấn luyện Bi-Encoder và lấy mẫu kịch bản (KA-15 Bi-Encoder Dataset Pipeline)
DEFAULT_UNSEEN_INTENT_RATIO: Final[float] = 0.20
DEFAULT_HARD_NEGATIVE_TOP_K: Final[int] = 3
DEFAULT_INCLUDE_ALIASES: Final[bool] = False

# Cấu hình huấn luyện tinh chỉnh Bi-Encoder (KA-17 Bi-Encoder Fine-Tuning Pipeline)
TRAINING_MODE_A: Final[str] = "mode_a"
TRAINING_MODE_A2: Final[str] = "mode_a2"
TRAINING_MODE_B: Final[str] = "mode_b"
SUPPORTED_TRAINING_MODES: Final[Sequence[str]] = (
    TRAINING_MODE_A,
    TRAINING_MODE_A2,
    TRAINING_MODE_B,
)
DEFAULT_TRAINING_MODE: Final[str] = TRAINING_MODE_A
DEFAULT_SHARED_WEIGHTS: Final[bool] = True

BIENCODER_CANDIDATE_LRS: Final[Sequence[float]] = (1e-5, 2e-5, 3e-5)
BIENCODER_DEFAULT_LR: Final[float] = 2e-5
BIENCODER_CANDIDATE_BATCH_SIZES: Final[Sequence[int]] = (16, 32)
BIENCODER_DEFAULT_BATCH_SIZE: Final[int] = 16
BIENCODER_MIN_EPOCHS: Final[int] = 3
BIENCODER_MAX_EPOCHS: Final[int] = 10
BIENCODER_DEFAULT_PATIENCE: Final[int] = 3
BIENCODER_WARMUP_RATIO: Final[float] = 0.1
BIENCODER_WEIGHT_DECAY: Final[float] = 0.01

DEFAULT_BIENCODER_CHECKPOINT_DIR: Final[Path] = CHECKPOINTS_DIR / "biencoder"
DEFAULT_BIENCODER_RESULTS_PATH: Final[Path] = REPORT_DIR / "biencoder_results.json"

SMOKE_RUN_BIENCODER_MAX_SAMPLES: Final[int] = 16
SMOKE_RUN_BIENCODER_EPOCHS: Final[int] = 1
SMOKE_RUN_BIENCODER_BATCH_SIZE: Final[int] = 4

# Các ngôn ngữ/nền tảng sinh mã đích được hỗ trợ
SUPPORTED_TARGETS: Final[Sequence[str]] = ("latex", "python", "cpp", "mathjax", "obsidian")
TEMPLATE_TARGET_FALLBACKS: Final[dict[str, tuple[str, ...]]] = {"latex": ("mathjax",)}
TARGET_CANDIDATE_OVERSAMPLE_FACTOR: Final[int] = 8
REQUIRED_TEMPLATE_KEYS: Final[Sequence[str]] = ("latex", "python", "cpp")
TARGET_LATEX: Final[str] = "latex"
POST_ACTION_TIDY_MATH: Final[str] = "tidy_math"
SUPPORTED_POST_ACTIONS: Final[Sequence[str]] = (POST_ACTION_TIDY_MATH,)


# Cấu hình trình biên dịch LaTeX (Tuân thủ nghiêm ngặt quy tắc bảo mật AGENTS.md)
# AGENTS.md: "Never run LaTeX with shell escape (always -no-shell-escape) and never compile text that did not come from the template bank."
DEFAULT_LATEX_ENGINE: Final[str] = "pdflatex"
SUPPORTED_LATEX_ENGINES: Final[Sequence[str]] = ("pdflatex", "xelatex", "lualatex")
LATEX_COMMAND_FLAGS: Final[Sequence[str]] = (
    "-no-shell-escape",
    "-interaction=nonstopmode",
    "-halt-on-error",
)
LATEX_COMPILE_TIMEOUT_SECONDS: Final[int] = 10
LATEX_STATIC_CHECK_ONLY: Final[str] = "static check only"

# Danh sách định dạng phần mở rộng hình ảnh được hỗ trợ trong tài liệu LaTeX
FIGURE_EXTS: Final[Sequence[str]] = (".png", ".jpg", ".jpeg", ".pdf")

# Cấu hình trích xuất tham số slot và chuẩn hóa kiểu dữ liệu (core/slots.py)
MAX_TEXT_SLOT_LENGTH: Final[int] = 200
SPOKEN_MATH_OPERATOR_MAP: Final[dict[str, str]] = {
    "cộng": "+", "trừ": "-", "âm": "-", "dương": "+", "chia": "/",
    "bằng không": "= 0", "bằng": "=", "bình phương": "^2",
    "tiến tới": r"\\to",
}
VIETNAMESE_NUMBER_UNITS: Final[dict[str, int]] = {
    "không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4,
    "tư": 4, "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9,
}
VIETNAMESE_NUMBER_TENS: Final[dict[str, int]] = {
    "mười": 10, "hai mươi": 20, "ba mươi": 30, "bốn mươi": 40,
    "năm mươi": 50, "sáu mươi": 60, "bảy mươi": 70,
    "tám mươi": 80, "chín mươi": 90,
}

# Danh sách các từ điều khiển (control words) LaTeX bị cấm vì lý do an toàn bảo mật
FORBIDDEN_LATEX_CONTROL_WORDS: Final[Sequence[str]] = (
    "input",
    "include",
    "write",
    "openout",
    "openin",
    "read",
    "immediate",
    "catcode",
    "directlua",
)
FORBIDDEN_CONTROL_WORDS: Final[Sequence[str]] = FORBIDDEN_LATEX_CONTROL_WORDS

# Cấu hình tiền xử lý văn bản và phân đoạn từ vựng (KA-02)
# Theo bài báo PhoBERT (Nguyen & Tuan Nguyen, 2020 - arXiv:2003.00744), PhoBERT là mô hình cased (phân biệt hoa thường).
# Cờ LOWERCASE_PHOBERT mặc định False nhằm bảo toàn kiểu chữ cho tên biến, nhãn LaTeX và ký hiệu toán học.
LOWERCASE_PHOBERT: Final[bool] = False
DEFAULT_SEGMENTER: Final[str] = "pyvi"
SUPPORTED_SEGMENTERS: Final[Sequence[str]] = ("pyvi", "rdr")
NORMALIZE_MATH_LOOKALIKES: Final[bool] = True
LATEX_COMMAND_REGEX: Final[str] = r"\\[A-Za-z]+"

# Hạt giống mặc định cho tính tái lập (reproducibility)
DEFAULT_SEED: Final[int] = 42

# Cấu hình giao diện dòng lệnh (CLI)
CLI_BANNER_TITLE: Final[str] = "Vietnamese LaTeX Knowledge Assistant"
CLI_PROMPT: Final[str] = "latex-assistant> "
CLI_EXIT_COMMANDS: Final[Sequence[str]] = ("exit", "quit", ":q", "q")

# Cấu hình thực nghiệm đánh giá học thuật PhoBERT RoBERTa
DEFAULT_EVAL_WARMUP_RUNS: Final[int] = 5
DEFAULT_EVAL_TIMING_RUNS: Final[int] = 3

# Cấu hình huấn luyện JointBERT/BiJoint-BERT-NLU (Prompt Pack §13–14)
JOINT_SLOT_LEARNING_RATES: Final[Sequence[float]] = (1e-5, 5e-5)
BIJOINT_LEARNING_RATE: Final[float] = 5e-5
NLU_EXPERIMENT_SEEDS: Final[Sequence[int]] = (42, 43, 44, 45, 46)
NLU_MAX_EPOCHS: Final[int] = 50
NLU_EARLY_STOPPING_PATIENCE: Final[int] = 5
NLU_BATCH_SIZE: Final[int] = 32
NLU_ADAMW_EPS: Final[float] = 1e-8
NLU_ADAMW_BETAS: Final[tuple[float, float]] = (0.9, 0.999)
NLU_IGNORE_INDEX: Final[int] = -100
BIO_GENERATION_MAX_ATTEMPTS_PER_SAMPLE: Final[int] = 50
BIJOINT_PAPER_LOSS_WEIGHTS: Final[Sequence[tuple[float, float, float]]] = (
    (0.2, 0.4, 0.4), (0.4, 0.3, 0.3), (0.2, 0.3, 0.5),
    (0.4, 0.4, 0.2), (0.3, 0.6, 0.1),
)


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """
    Thiết lập hạt giống ngẫu nhiên đồng bộ cho random, numpy và torch (nếu đã cài đặt).
    Đảm bảo tính tái lập (deterministic reproducibility) cho toàn bộ quy trình huấn luyện và đánh giá.

    Tham số:
        seed: Số nguyên chỉ định hạt giống (mặc định 42).

    Trả về:
        Hạt giống số nguyên đã được thiết lập.
    """
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def get_default_device() -> str:
    """
    Xác định thiết bị tính toán mặc định (GPU 'cuda' nếu khả dụng, ngược lại dùng CPU).
    Hỗ trợ thực thi ngoại tuyến không cần internet.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def verify_hf_model_on_hub(model_name: str, timeout: float = 5.0) -> bool:
    """
    Xác thực sự tồn tại của mô hình trên HuggingFace Hub qua API chính thức.
    Sử dụng thư viện chuẩn urllib để hoạt động độc lập ngay cả khi chưa cài thư viện bên thứ ba.

    Tham số:
        model_name: Định danh mô hình trên Hub (ví dụ: 'vinai/phobert-base-v2').
        timeout: Thời gian chờ phản hồi tính bằng giây.

    Trả về:
        True nếu mô hình tồn tại và phản hồi mã 200, False nếu không tồn tại hoặc lỗi mạng.
    """
    url = f"https://huggingface.co/api/models/{model_name}"
    headers = {"User-Agent": "Vietnamese-LaTeX-Assistant/1.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("id") == model_name or data.get("modelId") == model_name
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False
    return False


def load_calibrated_thresholds(
    thresholds_path: Path | str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_preprocessing_fingerprint: str | None = None,
) -> dict[str, float]:
    """
    Nạp các tham số hiệu chuẩn (tau, theta, temperature) từ tệp JSON (mặc định: models/thresholds.json).
    Nếu tệp không tồn tại hoặc lỗi đọc, trả về các giá trị cấu hình mặc định (TAU_INTENT_THRESHOLD,
    THETA_RETRIEVAL_THRESHOLD, DEFAULT_TEMPERATURE).

    Tham số:
        thresholds_path: Đường dẫn tệp thresholds.json (tùy chọn).

    Trả về:
        Dictionary chứa các tham số: 'tau', 'theta', 'temperature'.
    """
    target_path = Path(thresholds_path) if thresholds_path is not None else DEFAULT_THRESHOLDS_PATH
    if target_path.is_file():
        try:
            with target_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("smoke_mode") is True:
                raise ValueError("Threshold artifact được tạo từ smoke checkpoint.")
            if expected_checkpoint_sha256 is not None:
                if data.get("classifier_checkpoint_sha256") != expected_checkpoint_sha256:
                    raise ValueError("Checkpoint hash trong thresholds không khớp classifier hiện tại.")
            if expected_preprocessing_fingerprint is not None:
                if data.get("preprocessing_fingerprint") != expected_preprocessing_fingerprint:
                    raise ValueError("Preprocessing fingerprint trong thresholds không khớp.")
            values = {
                "tau": float(data.get("tau", TAU_INTENT_THRESHOLD)),
                "theta": float(data.get("theta", THETA_RETRIEVAL_THRESHOLD)),
                "temperature": float(data.get("temperature", DEFAULT_TEMPERATURE)),
            }
            if (not all(math.isfinite(value) for value in values.values())
                    or not 0.0 < values["tau"] <= 1.0
                    or not 0.0 <= values["theta"] <= 1.0
                    or values["temperature"] <= 0.0):
                raise ValueError("Giá trị threshold nằm ngoài miền hợp lệ.")
            return values
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            logging.getLogger(__name__).warning(
                "Không dùng thresholds từ %s; hash/fingerprint không khớp hoặc artifact lỗi. Dùng default tau/theta.",
                target_path,
            )
    elif expected_checkpoint_sha256 is not None or expected_preprocessing_fingerprint is not None:
        logging.getLogger(__name__).warning(
            "Threshold artifact %s is absent; using default tau/theta.", target_path
        )
    return {
        "tau": TAU_INTENT_THRESHOLD,
        "theta": THETA_RETRIEVAL_THRESHOLD,
        "temperature": DEFAULT_TEMPERATURE,
    }


def checkpoint_sha256(checkpoint_dir: Path | str) -> str:
    """Băm tất định các metadata/trọng số của classifier để khóa threshold vào checkpoint."""
    path = Path(checkpoint_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Không thấy classifier checkpoint: {path}")
    relevant = ("config.json", "pytorch_model.bin", "model.safetensors", "preprocessing_fingerprint.json",
                "id2label.json", "label2id.json")
    digest = hashlib.sha256()
    found = False
    for name in relevant:
        file_path = path / name
        if file_path.is_file():
            found = True
            digest.update(name.encode("utf-8"))
            with file_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    if not found:
        raise FileNotFoundError(f"Checkpoint không chứa file metadata/trọng số: {path}")
    return digest.hexdigest()


def checkpoint_is_smoke(checkpoint_dir: Path | str) -> bool:
    """Nhận diện classifier smoke theo metadata đã lưu trong checkpoint config."""
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.is_file():
        return True
    try:
        return bool(json.loads(config_path.read_text(encoding="utf-8")).get("smoke_mode", False))
    except (OSError, json.JSONDecodeError):
        return True


def resolve_hf_snapshot(model_name: str, cache_root: Path | str | None = None) -> Path | None:
    """Tìm snapshot Hub ở modelRoot và các cache mặc định của Hugging Face."""
    cache_roots = [Path(cache_root or HF_CACHE_DIR).expanduser() / "hub"]
    if cache_root is None:
        hub_cache = os.environ.get("HF_HUB_CACHE")
        hf_home = os.environ.get("HF_HOME")
        cache_roots.extend(
            Path(root).expanduser()
            for root in (
                hub_cache,
                str(Path(hf_home).expanduser() / "hub") if hf_home else None,
                str(Path.home() / ".cache" / "huggingface" / "hub"),
            )
            if root
        )
    repo_name = f"models--{model_name.replace('/', '--')}"
    for root in dict.fromkeys(cache_roots):
        repo = root / repo_name
        snapshots = repo / "snapshots"
        if not snapshots.is_dir():
            continue
        ref = repo / "refs" / "main"
        try:
            candidate = snapshots / ref.read_text(encoding="utf-8").strip()
            if candidate.is_dir():
                return candidate
        except OSError:
            pass
        candidates = sorted(
            (item for item in snapshots.iterdir() if item.is_dir()), key=lambda item: item.name
        )
        if candidates:
            return candidates[-1]
    return None


def resolve_phobert_tokenizer_dir() -> Path | None:
    """Trả thư mục tokenizer PhoBERT local đã có vocab và mã BPE."""
    candidates = (PHOBERT_DIR, resolve_hf_snapshot(PHOBERT_MODEL_NAME))
    return next((path for path in candidates if path is not None and
                 (path / "vocab.txt").is_file() and (path / "bpe.codes").is_file()), None)


def _has_transformer_weights(path: Path) -> bool:
    direct_names = ("pytorch_model.bin", "model.safetensors")
    if any((path / name).is_file() for name in direct_names):
        return True
    return any(path.rglob("pytorch_model*.bin")) or any(path.rglob("*.safetensors"))


def classifier_artifact_status(
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    tokenizer_dir: Path | str | None = None,
) -> tuple[bool, str]:
    """Kiểm tra tối thiểu toàn bộ artifact mà IntentClassifier loader cần."""
    checkpoint = Path(checkpoint_dir).expanduser()
    if not checkpoint.is_dir():
        return False, f"Không tìm thấy checkpoint PhoBERT: {checkpoint}"
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        return False, f"Checkpoint thiếu config.json: {checkpoint}"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, f"Checkpoint có config.json không đọc được: {checkpoint}"
    if config.get("smoke_mode") is True:
        return False, f"Checkpoint chỉ dùng smoke test, không phải model runtime: {checkpoint}"
    if not config.get("model_type"):
        return False, f"Checkpoint thiếu model_type trong config.json: {checkpoint}"
    if not any((checkpoint / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")):
        return False, f"Checkpoint thiếu trọng số model.safetensors hoặc pytorch_model.bin: {checkpoint}"
    label_files = (checkpoint / "id2label.json", checkpoint / "label2id.json")
    label_mapping: dict[str, object] | None = None
    for label_file in label_files:
        if label_file.is_file():
            try:
                raw_mapping = json.loads(label_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False, f"Checkpoint có ánh xạ nhãn không đọc được: {label_file.name}"
            if isinstance(raw_mapping, dict) and raw_mapping:
                label_mapping = raw_mapping
                break
    if label_mapping is None:
        return False, f"Checkpoint thiếu ánh xạ nhãn id2label.json/label2id.json: {checkpoint}"
    fingerprint_path = checkpoint / FINGERPRINT_METADATA_FILENAME
    fingerprint_value: str | None = None
    try:
        if fingerprint_path.is_file():
            metadata = json.loads(fingerprint_path.read_text(encoding="utf-8"))
            fingerprint_value = metadata.get("fingerprint_sha256") or metadata.get("preprocessing_fingerprint")
        else:
            fingerprint_value = config.get("preprocessing_fingerprint")
    except (OSError, json.JSONDecodeError):
        return False, f"Checkpoint có metadata fingerprint không hợp lệ: {checkpoint}"
    if not fingerprint_value:
        return False, f"Checkpoint thiếu {FINGERPRINT_METADATA_FILENAME}"
    try:
        from core.preprocess import preprocess_fingerprint

        if fingerprint_value != preprocess_fingerprint():
            return False, "Fingerprint preprocessing của checkpoint không khớp môi trường hiện tại."
    except (ImportError, RuntimeError) as error:
        return False, f"Không kiểm tra được fingerprint preprocessing: {error}"
    checkpoint_tokenizer = checkpoint if (checkpoint / "vocab.txt").is_file() and (checkpoint / "bpe.codes").is_file() else None
    tokenizer = (
        checkpoint_tokenizer
        or (Path(tokenizer_dir).expanduser() if tokenizer_dir is not None else None)
        or resolve_phobert_tokenizer_dir()
    )
    if tokenizer is None or not (tokenizer / "vocab.txt").is_file() or not (tokenizer / "bpe.codes").is_file():
        return False, "Thiếu tokenizer PhoBERT (cần vocab.txt và bpe.codes) trong checkpoint hoặc HF cache."
    return True, f"Checkpoint PhoBERT và tokenizer hợp lệ tại {checkpoint}"


def resolve_sbert_model_dir() -> Path | None:
    """Tìm SentenceTransformer local ở thư mục cấu hình hoặc snapshot Hugging Face."""
    candidates = (SBERT_DIR, resolve_hf_snapshot(SBERT_MODEL_NAME))
    for candidate in candidates:
        if candidate is not None and _is_sentence_transformer_dir(candidate):
            return candidate
    return None


def _is_sentence_transformer_dir(candidate: Path) -> bool:
    if not candidate.is_dir():
        return False
    try:
        modules_path = candidate / "modules.json"
        has_modules = False
        if modules_path.is_file():
            modules = json.loads(modules_path.read_text(encoding="utf-8"))
            has_modules = (
                isinstance(modules, list)
                and bool(modules)
                and all(
                    isinstance(item, dict)
                    and isinstance(item.get("path"), str)
                    and (candidate / item["path"]).is_dir()
                    for item in modules
                )
            )
        has_config = (candidate / "config.json").is_file()
        has_weights = _has_transformer_weights(candidate)
        has_tokenizer = any(
            next(candidate.rglob(name), None) is not None
            for name in ("tokenizer.json", "vocab.txt", "spiece.model")
        )
        return (has_modules or has_config) and has_weights and has_tokenizer
    except (OSError, ValueError):
        return False


def retrieval_model_status(model_dir: Path | str | None = None) -> tuple[bool, str, Path | None]:
    """Kiểm tra model retrieval tách biệt với cache index được xây trên registry."""
    resolved = Path(model_dir).expanduser() if model_dir is not None else resolve_sbert_model_dir()
    if resolved is None or not _is_sentence_transformer_dir(resolved):
        return False, f"Không tìm thấy model retrieval hợp lệ. Cần đặt {SBERT_DIR} hoặc Hugging Face cache.", None
    return True, f"Model retrieval có sẵn tại {resolved}", resolved

