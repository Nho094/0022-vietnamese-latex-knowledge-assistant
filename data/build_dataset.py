"""
Module tổng hợp và kiểm định phân vùng tập dữ liệu (Dataset Compilation and Split Validation Pipeline).

Tuân thủ nghiêm ngặt các quy tắc kiến trúc và tính trung thực trong AGENTS.md:
- Honesty Rule #1: Hệ thống kịch bản đơn giản hóa lấy cảm hứng từ Retrieve-and-Fill (RAF),
  điền slot bằng biểu thức chính quy regex.
- Honesty Rule #3: Bất biến tập kiểm thử (Test Set Invariant): Tuyệt đối không tinh chỉnh trên test set.
  Các câu trong test set phải do chủ sở hữu viết độc quyền (owner-written), không sinh tự động hay thêm bớt bừa bãi.
- Honesty Rule #4: Các phân vùng train/val/test tuyệt đối không chia sẻ bất kỳ câu trùng lặp nào
  (so khớp chuẩn hóa không phân biệt hoa thường và khoảng trắng).
- Ràng buộc phân vùng nghiêm ngặt:
  1. Câu kiểm thử (test) là độc quyền từ tác giả / chủ sở hữu (exclusively owner-written).
  2. Câu kiểm định (val) có cách diễn đạt độc lập, không trùng với câu huấn luyện (train).
  3. Dừng thực thi (abort) ngay lập tức nếu phát hiện bất kỳ câu test nào trùng lặp với train hoặc val.
  4. Phát cảnh báo nếu bất kỳ ý định (intent) nào có ít hơn 10 mẫu train hoặc ít hơn 5 mẫu test.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Bổ sung thư mục gốc dự án vào sys.path để hỗ trợ thực thi trực tiếp dạng script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import (
    DATA_DIR,
    DEFAULT_INTENTS_PATH,
    DEFAULT_OUT_OF_SCOPE_CATEGORY,
    DEFAULT_SEED,
    MIN_TEST_SAMPLES_PER_INTENT,
    MIN_TRAIN_SAMPLES_PER_INTENT,
    OUT_OF_SCOPE_INTENT,
    PHOATIS_DIR,
    RAW_SENTENCES_PATH,
    SYNTH_TRAIN_SAMPLES_PER_INTENT,
    set_seed,
)


@dataclass(frozen=True)
class DatasetRecord:
    """Bản ghi đại diện cho một mẫu dữ liệu trong quy trình tổng hợp."""

    text: str
    label: str
    category: str
    split: str
    origin: str


@dataclass
class DatasetBuildResult:
    """Kết quả tổng hợp và kiểm định các phân vùng tập dữ liệu."""

    train_count: int
    val_count: int
    test_count: int
    intent_counts: dict[str, dict[str, int]]
    warnings_list: list[str]
    train_path: Path
    val_path: Path
    test_path: Path


def normalize_for_dedup(text: str) -> str:
    """
    Chuẩn hóa văn bản nhằm phát hiện trùng lặp xuyên tập (cross-split deduplication).
    Thực hiện đưa về dạng chuẩn Unicode NFC, chuyển chữ thường và gom cụm khoảng trắng liên tiếp.

    Tham số:
        text: Chuỗi văn bản thô đầu vào.

    Trả về:
        Chuỗi văn bản chuẩn hóa chữ thường không chứa khoảng trắng thừa.
    """
    nfc_text = unicodedata.normalize("NFC", text)
    return " ".join(nfc_text.strip().lower().split())


def load_intents_catalog(intents_path: Path = DEFAULT_INTENTS_PATH) -> dict[str, str]:
    """
    Đọc danh mục ý định (Intent Bank) từ tệp JSON và trích xuất ánh xạ intent -> danh mục (category).

    Tham số:
        intents_path: Đường dẫn tệp intents.json.

    Trả về:
        Từ điển ánh xạ từ tên intent sang tên category chuẩn hóa.

    Ngoại lệ:
        FileNotFoundError: Nếu tệp không tồn tại.
        ValueError: Nếu định dạng JSON không hợp lệ.
    """
    if not intents_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp danh mục ý định tại: {intents_path}")

    with open(intents_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Định dạng tệp intents.json không hợp lệ, kỳ vọng danh sách (list): {intents_path}")

    intent_map: dict[str, str] = {}
    for item in data:
        name = item.get("intent")
        if not name:
            continue
        category = item.get("category", DEFAULT_OUT_OF_SCOPE_CATEGORY)
        intent_map[name] = category

    return intent_map


def load_owner_sentences(
    csv_path: Path = RAW_SENTENCES_PATH,
    valid_intents: dict[str, str] | None = None,
) -> list[DatasetRecord]:
    """
    Nạp các câu mẫu do chủ sở hữu viết từ tệp CSV thô (data/raw_sentences.csv).
    Kiểm thực nhãn ý định hợp lệ và gán category theo danh mục chuẩn.

    Tham số:
        csv_path: Đường dẫn tệp raw_sentences.csv.
        valid_intents: Từ điển các intent hợp lệ kèm category (tùy chọn).

    Trả về:
        Danh sách DatasetRecord với nguồn gốc origin='owner_written'.

    Ngoại lệ:
        FileNotFoundError: Nếu tệp CSV không tồn tại.
        ValueError: Nếu phát hiện nhãn intent không thuộc danh mục hoặc phân vùng split không hợp lệ.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp câu mẫu chủ sở hữu tại: {csv_path}")

    records: list[DatasetRecord] = []
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            raw_text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            split = (row.get("split") or "").strip().lower()

            if not raw_text or not label:
                continue

            if split not in ("train", "val", "test"):
                raise ValueError(
                    f"Dòng {idx}: Phân vùng split '{split}' không hợp lệ. Chỉ chấp nhận 'train', 'val', 'test'."
                )

            if valid_intents is not None and label not in valid_intents:
                raise ValueError(
                    f"Dòng {idx}: Nhãn intent '{label}' không tồn tại trong danh mục intents.json chuẩn."
                )

            category = valid_intents[label] if valid_intents and label in valid_intents else row.get("category", DEFAULT_OUT_OF_SCOPE_CATEGORY)

            records.append(
                DatasetRecord(
                    text=raw_text,
                    label=label,
                    category=category,
                    split=split,
                    origin="owner_written",
                )
            )

    return records


def load_synthetic_sentences(
    intents_path: Path = DEFAULT_INTENTS_PATH,
    seed: int = DEFAULT_SEED,
    samples_per_intent: int = SYNTH_TRAIN_SAMPLES_PER_INTENT,
    valid_intents: dict[str, str] | None = None,
) -> list[DatasetRecord]:
    """
    Sinh các câu huấn luyện tổng hợp (Synthetic Training Sentences) thông qua module synth_sentences.
    Tuân thủ quy tắc phân vùng: dữ liệu tổng hợp chỉ được dùng cho tập huấn luyện (split='train').

    Tham số:
        intents_path: Đường dẫn tệp intents.json.
        seed: Hạt giống ngẫu nhiên tất định.
        samples_per_intent: Số mẫu sinh cho mỗi intent.
        valid_intents: Danh mục intent hợp lệ.

    Trả về:
        Danh sách DatasetRecord cho phân vùng train với origin='synthetic'.
    """
    from data.synth_sentences import generate_synthetic_sentences

    raw_synth = generate_synthetic_sentences(
        intents_path=intents_path,
        seed=seed,
        samples_per_intent=samples_per_intent,
        exclude_val_test=False,  # Việc khử trùng lặp xuyên tập sẽ được thực hiện tại pipeline kiểm định
        include_out_of_scope=False,
    )

    records: list[DatasetRecord] = []
    for item in raw_synth:
        text = item["text"]
        label = item["label"]
        category = valid_intents.get(label, item.get("category", DEFAULT_OUT_OF_SCOPE_CATEGORY)) if valid_intents else item.get("category", DEFAULT_OUT_OF_SCOPE_CATEGORY)

        records.append(
            DatasetRecord(
                text=text,
                label=label,
                category=category,
                split="train",
                origin="synthetic",
            )
        )

    return records


def load_out_of_scope_samples(
    phoatis_dir: Path = PHOATIS_DIR,
    include_test: bool = False,
    seed: int = DEFAULT_SEED,
    valid_intents: dict[str, str] | None = None,
) -> list[DatasetRecord]:
    """
    Nạp dữ liệu ngoài phạm vi (Out-of-Scope / OOS) từ các tệp PhoATIS/casual đã lấy mẫu sẵn.
    Mặc định không nạp OOS cho phân vùng test để bảo toàn nguyên tắc test độc quyền owner-written.

    Tham số:
        phoatis_dir: Thư mục chứa các tệp out_of_scope_{train,val,test}.csv.
        include_test: Có nạp câu OOS cho phân vùng test hay không.
        seed: Hạt giống sinh ngẫu nhiên khi cần lấy mẫu dự phòng.
        valid_intents: Danh mục intent hợp lệ.

    Trả về:
        Danh sách DatasetRecord nhãn OUT_OF_SCOPE.
    """
    records: list[DatasetRecord] = []
    splits_to_load = ["train", "val"]
    if include_test:
        splits_to_load.append("test")

    category = (
        valid_intents.get(OUT_OF_SCOPE_INTENT, DEFAULT_OUT_OF_SCOPE_CATEGORY)
        if valid_intents
        else DEFAULT_OUT_OF_SCOPE_CATEGORY
    )

    for split in splits_to_load:
        csv_file = phoatis_dir / f"out_of_scope_{split}.csv"
        if csv_file.is_file():
            with open(csv_file, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = (row.get("text") or "").strip()
                    if text:
                        records.append(
                            DatasetRecord(
                                text=text,
                                label=OUT_OF_SCOPE_INTENT,
                                category=category,
                                split=split,
                                origin="out_of_scope",
                            )
                        )
        else:
            # Dự phòng: Nạp từ các tập câu hội thoại tiếng Việt thông thường (casual pools)
            try:
                from data.fetch_external import (
                    CASUAL_TEST_POOL,
                    CASUAL_TRAIN_POOL,
                    CASUAL_VAL_POOL,
                )

                pool_map = {
                    "train": CASUAL_TRAIN_POOL,
                    "val": CASUAL_VAL_POOL,
                    "test": CASUAL_TEST_POOL,
                }
                for text in pool_map.get(split, []):
                    records.append(
                        DatasetRecord(
                            text=text,
                            label=OUT_OF_SCOPE_INTENT,
                            category=category,
                            split=split,
                            origin="out_of_scope_casual",
                        )
                    )
            except ImportError:
                pass

    return records


def validate_split_integrity(
    train_records: Sequence[DatasetRecord],
    val_records: Sequence[DatasetRecord],
    test_records: Sequence[DatasetRecord],
    valid_intents: dict[str, str],
    allow_non_owner_test: bool = False,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """
    Kiểm định toàn vẹn các phân vùng dữ liệu theo quy chuẩn AGENTS.md và yêu cầu tác vụ:
    1. Kiểm tra tập test chỉ bao gồm câu do chủ sở hữu viết (owner-written).
    2. Chuẩn hóa văn bản và kiểm tra không trùng lặp câu giữa test và train/val (Honesty Rule #4).
    3. Kiểm tra câu tập val không trùng lặp diễn đạt với train.
    4. Thống kê số lượng mẫu và phát cảnh báo nếu intent có ít hơn 10 train hoặc ít hơn 5 test.

    Tham số:
        train_records: Danh sách bản ghi tập huấn luyện.
        val_records: Danh sách bản ghi tập kiểm định.
        test_records: Danh sách bản ghi tập kiểm thử.
        valid_intents: Từ điển danh mục ý định chuẩn.
        allow_non_owner_test: Cho phép câu không phải owner-written trong tập test (mặc định False).

    Trả về:
        Bộ tuple gồm:
        - Từ điển thống kê số lượng mẫu: {intent: {'train': int, 'val': int, 'test': int, 'total': int}}
        - Danh sách các thông điệp cảnh báo được phát ra.

    Ngoại lệ:
        ValueError: Nếu phát hiện vi phạm phân vùng (câu test trùng train/val, val trùng train,
                    hoặc tập test chứa câu không phải do chủ sở hữu viết).
    """
    # 1. Ràng buộc: Tập test độc quyền câu do chủ sở hữu viết
    if not allow_non_owner_test:
        non_owner_test = [r for r in test_records if r.origin != "owner_written"]
        if non_owner_test:
            sample_bad = non_owner_test[0]
            raise ValueError(
                f"Vi phạm quy tắc phân vùng: Tập test phải độc quyền do chủ sở hữu viết (owner-written). "
                f"Phát hiện mẫu không hợp lệ với nguồn gốc origin='{sample_bad.origin}': '{sample_bad.text}'"
            )

    # 2. Xây dựng tập bản đồ chuẩn hóa câu cho từng phân vùng
    train_norms: dict[str, str] = {}
    for r in train_records:
        norm = normalize_for_dedup(r.text)
        train_norms[norm] = r.text

    val_norms: dict[str, str] = {}
    for r in val_records:
        norm = normalize_for_dedup(r.text)
        val_norms[norm] = r.text

    test_norms: dict[str, str] = {}
    for r in test_records:
        norm = normalize_for_dedup(r.text)
        test_norms[norm] = r.text

    # 3. Kiểm tra câu kiểm thử (test) trùng lặp với train hoặc val (BẮT BUỘC DỪNG / ABORT)
    dup_test_train = set(test_norms.keys()) & set(train_norms.keys())
    if dup_test_train:
        sample_dup_norm = next(iter(dup_test_train))
        raise ValueError(
            f"DỪNG THỰC THI (ABORT): Phát hiện {len(dup_test_train)} câu kiểm thử (test) trùng lặp với tập train! "
            f"Ví dụ câu trùng: '{test_norms[sample_dup_norm]}'"
        )

    dup_test_val = set(test_norms.keys()) & set(val_norms.keys())
    if dup_test_val:
        sample_dup_norm = next(iter(dup_test_val))
        raise ValueError(
            f"DỪNG THỰC THI (ABORT): Phát hiện {len(dup_test_val)} câu kiểm thử (test) trùng lặp với tập val! "
            f"Ví dụ câu trùng: '{test_norms[sample_dup_norm]}'"
        )

    # 4. Kiểm tra câu kiểm định (val) trùng lặp cách diễn đạt với train
    dup_val_train = set(val_norms.keys()) & set(train_norms.keys())
    if dup_val_train:
        sample_dup_norm = next(iter(dup_val_train))
        raise ValueError(
            f"DỪNG THỰC THI (ABORT): Phát hiện {len(dup_val_train)} câu kiểm định (val) trùng lặp với tập train! "
            f"Ví dụ câu trùng: '{val_norms[sample_dup_norm]}'"
        )

    # 5. Thống kê số lượng mẫu trên từng phân vùng và ý định
    train_counter = Counter(r.label for r in train_records)
    val_counter = Counter(r.label for r in val_records)
    test_counter = Counter(r.label for r in test_records)

    intent_counts: dict[str, dict[str, int]] = {}
    warnings_list: list[str] = []

    for intent in valid_intents:
        t_cnt = train_counter.get(intent, 0)
        v_cnt = val_counter.get(intent, 0)
        te_cnt = test_counter.get(intent, 0)

        intent_counts[intent] = {
            "train": t_cnt,
            "val": v_cnt,
            "test": te_cnt,
            "total": t_cnt + v_cnt + te_cnt,
        }

        # Kiểm tra ngưỡng tối thiểu số mẫu
        if t_cnt < MIN_TRAIN_SAMPLES_PER_INTENT:
            msg = (
                f"[CẢNH BÁO] Ý định '{intent}' chỉ có {t_cnt} mẫu train "
                f"(dưới ngưỡng tối thiểu {MIN_TRAIN_SAMPLES_PER_INTENT} mẫu)."
            )
            warnings_list.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

        if te_cnt < MIN_TEST_SAMPLES_PER_INTENT:
            msg = (
                f"[CẢNH BÁO] Ý định '{intent}' chỉ có {te_cnt} mẫu test "
                f"(dưới ngưỡng tối thiểu {MIN_TEST_SAMPLES_PER_INTENT} mẫu)."
            )
            warnings_list.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    return intent_counts, warnings_list


def export_split_csv(
    records: Sequence[DatasetRecord],
    output_path: Path,
) -> int:
    """
    Xuất danh sách bản ghi ra tệp CSV với 3 cột chuẩn hóa: text, label, category.
    Sử dụng định dạng UTF-8-SIG tương thích chuẩn hệ thống.

    Tham số:
        records: Danh sách các DatasetRecord cần ghi.
        output_path: Đường dẫn tệp CSV đích.

    Trả về:
        Số lượng bản ghi đã ghi vào tệp.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "category"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "text": r.text,
                "label": r.label,
                "category": r.category,
            })
    return len(records)


def format_summary_table(
    intent_counts: dict[str, dict[str, int]],
    valid_intents: dict[str, str],
    total_train: int,
    total_val: int,
    total_test: int,
) -> str:
    """
    Định dạng bảng tổng hợp số lượng mẫu theo từng ý định và phân vùng.

    Tham số:
        intent_counts: Bảng đếm số mẫu {intent: {'train': x, 'val': y, 'test': z, 'total': n}}.
        valid_intents: Ánh xạ intent -> category.
        total_train: Tổng số mẫu train.
        total_val: Tổng số mẫu val.
        total_test: Tổng số mẫu test.

    Trả về:
        Chuỗi văn bản định dạng bảng Markdown/ASCII trực quan.
    """
    lines: list[str] = [
        "=" * 86,
        "   BÁO CÁO TỔNG HỢP VÀ PHÂN BỔ DỮ LIỆU (DATASET COMPILATION SUMMARY)",
        "=" * 86,
        f"Tổng số mẫu tập Train: {total_train:,} câu",
        f"Tổng số mẫu tập Val:   {total_val:,} câu",
        f"Tổng số mẫu tập Test:  {total_test:,} câu",
        f"Tổng cộng toàn bộ:     {total_train + total_val + total_test:,} câu",
        f"Tổng số ý định:        {len(valid_intents):,} intents",
        "-" * 86,
        f"{'Intent Name':<38} | {'Category':<15} | {'Train':>5} | {'Val':>4} | {'Test':>4} | {'Total':>5}",
        "-" * 86,
    ]

    for intent, counts in sorted(intent_counts.items(), key=lambda x: x[0]):
        cat = valid_intents.get(intent, "unknown")
        lines.append(
            f"{intent:<38} | {cat:<15} | {counts['train']:>5} | {counts['val']:>4} | {counts['test']:>4} | {counts['total']:>5}"
        )

    lines.append("=" * 86)
    return "\n".join(lines)


def compile_and_validate_dataset(
    raw_sentences_path: Path = RAW_SENTENCES_PATH,
    intents_path: Path = DEFAULT_INTENTS_PATH,
    output_dir: Path = DATA_DIR,
    phoatis_dir: Path = PHOATIS_DIR,
    include_synthetic: bool = True,
    synthetic_samples_per_intent: int = SYNTH_TRAIN_SAMPLES_PER_INTENT,
    include_oos: bool = True,
    include_oos_test: bool = False,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
) -> DatasetBuildResult:
    """
    Quy trình tích hợp chính: nạp nguồn dữ liệu, tổng hợp các tập, kiểm định toàn vẹn
    và xuất các tệp train.csv, val.csv, test.csv.

    Tham số:
        raw_sentences_path: Tệp câu gốc của chủ sở hữu.
        intents_path: Tệp intents.json.
        output_dir: Thư mục xuất các tệp CSV.
        phoatis_dir: Thư mục chứa dữ liệu ngoại phạm vi PhoATIS.
        include_synthetic: Có nạp câu tổng hợp vào tập huấn luyện hay không.
        synthetic_samples_per_intent: Số mẫu sinh cho mỗi intent.
        include_oos: Có nạp câu ngoại phạm vi OUT_OF_SCOPE hay không.
        include_oos_test: Có nạp OOS vào tập test hay không (mặc định False để bảo đảm test độc quyền owner-written).
        seed: Hạt giống ngẫu nhiên tái lập.
        dry_run: Nếu True, chỉ chạy kiểm định và hiển thị báo cáo mà không ghi đè tệp CSV.

    Trả về:
        Đối tượng DatasetBuildResult chứa thống kê và đường dẫn tệp.
    """
    set_seed(seed)

    # 1. Nạp danh mục ý định chuẩn
    valid_intents = load_intents_catalog(intents_path)

    # 2. Nạp dữ liệu tác giả viết (owner-written)
    owner_records = load_owner_sentences(raw_sentences_path, valid_intents=valid_intents)

    # Phân tách ban đầu theo split
    train_pool: list[DatasetRecord] = [r for r in owner_records if r.split == "train"]
    val_pool: list[DatasetRecord] = [r for r in owner_records if r.split == "val"]
    test_pool: list[DatasetRecord] = [r for r in owner_records if r.split == "test"]

    # 3. Nạp dữ liệu tổng hợp (chỉ bổ sung vào tập train)
    if include_synthetic:
        synth_records = load_synthetic_sentences(
            intents_path=intents_path,
            seed=seed,
            samples_per_intent=synthetic_samples_per_intent,
            valid_intents=valid_intents,
        )
        train_pool.extend(synth_records)

    # 4. Nạp dữ liệu ngoại phạm vi (OUT_OF_SCOPE)
    if include_oos:
        oos_records = load_out_of_scope_samples(
            phoatis_dir=phoatis_dir,
            include_test=include_oos_test,
            seed=seed,
            valid_intents=valid_intents,
        )
        for r in oos_records:
            if r.split == "train":
                train_pool.append(r)
            elif r.split == "val":
                val_pool.append(r)
            elif r.split == "test" and include_oos_test:
                test_pool.append(r)

    # 5. Khử trùng lặp nội bộ trong từng phân vùng trước khi kiểm tra liên phân vùng
    def deduplicate_split(records: list[DatasetRecord]) -> list[DatasetRecord]:
        seen: set[str] = set()
        deduped: list[DatasetRecord] = []
        for r in records:
            norm = normalize_for_dedup(r.text)
            if norm not in seen:
                seen.add(norm)
                deduped.append(r)
        return deduped

    clean_train = deduplicate_split(train_pool)
    clean_val = deduplicate_split(val_pool)
    clean_test = deduplicate_split(test_pool)

    # 6. Kiểm định toàn vẹn phân vùng nghiêm ngặt
    intent_counts, warnings_list = validate_split_integrity(
        train_records=clean_train,
        val_records=clean_val,
        test_records=clean_test,
        valid_intents=valid_intents,
        allow_non_owner_test=include_oos_test,
    )

    # 7. Xuất các tệp CSV phân vùng
    train_csv_path = output_dir / "train.csv"
    val_csv_path = output_dir / "val.csv"
    test_csv_path = output_dir / "test.csv"

    if not dry_run:
        export_split_csv(clean_train, train_csv_path)
        export_split_csv(clean_val, val_csv_path)
        export_split_csv(clean_test, test_csv_path)

    return DatasetBuildResult(
        train_count=len(clean_train),
        val_count=len(clean_val),
        test_count=len(clean_test),
        intent_counts=intent_counts,
        warnings_list=warnings_list,
        train_path=train_csv_path,
        val_path=val_csv_path,
        test_path=test_csv_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """
    Giao diện dòng lệnh (CLI) thực thi pipeline tổng hợp và kiểm định tập dữ liệu.
    """
    parser = argparse.ArgumentParser(
        description="Quy trình tổng hợp và kiểm định phân chia tập dữ liệu tiếng Việt cho trợ lý LaTeX."
    )
    parser.add_argument(
        "--raw-sentences",
        type=Path,
        default=RAW_SENTENCES_PATH,
        help="Đường dẫn tệp CSV câu gốc do chủ sở hữu viết (mặc định: data/raw_sentences.csv)",
    )
    parser.add_argument(
        "--intents",
        type=Path,
        default=DEFAULT_INTENTS_PATH,
        help="Đường dẫn tệp JSON danh mục ý định chuẩn (mặc định: data/intents.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Thư mục xuất các tệp train.csv, val.csv, test.csv (mặc định: data/)",
    )
    parser.add_argument(
        "--phoatis-dir",
        type=Path,
        default=PHOATIS_DIR,
        help="Thư mục chứa dữ liệu ngoại phạm vi PhoATIS (mặc định: data/external/phoatis)",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Không bổ sung câu tổng hợp vào tập huấn luyện",
    )
    parser.add_argument(
        "--no-oos",
        action="store_true",
        help="Không bổ sung câu ngoài phạm vi OUT_OF_SCOPE",
    )
    parser.add_argument(
        "--include-oos-test",
        action="store_true",
        help="Cho phép đưa mẫu OUT_OF_SCOPE vào tập test (mặc định tắt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Hạt giống ngẫu nhiên tái lập (mặc định: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ kiểm định và in báo cáo mà không ghi đè tệp CSV",
    )

    args = parser.parse_args(argv)

    try:
        valid_intents = load_intents_catalog(args.intents)
        result = compile_and_validate_dataset(
            raw_sentences_path=args.raw_sentences,
            intents_path=args.intents,
            output_dir=args.output_dir,
            phoatis_dir=args.phoatis_dir,
            include_synthetic=not args.no_synthetic,
            synthetic_samples_per_intent=SYNTH_TRAIN_SAMPLES_PER_INTENT,
            include_oos=not args.no_oos,
            include_oos_test=args.include_oos_test,
            seed=args.seed,
            dry_run=args.dry_run,
        )

        table_str = format_summary_table(
            intent_counts=result.intent_counts,
            valid_intents=valid_intents,
            total_train=result.train_count,
            total_val=result.val_count,
            total_test=result.test_count,
        )
        print(table_str)

        if result.warnings_list:
            print(f"\n[DANH SÁCH CẢNH BÁO: {len(result.warnings_list)} CẢNH BÁO ĐÃ PHÁT]")
            for w in result.warnings_list:
                print(f"  {w}")

        if not args.dry_run:
            print("\n[HOÀN TẤT] Đã xuất thành công các phân vùng dữ liệu:")
            print(f"  + Train: {result.train_path} ({result.train_count} mẫu)")
            print(f"  + Val:   {result.val_path} ({result.val_count} mẫu)")
            print(f"  + Test:  {result.test_path} ({result.test_count} mẫu)")
        else:
            print("\n[CHẾ ĐỘ THỬ NGHIỆM - DRY RUN] Toàn bộ kiểm định hợp lệ, không có tệp nào được ghi.")

        return 0

    except Exception as exc:
        print(f"\n[LỖI NGHIÊM TRỌNG / ABORT] Quy trình tổng hợp dữ liệu bị hủy bỏ: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
