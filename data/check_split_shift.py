"""Báo cáo kiểm toán độ lệch giữa các split mà không ghi vào dữ liệu nguồn."""
from __future__ import annotations

import csv
import json
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.config import (
    DEFAULT_SEED,
    OUT_OF_SCOPE_INTENT,
    REPORT_DIR,
    SPLIT_SHIFT_BOILERPLATE_WARNING_THRESHOLD,
    SPLIT_SHIFT_TRIGRAM_SAMPLE_SIZE,
    SPLIT_SHIFT_TRIGRAM_WARNING_THRESHOLD,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
)

SPLITS = {"train": TRAIN_CSV_PATH, "val": VAL_CSV_PATH, "test": TEST_CSV_PATH}
OUTPUT = REPORT_DIR / "split_shift.json"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize_text(value: str) -> str:
    """Chuẩn hóa khoảng trắng/chữ hoa để đếm câu boilerplate và giao nhau."""
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def char_trigrams(value: str) -> set[str]:
    normalized = normalize_text(value)
    return {normalized[index:index + 3] for index in range(max(0, len(normalized) - 2))}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return {
        "min": ordered[0], "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered), "p95": ordered[p95_index], "max": ordered[-1],
    }


def nearest_jaccard(source: list[set[str]], target: list[set[str]]) -> list[float]:
    """Tính Jaccard lân cận gần nhất bằng inverted index để bỏ qua cặp giao bằng 0."""
    postings: dict[str, list[int]] = defaultdict(list)
    for target_index, grams in enumerate(target):
        for gram in grams:
            postings[gram].append(target_index)
    scores: list[float] = []
    for grams in source:
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(postings.get(gram, ()))
        best = 0.0
        for candidate in candidates:
            other = target[candidate]
            union = len(grams | other)
            if union:
                best = max(best, len(grams & other) / union)
        scores.append(best)
    return scores


def build_report() -> dict[str, Any]:
    rows = {name: read_rows(path) for name, path in SPLITS.items()}
    all_sentences = Counter(
        normalize_text(sentence)
        for split_rows in rows.values()
        for row in split_rows
        for sentence in SENTENCE_SPLIT.split(row["text"].strip())
        if len(normalize_text(sentence)) >= 20
    )
    boilerplate = all_sentences.most_common(1)[0][0] if all_sentences else ""
    rng = random.Random(DEFAULT_SEED)
    grams: dict[str, list[set[str]]] = {}
    for name, split_rows in rows.items():
        sample = split_rows if len(split_rows) <= SPLIT_SHIFT_TRIGRAM_SAMPLE_SIZE else rng.sample(
            split_rows, SPLIT_SHIFT_TRIGRAM_SAMPLE_SIZE
        )
        grams[name] = [char_trigrams(row["text"]) for row in sample]
    report: dict[str, Any] = {
        "source": "data/train.csv, data/val.csv, data/test.csv (test is ai_draft)",
        "most_frequent_sentence": boilerplate,
        "thresholds": {
            "nearest_trigram_jaccard_warning": SPLIT_SHIFT_TRIGRAM_WARNING_THRESHOLD,
            "boilerplate_share_warning": SPLIT_SHIFT_BOILERPLATE_WARNING_THRESHOLD,
        },
        "trigram_method": (
            f"nearest-neighbor Jaccard on deterministic samples capped at "
            f"{SPLIT_SHIFT_TRIGRAM_SAMPLE_SIZE} rows per split; descriptive, not a deduplication proof"
        ),
        "splits": {}, "warnings": [],
    }
    for name, split_rows in rows.items():
        lengths = [len(row["text"]) for row in split_rows]
        boilerplate_count = sum(boilerplate in normalize_text(row["text"]) for row in split_rows)
        split_sentences = Counter(
            normalize_text(sentence)
            for row in split_rows
            for sentence in SENTENCE_SPLIT.split(row["text"].strip())
            if len(normalize_text(sentence)) >= 20
        )
        split_boilerplate = split_sentences.most_common(1)[0][0] if split_sentences else ""
        split_boilerplate_share = (
            sum(split_boilerplate in normalize_text(row["text"]) for row in split_rows) / len(split_rows)
            if split_rows else 0.0
        )
        item: dict[str, Any] = {
            "count": len(split_rows),
            "mean_char_length": statistics.fmean(lengths) if lengths else 0.0,
            "most_frequent_sentence_share": boilerplate_count / len(split_rows) if split_rows else 0.0,
            "split_most_frequent_sentence": split_boilerplate,
            "split_most_frequent_sentence_share": split_boilerplate_share,
            "label_coverage": len({row["label"] for row in split_rows}),
            "label_count": dict(sorted(Counter(row["label"] for row in split_rows).items())),
            "out_of_scope_count": sum(row["label"] == OUT_OF_SCOPE_INTENT for row in split_rows),
            "nearest_trigram_jaccard_to": {},
        }
        for target_name in rows:
            if target_name != name:
                item["nearest_trigram_jaccard_to"][target_name] = percentiles(
                    nearest_jaccard(grams[name], grams[target_name])
                )
        report["splits"][name] = item
        if item["most_frequent_sentence_share"] > SPLIT_SHIFT_BOILERPLATE_WARNING_THRESHOLD:
            report["warnings"].append(
                f"WARNING: {name} boilerplate share {item['most_frequent_sentence_share']:.4f} exceeds "
                f"{SPLIT_SHIFT_BOILERPLATE_WARNING_THRESHOLD:.2f}."
            )
        if split_boilerplate_share > SPLIT_SHIFT_BOILERPLATE_WARNING_THRESHOLD:
            report["warnings"].append(
                f"WARNING: {name} split-specific boilerplate share {split_boilerplate_share:.4f} exceeds "
                f"{SPLIT_SHIFT_BOILERPLATE_WARNING_THRESHOLD:.2f}."
            )
        for target_name, distribution in item["nearest_trigram_jaccard_to"].items():
            if distribution["p95"] > SPLIT_SHIFT_TRIGRAM_WARNING_THRESHOLD:
                report["warnings"].append(
                    f"WARNING: {name}->{target_name} nearest trigram Jaccard p95 "
                    f"{distribution['p95']:.4f} exceeds {SPLIT_SHIFT_TRIGRAM_WARNING_THRESHOLD:.2f}."
                )
    return report


def main() -> int:
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    for warning in report["warnings"]:
        print(warning)
    for name, split in report["splits"].items():
        print(f"{name}: n={split['count']} mean_chars={split['mean_char_length']:.2f} "
              f"boilerplate_share={split['most_frequent_sentence_share']:.4f} "
              f"split_boilerplate_share={split['split_most_frequent_sentence_share']:.4f} "
              f"labels={split['label_coverage']} OOS={split['out_of_scope_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
