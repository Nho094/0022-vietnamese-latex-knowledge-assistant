"""KA-10 runner: emits explicit BLOCKED rows when frozen real artifacts are absent.

The runner deliberately does not synthesize predictions or relabel smoke artifacts.
Real model modes need saved predictions from the five-seed KA-05/06 runs.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "test.csv"
PROGRESS = ROOT / "PROGRESS.md"
OUT = ROOT / "report" / "ablation.csv"
MODES = ("tfidf_only", "phobert_only", "retrieval_only", "phobert_retrieval",
         "phobert_retrieval_fuzzy", "two_tier")
COLUMNS = ("mode", "accuracy", "macro-F1", "in_scope_accuracy", "OOS_rejection_rate",
           "false_rejection_rate", "automatic_answer_share", "need_choice_rate",
           "top3_hit_rate", "median_latency_ms", "accuracy_ci95", "macro_f1_ci95",
           "status")


def check_test_hash() -> str:
    actual = hashlib.sha256(TEST.read_bytes()).hexdigest()
    text = PROGRESS.read_text(encoding="utf-8")
    matches = re.findall(r"[0-9a-f]{64}", text)
    if actual not in matches:
        raise SystemExit(f"BLOCKED: test hash mismatch; actual={actual}")
    return actual


def main() -> int:
    digest = check_test_hash()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for mode in MODES:
            writer.writerow({"mode": mode, **{name: "" for name in COLUMNS[1:-1]},
                             "status": "BLOCKED: no frozen five-seed predictions/artifact"})
    md = ROOT / "report" / "ablation.md"
    md.write_text(
        "# KA-10 ablation\n\n"
        f"Test CSV SHA-256 verified: `{digest}`. No ablation metrics are reported because the required frozen, "
        "five-seed real checkpoint predictions are unavailable. Empty metric cells are intentionally not estimates.\n\n"
        "The separate KA-04 TF-IDF baseline is in `report/baseline_results.json`; its test is AI drafted and is not the final benchmark.\n",
        encoding="utf-8",
    )
    for mode in MODES:
        print(f"{mode}: BLOCKED (missing frozen five-seed real predictions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
