"""Đọc gold spans và tạo BIO theo từ, tách biểu thức toán trước khi gọi pyvi.

CSV gold có cột text,intent,spans; spans là JSON [{"start":0,"end":2,"slot":"a"}].
Offsets là chỉ số ký tự trên text gốc, nhờ vậy nhãn không bị suy đoán từ regex.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MATH_RE = re.compile(
    r"(?<![\wÀ-ỹ])(?:\\[A-Za-z]+|\d+(?:\.\d+)?[A-Za-z](?:\^[+-]?\d+)?|"
    r"[A-Za-z](?:\^[+-]?\d+)?|\d+(?:\.\d+)?)(?![\wÀ-ỹ])|[+*/=^()-]"
)


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Tách toán bằng regex; pyvi chỉ chạy trên các khoảng văn bản thông thường."""
    from pyvi import ViTokenizer

    output: list[tuple[str, int, int]] = []
    cursor = 0
    for match in MATH_RE.finditer(text):
        _append_segment(output, text, cursor, match.start(), ViTokenizer)
        value = match.group()
        coefficient = re.fullmatch(r"(\d+(?:\.\d+)?)([A-Za-z](?:\^[+-]?\d+)?)", value)
        if coefficient:
            split_at = match.start() + len(coefficient.group(1))
            output.extend(((coefficient.group(1), match.start(), split_at),
                           (coefficient.group(2), split_at, match.end())))
        else:
            output.append((value, match.start(), match.end()))
        cursor = match.end()
    _append_segment(output, text, cursor, len(text), ViTokenizer)
    return output


def _append_segment(output: list[tuple[str, int, int]], text: str, start: int, end: int, tokenizer: Any) -> None:
    segment = text[start:end]
    search_from = start
    for match in re.finditer(r"\S+", tokenizer.tokenize(segment)):
        token = match.group()
        pieces = token.split("_")
        found = text.find(pieces[0], search_from, end)
        if found < 0:
            continue
        last = found + len(pieces[0])
        for piece in pieces[1:]:
            next_start = text.find(piece, last, end)
            if next_start >= 0:
                last = next_start + len(piece)
        output.append((token, found, last))
        search_from = last


def bio_labels(text: str, spans: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    tokens = tokenize_with_offsets(text)
    labels = ["O"] * len(tokens)
    occupied: list[tuple[int, int]] = []
    for span in spans:
        start, end, slot = int(span["start"]), int(span["end"]), str(span["slot"])
        if start < 0 or end <= start or end > len(text) or not slot:
            raise ValueError(f"Gold span không hợp lệ: {span}")
        if any(start < previous_end and end > previous_start
               for previous_start, previous_end in occupied):
            raise ValueError(f"Gold spans bị chồng lấn: {span}")
        occupied.append((start, end))
        hits = [i for i, (_, left, right) in enumerate(tokens) if left < end and right > start]
        if not hits:
            raise ValueError(f"Gold span không khớp token nào: {span}")
        for n, i in enumerate(hits):
            labels[i] = f"{'B' if n == 0 else 'I'}-{slot}"
    return [token for token, _, _ in tokens], labels


def convert(input_path: Path, output_path: Path) -> int:
    rows: list[dict[str, str]] = []
    with input_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            tokens, labels = bio_labels(row["text"], json.loads(row["spans"]))
            rows.append({"text": row["text"], "intent": row["intent"],
                         "tokens": json.dumps(tokens, ensure_ascii=False),
                         "bio": json.dumps(labels, ensure_ascii=False),
                         "spans": json.dumps(json.loads(row["spans"]), ensure_ascii=False)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["text", "intent", "tokens", "bio", "spans"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def make_synthetic_training(output_path: Path, samples_per_intent: int = 2, seed: int = 42) -> int:
    """Sinh câu và span cùng lúc từ frame KA-02 để gold BIO không suy đoán bằng regex slot."""
    from core.config import BIO_GENERATION_MAX_ATTEMPTS_PER_SAMPLE
    from data.synth_sentences import (
        DEFAULT_INTENTS_PATH,
        SENTENCE_FRAMES,
        VIETNAMESE_PROMPT_STARTERS,
        generate_slot_value,
    )

    rng = random.Random(seed)
    intents = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))
    intents_by_name = {row["intent"]: row for row in intents}
    records: list[dict[str, str]] = []
    gold_path = Path(__file__).with_name("slot_gold.csv")
    gold_rows: list[dict[str, str]] = []
    if gold_path.exists():
        with gold_path.open(encoding="utf-8", newline="") as stream:
            gold_rows = list(csv.DictReader(stream))

    def normalize_sentence(sentence: str) -> str:
        return " ".join(unicodedata.normalize("NFC", sentence).casefold().split())

    seen: set[str] = set()
    for row in gold_rows:
        normalized = normalize_sentence(row["text"])
        if normalized in seen:
            raise ValueError(f"Câu gold bị trùng lặp: {row['text']}")
        if row.get("split") not in {"val", "test"}:
            raise ValueError("slot_gold.csv chỉ được có split val hoặc test")
        if row["intent"] not in intents_by_name:
            raise ValueError(f"Intent gold không tồn tại: {row['intent']}")
        declared_slots = set(intents_by_name[row["intent"]].get("slots", {}))
        spans = json.loads(row["spans"])
        unknown = {str(span.get("slot", "")) for span in spans} - declared_slots
        if unknown:
            raise ValueError(f"Slot gold chưa khai báo cho {row['intent']}: {sorted(unknown)}")
        seen.add(normalized)
    for item in intents:
        frames = SENTENCE_FRAMES.get(item["intent"], [])
        slots = item.get("slots", {})
        generated = 0
        attempts = 0
        while generated < samples_per_intent and attempts < samples_per_intent * BIO_GENERATION_MAX_ATTEMPTS_PER_SAMPLE:
            attempts += 1
            rendered = rng.choice(frames).replace("{starter}", rng.choice(VIETNAMESE_PROMPT_STARTERS))
            spans: list[dict[str, Any]] = []
            for slot in re.findall(r"\{([A-Za-z0-9_]+)\}", rendered):
                value = generate_slot_value(slot, slots.get(slot, {}), rng)
                placeholder = "{" + slot + "}"
                offset = rendered.index(placeholder)
                rendered = rendered[:offset] + value + rendered[offset + len(placeholder):]
                spans.append({"start": offset, "end": offset + len(value), "slot": slot})
            normalized = normalize_sentence(rendered)
            if normalized in seen:
                continue
            seen.add(normalized)
            tokens, labels = bio_labels(rendered, spans)
            records.append({"split": "train", "text": rendered, "intent": item["intent"],
                            "tokens": json.dumps(tokens, ensure_ascii=False),
                            "bio": json.dumps(labels, ensure_ascii=False),
                            "spans": json.dumps(spans, ensure_ascii=False)})
            generated += 1
        if generated != samples_per_intent:
            raise ValueError(f"Không sinh đủ câu khác biệt cho intent {item['intent']}")

    # Gold validation/test are owner-authored and kept separate from generated training data.
    for row in gold_rows:
        tokens, labels = bio_labels(row["text"], json.loads(row["spans"]))
        records.append({"split": row.get("split", "test"), "text": row["text"],
                        "intent": row["intent"], "tokens": json.dumps(tokens, ensure_ascii=False),
                        "bio": json.dumps(labels, ensure_ascii=False),
                        "spans": json.dumps(json.loads(row["spans"]), ensure_ascii=False)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "text", "intent", "tokens", "bio", "spans"])
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("slot_gold.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("slot_gold_bio.csv"))
    parser.add_argument("--synthetic-train", action="store_true",
                        help="Sinh BIO train từ KA-02 và gộp slot_gold.csv nếu có.")
    parser.add_argument("--samples-per-intent", type=int, default=2)
    args = parser.parse_args()
    count = make_synthetic_training(args.output, args.samples_per_intent) if args.synthetic_train else convert(args.input, args.output)
    print(f"BIO sentences: {count}")


if __name__ == "__main__":
    main()
