"""Kiểm tra owner-written benchmark trước khi dùng làm primary test."""
from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = ROOT / "data" / "test_owner.csv"
INTENTS = ROOT / "data" / "intents.json"
PROGRESS = ROOT / "PROGRESS.md"
SPLITS = [ROOT / "data" / f"{name}.csv" for name in ("train", "val", "test")]
REQUIRED = ["text", "label", "category", "written_by"]


def key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def main() -> int:
    if not OWNER.is_file():
        raise SystemExit("BLOCKED: data/test_owner.csv is absent; owner must complete data/test_owner_template.csv")
    with OWNER.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != REQUIRED:
            raise SystemExit(f"Invalid owner-test columns: {reader.fieldnames}; expected {REQUIRED}")
        owner_rows = list(reader)
    intents = json.loads(INTENTS.read_text(encoding="utf-8"))
    lookup = {item["intent"]: item["category"] for item in intents}
    if not owner_rows:
        raise SystemExit("Owner test is empty")
    seen: set[str] = set()
    for line, row in enumerate(owner_rows, start=2):
        if any(not row[name].strip() for name in REQUIRED):
            raise SystemExit(f"Blank required owner-test value at CSV line {line}")
        if row["label"] not in lookup or lookup[row["label"]] != row["category"]:
            raise SystemExit(f"Unknown/mismatched label or category at CSV line {line}")
        normalized = key(row["text"])
        if normalized in seen:
            raise SystemExit(f"Duplicate owner-test row at CSV line {line}")
        seen.add(normalized)
    for split_path in SPLITS:
        with split_path.open(encoding="utf-8-sig", newline="") as stream:
            overlap = seen & {key(row["text"]) for row in csv.DictReader(stream)}
        if overlap:
            raise SystemExit(f"Owner test overlaps {split_path.name}: {len(overlap)} rows")
    digest = hashlib.sha256(OWNER.read_bytes()).hexdigest()
    progress = PROGRESS.read_text(encoding="utf-8")
    marker = "`data/test_owner.csv` SHA-256: `"
    lines = [line for line in progress.splitlines() if not line.startswith(marker)]
    lines.extend(["", f"{marker}{digest}`"])
    PROGRESS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Owner test valid: {len(owner_rows)} rows; sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
