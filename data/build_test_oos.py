"""Tạo benchmark OOS tách riêng từ phần PhoATIS/casual test đã có sẵn."""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "external" / "phoatis" / "out_of_scope_test.csv"
TRAIN = ROOT / "data" / "train.csv"
VAL = ROOT / "data" / "val.csv"
OUTPUT = ROOT / "data" / "test_oos.csv"


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def main() -> int:
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    forbidden: set[str] = set()
    for split_path in (TRAIN, VAL):
        with split_path.open(encoding="utf-8-sig", newline="") as stream:
            forbidden.update(normalized(row["text"]) for row in csv.DictReader(stream))
    emitted: set[str] = set()
    fields = ["text", "label", "category", "source"]
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in source_rows:
            text = row["text"].strip()
            key = normalized(text)
            if not text or key in forbidden or key in emitted:
                continue
            writer.writerow({"text": text, "label": "OUT_OF_SCOPE", "category": "tex_misc",
                             "source": row.get("source", "phoatis_test")})
            emitted.add(key)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(emitted)} rows (source candidates={len(source_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
