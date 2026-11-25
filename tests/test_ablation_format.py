from __future__ import annotations

import csv
import math
import subprocess
import sys

from train.ablation import COLUMNS, OUT, ROOT


def test_ablation_has_required_columns_and_no_nan() -> None:
    subprocess.run([sys.executable, "train/ablation.py"], cwd=ROOT, check=True)
    with OUT.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert set(COLUMNS).issubset(reader.fieldnames or [])
        rows = list(reader)
    assert rows
    for row in rows:
        for value in row.values():
            assert value is not None and value.lower() != "nan"
            if value:
                try:
                    assert not math.isnan(float(value))
                except ValueError:
                    pass
        assert row["status"].startswith("BLOCKED:")
