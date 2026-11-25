#!/usr/bin/env bash
set -euo pipefail
ruff check .
pytest -q -m 'not slow and not needs_weights'
python -m core.validator
python train/ablation_stats.py --verify-test
if grep -RInE 'from src|import src' --include='*.py' . \
  --exclude-dir=.git --exclude-dir=legacy --exclude-dir=AI --exclude-dir=.venv --exclude-dir=venv; then
  echo 'Unexpected src import outside legacy/' >&2
  exit 1
fi
echo 'No src imports outside legacy/'
python data/check_split_shift.py
python report/build_report.py
