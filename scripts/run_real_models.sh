#!/usr/bin/env bash
set -euo pipefail
# Run in the project environment with network access for the official pretrained weights.
python train/finetune_phobert.py \
  --target intent --seeds 42 43 44 45 46 \
  --checkpoint-dir models/checkpoints/phobert_classifier_real \
  --results-path report/phobert_results.json
python train/triplet_sbert.py --mode mode_a \
  --checkpoint-dir models/checkpoints/biencoder_real \
  --results-path report/biencoder_results.json
