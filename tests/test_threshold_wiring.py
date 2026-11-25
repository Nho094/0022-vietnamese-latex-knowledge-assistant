from __future__ import annotations

import json
import logging
from pathlib import Path

from core.config import checkpoint_sha256
from core.pipeline import Pipeline
from core.preprocess import preprocess_fingerprint


def _checkpoint(path: Path) -> str:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"smoke_mode": False}), encoding="utf-8")
    (path / "pytorch_model.bin").write_bytes(b"test checkpoint bytes")
    return checkpoint_sha256(path)


def test_pipeline_loads_thresholds_matching_checkpoint_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "classifier"
    checkpoint_hash = _checkpoint(checkpoint)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({
        "tau": 0.61, "theta": 0.27, "temperature": 1.4,
        "classifier_checkpoint_sha256": checkpoint_hash,
        "preprocessing_fingerprint": preprocess_fingerprint(),
        "smoke_mode": False,
    }), encoding="utf-8")
    pipeline = Pipeline(mode="phobert_only", checkpoint_dir=checkpoint, thresholds_path=thresholds)
    assert pipeline.tau == 0.61
    assert pipeline.theta == 0.27
    assert pipeline.temperature == 1.4


def test_pipeline_uses_defaults_and_warns_on_checkpoint_mismatch(tmp_path: Path, caplog) -> None:
    checkpoint = tmp_path / "classifier"
    _checkpoint(checkpoint)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({
        "tau": 0.61, "theta": 0.27, "temperature": 1.4,
        "classifier_checkpoint_sha256": "wrong-hash",
        "preprocessing_fingerprint": preprocess_fingerprint(),
        "smoke_mode": False,
    }), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        pipeline = Pipeline(mode="phobert_only", checkpoint_dir=checkpoint, thresholds_path=thresholds)
    assert pipeline.tau != 0.61
    assert pipeline.theta != 0.27
    assert "Dùng default tau/theta" in caplog.text


def test_pipeline_ignores_thresholds_for_smoke_checkpoint(tmp_path: Path, caplog) -> None:
    checkpoint = tmp_path / "classifier"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"smoke_mode": True}), encoding="utf-8")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"tau": 0.9, "theta": 0.9, "temperature": 1.0}), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        pipeline = Pipeline(mode="phobert_only", checkpoint_dir=checkpoint, thresholds_path=thresholds)
    assert pipeline.tau != 0.9
    assert "checkpoint thật" in caplog.text
