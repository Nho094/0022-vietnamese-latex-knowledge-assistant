from __future__ import annotations

from train.nlu_training import regex_baseline_metrics


def test_regex_baseline_uses_the_same_gold_sentence_spans() -> None:
    text = "2x^2 + 3x + 5 = 0"
    spans = [
        {"start": 0, "end": 1, "slot": "a"},
        {"start": 7, "end": 8, "slot": "b"},
        {"start": 12, "end": 13, "slot": "c"},
    ]
    metrics = regex_baseline_metrics([{"text": text, "intent": "QUADRATIC_EQUATION", "spans": spans}])
    assert metrics["slot_f1"] == 1.0
    assert metrics["exact_slot_accuracy"] == 1.0
