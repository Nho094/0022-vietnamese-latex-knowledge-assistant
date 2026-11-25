from __future__ import annotations

from data.make_bio import bio_labels


def test_math_pre_tokenization_keeps_slot_span_alignment() -> None:
    text = "Tạo 2x^2 + 3"
    tokens, labels = bio_labels(text, [{"start": 4, "end": 8, "slot": "coefficient"}])
    assert tokens[:3] == ["Tạo", "2", "x^2"]
    assert labels[:3] == ["O", "B-coefficient", "I-coefficient"]
