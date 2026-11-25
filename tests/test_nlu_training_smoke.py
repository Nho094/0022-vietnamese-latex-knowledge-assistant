from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

from train import nlu_training


def test_runner_trains_and_evaluates_both_heads_on_a_tiny_local_encoder(
    tmp_path: Path, monkeypatch,
) -> None:
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
             "alpha": 4, "beta": 5, "math": 6}
    tokenizer_object = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_object.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object, bos_token="<s>",
                                        pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    model_dir = tmp_path / "tiny-phobert"
    model_dir.mkdir()
    tokenizer.save_pretrained(model_dir)
    config = RobertaConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
                           num_attention_heads=2, intermediate_size=16, max_position_embeddings=32)
    RobertaModel(config).save_pretrained(model_dir)
    monkeypatch.setattr(nlu_training, "PHOBERT_DIR", model_dir)

    bio_path = tmp_path / "tiny_bio.csv"
    records = []
    for split, prefix in (("train", "alpha"), ("val", "beta"), ("test", "alpha")):
        for intent, token in (("INTENT_A", prefix), ("INTENT_B", "math")):
            records.append({"split": split, "text": token, "intent": intent,
                            "tokens": json.dumps([token]), "bio": json.dumps(["O"]),
                            "spans": "[]"})
    with bio_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    # This environment does not install seqeval; isolate trainer control-flow from that optional package.
    seqeval = types.ModuleType("seqeval")
    metrics = types.ModuleType("seqeval.metrics")
    metrics.f1_score = lambda gold, pred: 1.0 if gold == pred else 0.0
    monkeypatch.setitem(sys.modules, "seqeval", seqeval)
    monkeypatch.setitem(sys.modules, "seqeval.metrics", metrics)

    experiment_results = {}
    for architecture, weights in (
        ("joint", [(1.0, 0.0, 0.0)]),
        ("separate", [(1.0, 0.0, 0.0)]),
        ("intent_only", [(1.0, 0.0, 0.0)]),
        ("bijoint", [(0.2, 0.3, 0.5)]),
    ):
        result = nlu_training.run_experiment(
            bio_path, architecture, [1e-3], weights, [7], 1, 1, 2, "cpu",
            tmp_path / architecture,
        )
        assert result["split_counts"] == {"train": 2, "val": 2, "test": 2}
        assert len(result["per_seed"]) == 1
        assert "intent_accuracy" in result["aggregate_test"]
        experiment_results[architecture] = result
    comparison = nlu_training.compare_intent_predictions(
        experiment_results["joint"], experiment_results["intent_only"]
    )
    assert comparison["paired_predictions"] == 2
