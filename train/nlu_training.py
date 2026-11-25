"""Huấn luyện và đánh giá Joint/BiJoint trên dữ liệu BIO theo cùng giao thức."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from core.config import (
    DEFAULT_CLASSIFIER_DROPOUT,
    NLU_ADAMW_BETAS,
    NLU_ADAMW_EPS,
    NLU_IGNORE_INDEX,
    PHOBERT_DIR,
    PHOBERT_MAX_SEQUENCE_LIMIT,
    PHOBERT_MODEL_NAME,
    set_seed,
)
from core.slots import extract_slots


@dataclass
class EncodedRows:
    rows: list[dict[str, Any]]
    intents: list[str]
    slot_labels: list[str]


class IntentOnlyHead(nn.Module):
    """Head KA-05 chỉ dự đoán intent trên h_0, không có nhánh slot."""

    def __init__(self, hidden_size: int, num_intents: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(DEFAULT_CLASSIFIER_DROPOUT)
        self.classifier = nn.Linear(hidden_size, num_intents)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(hidden[:, 0]))


def read_bio_csv(path: Path) -> EncodedRows:
    """Nạp CSV do data/make_bio.py sinh, không lấy test từ nguồn khác."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"split", "intent", "tokens", "bio", "spans"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} cần các cột: {sorted(required)}")
    for row in rows:
        row["tokens"] = json.loads(row["tokens"])
        row["bio"] = json.loads(row["bio"])
        row["spans"] = json.loads(row["spans"])
        if len(row["tokens"]) != len(row["bio"]):
            raise ValueError("Số token và nhãn BIO không khớp")
    splits = {row["split"] for row in rows}
    if not {"train", "val", "test"}.issubset(splits):
        raise ValueError("Cần có train tổng hợp và val/test owner-written trong slot_gold.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    intents = sorted({row["intent"] for row in train_rows})
    tags = sorted({tag for row in train_rows for tag in row["bio"]} | {"O"})
    return EncodedRows(rows, intents, tags)


def _make_batches(rows: list[dict[str, Any]], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda batch: batch)


def _tokenize_words(tokenizer: Any, words: list[list[str]]) -> Any:
    """Dùng cùng giới hạn chuỗi cho train/validation/test để căn chỉnh token ổn định."""
    return tokenizer(words, is_split_into_words=True, padding=True, truncation=True,
                     max_length=PHOBERT_MAX_SEQUENCE_LIMIT, return_tensors="pt")


def _encode_batch(batch: list[dict[str, Any]], tokenizer: Any, intent2id: dict[str, int],
                  tag2id: dict[str, int], device: torch.device,
                  word_level: bool) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    words = [row["tokens"] for row in batch]
    tokenized = _tokenize_words(tokenizer, words)
    aligned: list[list[int]] = []
    for index, row in enumerate(batch):
        word_ids = tokenized.word_ids(batch_index=index)
        if word_level:
            aligned.append([tag2id[tag] for tag in row["bio"]])
        else:
            previous: int | None = None
            labels: list[int] = []
            for word_id in word_ids:
                if word_id is None:
                    labels.append(NLU_IGNORE_INDEX)
                elif word_id != previous and word_id < len(row["bio"]):
                    labels.append(tag2id[row["bio"][word_id]])
                else:
                    labels.append(NLU_IGNORE_INDEX)
                previous = word_id
            aligned.append(labels)
    if word_level:
        width = max(len(tags) for tags in aligned)
        padded = [tags + [NLU_IGNORE_INDEX] * (width - len(tags)) for tags in aligned]
        slot_labels = torch.tensor(padded, dtype=torch.long, device=device)
    else:
        slot_labels = torch.tensor(aligned, dtype=torch.long, device=device)
    inputs = {key: value.to(device) for key, value in tokenized.items()}
    intent_labels = torch.tensor([intent2id[row["intent"]] for row in batch],
                                 dtype=torch.long, device=device)
    return inputs, intent_labels, slot_labels


def _entities(tags: list[str]) -> set[tuple[str, int, int]]:
    found: set[tuple[str, int, int]] = set()
    active: str | None = None
    start = 0
    for index, tag in enumerate([*tags, "O"]):
        prefix, _, slot = tag.partition("-")
        if active is not None and (prefix != "I" or slot != active):
            found.add((active, start, index))
            active = None
        if prefix == "B" or prefix == "I" and active is None:
            active, start = slot, index
    return found


def _metrics(gold_intents: list[str], pred_intents: list[str],
             gold_tags: list[list[str]], pred_tags: list[list[str]]) -> dict[str, float]:
    try:
        from seqeval.metrics import f1_score

        slot_f1 = float(f1_score(gold_tags, pred_tags))
    except ImportError:
        gold_entities = {(sentence, *entity) for sentence, tags in enumerate(gold_tags)
                         for entity in _entities(tags)}
        predicted_entities = {(sentence, *entity) for sentence, tags in enumerate(pred_tags)
                              for entity in _entities(tags)}
        true_positive = len(gold_entities & predicted_entities)
        denominator = len(gold_entities) + len(predicted_entities)
        slot_f1 = 2 * true_positive / denominator if denominator else 0.0

    intent_acc = sum(gold == pred for gold, pred in zip(gold_intents, pred_intents)) / max(len(gold_intents), 1)
    exact_slots = sum(gold == pred for gold, pred in zip(gold_tags, pred_tags)) / max(len(gold_tags), 1)
    sentence_acc = sum(gold_i == pred_i and gold_s == pred_s
                       for gold_i, pred_i, gold_s, pred_s in zip(gold_intents, pred_intents, gold_tags, pred_tags)) / max(len(gold_intents), 1)
    return {"intent_accuracy": intent_acc, "slot_f1": slot_f1,
            "exact_slot_accuracy": exact_slots, "sentence_accuracy": sentence_acc}


def regex_baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """So sánh regex KA-08 với gold span trên đúng các câu đầu vào này."""
    true_entities = 0
    predicted_entities = 0
    correct_entities = 0
    exact = 0
    evaluated = 0
    for row in rows:
        gold: set[tuple[str, str]] = set()
        for span in row["spans"]:
            value = row["text"][int(span["start"]):int(span["end"])].strip()
            gold.add((str(span["slot"]), value))
        extracted = extract_slots(row["text"], row["intent"])
        predicted = {(name, str(value).strip()) for name, value in extracted.slots.items()}
        true_entities += len(gold)
        predicted_entities += len(predicted)
        correct_entities += len(gold & predicted)
        exact += int(gold == predicted)
        evaluated += 1
    denominator = true_entities + predicted_entities
    return {"slot_f1": (2 * correct_entities / denominator) if denominator else 0.0,
            "exact_slot_accuracy": exact / max(evaluated, 1), "sentences": float(evaluated)}


def compare_intent_predictions(model_a: dict[str, Any], model_b: dict[str, Any]) -> dict[str, Any]:
    """McNemar Yates trên các dự đoán cùng gold và cùng seed của hai mô hình."""
    if len(model_a["per_seed"]) != len(model_b["per_seed"]):
        raise ValueError("McNemar cần cùng số lượng seed giữa hai mô hình")
    b = c = 0
    paired = 0
    for seed_a, seed_b in zip(model_a["per_seed"], model_b["per_seed"]):
        prediction_a = seed_a["test_intent_predictions"]
        prediction_b = seed_b["test_intent_predictions"]
        gold_a, gold_b = prediction_a["gold_intents"], prediction_b["gold_intents"]
        if gold_a != gold_b:
            raise ValueError("McNemar chỉ hợp lệ khi hai mô hình dùng đúng cùng test labels")
        if (len(prediction_a["predicted_intents"]) != len(gold_a)
                or len(prediction_b["predicted_intents"]) != len(gold_b)):
            raise ValueError("Thiếu dự đoán cho một hoặc nhiều test samples")
        for gold, pred_a, pred_b in zip(gold_a, prediction_a["predicted_intents"],
                                        prediction_b["predicted_intents"]):
            b += int(gold == pred_a and gold != pred_b)
            c += int(gold != pred_a and gold == pred_b)
            paired += 1
    discordant = b + c
    statistic = (max(0, abs(b - c) - 1) ** 2 / discordant) if discordant else 0.0
    p_value = math.erfc(math.sqrt(statistic / 2)) if discordant else 1.0
    return {"b": b, "c": c, "statistic_yates": statistic, "p_value": p_value,
            "paired_predictions": paired}


def run_experiment(data_path: Path, architecture: str, learning_rates: list[float],
                   weights: list[tuple[float, float, float]], seeds: list[int], epochs: int,
                   patience: int, batch_size: int, device_name: str | None,
                   output_dir: Path) -> dict[str, Any]:
    """Chọn lr/weights trên VAL, sau đó đánh giá cấu hình đã chọn trên TEST một lần."""
    from transformers import AutoModel, AutoTokenizer

    from train.bijoint_nlu import BiJointNLU
    from train.joint_intent_slot import JointIntentSlotModel, SeparateIntentSlotHeads

    corpus = read_bio_csv(data_path)
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    split = {name: [r for r in corpus.rows if r["split"] == name] for name in ("train", "val", "test")}
    model_source = PHOBERT_DIR if (PHOBERT_DIR / "config.json").exists() else PHOBERT_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(str(model_source), use_fast=True)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    intent2id = {label: index for index, label in enumerate(corpus.intents)}
    tag2id = {label: index for index, label in enumerate(corpus.slot_labels)}
    id2tag = {index: label for label, index in tag2id.items()}

    def one_run(lr: float, loss_weights: tuple[float, float, float], seed: int,
                evaluate_test: bool) -> tuple[dict[str, float], dict[str, float] | None,
                                               nn.Module, dict[str, list[str]] | None]:
        set_seed(seed)
        local_only = model_source == PHOBERT_DIR
        backbone = AutoModel.from_pretrained(str(model_source), local_files_only=local_only)
        hidden_dim = int(backbone.config.hidden_size)
        if architecture == "separate":
            slot_backbone = AutoModel.from_pretrained(str(model_source), local_files_only=local_only)
            head: nn.Module = SeparateIntentSlotHeads(hidden_dim, len(intent2id), len(tag2id))
            model = nn.ModuleDict({"encoder_intent": backbone, "encoder_slot": slot_backbone,
                                   "head": head}).to(device)
        else:
            if architecture == "intent_only":
                head = IntentOnlyHead(hidden_dim, len(intent2id))
            else:
                head = (JointIntentSlotModel(hidden_dim, len(intent2id), len(tag2id))
                        if architecture == "joint" else
                        BiJointNLU(hidden_dim, len(intent2id), len(tag2id)))
            model = nn.ModuleDict({"encoder": backbone, "head": head}).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=NLU_ADAMW_EPS,
                                      betas=NLU_ADAMW_BETAS)
        best_state: dict[str, torch.Tensor] | None = None
        best_val: dict[str, float] = {}
        stale = 0
        saved_predictions: dict[str, list[str]] | None = None

        def evaluate(name: str) -> dict[str, float]:
            nonlocal saved_predictions
            model.eval()
            gold_i: list[str] = []
            pred_i: list[str] = []
            gold_s: list[list[str]] = []
            pred_s: list[list[str]] = []
            with torch.no_grad():
                for batch in _make_batches(split[name], batch_size, False):
                    word_level = architecture == "bijoint"
                    inputs, intent_ids, slot_ids = _encode_batch(batch, tokenizer, intent2id,
                                                                  tag2id, device, word_level)
                    if architecture == "separate":
                        intent_hidden = model["encoder_intent"](**inputs).last_hidden_state
                        encoded = model["encoder_slot"](**inputs).last_hidden_state
                    else:
                        encoded = model["encoder"](**inputs).last_hidden_state
                    if architecture == "intent_only":
                        logits = model["head"](encoded)
                        ip = logits.argmax(-1).tolist()
                        for i, row in enumerate(batch):
                            gold_i.append(row["intent"])
                            pred_i.append(corpus.intents[ip[i]])
                        continue
                    if word_level:
                        tokenized = _tokenize_words(tokenizer, [r["tokens"] for r in batch])
                        word_id_lists = [tokenized.word_ids(batch_index=i) for i in range(len(batch))]
                        vectors: list[list[int]] = []
                        for ids in word_id_lists:
                            previous: int | None = None
                            positions: list[int] = []
                            for pos, word_id in enumerate(ids):
                                if word_id is not None and word_id != previous:
                                    positions.append(pos)
                                previous = word_id
                            vectors.append(positions)
                        word_hidden, word_mask = _select_word_hidden(encoded, vectors)
                        slot_ids = _pad_labels([r["bio"] for r in batch], tag2id, device)
                        out = model["head"](word_hidden, word_mask, intent_ids, slot_ids, loss_weights)
                        ip = out.final_intent_logits.argmax(-1).tolist()
                        sp = out.slot_logits.argmax(-1).tolist()
                        for i, row in enumerate(batch):
                            pred_s.append([id2tag[v] for v in sp[i][:len(row["bio"])]] )
                    else:
                        out = (model["head"](intent_hidden, encoded, slot_ids, intent_ids)
                               if architecture == "separate" else
                               model["head"](encoded, slot_ids, intent_ids))
                        ip = out.intent_logits.argmax(-1).tolist()
                        sp = out.slot_logits.argmax(-1).tolist()
                        tokenized = _tokenize_words(tokenizer, [r["tokens"] for r in batch])
                        for i, row in enumerate(batch):
                            tags: list[str] = []
                            seen: set[int] = set()
                            for pos, word_id in enumerate(tokenized.word_ids(batch_index=i)):
                                if word_id is not None and word_id not in seen and word_id < len(row["bio"]):
                                    seen.add(word_id)
                                    tags.append(id2tag[sp[i][pos]])
                            pred_s.append(tags)
                    for i, row in enumerate(batch):
                        gold_i.append(row["intent"])
                        pred_i.append(corpus.intents[ip[i]])
                        gold_s.append(row["bio"])
            if name == "test":
                saved_predictions = {"gold_intents": gold_i, "predicted_intents": pred_i}
            if architecture == "intent_only":
                accuracy = sum(gold == pred for gold, pred in zip(gold_i, pred_i)) / max(len(gold_i), 1)
                if name == "test":
                    saved_predictions = {"gold_intents": gold_i, "predicted_intents": pred_i}
                return {"intent_accuracy": accuracy, "sentence_accuracy": accuracy}
            return _metrics(gold_i, pred_i, gold_s, pred_s)

        for _epoch in range(epochs):
            model.train()
            for batch in _make_batches(split["train"], batch_size, True):
                word_level = architecture == "bijoint"
                inputs, intent_ids, slot_ids = _encode_batch(batch, tokenizer, intent2id,
                                                              tag2id, device, word_level)
                if architecture == "separate":
                    intent_hidden = model["encoder_intent"](**inputs).last_hidden_state
                    hidden = model["encoder_slot"](**inputs).last_hidden_state
                else:
                    hidden = model["encoder"](**inputs).last_hidden_state
                if architecture == "intent_only":
                    logits = model["head"](hidden)
                    loss = nn.functional.cross_entropy(logits, intent_ids)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if word_level:
                    tokenized = _tokenize_words(tokenizer, [r["tokens"] for r in batch])
                    positions = _first_word_positions(tokenized, len(batch))
                    word_hidden, word_mask = _select_word_hidden(hidden, positions)
                    slot_ids = _pad_labels([r["bio"] for r in batch], tag2id, device)
                    output = model["head"](word_hidden, word_mask, intent_ids, slot_ids, loss_weights)
                    loss = output.total_loss
                else:
                    output = (model["head"](intent_hidden, hidden, slot_ids, intent_ids)
                              if architecture == "separate" else
                              model["head"](hidden, slot_ids, intent_ids))
                    loss = output.loss
                if loss is None:
                    raise RuntimeError("Loss không được tạo dù labels đã truyền")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scores = evaluate("val")
            if scores["sentence_accuracy"] > best_val.get("sentence_accuracy", -1.0):
                best_val = scores
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("Không có checkpoint validation")
        model.load_state_dict(best_state)
        test_metrics = evaluate("test") if evaluate_test else None
        return best_val, test_metrics, model, saved_predictions

    configs = [(lr, weight) for lr in learning_rates for weight in weights]
    selection_runs: list[dict[str, Any]] = []
    best = configs[0]
    best_score = -1.0
    for lr, weight in configs:
        val_metrics, _, _, _ = one_run(lr, weight, seeds[0], False)
        selection_runs.append({"learning_rate": lr, "weights": weight, "val": val_metrics})
        if val_metrics["sentence_accuracy"] > best_score:
            best_score, best = val_metrics["sentence_accuracy"], (lr, weight)

    test_runs: list[dict[str, Any]] = []
    for seed in seeds:
        val_metrics, test_metrics, model, predictions = one_run(best[0], best[1], seed, True)
        test_runs.append({"seed": seed, "val": val_metrics, "test": test_metrics,
                          "test_intent_predictions": predictions})
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / f"{architecture}_seed_{seed}.pt")
    metric_names = list(test_runs[0]["test"])
    aggregate = {name: {"mean": statistics.mean(row["test"][name] for row in test_runs),
                        "std": statistics.stdev(row["test"][name] for row in test_runs)
                        if len(test_runs) > 1 else 0.0} for name in metric_names}
    regex_test = regex_baseline_metrics(split["test"])
    return {"architecture": architecture, "gold_bio_sha256": data_hash,
            "split_counts": {name: len(rows) for name, rows in split.items()},
            "selected_learning_rate": best[0],
            "selected_loss_weights": best[1], "validation_search": selection_runs,
            "per_seed": test_runs, "aggregate_test": aggregate,
            "regex_baseline_test": regex_test,
            "intents": corpus.intents, "slot_labels": corpus.slot_labels}


def _first_word_positions(tokenized: Any, batch_size: int) -> list[list[int]]:
    positions: list[list[int]] = []
    for index in range(batch_size):
        ids = tokenized.word_ids(batch_index=index)
        prev: int | None = None
        rows: list[int] = []
        for pos, word_id in enumerate(ids):
            if word_id is not None and word_id != prev:
                rows.append(pos)
            prev = word_id
        positions.append(rows)
    return positions


def _select_word_hidden(hidden: torch.Tensor, positions: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    max_words = max(map(len, positions))
    output = hidden.new_zeros((len(positions), max_words + 1, hidden.shape[-1]))
    mask = hidden.new_zeros((len(positions), max_words))
    for batch_index, pos_list in enumerate(positions):
        output[batch_index, 0] = hidden[batch_index, 0]
        for index, pos in enumerate(pos_list[:max_words]):
            output[batch_index, index + 1] = hidden[batch_index, pos]
            mask[batch_index, index] = 1
    return output, mask


def _pad_labels(rows: list[list[str]], tag2id: dict[str, int], device: torch.device) -> torch.Tensor:
    width = max(map(len, rows))
    return torch.tensor([[tag2id[tag] for tag in row] + [NLU_IGNORE_INDEX] * (width - len(row)) for row in rows],
                        dtype=torch.long, device=device)
