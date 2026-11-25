"""JointBERT PhoBERT head và tiện ích căn chỉnh nhãn BIO cấp từ.

Loss tuân theo JointBERT: CE intent cộng CE slot trung bình trên các từ có nhãn.
Mô-đun smoke dùng encoder đồ chơi để chạy CPU ngoại tuyến; huấn luyện đầy đủ nhận
hidden states từ backbone PhoBERT đã tokenize bởi caller. Không dùng CRF.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    JOINT_SLOT_LEARNING_RATES,
    NLU_BATCH_SIZE,
    NLU_EARLY_STOPPING_PATIENCE,
    NLU_EXPERIMENT_SEEDS,
    NLU_IGNORE_INDEX,
    NLU_MAX_EPOCHS,
)


@dataclass
class JointOutput:
    intent_logits: torch.Tensor
    slot_logits: torch.Tensor
    loss: torch.Tensor | None = None


def align_word_labels(word_ids: list[int | None], word_labels: list[int],
                      ignore_index: int = NLU_IGNORE_INDEX) -> list[int]:
    """Gán nhãn chỉ cho sub-token đầu tiên; các sub-token tiếp theo bị bỏ qua."""
    aligned: list[int] = []
    previous: int | None = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(ignore_index)
        elif word_id != previous:
            aligned.append(word_labels[word_id])
        else:
            aligned.append(ignore_index)
        previous = word_id
    return aligned


class JointIntentSlotModel(nn.Module):
    """Hai head trên cùng H; dùng hidden state sub-token đầu của mỗi từ."""

    def __init__(self, hidden_size: int, num_intents: int, num_slot_labels: int) -> None:
        super().__init__()
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.slot_head = nn.Linear(hidden_size, num_slot_labels)

    def forward(self, hidden: torch.Tensor, slot_labels: torch.Tensor | None = None,
                intent_labels: torch.Tensor | None = None) -> JointOutput:
        intent_logits = self.intent_head(hidden[:, 0])
        slot_logits = self.slot_head(hidden)
        loss = None
        if intent_labels is not None and slot_labels is not None:
            intent_loss = nn.functional.cross_entropy(intent_logits, intent_labels)
            token_loss = nn.functional.cross_entropy(
                slot_logits.reshape(-1, slot_logits.shape[-1]), slot_labels.reshape(-1),
                ignore_index=NLU_IGNORE_INDEX, reduction="none",
            ).reshape(slot_labels.shape)
            counts = (slot_labels != NLU_IGNORE_INDEX).sum(dim=1).clamp_min(1)
            slot_loss = (token_loss.sum(dim=1) / counts).mean()
            loss = intent_loss + slot_loss
        return JointOutput(intent_logits, slot_logits, loss)


class SeparateIntentSlotHeads(nn.Module):
    """Hai head tách rời nhận H từ hai PhoBERT độc lập (no-joint ablation)."""

    def __init__(self, hidden_size: int, num_intents: int, num_slot_labels: int) -> None:
        super().__init__()
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.slot_head = nn.Linear(hidden_size, num_slot_labels)

    def forward(self, intent_hidden: torch.Tensor, slot_hidden: torch.Tensor,
                slot_labels: torch.Tensor | None = None,
                intent_labels: torch.Tensor | None = None) -> JointOutput:
        intent_logits = self.intent_head(intent_hidden[:, 0])
        slot_logits = self.slot_head(slot_hidden)
        loss = None
        if slot_labels is not None and intent_labels is not None:
            loss_intent = nn.functional.cross_entropy(intent_logits, intent_labels)
            token_losses = nn.functional.cross_entropy(
                slot_logits.flatten(0, 1), slot_labels.flatten(), ignore_index=NLU_IGNORE_INDEX,
                reduction="none",
            ).reshape_as(slot_labels)
            counts = (slot_labels != NLU_IGNORE_INDEX).sum(1).clamp_min(1)
            loss_slot = (token_losses.sum(1) / counts).mean()
            loss = loss_intent + loss_slot
        return JointOutput(intent_logits, slot_logits, loss)


def _smoke() -> None:
    torch.manual_seed(7)
    model = JointIntentSlotModel(8, 3, 5)
    hidden = torch.randn(2, 4, 8)
    labels = torch.tensor([[0, 1, NLU_IGNORE_INDEX, 2], [0, 3, 4, NLU_IGNORE_INDEX]])
    output = model(hidden, labels, torch.tensor([1, 2]))
    assert output.loss is not None and torch.isfinite(output.loss)
    assert align_word_labels([None, 0, 0, 1], [2, 3]) == [NLU_IGNORE_INDEX, 2, NLU_IGNORE_INDEX, 3]
    print("joint intent-slot CPU smoke: ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data", type=Path, default=Path("data/slot_gold_bio.csv"))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=list(JOINT_SLOT_LEARNING_RATES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(NLU_EXPERIMENT_SEEDS))
    parser.add_argument("--epochs", type=int, default=NLU_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=NLU_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=NLU_BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/joint_intent_slot"))
    parser.add_argument("--results", type=Path, default=Path("report/joint_intent_slot_results.json"))
    parser.add_argument("--skip-no-joint-ablation", action="store_true")
    parser.add_argument("--skip-ka05-baseline", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
        return
    from train.nlu_training import run_experiment

    result = run_experiment(args.data, "joint", args.learning_rates, [(1.0, 0.0, 0.0)],
                            args.seeds, args.epochs, args.patience, args.batch_size,
                            args.device, args.output)
    if not args.skip_ka05_baseline:
        ka05 = run_experiment(args.data, "intent_only", args.learning_rates, [(1.0, 0.0, 0.0)],
                              args.seeds, args.epochs, args.patience, args.batch_size,
                              args.device, args.output / "ka05_intent_only")
        from train.nlu_training import compare_intent_predictions

        result["ka05_intent_only"] = ka05
        result["intent_mcnemar_joint_vs_ka05"] = compare_intent_predictions(result, ka05)
    if not args.skip_no_joint_ablation:
        separate = run_experiment(args.data, "separate", [result["selected_learning_rate"]],
                                  [(1.0, 0.0, 0.0)], args.seeds, args.epochs,
                                  args.patience, args.batch_size, args.device,
                                  args.output / "separate")
        from train.nlu_training import compare_intent_predictions

        result["no_joint_ablation"] = separate
        result["intent_mcnemar_joint_vs_separate"] = compare_intent_predictions(result, separate)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate_test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
