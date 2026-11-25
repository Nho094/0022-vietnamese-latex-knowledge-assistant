"""BiJoint-BERT-NLU head theo công thức KA-12, không có CRF.

Attention chuẩn hóa trên các từ hợp lệ; q là tổng vector xác suất slot theo
loại (không phải tổng nhãn argmax), sau đó ghép với h_0 cho intent cuối.
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
    BIJOINT_LEARNING_RATE,
    BIJOINT_PAPER_LOSS_WEIGHTS,
    NLU_BATCH_SIZE,
    NLU_EARLY_STOPPING_PATIENCE,
    NLU_EXPERIMENT_SEEDS,
    NLU_IGNORE_INDEX,
    NLU_MAX_EPOCHS,
)


@dataclass
class BiJointOutput:
    attention: torch.Tensor
    intent_label_embedding: torch.Tensor
    slot_context: torch.Tensor
    slot_logits: torch.Tensor
    slot_probability_sum: torch.Tensor
    final_intent_logits: torch.Tensor
    losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    total_loss: torch.Tensor | None


class BiJointNLU(nn.Module):
    """Triển khai các tensor w, alpha, v, q và y_final đúng đặc tả."""

    def __init__(self, hidden_size: int, num_intents: int, num_slots: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.temporary_intent = nn.Linear(hidden_size, num_intents)
        self.intent_embedding = nn.Linear(num_intents, hidden_size, bias=False)
        self.slot_head = nn.Linear(hidden_size * 2, num_slots)
        self.final_intent = nn.Linear(hidden_size + num_slots, num_intents)

    def forward(self, hidden: torch.Tensor, word_mask: torch.Tensor,
                intent_labels: torch.Tensor | None = None,
                slot_labels: torch.Tensor | None = None,
                weights: tuple[float, float, float] = (0.2, 0.3, 0.5)) -> BiJointOutput:
        h0, words = hidden[:, 0], hidden[:, 1:]
        temporary_logits = self.temporary_intent(h0)
        y_tmp = torch.softmax(temporary_logits, dim=-1)
        w = self.intent_embedding(y_tmp)
        scores = torch.einsum("bd,bnd->bn", w, words).masked_fill(~word_mask.bool(), -torch.inf)
        alpha = torch.softmax(scores, dim=-1)
        s = alpha.unsqueeze(-1) * w.unsqueeze(1)
        v = torch.cat((s, words), dim=-1)
        slot_logits = self.slot_head(v)
        slot_probs = torch.softmax(slot_logits, dim=-1) * word_mask.unsqueeze(-1)
        q = slot_probs.sum(dim=1)
        final_logits = self.final_intent(torch.cat((q, h0), dim=-1))
        losses = total = None
        if intent_labels is not None and slot_labels is not None:
            l_tmp = nn.functional.cross_entropy(temporary_logits, intent_labels)
            l_st = nn.functional.cross_entropy(slot_logits.reshape(-1, self.num_slots),
            slot_labels.reshape(-1), ignore_index=NLU_IGNORE_INDEX)
            l_final = nn.functional.cross_entropy(final_logits, intent_labels)
            losses = (l_tmp, l_st, l_final)
            total = sum(float(weight) * loss for weight, loss in zip(weights, losses))
        return BiJointOutput(alpha, w, v, slot_logits, q, final_logits, losses, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data", type=Path, default=Path("data/slot_gold_bio.csv"))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[BIJOINT_LEARNING_RATE])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(NLU_EXPERIMENT_SEEDS))
    parser.add_argument("--epochs", type=int, default=NLU_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=NLU_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=NLU_BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/bijoint_nlu"))
    parser.add_argument("--results", type=Path, default=Path("report/bijoint_nlu_results.json"))
    parser.add_argument("--joint-results", type=Path, default=Path("report/joint_intent_slot_results.json"))
    args = parser.parse_args()
    if args.smoke:
        torch.manual_seed(11)
        model = BiJointNLU(8, 3, 5)
        hidden = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.float32)
        labels = torch.tensor([[0, 1, 2, -100], [1, 2, 3, 4]])
        out = model(hidden, mask, torch.tensor([1, 2]), labels)
        assert out.intent_label_embedding.shape == (2, 8)
        assert out.attention.shape == (2, 4) and torch.allclose(out.attention.sum(-1), torch.ones(2))
        assert out.slot_context.shape == (2, 4, 16) and out.slot_probability_sum.shape == (2, 5)
        assert out.final_intent_logits.shape == (2, 3) and out.total_loss is not None
        print("BiJoint NLU CPU smoke: ok")
        return
    from train.nlu_training import run_experiment

    weights = list(BIJOINT_PAPER_LOSS_WEIGHTS)
    if args.full_grid:
        weights = [(tmp / 10, slot / 10, final / 10)
                   for tmp in range(1, 9) for slot in range(1, 10 - tmp)
                   for final in [10 - tmp - slot] if final >= 1]
    result = run_experiment(args.data, "bijoint", args.learning_rates, weights,
                            args.seeds, args.epochs, args.patience, args.batch_size,
                            args.device, args.output)
    if args.joint_results.exists():
        previous = json.loads(args.joint_results.read_text(encoding="utf-8"))
        from train.nlu_training import compare_intent_predictions

        result["intent_mcnemar_bijoint_vs_joint"] = compare_intent_predictions(result, previous)
        if "ka05_intent_only" in previous:
            result["intent_mcnemar_bijoint_vs_ka05"] = compare_intent_predictions(
                result, previous["ka05_intent_only"]
            )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate_test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
