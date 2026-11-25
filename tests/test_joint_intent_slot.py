from __future__ import annotations

import torch

from train.joint_intent_slot import JointIntentSlotModel, align_word_labels


def test_only_first_subtoken_gets_word_label() -> None:
    assert align_word_labels([None, 0, 0, 1, 1, None], [2, 3]) == [-100, 2, -100, 3, -100, -100]


def test_joint_loss_is_intent_plus_mean_sentence_slot_loss() -> None:
    torch.manual_seed(2)
    model = JointIntentSlotModel(4, 2, 3)
    hidden = torch.randn(2, 3, 4)
    slots = torch.tensor([[0, 1, -100], [2, -100, -100]])
    intents = torch.tensor([0, 1])
    output = model(hidden, slots, intents)
    expected = torch.nn.functional.cross_entropy(output.intent_logits, intents)
    raw = torch.nn.functional.cross_entropy(output.slot_logits.reshape(-1, 3), slots.reshape(-1),
                                            ignore_index=-100, reduction="none").reshape(2, 3)
    expected += (raw.sum(1) / torch.tensor([2, 1])).mean()
    assert torch.allclose(output.loss, expected)
