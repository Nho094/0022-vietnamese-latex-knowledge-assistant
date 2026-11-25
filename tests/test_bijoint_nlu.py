from __future__ import annotations

import torch

from train.bijoint_nlu import BiJointNLU


def test_bijoint_intermediate_shapes_and_weighted_loss() -> None:
    model = BiJointNLU(4, 2, 3)
    hidden = torch.randn(2, 4, 4)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    slot_labels = torch.tensor([[0, 1, -100], [1, 2, 0]])
    output = model(hidden, mask, torch.tensor([0, 1]), slot_labels, (0.2, 0.3, 0.5))
    assert output.intent_label_embedding.shape == (2, 4)
    assert output.attention.shape == (2, 3)
    assert torch.allclose(output.attention.sum(-1), torch.ones(2))
    assert output.slot_context.shape == (2, 3, 8)
    assert output.slot_probability_sum.shape == (2, 3)
    assert output.final_intent_logits.shape == (2, 2)
    assert output.losses is not None and output.total_loss is not None
    expected = sum(w * loss for w, loss in zip((0.2, 0.3, 0.5), output.losses))
    assert torch.allclose(output.total_loss, expected)
