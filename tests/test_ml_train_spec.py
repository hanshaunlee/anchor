"""
Specification-based ML train tests: focal loss properties, not just shape.
"""
import pytest

pytest.importorskip("torch")
import torch

from ml.train import focal_loss


def test_focal_loss_higher_for_wrong_predictions() -> None:
    """Focal loss down-weights easy examples: wrong labels should give higher loss than correct."""
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [2.0, -1.0], [-1.0, 2.0]])  # confident correct
    correct = torch.tensor([0, 1, 0, 1])
    wrong = torch.tensor([1, 0, 1, 0])
    loss_correct = focal_loss(logits, correct)
    loss_wrong = focal_loss(logits, wrong)
    assert loss_wrong.item() > loss_correct.item()


def test_focal_loss_reduction_sum_different_from_mean() -> None:
    """Sum reduction should scale with batch size vs mean."""
    logits = torch.randn(10, 2)
    targets = torch.randint(0, 2, (10,))
    loss_mean = focal_loss(logits, targets, reduction="mean")
    loss_sum = focal_loss(logits, targets, reduction="sum")
    assert loss_sum.item() == pytest.approx(loss_mean.item() * 10, rel=1e-5)


def test_focal_loss_zero_for_perfect_confidence() -> None:
    """With infinite logit for correct class, CE is 0 so focal is 0."""
    logits = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
    targets = torch.tensor([0, 1])
    loss = focal_loss(logits, targets)
    assert loss.item() < 1e-5
