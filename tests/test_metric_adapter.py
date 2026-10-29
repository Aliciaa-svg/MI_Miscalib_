from __future__ import annotations

import torch

from mi_miscalib.patching.metric_adapter import make_tsld_metric_for_eap


def test_tsld_metric_reads_last_real_position(digit_tokenizer):
    metric = make_tsld_metric_for_eap(digit_tokenizer)
    logits = torch.zeros(1, 6, 10)
    logits[0, 3, 7:10] = 4.0  # high-confidence first tokens "7", "8", "9" at position 3
    value = metric(logits, logits, torch.tensor([4]), None)
    assert value.ndim == 0
    assert value.item() == 4.0
    assert metric(logits, logits, torch.tensor([6]), None).item() == 0.0


def test_tsld_metric_batch_and_sign(digit_tokenizer):
    metric = make_tsld_metric_for_eap(digit_tokenizer)
    logits = torch.zeros(2, 3, 10)
    logits[0, -1, 9] = 7.0  # favours 90 / 95 / 99
    logits[1, -1, 0] = 6.0  # favours 0
    values = metric(logits, logits, torch.tensor([3, 3]), None, mean=False)
    assert values.shape == (2,)
    assert values[0] > 0 > values[1]


def test_tsld_metric_is_differentiable(digit_tokenizer):
    metric = make_tsld_metric_for_eap(digit_tokenizer)
    logits = torch.randn(1, 5, 10, requires_grad=True)
    metric(logits, logits.detach(), torch.tensor([5]), None).backward()
    assert logits.grad is not None
    assert logits.grad[0, :4].abs().sum() == 0  # only POS_END receives gradient
