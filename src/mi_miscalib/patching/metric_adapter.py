from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from mi_miscalib.metrics.logit_difference import HIGH_CONF_SET, LOW_CONF_SET


def make_tsld_metric_for_eap(
    tokenizer: object,
    high_conf_set: list[int] = HIGH_CONF_SET,
    low_conf_set: list[int] = LOW_CONF_SET,
) -> Callable[..., Tensor]:
    def _first_token_id(n: int) -> int:
        ids = tokenizer(str(n), add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"Tokenizer returned empty IDs for candidate {n}")
        return ids[0]

    high_token_ids = [_first_token_id(c) for c in high_conf_set]
    low_token_ids = [_first_token_id(c) for c in low_conf_set]

    def metric_fn(
        clean_logits: Tensor,
        corrupted_logits: Tensor,
        input_length: Tensor,
        labels: object,
        mean: bool = True,
        prob: bool = False,
        loss: bool = False,
    ) -> Tensor:
        results = []
        for b in range(clean_logits.shape[0]):
            last_logits = clean_logits[b, input_length[b] - 1]
            results.append(last_logits[high_token_ids].mean() - last_logits[low_token_ids].mean())
        result = torch.stack(results)
        return result.mean() if mean else result

    return metric_fn
