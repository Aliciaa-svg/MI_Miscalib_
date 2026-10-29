from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mi_miscalib.metrics.logit_difference import (
    ALL_CANDIDATES,
    HIGH_CONF_SET,
    LOW_CONF_SET,
    build_candidate_token_map,
    compute_tsld,
    score_candidates,
)


class FixedLogitModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def __call__(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        return SimpleNamespace(logits=self.logits.expand(batch, seq_len, -1).clone())


def test_candidate_sets():
    assert HIGH_CONF_SET == [70, 75, 80, 85, 90, 95, 99]
    assert LOW_CONF_SET == [0, 10, 15, 20, 25, 30]
    assert ALL_CANDIDATES == sorted(set(HIGH_CONF_SET) | set(LOW_CONF_SET))


def test_build_candidate_token_map(digit_tokenizer):
    token_map = build_candidate_token_map(digit_tokenizer, [5, 85, 100])
    assert token_map == {5: [5], 85: [8, 5], 100: [1, 0, 0]}


def test_compute_tsld():
    scores = {c: 2.0 for c in HIGH_CONF_SET} | {c: -1.0 for c in LOW_CONF_SET}
    assert compute_tsld(scores) == pytest.approx(3.0)
    assert compute_tsld(scores, high_conf_set=[70], low_conf_set=[0]) == pytest.approx(3.0)


def test_score_candidates_first_token(digit_tokenizer):
    logits = torch.arange(10, dtype=torch.float)  # logit of token d is d
    model = FixedLogitModel(logits)
    token_map = build_candidate_token_map(digit_tokenizer)
    scores = score_candidates(model, torch.zeros(1, 4, dtype=torch.long), token_map)
    assert scores[85] == 8.0 and scores[0] == 0.0 and scores[30] == 3.0
    assert compute_tsld(scores) > 0


def test_score_candidates_teacher_forcing(digit_tokenizer):
    logits = torch.zeros(10)
    logits[9] = 5.0
    model = FixedLogitModel(logits)
    token_map = build_candidate_token_map(digit_tokenizer, [9, 99, 90])
    scores = score_candidates(
        model, torch.zeros(1, 4, dtype=torch.long), token_map, method="teacher_forcing_logprob_sum"
    )
    log_p9 = torch.log_softmax(logits, -1)[9].item()
    log_p0 = torch.log_softmax(logits, -1)[0].item()
    assert scores[9] == pytest.approx(log_p9)
    assert scores[99] == pytest.approx(2 * log_p9)
    assert scores[90] == pytest.approx(log_p9 + log_p0)


def test_score_candidates_rejects_unknown_method(digit_tokenizer):
    with pytest.raises(ValueError):
        score_candidates(FixedLogitModel(torch.zeros(10)), torch.zeros(1, 2, dtype=torch.long),
                         build_candidate_token_map(digit_tokenizer), method="argmax")
