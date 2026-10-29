from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Confidence values (integers, not words) the model may verbalize in Step 2.
HIGH_CONF_SET: list[int] = [70, 75, 80, 85, 90, 95, 99]
LOW_CONF_SET: list[int] = [0, 10, 15, 20, 25, 30]
ALL_CANDIDATES: list[int] = sorted(set(HIGH_CONF_SET + LOW_CONF_SET))

SCORING_METHODS = ("first_token", "teacher_forcing_logprob_sum")


def build_candidate_token_map(
    tokenizer: AutoTokenizer,
    candidates: Iterable[int] = ALL_CANDIDATES,
) -> dict[int, list[int]]:
    token_map: dict[int, list[int]] = {}
    for n in candidates:
        ids = tokenizer(str(n), add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"Tokenizer returned empty token IDs for candidate {n}.")
        token_map[n] = ids
    return token_map


def score_candidates(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    token_map: dict[int, list[int]],
    method: str = "first_token",
) -> dict[int, float]:
    if method not in SCORING_METHODS:
        raise ValueError(f"Unsupported scoring method: {method}. Supported: {SCORING_METHODS}.")

    scores: dict[int, float] = {}
    if method == "first_token":
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
        for n, tok_ids in token_map.items():
            scores[n] = float(logits[0, tok_ids[0]].item())
        return scores

    prompt_len = int(input_ids.shape[1])
    with torch.no_grad():
        for n, tok_ids in token_map.items():
            target = torch.tensor([tok_ids], device=input_ids.device, dtype=input_ids.dtype)
            logits = model(torch.cat([input_ids, target], dim=1)).logits
            # Logits at position i predict token i + 1.
            pred_slice = logits[:, prompt_len - 1 : prompt_len - 1 + len(tok_ids), :]
            log_probs = F.log_softmax(pred_slice, dim=-1)
            token_log_probs = log_probs.gather(dim=2, index=target.unsqueeze(-1)).squeeze(-1)
            scores[n] = float(token_log_probs.sum().item())
    return scores


def compute_tsld(
    candidate_scores: dict[int, float],
    high_conf_set: list[int] = HIGH_CONF_SET,
    low_conf_set: list[int] = LOW_CONF_SET,
) -> float:
    high_scores = [candidate_scores[c] for c in high_conf_set]
    low_scores = [candidate_scores[c] for c in low_conf_set]
    return sum(high_scores) / len(high_scores) - sum(low_scores) / len(low_scores)


def compute_tsld_for_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    token_map: dict[int, list[int]],
    high_conf_set: list[int] = HIGH_CONF_SET,
    low_conf_set: list[int] = LOW_CONF_SET,
    method: str = "first_token",
    device: str = "cuda",
) -> float:
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer([prompt_str], return_tensors="pt").input_ids.to(device)
    scores = score_candidates(model, input_ids, token_map, method=method)
    return compute_tsld(scores, high_conf_set, low_conf_set)


def compute_delta_tsld(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    clean_messages: list[dict],
    corrupt_messages: list[dict],
    token_map: dict[int, list[int]],
    high_conf_set: list[int] = HIGH_CONF_SET,
    low_conf_set: list[int] = LOW_CONF_SET,
    method: str = "first_token",
    device: str = "cuda",
) -> tuple[float, float, float]:
    kwargs = dict(
        token_map=token_map,
        high_conf_set=high_conf_set,
        low_conf_set=low_conf_set,
        method=method,
        device=device,
    )
    clean_tsld = compute_tsld_for_prompt(model, tokenizer, clean_messages, **kwargs)
    corrupt_tsld = compute_tsld_for_prompt(model, tokenizer, corrupt_messages, **kwargs)
    return clean_tsld, corrupt_tsld, corrupt_tsld - clean_tsld
