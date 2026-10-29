from __future__ import annotations

from typing import Any, Callable

import torch
from tqdm import tqdm

from mi_miscalib.circuit.discovery import ComponentAttribution
from mi_miscalib.data.load_bucket import ConfidenceRecord
from mi_miscalib.models.adapter import apply_chat_template

ALIGNMENT_MODES = ("relaxed", "len_equal")


def component_key(component: ComponentAttribution) -> str:
    if component.component_type == "attn":
        return f"L{component.layer}H{component.head}"
    return f"M{component.layer}"


def component_hook_name(component: ComponentAttribution) -> str:
    if component.component_type == "attn":
        return f"blocks.{component.layer}.attn.hook_result"
    return f"blocks.{component.layer}.hook_mlp_out"


def build_patch_hooks(
    components: list[ComponentAttribution],
    donor_cache: Any,
) -> list[tuple[str, Callable]]:
    hooks: list[tuple[str, Callable]] = []
    for component in components:
        hook_name = component_hook_name(component)
        donor = donor_cache[hook_name]

        if component.component_type == "attn":
            def attn_hook(value: torch.Tensor, hook: Any, donor=donor, head=component.head):
                # value: (batch, pos, n_heads, d_model)
                value[:, -1, head, :] = donor[:, -1, head, :].to(value.device, value.dtype)
                return value

            hooks.append((hook_name, attn_hook))
        else:
            def mlp_hook(value: torch.Tensor, hook: Any, donor=donor):
                # value: (batch, pos, d_model)
                value[:, -1, :] = donor[:, -1, :].to(value.device, value.dtype)
                return value

            hooks.append((hook_name, mlp_hook))
    return hooks


def _tsld_at_last_position(logits: torch.Tensor, metric_fn: Callable) -> float:
    input_length = torch.full(
        (logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device
    )
    labels = torch.zeros(logits.shape[0], 2, dtype=torch.long, device=logits.device)
    return float(metric_fn(logits, logits, input_length, labels, mean=True).item())


def paired_resample_tsld(
    model: Any,
    clean_str: str,
    corrupt_str: str,
    component_sets: dict[str, list[ComponentAttribution]],
    metric_fn: Callable,
) -> tuple[float, dict[str, float]]:
    hook_names = sorted({
        component_hook_name(c) for comps in component_sets.values() for c in comps
    })
    clean_tokens = model.to_tokens(clean_str)
    corrupt_tokens = model.to_tokens(corrupt_str)

    with torch.no_grad():
        _, donor_cache = model.run_with_cache(corrupt_tokens, names_filter=hook_names)
        baseline = _tsld_at_last_position(model(clean_tokens), metric_fn)
        ablated = {}
        for name, components in component_sets.items():
            logits = model.run_with_hooks(
                clean_tokens, fwd_hooks=build_patch_hooks(components, donor_cache)
            )
            ablated[name] = _tsld_at_last_position(logits, metric_fn)
    return baseline, ablated


def run_component_ablations(
    model: Any,
    tokenizer: Any,
    records: list[ConfidenceRecord],
    component_sets: dict[str, list[ComponentAttribution]],
    metric_fn: Callable,
) -> tuple[float, dict[str, float]]:
    baselines: list[float] = []
    ablated: dict[str, list[float]] = {name: [] for name in component_sets}
    for rec in tqdm(records, desc="[ablation] records", dynamic_ncols=True):
        baseline, rec_ablated = paired_resample_tsld(
            model,
            apply_chat_template(tokenizer, rec.clean_conversation),
            apply_chat_template(tokenizer, rec.corrupt_conversation),
            component_sets,
            metric_fn,
        )
        baselines.append(baseline)
        for name, value in rec_ablated.items():
            ablated[name].append(value)

    if not baselines:
        raise ValueError("No records to ablate.")
    mean_baseline = sum(baselines) / len(baselines)
    return mean_baseline, {name: sum(v) / len(v) for name, v in ablated.items()}


def filter_aligned_records(
    records: list[ConfidenceRecord],
    tokenizer: Any,
    alignment_mode: str = "len_equal",
) -> list[ConfidenceRecord]:
    if alignment_mode not in ALIGNMENT_MODES:
        raise ValueError(f"Unknown alignment_mode {alignment_mode!r}; use one of {ALIGNMENT_MODES}.")
    kept = []
    for rec in records:
        if not rec.clean_conversation or not rec.corrupt_conversation:
            continue
        if alignment_mode == "len_equal":
            clean_n = len(tokenizer.encode(apply_chat_template(tokenizer, rec.clean_conversation)))
            corrupt_n = len(tokenizer.encode(apply_chat_template(tokenizer, rec.corrupt_conversation)))
            if clean_n != corrupt_n:
                continue
        kept.append(rec)
    return kept
