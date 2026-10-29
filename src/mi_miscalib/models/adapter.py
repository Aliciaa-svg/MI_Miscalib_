from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer


def load_hooked_transformer(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> "HookedTransformer":
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        dtype=dtype,
        device=device,
    )
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    model.eval()
    return model


def load_hf_tokenizer(model_name: str) -> "AutoTokenizer":
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def apply_chat_template(
    tokenizer: "AutoTokenizer",
    messages: list[dict],
    add_generation_prompt: bool = True,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except AttributeError as e:
        if "pass_eval_context" in str(e):
            raise RuntimeError(
                "apply_chat_template failed: jinja2 version too old. "
                "Upgrade with: pip install -U jinja2"
            ) from e
        raise
