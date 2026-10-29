from __future__ import annotations

import pytest
import torch

from mi_miscalib.data.load_bucket import ConfidenceRecord

CLEAN_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Question: What is the capital of France?"},
    {"role": "assistant", "content": '{"answer": "Berlin"}'},
    {"role": "user", "content": "How confident are you?"},
]
CORRUPT_CONVERSATION = [
    *CLEAN_CONVERSATION[:2],
    {"role": "assistant", "content": '{"answer": "Paris"}'},
    CLEAN_CONVERSATION[3],
]


def make_record(
    id: int = 1,
    delta: float = -6.0,
    clean_confidence: int = 85,
    is_unknown: bool = False,
) -> ConfidenceRecord:
    return ConfidenceRecord(
        id=id,
        question="What is the capital of France?",
        predicted_answer="Berlin",
        replacement_answer="Paris",
        clean_conversation=CLEAN_CONVERSATION,
        corrupt_conversation=CORRUPT_CONVERSATION,
        clean_confidence=clean_confidence,
        corrupted_confidence=40,
        delta_confidence=40 - clean_confidence,
        clean_ld_high_minus_low=5.0,
        corrupted_ld_high_minus_low=5.0 + delta,
        delta_high_minus_low=delta,
        bucket=3,
        is_unknown=is_unknown,
    )


class DigitTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(ch) for ch in str(text).strip()]}


@pytest.fixture
def digit_tokenizer() -> DigitTokenizer:
    return DigitTokenizer()


@pytest.fixture
def tiny_model():
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=2, d_model=16, n_ctx=64, d_head=4, n_heads=4, d_mlp=32, d_vocab=40,
        act_fn="silu", normalization_type="RMS", positional_embedding_type="rotary",
        rotary_dim=4, seed=0,
    )
    model = HookedTransformer(cfg)
    model.cfg.use_attn_result = True
    model.eval()
    model.to_tokens = lambda text: torch.tensor([[ord(ch) % 40 for ch in text]])
    return model
