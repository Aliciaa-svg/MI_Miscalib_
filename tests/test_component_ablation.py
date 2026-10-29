from __future__ import annotations

import pytest
import torch

from mi_miscalib.circuit.component_ablation import (
    build_patch_hooks,
    component_hook_name,
    component_key,
    paired_resample_tsld,
)
from mi_miscalib.circuit.discovery import ComponentAttribution
from mi_miscalib.patching.metric_adapter import make_tsld_metric_for_eap

HEAD = ComponentAttribution("attn", 1, 2, "a1.h2", -1.0, "positive")
MLP = ComponentAttribution("mlp", 0, None, "m0", -0.5, "positive")


def test_component_names():
    assert component_key(HEAD) == "L1H2" and component_key(MLP) == "M0"
    assert component_hook_name(HEAD) == "blocks.1.attn.hook_result"
    assert component_hook_name(MLP) == "blocks.0.hook_mlp_out"


def test_patch_hooks_only_touch_last_position_and_target_head():
    donor = {
        "blocks.1.attn.hook_result": torch.ones(1, 5, 4, 8),
        "blocks.0.hook_mlp_out": torch.ones(1, 7, 8),
    }
    (_, attn_hook), (_, mlp_hook) = build_patch_hooks([HEAD, MLP], donor)

    attn = attn_hook(torch.zeros(1, 6, 4, 8), None)
    assert attn[0, -1, 2].eq(1).all() and attn[0, -1, [0, 1, 3]].eq(0).all()
    assert attn[0, :-1].eq(0).all()

    mlp = mlp_hook(torch.zeros(1, 6, 8), None)
    assert mlp[0, -1].eq(1).all() and mlp[0, :-1].eq(0).all()


def test_paired_resample_patches_pos_end(tiny_model, digit_tokenizer):
    metric = make_tsld_metric_for_eap(digit_tokenizer)
    every_component = [ComponentAttribution("mlp", layer, None, f"m{layer}", 0.0) for layer in range(2)]
    every_component += [
        ComponentAttribution("attn", layer, head, f"a{layer}.h{head}", 0.0)
        for layer in range(2) for head in range(4)
    ]
    sets = {"head": [HEAD], "all": every_component}

    # Identical donor: patching changes nothing.
    baseline, ablated = paired_resample_tsld(tiny_model, "clean prompt", "clean prompt", sets, metric)
    assert ablated["head"] == pytest.approx(baseline, abs=1e-5)
    assert ablated["all"] == pytest.approx(baseline, abs=1e-5)

    # Donor ending in the same token: patching every component output at POS_END rebuilds
    # the donor's final residual stream, hence the donor's TSLD.
    donor_tsld, _ = paired_resample_tsld(tiny_model, "other prompt", "other prompt", {}, metric)
    baseline, ablated = paired_resample_tsld(tiny_model, "clean prompt", "other prompt", sets, metric)
    assert donor_tsld != pytest.approx(baseline, abs=1e-4)
    assert ablated["all"] == pytest.approx(donor_tsld, abs=1e-4)
    assert ablated["head"] != pytest.approx(baseline, abs=1e-6)
