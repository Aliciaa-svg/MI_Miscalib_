#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Single / joint / incremental component ablation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--tsld_jsonl", required=True)
    parser.add_argument("--base_json", required=True)
    parser.add_argument("--discovery_dir", required=True,
                        help="Output dir of discover_confidence_circuit.py (top_components.json)")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top_k_components", type=int, default=10)
    parser.add_argument("--n_eval", type=int, default=200, help="Number of Bucket-1 pairs.")
    parser.add_argument("--sort_by", choices=["dataset_order", "clean_confidence"],
                        default="dataset_order",
                        help="clean_confidence: take the most confident pairs first.")
    parser.add_argument("--alignment_mode", choices=["len_equal", "relaxed"], default="len_equal")
    parser.add_argument("--incremental", action="store_true",
                        help="Also ablate the top-1..top-K components cumulatively.")
    parser.add_argument("--no_single", action="store_true", help="Skip single-component ablations.")
    args = parser.parse_args()

    from mi_miscalib.circuit.component_ablation import (
        component_key,
        filter_aligned_records,
        run_component_ablations,
    )
    from mi_miscalib.circuit.discovery import ComponentAttribution
    from mi_miscalib.data.bucket_stratification import load_bucket1
    from mi_miscalib.models.adapter import load_hf_tokenizer, load_hooked_transformer
    from mi_miscalib.patching.metric_adapter import make_tsld_metric_for_eap

    with open(Path(args.discovery_dir) / "top_components.json") as f:
        components = [
            ComponentAttribution(**c) for c in json.load(f) if c["role"] == "positive"
        ][:args.top_k_components]
    keys = [component_key(c) for c in components]
    print(f"[ablation] Components: {keys}")

    bucket1, _ = load_bucket1(args.tsld_jsonl, args.base_json)
    tokenizer = load_hf_tokenizer(args.model)
    records = filter_aligned_records(bucket1, tokenizer, args.alignment_mode)
    print(f"[ablation] {len(records)}/{len(bucket1)} Bucket-1 pairs pass alignment ({args.alignment_mode})")
    if args.sort_by == "clean_confidence":
        records = sorted(records, key=lambda r: -r.clean_confidence)
    records = records[:args.n_eval]

    component_sets = {"joint": components}
    if not args.no_single:
        component_sets.update({f"single:{key}": [c] for key, c in zip(keys, components)})
    if args.incremental:
        component_sets.update({f"top{n}": components[:n] for n in range(1, len(components) + 1)})

    model = load_hooked_transformer(args.model, device=args.device)
    baseline, ablated = run_component_ablations(
        model, tokenizer, records, component_sets, make_tsld_metric_for_eap(tokenizer)
    )

    def entry(name: str, members: list[str]) -> dict:
        return {"components": members, "ablated_tsld": ablated[name],
                "delta_tsld": ablated[name] - baseline}

    result = {
        "model_name": args.model,
        "n_samples": len(records),
        "components": keys,
        "baseline_tsld": baseline,
        "joint": entry("joint", keys),
        "single": [] if args.no_single else [
            {"component": key, **entry(f"single:{key}", [key])} for key in keys
        ],
        "incremental": [
            {"top_n": n, **entry(f"top{n}", keys[:n])} for n in range(1, len(keys) + 1)
        ] if args.incremental else [],
        "run_config": vars(args),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBaseline TSLD: {baseline:.3f}  ({len(records)} pairs)")
    print(f"Joint top-{len(keys)}: delta={result['joint']['delta_tsld']:+.3f}")
    for row in result["single"]:
        print(f"  single {row['component']:>7}: delta={row['delta_tsld']:+.3f}")
    for row in result["incremental"]:
        print(f"  top-{row['top_n']:<2d}: ablated={row['ablated_tsld']:.3f}")
    print(f"[ablation] Saved {output_path}")


if __name__ == "__main__":
    main()
