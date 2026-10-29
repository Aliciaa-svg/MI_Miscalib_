#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence circuit discovery (EAP-IG + TSLD)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--tsld_jsonl", required=True)
    parser.add_argument("--base_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ig_steps", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=3000, help="Edges kept in the circuit.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_records", type=int, default=None, help="Use only the first N Bucket-1 records.")
    parser.add_argument("--reuse_graph_pt", default=None,
                        help="Load edge scores from an existing graph.pt instead of re-running EAP-IG.")
    parser.add_argument("--no_evaluate", action="store_true", help="Skip the top-k faithfulness evaluation.")
    parser.add_argument("--save_top_edges", type=int, default=500)
    parser.add_argument("--save_top_heads", type=int, default=30)
    parser.add_argument("--save_top_components", type=int, default=50)
    args = parser.parse_args()

    from mi_miscalib.circuit.discovery import (
        run_confidence_circuit_discovery,
        save_discovery_outputs,
    )
    from mi_miscalib.data.bucket_stratification import load_bucket1
    from mi_miscalib.models.adapter import load_hf_tokenizer, load_hooked_transformer

    bucket1, (thr_low, thr_high) = load_bucket1(args.tsld_jsonl, args.base_json, args.n_records)
    print(f"[discovery] Bucket 1: {len(bucket1)} records (thresholds [{thr_low:.2f}, {thr_high:.2f}])")

    tokenizer = load_hf_tokenizer(args.model)
    model = load_hooked_transformer(args.model, device=args.device)

    output_dir = Path(args.output_dir)
    t0 = time.time()
    result = run_confidence_circuit_discovery(
        model=model,
        tokenizer=tokenizer,
        records=bucket1,
        model_name=args.model,
        ig_steps=args.ig_steps,
        top_k=args.top_k,
        batch_size=args.batch_size,
        evaluate=not args.no_evaluate,
        graph_output_path=output_dir / "graph.pt",
        reuse_graph_path=args.reuse_graph_pt,
    )
    print(f"[discovery] Done in {time.time() - t0:.1f}s")

    print("\nTop-10 components by |score| (role=positive inflates confidence):")
    for c in result.component_attributions[:10]:
        print(f"  {c.name:>8}  score={c.score:+.4f}  role={c.role}")

    save_discovery_outputs(
        result,
        output_dir,
        save_top_edges=args.save_top_edges,
        save_top_heads=args.save_top_heads,
        save_top_components=args.save_top_components,
    )
    with open(output_dir / "run_config.json", "w") as f:
        json.dump({**vars(args), "bucket_thresholds": [thr_low, thr_high]}, f, indent=2)


if __name__ == "__main__":
    main()
