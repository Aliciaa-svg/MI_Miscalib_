#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Factual-retrieval circuit + overlap analysis")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--pairs_csv", required=True, help="match_bucket1_pairs.py output")
    parser.add_argument("--confidence_dir", required=True,
                        help="Output dir of discover_confidence_circuit.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ig_steps", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=3000, help="Edges kept in the circuit.")
    parser.add_argument("--overlap_k", type=int, default=50, help="Top heads used for Jaccard.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_records", type=int, default=None)
    parser.add_argument("--save_top_edges", type=int, default=500)
    parser.add_argument("--save_top_heads", type=int, default=30)
    parser.add_argument("--save_top_components", type=int, default=50)
    args = parser.parse_args()

    from mi_miscalib.circuit.discovery import (
        HeadAttribution,
        compute_circuit_overlap,
        compute_edge_overlap,
        run_retrieval_circuit_discovery,
        save_discovery_outputs,
    )
    from mi_miscalib.data.load_bucket import load_records_from_csv
    from mi_miscalib.models.adapter import load_hf_tokenizer, load_hooked_transformer

    records = load_records_from_csv(args.pairs_csv)[:args.n_records]
    print(f"[retrieval] {len(records)} counterfactual question pairs")

    confidence_dir = Path(args.confidence_dir)
    with open(confidence_dir / "discovery_result.json") as f:
        confidence_heads = [
            HeadAttribution(layer=h["layer"], head=h["head"], score=h["score"], role=h["role"])
            for h in json.load(f)["head_attributions"]
        ]

    tokenizer = load_hf_tokenizer(args.model)
    model = load_hooked_transformer(args.model, device=args.device)

    output_dir = Path(args.output_dir)
    graph_path = output_dir / "graph_retrieval.pt"
    t0 = time.time()
    result = run_retrieval_circuit_discovery(
        model=model,
        tokenizer=tokenizer,
        records=records,
        model_name=args.model,
        ig_steps=args.ig_steps,
        top_k=args.top_k,
        batch_size=args.batch_size,
        graph_output_path=graph_path,
    )
    print(f"[retrieval] Done in {time.time() - t0:.1f}s")
    save_discovery_outputs(
        result,
        output_dir,
        save_top_edges=args.save_top_edges,
        save_top_heads=args.save_top_heads,
        save_top_components=args.save_top_components,
    )

    overlap = compute_circuit_overlap(confidence_heads, result.head_attributions, top_k=args.overlap_k)
    overlap["edge_overlap"] = compute_edge_overlap(
        confidence_dir / "graph.pt", graph_path, top_k=args.top_k
    )
    with open(output_dir / "overlap_results.json", "w") as f:
        json.dump(overlap, f, indent=2)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[retrieval] Top-{args.overlap_k} head Jaccard: {overlap['jaccard_top_k']:.3f}; "
          f"head-score cosine: {overlap['cosine_similarity']:.3f}; "
          f"top-{args.top_k} edge Jaccard: {overlap['edge_overlap']['jaccard_top_k_edges']:.3f}")


if __name__ == "__main__":
    main()
