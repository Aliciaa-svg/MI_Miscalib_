#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Circuit validation (faithfulness / edge minimality)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--tsld_jsonl", required=True)
    parser.add_argument("--base_json", required=True)
    parser.add_argument("--discovery_dir", required=True,
                        help="Output dir of discover_confidence_circuit.py (graph.pt, discovery_result.json)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Edges in the circuit (default: top_k of the discovery run).")
    parser.add_argument("--run_edge_minimality", action="store_true")
    parser.add_argument("--edge_minimality_top_n", type=int, default=100)
    parser.add_argument("--edge_contribution_threshold", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_records", type=int, default=None)
    args = parser.parse_args()

    from mi_miscalib.circuit.validation import run_validation
    from mi_miscalib.data.bucket_stratification import load_bucket1
    from mi_miscalib.models.adapter import load_hf_tokenizer, load_hooked_transformer
    from mi_miscalib.patching.dataloader import make_eap_dataloader
    from mi_miscalib.patching.eap_wrapper import Graph
    from mi_miscalib.patching.metric_adapter import make_tsld_metric_for_eap

    discovery_dir = Path(args.discovery_dir)
    with open(discovery_dir / "discovery_result.json") as f:
        top_k = args.top_k or json.load(f)["top_k"]
    graph = Graph.from_pt(str(discovery_dir / "graph.pt"))

    bucket1, _ = load_bucket1(args.tsld_jsonl, args.base_json, args.n_records)
    tokenizer = load_hf_tokenizer(args.model)
    model = load_hooked_transformer(args.model, device=args.device)
    if (graph.cfg["n_layers"], graph.cfg["n_heads"]) != (model.cfg.n_layers, model.cfg.n_heads):
        raise ValueError(f"graph.pt does not match {args.model}: {graph.cfg}")

    dataloader = make_eap_dataloader(bucket1, tokenizer, batch_size=args.batch_size)
    result = run_validation(
        model=model,
        graph_scored=graph,
        dataloader=dataloader,
        metric_fn=make_tsld_metric_for_eap(tokenizer),
        top_k=top_k,
        model_name=args.model,
        run_edge_minimality=args.run_edge_minimality,
        edge_minimality_top_n=args.edge_minimality_top_n,
        edge_contribution_threshold=args.edge_contribution_threshold,
    )

    summary = dataclasses.asdict(result)
    edge_minimality = summary.pop("edge_minimality_scores")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "validation_result.json", "w") as f:
        json.dump(summary, f, indent=2)
    if args.run_edge_minimality:
        with open(output_dir / "minimality_per_edge.json", "w") as f:
            json.dump(edge_minimality, f, indent=2)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump({**vars(args), "top_k": top_k}, f, indent=2)

    print(f"\nRecords:                   {result.n_records}")
    print(f"Full-model TSLD:           {result.full_model_metric:.3f}")
    print(f"Faithfulness (top-{top_k}): {result.faithfulness:.3f}")
    if args.run_edge_minimality:
        print(f"Minimal edges:             {len(result.minimal_edges)} / {len(edge_minimality)}")
    print(f"[validation] Saved {output_dir / 'validation_result.json'}")


if __name__ == "__main__":
    main()
