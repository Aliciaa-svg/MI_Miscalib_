#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

DEFAULT_TOP_K = "500,1000,1500,2000,2500,3000,3500,4000,5000,6000"


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithfulness curve over top-k edges")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--tsld_jsonl", required=True)
    parser.add_argument("--base_json", required=True)
    parser.add_argument("--graph_pt", required=True, help="Scored graph from discovery")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top_k_list", default=DEFAULT_TOP_K, help="Comma-separated k values")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_records", type=int, default=None)
    args = parser.parse_args()

    from mi_miscalib.circuit.validation import compute_faithfulness_curve
    from mi_miscalib.data.bucket_stratification import load_bucket1
    from mi_miscalib.models.adapter import load_hf_tokenizer, load_hooked_transformer
    from mi_miscalib.patching.dataloader import make_eap_dataloader
    from mi_miscalib.patching.eap_wrapper import Graph
    from mi_miscalib.patching.metric_adapter import make_tsld_metric_for_eap

    top_k_list = sorted({int(k) for k in args.top_k_list.split(",") if k.strip()})
    graph = Graph.from_pt(args.graph_pt)

    bucket1, _ = load_bucket1(args.tsld_jsonl, args.base_json, args.n_records)
    tokenizer = load_hf_tokenizer(args.model)
    model = load_hooked_transformer(args.model, device=args.device)
    if (graph.cfg["n_layers"], graph.cfg["n_heads"]) != (model.cfg.n_layers, model.cfg.n_heads):
        raise ValueError(f"graph.pt does not match {args.model}: {graph.cfg}")

    dataloader = make_eap_dataloader(bucket1, tokenizer, batch_size=args.batch_size)
    curve, clean_metric, corrupt_metric = compute_faithfulness_curve(
        model, dataloader, make_tsld_metric_for_eap(tokenizer), graph, top_k_list
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model_name": args.model,
            "graph_pt": args.graph_pt,
            "n_records": len(dataloader.dataset),
            "top_k_list": top_k_list,
            "clean_metric": clean_metric,
            "corrupt_metric": corrupt_metric,
            "curve": curve,
            "run_config": vars(args),
        }, f, indent=2)

    for row in curve:
        normalized = row["faithfulness_normalized"]
        print(f"  k={row['top_k']:<5d} faithfulness={row['faithfulness']:.3f}  "
              f"normalized={'n/a' if normalized is None else f'{normalized:.3f}'}")
    print(f"[faithfulness] Saved {output_path}")


if __name__ == "__main__":
    main()
