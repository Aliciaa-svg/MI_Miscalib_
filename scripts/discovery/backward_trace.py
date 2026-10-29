#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def trace_components(
    all_edges: list[dict],
    components: list[dict],
    top_k_per_component: int,
) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    for comp in components:
        layer = int(comp["layer"])
        if comp["component_type"] == "attn":
            key = f"L{layer}H{int(comp['head'])}"
            incoming = [
                e for e in all_edges
                if e.get("dst_layer") == layer and e.get("dst_head") == int(comp["head"])
            ]
        else:
            key = f"M{layer}"
            incoming = [e for e in all_edges if e.get("dst") == f"m{layer}"]
        incoming.sort(key=lambda e: abs(e["score"]), reverse=True)
        results[key] = incoming[:top_k_per_component]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backward trace of top circuit components")
    parser.add_argument("--discovery_dir", required=True,
                        help="Output dir of discover_confidence_circuit.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k_components", type=int, default=20)
    parser.add_argument("--top_k_per_component", type=int, default=20)
    args = parser.parse_args()

    from mi_miscalib.patching.eap_wrapper import Graph, get_edge_scores

    discovery_dir = Path(args.discovery_dir)
    with open(discovery_dir / "top_components.json") as f:
        components = json.load(f)[:args.top_k_components]

    all_edges = get_edge_scores(Graph.from_pt(str(discovery_dir / "graph.pt")))
    trace = trace_components(all_edges, components, args.top_k_per_component)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "backward_trace.json", "w") as f:
        json.dump(trace, f, indent=2)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    for key, edges in trace.items():
        print(f"  {key}: top upstream = {[e['src'] for e in edges[:3]]}")
    print(f"[backward_trace] Saved {output_dir / 'backward_trace.json'}")


if __name__ == "__main__":
    main()
