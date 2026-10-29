from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from eap.graph import Graph
from eap.metrics import logit_diff
from mi_miscalib.data.load_bucket import ConfidenceRecord
from mi_miscalib.patching.dataloader import make_eap_dataloader, make_retrieval_dataloader
from mi_miscalib.patching.eap_wrapper import (
    EvalResult,
    build_eap_graph,
    count_active_edges,
    evaluate_circuit,
    get_edge_scores,
    get_top_edges,
    run_attribution,
)
from mi_miscalib.patching.metric_adapter import make_tsld_metric_for_eap


@dataclass
class HeadAttribution:
    layer: int
    head: int
    score: float
    role: str = "neutral"
    edge_count: int = 0


@dataclass
class ComponentAttribution:
    component_type: str
    layer: int
    head: Optional[int]
    name: str
    score: float
    role: str = "neutral"
    edge_count: int = 0


@dataclass
class CircuitDiscoveryResult:
    model_name: str
    n_records: int
    n_edges_total: int
    n_edges_top: int
    head_attributions: list[HeadAttribution]
    edge_scores: list[dict]
    eval_result: Optional[EvalResult]
    ig_steps: int
    top_k: int
    component_attributions: list[ComponentAttribution] = field(default_factory=list)


def _head_role_from_score(score: float) -> str:
    return "CMH" if score < 0 else ("CAH" if score > 0 else "neutral")


def _component_role_from_score(score: float) -> str:
    return "positive" if score < 0 else ("negative" if score > 0 else "neutral")


def _aggregate_head_attributions(edge_scores: list[dict]) -> list[HeadAttribution]:
    scores: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int] = {}
    for edge in edge_scores:
        layer, head = edge.get("src_layer"), edge.get("src_head")
        if layer is None or head is None:
            continue
        key = (layer, head)
        scores[key] = scores.get(key, 0.0) + edge["score"]
        counts[key] = counts.get(key, 0) + 1

    heads = [
        HeadAttribution(
            layer=layer,
            head=head,
            score=score,
            role=_head_role_from_score(score),
            edge_count=counts[(layer, head)],
        )
        for (layer, head), score in scores.items()
    ]
    heads.sort(key=lambda h: abs(h.score), reverse=True)
    return heads


def _aggregate_component_attributions(edge_scores: list[dict]) -> list[ComponentAttribution]:
    scores: dict[tuple[str, int, Optional[int]], float] = {}
    counts: dict[tuple[str, int, Optional[int]], int] = {}
    for edge in edge_scores:
        layer, head = edge.get("src_layer"), edge.get("src_head")
        if layer is None:
            continue
        if head is not None:
            key = ("attn", int(layer), int(head))
        elif str(edge.get("src", "")).startswith("m"):
            key = ("mlp", int(layer), None)
        else:
            continue  # input node
        scores[key] = scores.get(key, 0.0) + edge["score"]
        counts[key] = counts.get(key, 0) + 1

    components = [
        ComponentAttribution(
            component_type=ctype,
            layer=layer,
            head=head,
            name=f"a{layer}.h{head}" if ctype == "attn" else f"m{layer}",
            score=score,
            role=_component_role_from_score(score),
            edge_count=counts[(ctype, layer, head)],
        )
        for (ctype, layer, head), score in scores.items()
    ]
    components.sort(key=lambda c: abs(c.score), reverse=True)
    return components


def _discover(
    model: Any,
    graph: Graph,
    dataloader: Any,
    metric_fn: Callable,
    model_name: str,
    ig_steps: int,
    top_k: int,
    run_eap: bool,
    evaluate: bool,
    graph_output_path: str | Path | None,
) -> CircuitDiscoveryResult:
    n_edges_total = len(graph.edges)
    n_usable = len(dataloader.dataset)
    print(f"[discovery] Graph: {len(graph.nodes)} nodes, {n_edges_total} edges; "
          f"{n_usable} usable pairs.")

    if run_eap:
        print(f"[discovery] Running EAP-IG (ig_steps={ig_steps})...")
        run_attribution(model, graph, dataloader, metric_fn, ig_steps=ig_steps)

    if graph_output_path is not None:
        save_path = Path(graph_output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        graph.to_pt(str(save_path))
        print(f"[discovery] Saved scored graph to {save_path}")

    edge_scores = get_edge_scores(graph)
    get_top_edges(graph, n=top_k, absolute=True)

    eval_result = None
    if evaluate:
        eval_result = evaluate_circuit(model, graph, dataloader, metric_fn)
        print(f"[discovery] Faithfulness of top-{top_k} circuit: "
              f"{eval_result.faithfulness:.3f} ({eval_result.n_edges} edges)")

    return CircuitDiscoveryResult(
        model_name=model_name,
        n_records=n_usable,
        n_edges_total=n_edges_total,
        n_edges_top=count_active_edges(graph),
        head_attributions=_aggregate_head_attributions(edge_scores),
        edge_scores=edge_scores,
        eval_result=eval_result,
        ig_steps=ig_steps,
        top_k=top_k,
        component_attributions=_aggregate_component_attributions(edge_scores),
    )


def run_confidence_circuit_discovery(
    model: Any,
    tokenizer: Any,
    records: list[ConfidenceRecord],
    model_name: str,
    ig_steps: int = 5,
    top_k: int = 3000,
    batch_size: int = 1,
    evaluate: bool = True,
    graph_output_path: str | Path | None = None,
    reuse_graph_path: str | Path | None = None,
) -> CircuitDiscoveryResult:
    if reuse_graph_path is not None:
        graph = Graph.from_pt(str(reuse_graph_path))
        print(f"[discovery] Reusing scored graph {reuse_graph_path}")
    else:
        graph = build_eap_graph(model)

    return _discover(
        model=model,
        graph=graph,
        dataloader=make_eap_dataloader(records, tokenizer, batch_size=batch_size),
        metric_fn=make_tsld_metric_for_eap(tokenizer),
        model_name=model_name,
        ig_steps=ig_steps,
        top_k=top_k,
        run_eap=reuse_graph_path is None,
        evaluate=evaluate,
        graph_output_path=graph_output_path,
    )


def run_retrieval_circuit_discovery(
    model: Any,
    tokenizer: Any,
    records: list[ConfidenceRecord],
    model_name: str,
    ig_steps: int = 5,
    top_k: int = 3000,
    batch_size: int = 1,
    graph_output_path: str | Path | None = None,
) -> CircuitDiscoveryResult:
    return _discover(
        model=model,
        graph=build_eap_graph(model),
        dataloader=make_retrieval_dataloader(records, tokenizer, batch_size=batch_size),
        metric_fn=logit_diff,
        model_name=model_name,
        ig_steps=ig_steps,
        top_k=top_k,
        run_eap=True,
        evaluate=False,
        graph_output_path=graph_output_path,
    )


def compute_circuit_overlap(
    confidence_heads: list[HeadAttribution],
    retrieval_heads: list[HeadAttribution],
    top_k: int = 50,
) -> dict:
    import numpy as np

    conf_set = {(h.layer, h.head) for h in confidence_heads[:top_k]}
    retr_set = {(h.layer, h.head) for h in retrieval_heads[:top_k]}
    union = conf_set | retr_set
    jaccard = len(conf_set & retr_set) / len(union) if union else 0.0

    conf_map = {(h.layer, h.head): h.score for h in confidence_heads}
    retr_map = {(h.layer, h.head): h.score for h in retrieval_heads}
    all_heads = sorted(set(conf_map) | set(retr_map))
    conf_vec = np.array([conf_map.get(h, 0.0) for h in all_heads])
    retr_vec = np.array([retr_map.get(h, 0.0) for h in all_heads])
    norm = np.linalg.norm(conf_vec) * np.linalg.norm(retr_vec)
    cosine = float(np.dot(conf_vec, retr_vec) / norm) if norm > 0 else 0.0

    return {
        "jaccard_top_k": jaccard,
        "cosine_similarity": cosine,
        "shared_heads": sorted(conf_set & retr_set),
        "confidence_only": sorted(conf_set - retr_set),
        "retrieval_only": sorted(retr_set - conf_set),
        "top_k_used": top_k,
    }


def compute_edge_overlap(
    confidence_graph_pt: str | Path,
    retrieval_graph_pt: str | Path,
    top_k: int = 3000,
) -> dict:
    conf_graph = Graph.from_pt(str(confidence_graph_pt))
    retr_graph = Graph.from_pt(str(retrieval_graph_pt))
    conf_graph.apply_topn(top_k, absolute=True)
    retr_graph.apply_topn(top_k, absolute=True)

    conf_edges = {e.name for e in conf_graph.edges.values() if e.in_graph}
    retr_edges = {e.name for e in retr_graph.edges.values() if e.in_graph}
    union = conf_edges | retr_edges
    shared = conf_edges & retr_edges
    return {
        "jaccard_top_k_edges": len(shared) / len(union) if union else 0.0,
        "n_shared": len(shared),
        "n_confidence_only": len(conf_edges - retr_edges),
        "n_retrieval_only": len(retr_edges - conf_edges),
        "shared_edges": sorted(shared),
        "top_k_used": top_k,
    }


def save_discovery_outputs(
    result: CircuitDiscoveryResult,
    output_dir: str | Path,
    save_top_edges: int = 500,
    save_top_heads: int = 30,
    save_top_components: int = 50,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heads = [
        {"layer": h.layer, "head": h.head, "score": h.score, "role": h.role,
         "edge_count": h.edge_count}
        for h in result.head_attributions
    ]
    components = [
        {"component_type": c.component_type, "layer": c.layer, "head": c.head, "name": c.name,
         "score": c.score, "role": c.role, "edge_count": c.edge_count}
        for c in result.component_attributions
    ]
    eval_result = result.eval_result
    discovery = {
        "model_name": result.model_name,
        "n_records": result.n_records,
        "n_edges_total": result.n_edges_total,
        "n_edges_top": result.n_edges_top,
        "ig_steps": result.ig_steps,
        "top_k": result.top_k,
        "head_attributions": heads,
        "component_attributions": components[:save_top_components],
        "eval_result": None if eval_result is None else {
            "clean_metric": eval_result.clean_metric,
            "circuit_metric": eval_result.circuit_metric,
            "faithfulness": eval_result.faithfulness,
            "n_edges": eval_result.n_edges,
        },
        "top_edges": result.edge_scores[:save_top_edges],
    }

    for name, payload in (
        ("discovery_result.json", discovery),
        ("top_heads.json", heads[:save_top_heads]),
        ("top_components.json", components[:save_top_components]),
    ):
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[discovery] Saved results to {output_dir}")
