from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from tqdm import tqdm

from mi_miscalib.patching.eap_wrapper import (
    Graph,
    count_active_edges,
    evaluate_baseline,
    evaluate_circuit,
    get_top_edges,
    to_scalar_metric,
)


@dataclass
class ValidationResult:
    model_name: str
    top_k: int
    n_edges: int
    n_records: int
    full_model_metric: float
    circuit_metric: float
    faithfulness: float
    edge_minimality_scores: dict[str, float] = field(default_factory=dict)
    minimal_edges: list[str] = field(default_factory=list)


def clone_scored_graph(graph: Graph) -> Graph:
    return Graph.from_model(graph.cfg).copy_from(graph)


def compute_faithfulness_curve(
    model: Any,
    dataloader: Any,
    metric_fn: Callable,
    graph_scored: Graph,
    top_k_values: list[int],
) -> tuple[list[dict], float, float]:
    clean_metric = to_scalar_metric(evaluate_baseline(model, dataloader, metric_fn, quiet=True))
    corrupt_metric = to_scalar_metric(
        evaluate_baseline(model, dataloader, metric_fn, run_corrupted=True, quiet=True)
    )

    curve = []
    for k in tqdm(top_k_values, desc="[validation] faithfulness curve"):
        graph = get_top_edges(clone_scored_graph(graph_scored), n=k, absolute=True)
        result = evaluate_circuit(model, graph, dataloader, metric_fn, clean_metric=clean_metric)
        denom = clean_metric - corrupt_metric
        curve.append({
            "top_k": k,
            "n_edges": count_active_edges(graph),
            "circuit_metric": result.circuit_metric,
            "faithfulness": result.faithfulness,
            "faithfulness_normalized": (
                (result.circuit_metric - corrupt_metric) / denom if abs(denom) > 1e-8 else None
            ),
        })
    return curve, clean_metric, corrupt_metric


def compute_edge_minimality(
    model: Any,
    graph_topk: Graph,
    dataloader: Any,
    metric_fn: Callable,
    circuit_metric: float,
    full_model_metric: float,
    top_n_edges: int = 100,
    contribution_threshold: float = 0.05,
) -> tuple[dict[str, float], list[str]]:
    active = sorted(
        (edge for edge in graph_topk.edges.values() if edge.in_graph),
        key=lambda edge: abs(float(edge.score)),
        reverse=True,
    )
    scores: dict[str, float] = {}
    for edge in tqdm(active[:top_n_edges], desc="[validation] edge minimality"):
        graph = clone_scored_graph(graph_topk)
        graph.edges[edge.name].in_graph = False
        result = evaluate_circuit(model, graph, dataloader, metric_fn, clean_metric=full_model_metric)
        scores[edge.name] = circuit_metric - result.circuit_metric
    minimal = [name for name, drop in scores.items() if drop > contribution_threshold]
    return scores, minimal


def run_validation(
    model: Any,
    graph_scored: Graph,
    dataloader: Any,
    metric_fn: Callable,
    top_k: int,
    model_name: str,
    run_edge_minimality: bool = False,
    edge_minimality_top_n: int = 100,
    edge_contribution_threshold: float = 0.05,
) -> ValidationResult:
    graph_topk = get_top_edges(clone_scored_graph(graph_scored), n=top_k, absolute=True)
    circuit = evaluate_circuit(model, graph_topk, dataloader, metric_fn)
    print(f"[validation] top-{top_k}: faithfulness={circuit.faithfulness:.3f} "
          f"(circuit={circuit.circuit_metric:.3f}, full={circuit.clean_metric:.3f})")

    edge_scores: dict[str, float] = {}
    minimal_edges: list[str] = []
    if run_edge_minimality:
        edge_scores, minimal_edges = compute_edge_minimality(
            model, graph_topk, dataloader, metric_fn,
            circuit_metric=circuit.circuit_metric,
            full_model_metric=circuit.clean_metric,
            top_n_edges=edge_minimality_top_n,
            contribution_threshold=edge_contribution_threshold,
        )

    return ValidationResult(
        model_name=model_name,
        top_k=top_k,
        n_edges=circuit.n_edges,
        n_records=len(dataloader.dataset),
        full_model_metric=circuit.clean_metric,
        circuit_metric=circuit.circuit_metric,
        faithfulness=circuit.faithfulness,
        edge_minimality_scores=edge_scores,
        minimal_edges=minimal_edges,
    )
