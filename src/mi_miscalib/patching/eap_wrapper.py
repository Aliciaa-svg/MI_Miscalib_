from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from eap.attribute import attribute
from eap.evaluate import evaluate_baseline, evaluate_graph
from eap.graph import Graph

__all__ = [
    "EvalResult",
    "Graph",
    "build_eap_graph",
    "count_active_edges",
    "evaluate_baseline",
    "evaluate_circuit",
    "get_edge_scores",
    "get_top_edges",
    "run_attribution",
    "to_scalar_metric",
]


@dataclass
class EvalResult:
    clean_metric: float
    circuit_metric: float
    faithfulness: float
    n_edges: int


def to_scalar_metric(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.float().mean().item())
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        vals = [to_scalar_metric(v) for v in value]
        return float(sum(vals) / len(vals))
    return float(value)


def build_eap_graph(model: Any) -> Graph:
    return Graph.from_model(model)


def run_attribution(
    model: Any,
    graph: Graph,
    dataloader: Any,
    metric_fn: Callable,
    ig_steps: int = 5,
    method: str = "EAP-IG",
) -> Graph:
    attribute(model, graph, dataloader, metric_fn, ig_steps=ig_steps, method=method)
    return graph


def get_top_edges(graph: Graph, n: int, absolute: bool = True) -> Graph:
    graph.apply_topn(n, absolute=absolute)
    return graph


def evaluate_circuit(
    model: Any,
    graph: Graph,
    dataloader: Any,
    metric_fn: Callable,
    clean_metric: Optional[float] = None,
) -> EvalResult:
    circuit_metric = to_scalar_metric(evaluate_graph(model, graph, dataloader, metric_fn))
    if clean_metric is None:
        clean_metric = to_scalar_metric(evaluate_baseline(model, dataloader, metric_fn))
    faithfulness = circuit_metric / clean_metric if abs(clean_metric) > 1e-8 else 0.0
    return EvalResult(
        clean_metric=float(clean_metric),
        circuit_metric=circuit_metric,
        faithfulness=float(faithfulness),
        n_edges=count_active_edges(graph),
    )


def get_edge_scores(graph: Graph) -> list[dict]:
    edges = []
    for edge_name, edge in graph.edges.items():
        edges.append({
            "name": edge_name,
            "src": edge.parent.name,
            "dst": edge.child.name,
            "score": float(edge.score) if edge.score is not None else 0.0,
            "in_graph": bool(edge.in_graph),
            "src_layer": getattr(edge.parent, "layer", None),
            "src_head": getattr(edge.parent, "head", None),
            "dst_layer": getattr(edge.child, "layer", None),
            "dst_head": getattr(edge.child, "head", None),
            "qkv": getattr(edge, "qkv", None),
        })
    edges.sort(key=lambda e: abs(e["score"]), reverse=True)
    return edges


def count_active_edges(graph: Graph) -> int:
    return sum(1 for e in graph.edges.values() if bool(e.in_graph))
