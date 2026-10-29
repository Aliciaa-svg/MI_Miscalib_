from __future__ import annotations

import pytest

from mi_miscalib.circuit.discovery import (
    HeadAttribution,
    _aggregate_component_attributions,
    _aggregate_head_attributions,
    compute_circuit_overlap,
)
from mi_miscalib.patching.eap_wrapper import Graph, count_active_edges, get_edge_scores


def _edge(src, score, src_layer=None, src_head=None):
    return {"src": src, "score": score, "src_layer": src_layer, "src_head": src_head}


EDGES = [
    _edge("a1.h0", -2.0, 1, 0),
    _edge("a1.h0", -1.0, 1, 0),
    _edge("m1", 0.5, 1),
    _edge("m1", 0.25, 1),
    _edge("input", 9.0, 0),
]


def test_head_aggregation_and_roles():
    heads = _aggregate_head_attributions(EDGES)
    assert len(heads) == 1
    assert (heads[0].layer, heads[0].head, heads[0].score) == (1, 0, -3.0)
    assert heads[0].role == "CMH" and heads[0].edge_count == 2


def test_component_aggregation_skips_input_and_sorts():
    components = _aggregate_component_attributions(EDGES)
    assert [c.name for c in components] == ["a1.h0", "m1"]
    assert components[0].role == "positive"  # negative score = confidence-driving
    assert components[1].role == "negative" and components[1].score == pytest.approx(0.75)


def test_circuit_overlap():
    conf = [HeadAttribution(0, h, float(-h)) for h in range(4)]
    retr = [HeadAttribution(0, h, float(h)) for h in range(2, 6)]
    overlap = compute_circuit_overlap(conf, retr, top_k=4)
    assert overlap["jaccard_top_k"] == pytest.approx(2 / 6)
    assert overlap["shared_heads"] == [(0, 2), (0, 3)]
    assert overlap["cosine_similarity"] < 0


def test_graph_edge_scores_and_topn():
    graph = Graph.from_model({"n_layers": 2, "n_heads": 2, "parallel_attn_mlp": False})
    for i, edge in enumerate(graph.edges.values()):
        edge.score = float(i)
    scores = get_edge_scores(graph)
    assert abs(scores[0]["score"]) >= abs(scores[-1]["score"])
    graph.apply_topn(5, absolute=True)
    assert count_active_edges(graph) == 5
