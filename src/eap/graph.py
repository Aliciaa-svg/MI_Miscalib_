from typing import List, Dict, Union, Tuple, Literal, Optional, Set
from collections import defaultdict
from pathlib import Path 
import json
import heapq
import itertools
import math

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import numpy as np

from .visualization import EDGE_TYPE_COLORS, generate_random_color

class Node:
    name: str
    layer: int
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set['Node']
    parent_edges: Set['Edge']
    children: Set['Node']
    child_edges: Set['Edge']
    in_graph: bool
    qkv_inputs: Optional[List[str]]

    def __init__(self, name: str, layer:int, in_hook: List[str], out_hook: str, index: Tuple, qkv_inputs: Optional[List[str]]=None):
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook 
        self.index = index
        self.in_graph = True
        self.parents = set()
        self.children = set()
        self.parent_edges = set()
        self.child_edges = set()
        self.qkv_inputs = qkv_inputs

    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return f'Node({self.name}, in_graph: {self.in_graph})'
    
    def __hash__(self):
        return hash(self.name)

class LogitNode(Node):
    def __init__(self, n_layers:int):
        name = 'logits' 
        index = slice(None) 
        super().__init__(name, n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", '', index)
        
class MLPNode(Node):
    def __init__(self, layer: int):
        name = f'm{layer}' 
        index = slice(None) 
        super().__init__(name, layer, f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", index)

class AttentionNode(Node):
    head: int
    def __init__(self, layer:int, head:int):
        name = f'a{layer}.h{head}' 
        self.head = head
        index = (slice(None), slice(None), head) 
        super().__init__(name, layer, f'blocks.{layer}.hook_attn_in', f"blocks.{layer}.attn.hook_result", index, [f'blocks.{layer}.hook_{letter}_input' for letter in 'qkv'])

class InputNode(Node):
    def __init__(self):
        name = 'input' 
        index = slice(None) 
        super().__init__(name, 0, '', "hook_embed", index)  #"blocks.0.hook_resid_pre", index)

class Edge:
    name: str
    parent: Node 
    child: Node 
    hook: str
    index: Tuple
    score : Optional[float]
    in_graph: bool
    def __init__(self, parent: Node, child: Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None):
        self.name = f'{parent.name}->{child.name}' if qkv is None else f'{parent.name}->{child.name}<{qkv}>'
        self.parent = parent 
        self.child = child
        self.qkv = qkv
        self.score = None
        self.in_graph = True
        if isinstance(child, AttentionNode):
            if qkv is None:
                raise ValueError(f'Edge({self.name}): Edges to attention heads must have a non-none value for qkv.')
            self.hook = f'blocks.{child.layer}.hook_{qkv}_input'
            self.index = (slice(None), slice(None), child.head)
        else:
            self.index = child.index
            self.hook = child.in_hook

    def get_color(self):
        if self.qkv is not None:
            return EDGE_TYPE_COLORS[self.qkv]
        elif self.score < 0:
            return "#FF0000"
        else:
            return "#000000"

    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return f'Edge({self.name}, score: {self.score}, in_graph: {self.in_graph})'
    
    def __hash__(self):
        return hash(self.name)

class Graph:
    nodes: Dict[str, Node]
    edges: Dict[str, Edge]
    n_forward: int 
    n_backward: int
    cfg: HookedTransformerConfig

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.n_forward = 0
        self.n_backward = 0

    def add_edge(self, parent:Node, child:Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None):
        edge = Edge(parent, child, qkv)
        self.edges[edge.name] = edge
        parent.children.add(child)
        parent.child_edges.add(edge)
        child.parents.add(parent)
        child.parent_edges.add(edge)

    def prev_index(self, node: Node) -> Union[int, slice]:
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
        elif isinstance(node, MLPNode):
            if self.cfg['parallel_attn_mlp']:
                return 1 + node.layer * (self.cfg['n_heads'] + 1)
            else:
                return 1 + node.layer * (self.cfg['n_heads'] + 1) + self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            i =  1 + node.layer * (self.cfg['n_heads'] + 1)
            return i
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def forward_index(self, node:Node, attn_slice=True):
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            raise ValueError("No forward index for logits node")
        elif isinstance(node, MLPNode):
            return 1 + node.layer * (self.cfg['n_heads'] + 1) + self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            i =  1 + node.layer * (self.cfg['n_heads'] + 1)
            return slice(i, i + self.cfg['n_heads']) if attn_slice else i + node.head
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")
        

    def backward_index(self, node:Node, qkv=None, attn_slice=True):
        if isinstance(node, InputNode):
            raise ValueError(f"No backward for input node")
        elif isinstance(node, LogitNode):
            return -1
        elif isinstance(node, MLPNode):
            return (node.layer) * (3 * self.cfg['n_heads'] + 1) + 3 * self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            assert qkv in 'qkv', f'Must give qkv for AttentionNode, but got {qkv}'
            i = node.layer * (3 * self.cfg['n_heads'] + 1) + ('qkv'.index(qkv) * self.cfg['n_heads'])
            return slice(i, i + self.cfg['n_heads']) if attn_slice else i + node.head
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def get_scores(self, nonzero=False, in_graph=False, sort=True):
        s = torch.tensor([edge.score for edge in self.edges.values() if edge.score != 0 and (edge.in_graph or not in_graph)]) if nonzero else torch.tensor([edge.score for edge in self.edges.values()])
        return torch.sort(s).values if sort else s

    def get_dst_nodes(self):
        heads = []
        for layer in range(self.cfg['n_layers']):
            for letter in 'qkv':
                for attention_head in range(self.cfg['n_heads']):
                    heads.append(f'a{layer}.h{attention_head}<{letter}>')
            heads.append(f'm{layer}')
        heads.append('logits')
        return heads

    def count_included_edges(self):
        return sum(edge.in_graph for edge in self.edges.values())
    
    def count_included_nodes(self):
        return sum(node.in_graph for node in self.nodes.values())

    def apply_threshold(self, threshold: float, absolute: bool):
        threshold = float(threshold)
        for node in self.nodes.values():
            node.in_graph = True 
            
        for edge in self.edges.values():
            edge.in_graph = abs(edge.score) >= threshold if absolute else edge.score >= threshold
    
    def apply_topn(self, n:int, absolute: bool):
        a = abs if absolute else lambda x: x
        for node in self.nodes.values():
            node.in_graph = False

        sorted_edges = sorted(list(self.edges.values()), key = lambda edge: a(edge.score), reverse=True)
        for edge in sorted_edges[:n]:
            edge.in_graph = True 
            edge.parent.in_graph = True 
            edge.child.in_graph = True 

        for edge in sorted_edges[n:]:
            edge.in_graph = False

    def apply_greedy(self, n_edges, reset=True, absolute: bool = True):
        # if reset:
        #     for node in self.nodes.values():
        #         node.in_graph = False 
        #     for edge in self.edges.values():
        #         edge.in_graph = False
        #     self.nodes['logits'].in_graph = True

        # def abs_id(s: float):
        #     return abs(s) if absolute else s

        # candidate_edges = sorted([edge for edge in self.edges.values() if edge.child.in_graph], key = lambda edge: abs_id(edge.score), reverse=True)

        # edges = heapq.merge(candidate_edges, key = lambda edge: abs_id(edge.score), reverse=True)
        # while n_edges > 0:
        #     n_edges -= 1
        #     top_edge = next(edges)
        #     top_edge.in_graph = True
        #     parent = top_edge.parent
        #     if not parent.in_graph:
        #         parent.in_graph = True
        #         parent_parent_edges = sorted([parent_edge for parent_edge in parent.parent_edges], key = lambda edge: abs_id(edge.score), reverse=True)
        #         edges = heapq.merge(edges, parent_parent_edges, key = lambda edge: abs_id(edge.score), reverse=True)

        if reset:
            for node in self.nodes.values():
                node.in_graph = False 
            for edge in self.edges.values():
                edge.in_graph = False
            self.nodes['logits'].in_graph = True

        def abs_id(s: float):
            return abs(s) if absolute else s

        candidate_edges = [edge for edge in self.edges.values() if edge.child.in_graph]
        counter = itertools.count()
        heap = []
        for edge in candidate_edges:
            if edge.score is None:
                continue
            heapq.heappush(heap, (-abs_id(edge.score), next(counter), edge))

        while n_edges > 0 and heap:
            _, _, top_edge = heapq.heappop(heap)
            if top_edge.in_graph:
                continue
            top_edge.in_graph = True
            n_edges -= 1

            parent = top_edge.parent
            if not parent.in_graph:
                parent.in_graph = True
                for parent_edge in parent.parent_edges:
                    if parent_edge.score is None:
                        continue
                    heapq.heappush(heap, (-abs_id(parent_edge.score), next(counter), parent_edge))

    def prune_dead_nodes(self, prune_childless=True, prune_parentless=True):
        self.nodes['logits'].in_graph = any(parent_edge.in_graph for parent_edge in self.nodes['logits'].parent_edges)

        for node in reversed(self.nodes.values()):
            if isinstance(node, LogitNode):
                continue 
            
            if any(child_edge.in_graph for child_edge in node.child_edges) :
                node.in_graph = True
            else:
                if prune_childless:
                    node.in_graph = False
                    for parent_edge in node.parent_edges:
                        parent_edge.in_graph = False
                else: 
                    if any(child_edge.in_graph for child_edge in node.child_edges):
                        node.in_graph = True 
                    else:
                        node.in_graph = False

        if prune_parentless:
            for node in self.nodes.values():
                if not isinstance(node, InputNode) and node.in_graph and not any(parent_edge.in_graph for parent_edge in node.parent_edges):
                    node.in_graph = False 
                    for child_edge in node.child_edges:
                        child_edge.in_graph = False
    
    def reset_nodes_and_edges(self):
        for node in self.nodes.values():
            node.in_graph = True
            
        for edge in self.edges.values():
            edge.in_graph = True

    @classmethod
    def from_model(cls, model_or_config: Union[HookedTransformer,HookedTransformerConfig, Dict]):
        # print(cls)
        # print(model_or_config)
        graph = Graph()
        if isinstance(model_or_config, HookedTransformer):
            cfg = model_or_config.cfg
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp}
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp}
        else:
            graph.cfg = model_or_config
        # print(graph.cfg)
        
        input_node = InputNode()
        graph.nodes[input_node.name] = input_node
        residual_stream = [input_node]

        for layer in range(graph.cfg['n_layers']):
            attn_nodes = [AttentionNode(layer, head) for head in range(graph.cfg['n_heads'])]
            mlp_node = MLPNode(layer)
            
            for attn_node in attn_nodes: 
                graph.nodes[attn_node.name] = attn_node 
            graph.nodes[mlp_node.name] = mlp_node     
                                    
            if graph.cfg['parallel_attn_mlp']:
                for node in residual_stream:
                    for attn_node in attn_nodes:          
                        for letter in 'qkv':           
                            graph.add_edge(node, attn_node, qkv=letter)
                    graph.add_edge(node, mlp_node)
                
                residual_stream += attn_nodes
                residual_stream.append(mlp_node)

            else:
                for node in residual_stream:
                    for attn_node in attn_nodes:     
                        for letter in 'qkv':           
                            graph.add_edge(node, attn_node, qkv=letter)
                residual_stream += attn_nodes

                for node in residual_stream:
                    graph.add_edge(node, mlp_node)
                residual_stream.append(mlp_node)
                        
        logit_node = LogitNode(graph.cfg['n_layers'])
        for node in residual_stream:
            graph.add_edge(node, logit_node)
            
        graph.nodes[logit_node.name] = logit_node

        graph.n_forward = 1 + graph.cfg['n_layers'] * (graph.cfg['n_heads'] + 1)
        graph.n_backward = graph.cfg['n_layers'] * (3 * graph.cfg['n_heads'] + 1) + 1

        return graph


    def edge_matrices(self): 
        edge_scores = torch.zeros((self.n_forward, self.n_backward))
        edges_in_graph = torch.zeros((self.n_forward, self.n_backward)).bool()
        for edge in self.edges.values():
            edge_scores[self.forward_index(edge.parent, attn_slice=False), self.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)] = float(edge.score)
            edges_in_graph[self.forward_index(edge.parent, attn_slice=False), self.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)] = edge.in_graph
            
        return edge_scores, edges_in_graph


    def to_json(self, filename: str):
        # non serializable info
        d = {'cfg':self.cfg, 'nodes': {str(name): bool(node.in_graph) for name, node in self.nodes.items()}, 'edges':{str(name): {'score': None if edge.score is None else float(edge.score), 'in_graph': bool(edge.in_graph)} for name, edge in self.edges.items()}}
        with open(filename, 'w') as f:
            json.dump(d, f)

    def to_pt(self, filename: str):
        src_nodes = {node.name: node.in_graph for node in self.nodes.values() if not isinstance(node, LogitNode)}
        dst_nodes = self.get_dst_nodes()
        edge_scores, edges_in_graph = self.edge_matrices()
        d = {'cfg':self.cfg, 'src_nodes': src_nodes, 'dst_nodes': dst_nodes, 'edges': edge_scores, 'edges_in_graph': edges_in_graph}
        torch.save(d, filename)

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            d = json.load(f)
        g = Graph.from_model(d['cfg'])
        for name, in_graph in d['nodes'].items():
            g.nodes[name].in_graph = in_graph
        
        for name, info in d['edges'].items():
            g.edges[name].score = info['score']
            g.edges[name].in_graph = info['in_graph']

        return g
    
    @classmethod
    def from_pt(cls, filename):
        d = torch.load(filename)
        assert all([k in d.keys() for k in ['cfg', 'src_nodes', 'dst_nodes', 'edges', 'edges_in_graph']]), f"Bad torch circuit file format. Found keys - {d.keys()}, missing keys - {set(['cfg', 'src_nodes', 'dst_nodes', 'edges', 'edges_in_graph']) - set(d.keys())}"
        assert d['edges'].shape == d['edges_in_graph'].shape == (len(d['src_nodes']), len(d['dst_nodes'])), "Bad edges array shape"

        g = Graph.from_model(d['cfg'])

        for name, in_graph in d['src_nodes'].items():
            g.nodes[name].in_graph = in_graph

        # Enumerate over the tensor and fill the edge values in the graph
        for src_idx, src_name in enumerate(d['src_nodes']):
            for dst_idx, dst_name in enumerate(d['dst_nodes']):
                edge_name = f'{src_name}->{dst_name}'
                if edge_name in g.edges.keys():
                    g.edges[edge_name].score = d['edges'][src_idx, dst_idx]
                    g.edges[edge_name].in_graph = d['edges_in_graph'][src_idx, dst_idx]

        return g
    
    def __eq__(self, other):
        keys_equal = (set(self.nodes.keys()) == set(other.nodes.keys())) and (set(self.edges.keys()) == set(other.edges.keys()))
        if not keys_equal:
            return False
        
        for name, node in self.nodes.items():
            if node.in_graph != other.nodes[name].in_graph:
                return False 
            
        for name, edge in self.edges.items():
            if (edge.in_graph != other.edges[name].in_graph) or not np.allclose(edge.score, other.edges[name].score):
                return False
        return True
    
    def copy_from(self, other):
        for name, node in self.nodes.items():
            node.in_graph = other.nodes[name].in_graph
            
        for name, edge in self.edges.items():
            edge.in_graph = other.edges[name].in_graph
            edge.score = other.edges[name].score
        
        return self
        

    def to_graphviz(
        self,
        colorscheme: str = "Pastel2",
        minimum_penwidth: float = 0.6,
        maximum_penwidth: float = 5.0,
        layout: str="dot",
        seed: Optional[int] = None
    ) -> "pgv.AGraph":
        # Optional dependency: only needed for graph rendering.
        import pygraphviz as pgv

        g = pgv.AGraph(directed=True, bgcolor="white", overlap="false", splines="true", layout=layout)

        if seed is not None:
            np.random.seed(seed)

        colors = {node.name: generate_random_color(colorscheme) for node in self.nodes.values()}

        for node in self.nodes.values():
            if node.in_graph:
                g.add_node(node.name, 
                        fillcolor=colors[node.name], 
                        color="black", 
                        style="filled, rounded",
                        shape="box", 
                        fontname="Helvetica",
                        )

        scores = self.get_scores().abs()
        max_score = scores.max().item()
        min_score = scores.min().item()
        for edge in self.edges.values():
            if edge.in_graph:
                score = 0 if edge.score is None else edge.score
                normalized_score = (abs(score) - min_score) / (max_score - min_score) if max_score != min_score else abs(score)
                penwidth = max(minimum_penwidth, normalized_score * maximum_penwidth)
                g.add_edge(edge.parent.name,
                        edge.child.name,
                        penwidth=str(penwidth),
                        color=edge.get_color(),
                        )
        return g
    
    def calculate_entropy(self) -> Optional[float]:
        # Absolute scores of the edges in the graph
        scores = [abs(edge.score) for edge in self.edges.values() if edge.in_graph]
        
        # No valid scores: return None
        if not scores:
            return None
        
        # Normalize scores into a probability distribution
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores]

        # Shannon entropy
        edge_entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        
        return edge_entropy
    
    def calculate_weighted_entropy(self) -> Optional[float]:
        # Edge scores weighted by layer depth
        weighted_scores = []
        for edge in self.edges.values():
            if edge.score is not None:
                # Weight of the parent node's layer
                parent_weight = 1 / (edge.parent.layer + 1)
                child_weight = 1 / (edge.child.layer + 1)
                
                # Average of the parent and child weights
                edge_weight = (parent_weight + child_weight) / 2
                
                # Weighted score
                weighted_score = abs(edge.score) * edge_weight
                weighted_scores.append(weighted_score)

        # No valid weighted scores: return None
        if not weighted_scores:
            return None

        # Normalize weighted scores into a probability distribution
        total_weighted_score = sum(weighted_scores)
        probabilities = [score / total_weighted_score for score in weighted_scores]

        # Shannon entropy
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        return entropy

    def calculate_random_walk_entropy(self) -> float:
        total_entropy = 0.0

        # Compute entropy for each node
        for node_name, node in self.nodes.items():
            # Get outgoing edges
            outgoing_edges = [edge for edge in self.edges.values() if edge.parent == node]
            
            if not outgoing_edges:
                continue  # Skip nodes without outgoing edges
            
            # Normalize scores to probabilities
            scores = [edge.score for edge in outgoing_edges]
            max_score = max(scores)  # For numerical stability in softmax
            exp_scores = [math.exp(score - max_score) for score in scores]
            total_weight = sum(exp_scores)
            probabilities = [weight / total_weight for weight in exp_scores]

            # Compute entropy for this node
            entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
            total_entropy += entropy
        
        return total_entropy

