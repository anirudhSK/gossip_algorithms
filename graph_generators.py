import numpy as np
import networkx as nx
from typing import Dict, Any


def create_line_graph(n: int) -> nx.Graph:
    G = nx.path_graph(n)
    return G


def create_ring_graph(n: int) -> nx.Graph:
    G = nx.cycle_graph(n)
    return G


def create_clique_graph(n: int) -> nx.Graph:
    G = nx.complete_graph(n)
    return G


def create_random_graph(n: int, p: float) -> nx.Graph:
    # Erdős-Rényi random graph
    G = nx.erdos_renyi_graph(n, p)
    # Ensure connectivity
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p)
    return G


def create_grid_graph(rows: int, cols: int) -> nx.Graph:
    G = nx.grid_2d_graph(rows, cols)
    # Convert to simple graph with integer node labels
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G


def create_graph(graph_type: str, **kwargs) -> nx.Graph:
    if graph_type == 'line':
        return create_line_graph(kwargs['n'])
    elif graph_type == 'ring':
        return create_ring_graph(kwargs['n'])
    elif graph_type == 'clique':
        return create_clique_graph(kwargs['n'])
    elif graph_type == 'random':
        return create_random_graph(kwargs['n'], kwargs['p'])
    elif graph_type == 'grid':
        return create_grid_graph(kwargs['rows'], kwargs['cols'])
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")