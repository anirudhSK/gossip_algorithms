import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple


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


def create_geometric_random_graph(n: int, radius: float) -> nx.Graph:
    # Section VI-A: Geometric Random Graph G^d(n,r) for wireless networks
    G = nx.random_geometric_graph(n, radius)
    # Ensure connectivity
    while not nx.is_connected(G):
        G = nx.random_geometric_graph(n, radius)
    return G


def create_grid_graph(rows: int, cols: int) -> nx.Graph:
    G = nx.grid_2d_graph(rows, cols)
    # Convert to simple graph with integer node labels
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G


def get_graph_info(G: nx.Graph) -> Dict[str, Any]:
    n = len(G.nodes())
    m = len(G.edges())
    degrees = [G.degree(node) for node in G.nodes()]

    return {
        'num_nodes': n,
        'num_edges': m,
        'avg_degree': np.mean(degrees),
        'max_degree': max(degrees),
        'min_degree': min(degrees),
        'is_connected': nx.is_connected(G),
        'diameter': nx.diameter(G) if nx.is_connected(G) else float('inf')
    }


def create_graph(graph_type: str, **kwargs) -> nx.Graph:
    if graph_type == 'line':
        return create_line_graph(kwargs['n'])
    elif graph_type == 'ring':
        return create_ring_graph(kwargs['n'])
    elif graph_type == 'clique':
        return create_clique_graph(kwargs['n'])
    elif graph_type == 'random':
        return create_random_graph(kwargs['n'], kwargs['p'])
    elif graph_type == 'geometric':
        return create_geometric_random_graph(kwargs['n'], kwargs['radius'])
    elif graph_type == 'grid':
        return create_grid_graph(kwargs['rows'], kwargs['cols'])
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")