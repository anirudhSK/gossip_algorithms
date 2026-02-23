import networkx as nx


def create_graph(graph_type: str, **kwargs) -> nx.Graph:
    if graph_type == 'line':
        return nx.path_graph(kwargs['n'])
    elif graph_type == 'ring':
        return nx.cycle_graph(kwargs['n'])
    elif graph_type == 'clique':
        return nx.complete_graph(kwargs['n'])
    elif graph_type == 'random':
        # Erdős-Rényi random graph, resampled until connected
        G = nx.erdos_renyi_graph(kwargs['n'], kwargs['p'])
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(kwargs['n'], kwargs['p'])
        return G
    elif graph_type == 'grid':
        G = nx.grid_2d_graph(kwargs['rows'], kwargs['cols'])
        # grid_2d_graph uses (row, col) tuples; relabel to plain integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        return nx.relabel_nodes(G, mapping)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")