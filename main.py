#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from graph_generators import create_graph
from gossip_algorithms import NaturalRandomWalkGossip, DeterministicGossipAlgorithm
from analysis import plot_node_convergence, theoretical_convergence_bounds


def run_single_experiment(graph_type: str, n_nodes: int, epsilon: float, algorithm: str = 'natural', **graph_kwargs):
    # Create graph
    if graph_type == 'grid':
        # For grid, we need rows and cols
        rows = int(np.sqrt(n_nodes))
        cols = n_nodes // rows
        graph = create_graph(graph_type, rows=rows, cols=cols)
    else:
        graph = create_graph(graph_type, n=n_nodes, **graph_kwargs)

    # Create random initial values x(0) from paper
    np.random.seed(42)  # For reproducibility
    initial_values = np.random.uniform(-10, 10, len(graph.nodes()))
    true_average = np.mean(initial_values)

    # Run asynchronous gossip algorithm A(P)
    if algorithm.startswith('deterministic'):
        pattern_type = algorithm.split('_')[1] if '_' in algorithm else 'cyclic'
        sim = DeterministicGossipAlgorithm(graph, initial_values, pattern_type=pattern_type)
    else:
        sim = NaturalRandomWalkGossip(graph, initial_values)

    # Compute λ₂(W) from equation (7) - determines convergence rate
    lambda2 = sim.get_second_largest_eigenvalue()

    # Bounds from Theorem 3 equations (5) and (6)
    theoretical_lower, theoretical_upper = theoretical_convergence_bounds(lambda2, epsilon)

    # Run simulation until ε-averaging time T_ave(ε,P)
    # Use longer simulation for slow graphs like line graphs
    max_steps = 50000 if graph_type == 'line' else 15000
    result = sim.simulate(max_steps=max_steps, epsilon=epsilon)

    # Create directory for this graph
    alg_suffix = f"_{algorithm}" if algorithm != 'natural' else ""
    dir_name = f"{graph_type}_n{len(graph.nodes())}{alg_suffix}"
    os.makedirs(dir_name, exist_ok=True)

    # Save original network topology
    plt.figure(figsize=(12, 6))
    pos = None
    if graph_type == 'ring':
        pos = {i: (np.cos(2*np.pi*i/len(graph.nodes())), np.sin(2*np.pi*i/len(graph.nodes())))
               for i in range(len(graph.nodes()))}
    elif graph_type == 'line':
        pos = {i: (i, 0) for i in range(len(graph.nodes()))}

    nx.draw(graph, pos=pos, with_labels=True, node_color='lightblue',
            node_size=600, font_size=12, font_weight='bold', width=2)
    plt.title(f"Original {graph_type.title()} Network Topology (n={len(graph.nodes())})", fontsize=14)
    plt.savefig(f"{dir_name}/network_topology.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save communication pattern separately with more space
    plt.figure(figsize=(16, 8))

    # Create communication graph based on P matrix
    comm_graph = nx.DiGraph()
    comm_graph.add_nodes_from(graph.nodes())

    # Add directed edges based on probability matrix
    for i in range(len(graph.nodes())):
        for j in range(len(graph.nodes())):
            if sim.P[i, j] > 0:
                comm_graph.add_edge(i, j)

    # Choose colors based on algorithm
    if algorithm.startswith('deterministic'):
        edge_color = 'red'
        node_color = 'lightcoral'
    else:
        edge_color = 'blue'
        node_color = 'lightgreen'

    # For line graphs, use vertical spacing to avoid overlap
    if graph_type == 'line' and len(comm_graph.edges()) > 0:
        # Create two-row layout for better arrow visibility
        pos_comm = {}
        for i in range(len(graph.nodes())):
            # Alternate rows to separate forward and backward communications
            row = 0 if i % 4 < 2 else 1
            pos_comm[i] = (i, row * 2)

        nx.draw_networkx_nodes(comm_graph, pos_comm, node_color=node_color,
                              node_size=800, alpha=0.8)
        nx.draw_networkx_labels(comm_graph, pos_comm, font_size=12,
                               font_weight='bold')

        # Draw edges with different curves based on direction
        for edge in comm_graph.edges():
            i, j = edge
            if i < j:  # Forward direction - curve up
                connectionstyle = "arc3,rad=0.3"
            else:  # Backward direction - curve down
                connectionstyle = "arc3,rad=-0.3"

            nx.draw_networkx_edges(comm_graph, pos_comm, [(i, j)],
                                 edge_color=edge_color, arrows=True,
                                 arrowsize=20, arrowstyle='->',
                                 connectionstyle=connectionstyle, width=3)
    else:
        # For non-line graphs or when no edges exist
        nx.draw(comm_graph, pos=pos, with_labels=True,
                node_color=node_color, node_size=600, font_size=12,
                font_weight='bold', edge_color=edge_color, arrows=True,
                arrowsize=20, arrowstyle='->', width=3)

    plt.title(f"Communication Pattern: {algorithm} (P[i,j] > 0 = red arrow i→j)", fontsize=14)
    plt.savefig(f"{dir_name}/communication_pattern.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save probability matrix as ASCII text
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    with open(f"{dir_name}/probability_matrix.txt", 'w') as f:
        f.write(f"Probability Matrix P for {graph_type} graph (n={len(graph.nodes())}) - {algorithm} algorithm\n")
        f.write(f"P[i,j] = probability that node i contacts node j\n\n")
        f.write(str(sim.P))
        f.write("\n")

    # Create convergence visualization
    fig = plot_node_convergence(
        result['history'], true_average,
        node_indices=[0, 1, 2, len(graph.nodes())//2, len(graph.nodes())-1],
        title=f"{graph_type.title()} Graph (n={len(graph.nodes())}) - {algorithm}"
    )

    # Save convergence plot
    fig.savefig(f"{dir_name}/convergence.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        'graph_type': graph_type,
        'algorithm': algorithm,
        'n_nodes': len(graph.nodes()),
        'true_average': true_average,
        'initial_values': initial_values,
        'converged': result['converged'],
        'convergence_time': result['convergence_time'],
        'theoretical_lower': theoretical_lower,
        'theoretical_upper': theoretical_upper,
        'final_values': sim.current_values,
    }


def main():
    print("Gossip Algorithm Simulator")
    print("Based on: 'Randomized Gossip Algorithms' by Boyd et al. (2006)")

    epsilon = 1e-3
    print(f"Error tolerance ε = {epsilon}")
    print()

    # Test different graph topologies mentioned in the paper (all normalized to n=20)
    experiments = [
        {'graph_type': 'line', 'n_nodes': 20},
        {'graph_type': 'ring', 'n_nodes': 20},
        {'graph_type': 'clique', 'n_nodes': 20},  # Complete graph from Section I-B
        {'graph_type': 'random', 'n_nodes': 20, 'p': 0.3},  # Random graph
        {'graph_type': 'grid', 'n_nodes': 20},  # 4x5 grid (closest to 20 nodes)
        # Deterministic gossip algorithms with 0-1 probability matrices
        {'graph_type': 'line', 'n_nodes': 20, 'algorithm': 'deterministic_cyclic'},
        {'graph_type': 'line', 'n_nodes': 20, 'algorithm': 'deterministic_forward'},
        {'graph_type': 'line', 'n_nodes': 20, 'algorithm': 'deterministic_alternating'},
    ]

    for exp in experiments:
        try:
            result = run_single_experiment(epsilon=epsilon, **exp)

            # Print results for this graph
            alg_name = f" ({result['algorithm']})" if result['algorithm'] != 'natural' else ""
            graph_name = f"{result['graph_type']}{alg_name} (n={result['n_nodes']})"
            true_avg = result['true_average']
            conv_time = result['convergence_time'] if result['converged'] else "N/A"
            theory_lower = int(result['theoretical_lower']) if result['theoretical_lower'] != float('inf') else "∞"
            theory_upper = int(result['theoretical_upper']) if result['theoretical_upper'] != float('inf') else "∞"

            # Calculate the ratio from equation (3): ||x(t) - x_ave*1|| / ||x(0)||
            x_ave_vector = np.full(result['n_nodes'], true_avg)
            initial_norm = np.linalg.norm(result['initial_values'])
            final_deviation = np.linalg.norm(result['final_values'] - x_ave_vector)
            convergence_ratio = final_deviation / initial_norm if initial_norm > 0 else 0.0

            print(f"{graph_name:<20} ε: {convergence_ratio:8.5f}   Steps: {conv_time!s:<8}   Theory: [{theory_lower}, {theory_upper}]")

        except Exception as e:
            print(f"Error in {exp['graph_type']} experiment: {e}")

if __name__ == "__main__":
    main()