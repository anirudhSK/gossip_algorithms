import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import networkx as nx


def compute_convergence_metrics(history: np.ndarray, true_average: float) -> Dict[str, Any]:
    n_steps, n_nodes = history.shape

    # Compute L2 error over time as in Definition 1
    errors = []
    for t in range(n_steps):
        error = np.linalg.norm(history[t] - true_average)
        errors.append(error)

    # Find ε-convergence times for different ε values (Definition 1)
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    convergence_times = {}

    for eps in epsilons:
        conv_time = None
        for t, error in enumerate(errors):
            if error < eps:
                conv_time = t
                break
        convergence_times[f'eps_{eps}'] = conv_time

    return {
        'errors': np.array(errors),
        'convergence_times': convergence_times,
        'final_error': errors[-1] if errors else float('inf'),
        'min_error': min(errors) if errors else float('inf')
    }


def theoretical_convergence_bounds(W_eigenvalue: float, epsilon: float) -> tuple:
    # Bounds from Theorem 3 equations (5) and (6)
    if W_eigenvalue >= 1.0:
        return float('inf'), float('inf')

    # Upper bound (equation 5): T_ave(ε,P) ≤ 3 log ε^(-1) / log λ₂(W)^(-1)
    upper_bound = 3 * np.log(1/epsilon) / np.log(1/W_eigenvalue)

    # Lower bound (equation 6): T_ave(ε,P) ≥ 0.5 log ε^(-1) / log λ₂(W)^(-1)
    lower_bound = 0.5 * np.log(1/epsilon) / np.log(1/W_eigenvalue)

    return lower_bound, upper_bound

def theoretical_convergence_bound(W_eigenvalue: float, epsilon: float) -> float:
    # Keep old function for compatibility - returns upper bound
    _, upper = theoretical_convergence_bounds(W_eigenvalue, epsilon)
    return upper


def analyze_graph_properties(graph: nx.Graph) -> Dict[str, float]:
    if not nx.is_connected(graph):
        raise ValueError("Graph must be connected")

    # Compute adjacency matrix and transition matrix for natural random walk
    A = nx.adjacency_matrix(graph).toarray()
    n = len(graph.nodes())

    # Create degree matrix
    degrees = [graph.degree(node) for node in graph.nodes()]
    D = np.diag(degrees)

    # Natural random walk transition matrix
    D_inv = np.linalg.inv(D)
    P_natural = D_inv @ A

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(P_natural)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]

    return {
        'n_nodes': n,
        'n_edges': graph.number_of_edges(),
        'diameter': nx.diameter(graph),
        'average_clustering': nx.average_clustering(graph),
        'natural_walk_lambda2': eigenvalues[1] if len(eigenvalues) > 1 else 0.0,
        'spectral_gap': 1 - eigenvalues[1] if len(eigenvalues) > 1 else 1.0
    }


def plot_node_convergence(history: np.ndarray, true_average: float,
                         node_indices: Optional[List[int]] = None,
                         title: str = "Node Convergence to Average") -> plt.Figure:
    # Visualize convergence to x_ave as described in the paper
    n_steps, n_nodes = history.shape

    if node_indices is None:
        # Plot first few nodes by default
        node_indices = list(range(min(5, n_nodes)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Node values over time
    for node_idx in node_indices:
        ax1.plot(history[:, node_idx], label=f'Node {node_idx}', alpha=0.7)

    ax1.axhline(y=true_average, color='black', linestyle='--', linewidth=2,
                label=f'True Average ({true_average:.3f})')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Node Values')
    ax1.set_title(f'{title} - Node Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: L2 error over time (as in Definition 1)
    errors = []
    for t in range(n_steps):
        error = np.linalg.norm(history[t] - true_average)
        errors.append(error)

    ax2.semilogy(errors, 'b-', linewidth=2, label='L2 Error')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('L2 Error (log scale)')
    ax2.set_title(f'{title} - Convergence Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compare_algorithms(graph: nx.Graph, initial_values: np.ndarray,
                      algorithms: List[str] = ['natural'],
                      max_steps: int = 5000) -> Dict[str, Any]:
    results = {}

    for alg_name in algorithms:
        if alg_name == 'natural':
            from gossip_algorithms import NaturalRandomWalkGossip
            sim = NaturalRandomWalkGossip(graph, initial_values)
        elif alg_name == 'optimal':
            from gossip_algorithms import OptimalGossipAlgorithm
            sim = OptimalGossipAlgorithm(graph, initial_values)
        else:
            raise ValueError(f"Unknown algorithm: {alg_name}")

        # Run simulation
        result = sim.simulate(max_steps=max_steps)
        metrics = compute_convergence_metrics(result['history'], sim.true_average)

        results[alg_name] = {
            'simulator': sim,
            'simulation_result': result,
            'metrics': metrics,
            'lambda2': sim.get_second_largest_eigenvalue()
        }

    return results


def plot_algorithm_comparison(results: Dict[str, Any]) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot convergence errors - shows relationship between λ₂(W) and convergence rate
    for alg_name, data in results.items():
        errors = data['metrics']['errors']
        ax1.semilogy(errors, label=f'{alg_name} (λ₂={data["lambda2"]:.3f})', linewidth=2)

    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('L2 Error (log scale)')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot ε-averaging times for different ε values
    algorithms = list(results.keys())
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4]

    x = np.arange(len(epsilons))
    width = 0.35

    for i, alg_name in enumerate(algorithms):
        conv_times = []
        for eps in epsilons:
            conv_time = results[alg_name]['metrics']['convergence_times'][f'eps_{eps}']
            conv_times.append(conv_time if conv_time is not None else 0)

        ax2.bar(x + i*width, conv_times, width, label=alg_name)

    ax2.set_xlabel('Epsilon Values')
    ax2.set_ylabel('Convergence Time')
    ax2.set_title('Convergence Times for Different Epsilon')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels([f'{eps}' for eps in epsilons])
    ax2.legend()

    plt.tight_layout()
    return fig