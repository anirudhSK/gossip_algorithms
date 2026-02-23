import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def theoretical_convergence_bounds(W_eigenvalue: float, epsilon: float) -> tuple:
    # Bounds from Theorem 3 equations (5) and (6)
    if W_eigenvalue >= 1.0:
        return float('inf'), float('inf')

    # Upper bound (equation 5): T_ave(ε,P) ≤ 3 log ε^(-1) / log λ₂(W)^(-1)
    upper_bound = 3 * np.log(1/epsilon) / np.log(1/W_eigenvalue)

    # Lower bound (equation 6): T_ave(ε,P) ≥ 0.5 log ε^(-1) / log λ₂(W)^(-1)
    lower_bound = 0.5 * np.log(1/epsilon) / np.log(1/W_eigenvalue)

    return lower_bound, upper_bound

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
