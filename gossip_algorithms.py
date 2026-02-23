import numpy as np
import networkx as nx
from typing import Tuple, Dict, Any
import random


class GossipSimulator:
    # Asynchronous gossip algorithm from Section I-A of Boyd et al. (2006)

    def __init__(self, graph: nx.Graph, initial_values: np.ndarray):
        self.graph = graph
        self.n = len(graph.nodes())
        self.initial_values = initial_values.copy()
        self.current_values = initial_values.copy()
        # x_ave from equation after (1)
        self.true_average = np.mean(initial_values)
        self.history = [initial_values.copy()]
        self.time_steps = 0

        self.P = self._create_probability_matrix()

    def _create_probability_matrix(self, algorithm_type='uniform') -> np.ndarray:
        # Matrix P from Section I-A where P[i,j] is probability that node i contacts node j
        P = np.zeros((self.n, self.n))

        if algorithm_type == 'uniform':
            # Each node contacts each neighbor with equal probability
            for i in range(self.n):
                neighbors = list(self.graph.neighbors(i))
                if neighbors:
                    prob = 1.0 / len(neighbors)
                    for j in neighbors:
                        P[i, j] = prob

        return P

    def step(self) -> Tuple[int, int]:
        # One step of asynchronous algorithm A(P) from Section I-A

        # Each node has rate 1 Poisson process - select which clock ticks
        active_node = random.randint(0, self.n - 1)

        # Find neighbors of the active node
        neighbors = list(self.graph.neighbors(active_node))

        if not neighbors:
            # Isolated node, no update
            return active_node, -1

        # Choose which neighbor to contact based on probability matrix P
        probs = self.P[active_node, neighbors]
        if np.sum(probs) == 0:
            return active_node, -1

        # Normalize probabilities (in case they don't sum to 1)
        probs = probs / np.sum(probs)
        contacted_node = np.random.choice(neighbors, p=probs)

        # Pairwise averaging: both nodes set their values to the average (equation after (1))
        old_active = self.current_values[active_node]
        old_contacted = self.current_values[contacted_node]
        new_value = (old_active + old_contacted) / 2.0

        self.current_values[active_node] = new_value
        self.current_values[contacted_node] = new_value

        # Record history
        self.history.append(self.current_values.copy())
        self.time_steps += 1

        return active_node, contacted_node

    def simulate(self, max_steps: int = 10000, epsilon: float = 1e-6) -> Dict[str, Any]:
        # Run simulation until ε-averaging time T_ave(ε,P) from Definition 1
        convergence_times = []

        for step in range(max_steps):
            active, contacted = self.step()

            # Check convergence using L2 norm as in Definition 1
            error = np.linalg.norm(self.current_values - self.true_average)
            if error < epsilon:
                convergence_times.append(step + 1)
                break

        return {
            'converged': len(convergence_times) > 0,
            'convergence_time': convergence_times[0] if convergence_times else None,
            'final_error': np.linalg.norm(self.current_values - self.true_average),
            'total_steps': self.time_steps,
            'history': np.array(self.history)
        }

    def get_second_largest_eigenvalue(self) -> float:
        # Compute λ₂(W) from equation (7) and (25) - determines convergence rate per Theorem 3
        # W = I - D/(2n) + (P + P^T)/(2n) where D is diagonal matrix

        # Compute diagonal matrix D where D[i,i] = sum of P[i,:] + P[:,i]
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            D[i, i] = sum(self.P[i, :] + self.P[:, i])

        # Correct W matrix from equation (25)
        W = np.eye(self.n) - D/(2*self.n) + (self.P + self.P.T)/(2*self.n)

        eigenvalues = np.linalg.eigvals(W)
        eigenvalues = np.real(eigenvalues)  # Take real part
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order

        return eigenvalues[1] if len(eigenvalues) > 1 else 0.0


class NaturalRandomWalkGossip(GossipSimulator):
    # Natural gossip algorithm mentioned in Section VI-A

    def _create_probability_matrix(self) -> np.ndarray:
        # Uniform contact probabilities for natural random walk
        return super()._create_probability_matrix('uniform')


class DeterministicGossipAlgorithm(GossipSimulator):
    # Deterministic gossip with 0-1 probability matrix

    def __init__(self, graph: nx.Graph, initial_values: np.ndarray, pattern_type: str = 'cyclic'):
        self.pattern_type = pattern_type
        super().__init__(graph, initial_values)
        self.P = self._create_deterministic_probability_matrix()

    def _create_deterministic_probability_matrix(self) -> np.ndarray:
        P = np.zeros((self.n, self.n))

        if self.pattern_type == 'cyclic':
            # Each node contacts next neighbor in cyclic order
            for i in range(self.n):
                neighbors = list(self.graph.neighbors(i))
                if neighbors:
                    # Choose the neighbor with smallest index > i, or smallest if none
                    next_neighbor = min([j for j in neighbors if j > i],
                                      default=min(neighbors))
                    P[i, next_neighbor] = 1.0

        elif self.pattern_type == 'forward':
            # Each node contacts neighbor with higher index (if exists)
            for i in range(self.n):
                neighbors = list(self.graph.neighbors(i))
                higher_neighbors = [j for j in neighbors if j > i]
                if higher_neighbors:
                    P[i, min(higher_neighbors)] = 1.0
                elif neighbors:
                    # If no higher neighbors, contact smallest neighbor
                    P[i, min(neighbors)] = 1.0

        elif self.pattern_type == 'alternating':
            # Balanced alternating pattern - ensure better connectivity
            for i in range(self.n):
                neighbors = list(self.graph.neighbors(i))
                if neighbors:
                    # Simple alternating: even nodes contact first neighbor, odd nodes contact second
                    if i % 2 == 0 and len(neighbors) > 0:
                        P[i, sorted(neighbors)[0]] = 1.0
                    elif i % 2 == 1 and len(neighbors) > 1:
                        P[i, sorted(neighbors)[1]] = 1.0
                    elif neighbors:  # Fallback
                        P[i, neighbors[0]] = 1.0

        return P