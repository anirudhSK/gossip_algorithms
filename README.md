# Gossip Algorithm Simulator

Implementation of asynchronous gossip algorithms based on Boyd et al. (2006) "Randomized Gossip Algorithms".

## Quick Start
```bash
python3 main.py
```

## Key Features
- **Natural Random Walk Gossip**: Standard randomized approach from the paper
- **Deterministic Gossip**: 0-1 probability matrices with cyclic, forward, and alternating patterns
- **Multiple Graph Types**: Line, ring, clique, random, grid topologies
- **Convergence Analysis**: Theoretical bounds vs simulation results using λ₂(W) eigenvalue analysis
- **Visualizations**: Separate network topology and communication pattern diagrams

## Files
- `main.py` - Main simulation runner, creates experiment directories with results
- `gossip_algorithms.py` - Core algorithm implementations (NaturalRandomWalkGossip, DeterministicGossipAlgorithm)
- `graph_generators.py` - Graph creation functions for different topologies
- `analysis.py` - Convergence analysis, theoretical bounds, and visualization tools

## Key Implementation Details
- **Sum Preservation**: Mathematical property that average is preserved at every gossip step
- **Eigenvalue Analysis**: λ₂(W) < 1 required for convergence (Theorem 3 from paper)
- **Communication vs Network Topology**: Deterministic patterns can fragment communication graph even on connected networks
- **Alternating Pattern Issue**: Creates isolated pairs, preventing convergence (λ₂(W) = 1)

## Output
Each run creates directories like `line_n20/`, `ring_n20_deterministic_cyclic/` containing:
- `network_topology.png` - Original graph structure
- `communication_pattern.png` - Who talks to whom (P[i,j] > 0 = red arrow)
- `convergence.png` - Node values and L2 error over time
- `probability_matrix.txt` - ASCII dump of P matrix

## Theoretical Background
Implements equations from Boyd et al. (2006):
- Pairwise averaging: x_i ← (x_i + x_j)/2, x_j ← (x_i + x_j)/2
- Convergence bounds: 0.5 log(ε⁻¹)/log(λ₂(W)⁻¹) ≤ T_ave(ε,P) ≤ 3 log(ε⁻¹)/log(λ₂(W)⁻¹)
- Matrix W = I - D/(2n) + (P + P^T)/(2n) where D is degree matrix