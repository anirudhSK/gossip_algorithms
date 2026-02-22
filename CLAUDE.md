# Claude Code Context

## Project Purpose
Gossip algorithm simulator implementing Boyd et al. (2006) paper. Focus on asynchronous model only.

## User Preferences & Requirements
- **No docstrings** - User explicitly requested removal of all docstrings
- **Concise output** - Clean terminal output with theoretical bounds
- **Separate visualizations** - Network topology and communication patterns in separate PNG files
- **Greek symbols** - Use ε instead of "ratio" in output
- **Mathematical accuracy** - User questioned 6-decimal precision, led to sum preservation discovery

## Key Implementation Decisions
- **Deterministic algorithms**: Added 3 patterns (cyclic, forward, alternating) with 0-1 probability matrices
- **Eigenvalue bug fix**: Had to correct W matrix calculation from equation (25) in paper
- **Visualization separation**: Split into network_topology.png and communication_pattern.png for clarity
- **Two-row layout**: Line graphs use vertical spacing to avoid arrow overlap in communication patterns

## Important Concepts Discovered
- **Sum preservation**: Average mathematically preserved at every step (not floating point error)
- **Communication vs topology**: Deterministic patterns can fragment communication graph
- **Alternating pattern failure**: Creates isolated pairs, λ₂(W) = 1, no convergence
- **Strong connectivity requirement**: Communication graph must be strongly connected for convergence

## Commands That Work
- `python3 main.py` - Run all experiments
- Results autogenerate in directories like `line_n20_deterministic_cyclic/`

## Issues to Avoid
- Don't assume floating point errors cause average drift - it's mathematically preserved
- Alternating pattern on rings doesn't converge - known limitation
- Line graphs need special visualization handling for clear arrow display