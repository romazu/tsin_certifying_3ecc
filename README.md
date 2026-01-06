# Tsin Certifying Algorithm for 3-Edge-Connectivity

[![Tests](https://img.shields.io/github/actions/workflow/status/romazu/tsin_certifying_3ecc/tests.yml?branch=main)](https://github.com/romazu/tsin_certifying_3ecc/actions?query=branch%3Amain)

Python implementation of Tsin's linear-time certifying algorithm for 3-edge-connectivity.

Given a connected graph, the algorithm:
- Decomposes the graph into 3-edge-connected components
- Generates a Mader construction sequence for each component (certificate of 3-edge-connectivity)
- Identifies bridges and cut-pairs
- Builds a cactus representation of cut-pairs

Based on: [A simple certifying algorithm for 3-edge-connectivity](https://doi.org/10.1016/j.tcs.2023.113760) ([arXiv](https://arxiv.org/abs/2002.04727)), Y. H. Tsin, 2023

## Usage

```python
from ecc_cert import Solver

#        1
#       /|\
#      / | \
#     /  |  \
#    2---3---4
#    |  /    |
#    | /     |
#    |/      |
#    5-------6

# adjacency
graph = {
    1: [2, 3, 4],
    2: [1, 3, 5],
    3: [1, 2, 4, 5],
    4: [1, 3, 6],
    5: [2, 3, 6],
    6: [4, 5]
}

solver = Solver(graph, num_vertices=6)
solver.certifying_3_edge_connectivity(start_vertex=1)

decomposition = solver.materialize_decomposition()
print(decomposition.components)  # [6, 1]
print(decomposition.sigma)       # {6: [6], 1: [1, 2, 3, 4, 5]}
print(decomposition.mader)       # {1: [[1, 4, 3], [1, 2, 3], [1, 3], [2, 5, 4], [3, 5]]}
print(decomposition.cycles)      # {1: [[6]]}
print(decomposition.bridges)     # []
```

## Tests

```bash
pytest -v
```

## graph.guide

Traces from this algorithm are used on [graph.guide](https://graph.guide/algorithm/tsin-certifying-3-edge-connectivity?graph=tsin1_mod) for graph algorithm visualization.

## Literature

- Tsin, Y.H. (2023). *A simple certifying algorithm for 3-edge-connectivity.* Theoretical Computer Science, Volume 951, 113760. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304397523000737) / [arXiv](https://arxiv.org/abs/2002.04727)

- Norouzi, N. & Tsin, Y.H. (2014). *A simple 3-edge connected component algorithm revisited.* Information Processing Letters, 114(1–2), 50–55. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0020019013002470)

- Tsin, Y.H. (2007). *A Simple 3-Edge-Connected Component Algorithm.* Theory of Computing Systems, 40, 125–142. [Springer](https://link.springer.com/article/10.1007/s00224-005-1269-4)

- Mehlhorn, K., Neumann, A. & Schmidt, J.M. (2017). *Certifying 3-edge connectivity.* Algorithmica, 77(2), 309–335. [Springer](https://link.springer.com/article/10.1007/s00453-015-0075-x)

- Mader, W. (1978). *A reduction for edge-connectivity in graphs.* In: Bollobás, B. (Ed.), Advances in Graph Theory. Annals of Discrete Mathematics, vol. 3, pp. 145–164. [ScienceDirect](https://www.sciencedirect.com/science/chapter/bookseries/abs/pii/S0167506008705041)
