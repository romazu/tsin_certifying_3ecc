from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Decomposition:
    """3-edge-connectivity decomposition result."""
    components: List
    sigma: Dict[int, List[int]]
    cs: Dict[int, List[List[int]]]
    bridges: List[List[int]]
    cycles: Dict[int, List[List[int]]]
    mader: Dict[int, List[List[int]]]  # CS with extracted K32 core
    ecc1: Dict[int, List[int]]

    def is_equal_strict(self, other: 'Decomposition') -> bool:
        """Strict equality check (order-sensitive, catches algorithm changes)."""
        if not isinstance(other, Decomposition):
            raise ValueError("Decomposition can be compared only to another Decomposition")

        # Compare components (order matters)
        if self.components != other.components:
            return False

        # Compare sigma dictionaries (order matters)
        if self.sigma != other.sigma:
            return False

        # Compare CS paths dictionaries
        if self.cs != other.cs:
            return False

        # Compare bridges (order matters)
        if self.bridges != other.bridges:
            return False

        # Compare cycles dictionaries (directions matter)
        if self.cycles != other.cycles:
            return False

        # Compare mader dictionaries (K32 core extraction)
        if self.mader != other.mader:
            return False

        # Compare 1-edge-connected component mapping
        if self.ecc1 != other.ecc1:
            return False

        return True


def HypercubeEdges(n):
    """Generate edges of n-dimensional hypercube (vertices 1..2^n)."""
    if n <= 0:
        return []

    num_vertices = 1 << n  # 2^n
    edges = []

    for v in range(num_vertices):
        for bit in range(n):
            u = v ^ (1 << bit)  # flip one bit
            if v < u:  # avoid duplicates
                edges.append([v + 1, u + 1])  # 1-based indexing

    return edges


def build_adjacency_list(edges, num_vertices):
    """Build adjacency list from edges."""
    adj = [[] for _ in range(num_vertices + 1)]  # 0-indexed but we use 1..num_vertices
    for edge in edges:
        adj[edge[0]].append(edge[1])
    return adj


def to_adj_string(edges, num_vertices):
    """Convert edges to adjacency list string."""
    adj = build_adjacency_list(edges, num_vertices)

    lines = []
    for v in range(1, num_vertices + 1):
        neighbors = ' '.join(str(u) for u in adj[v])
        lines.append(f"{v}: {neighbors}")
    return '\n'.join(lines)


def build_adjacency(num_vertices, edges) -> Dict[int, List[int]]:
    """Build adjacency dict from undirected edges."""
    # Build adjacency dictionary using Python dict
    adjacency = {}
    for v in range(1, num_vertices + 1):
        adjacency[v] = []

    for edge in edges:
        # Add both directions for undirected edge
        adjacency[edge[0]].append(edge[1])
        adjacency[edge[1]].append(edge[0])

    return adjacency
