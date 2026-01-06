from dataclasses import dataclass
from typing import List

from utils import Decomposition


@dataclass
class Graph:
    edges: List[List[int]]
    num_vertices: int


@dataclass
class Example:
    name: str
    graph: Graph
    start_vertex: int
    decomposition: Decomposition


# Test graph: Ladder (10 vertices)
Ladder = Example(
    name='ladder',
    graph=Graph(
        edges=[
            [1, 2],
            [1, 3],
            [2, 4],

            [3, 4],
            [3, 5],
            [4, 6],

            [5, 6],
            [5, 7],
            [6, 8],

            [7, 8],
            [7, 9],
            [8, 10],

            [9, 10],
        ],
        num_vertices=10
    ),
    start_vertex=1,
    decomposition=Decomposition(
        components=[10, 9, 8, 5, 4, 2, 1],
        sigma={10: [10], 9: [9], 8: [8, 7], 5: [5, 6], 4: [4, 3], 2: [2], 1: [1]},
        cs={
            4: [[3, 4], [4, 3], [4, 3]],
            5: [[6, 5], [5, 6], [5, 6]],
            8: [[7, 8], [8, 7], [8, 7]]
        },
        bridges=[],
        cycles={
            8: [[10, 9]],
            5: [[8]],
            4: [[5]],
            1: [[4, 2]]
        },
        mader={
            4: [[3, 4], [3, 4], [3, 4]],
            5: [[6, 5], [6, 5], [6, 5]],
            8: [[7, 8], [7, 8], [7, 8]]
        }
    )
)

# Test graph: Shuriken (12 vertices, disconnected 3ECC core component)
Shuriken = Example(
    name='shuriken',
    graph=Graph(
        edges=[
            [5, 1],
            [5, 2],
            [6, 1],
            [6, 2],

            [7, 2],
            [7, 3],
            [8, 2],
            [8, 3],

            [9, 3],
            [9, 4],
            [10, 3],
            [10, 4],

            [11, 4],
            [11, 1],
            [12, 4],
            [12, 1],
        ],
        num_vertices=12
    ),
    start_vertex=1,
    decomposition=Decomposition(
        components=[6, 8, 10, 11, 12, 9, 7, 5, 1],
        sigma={6: [6], 8: [8], 10: [10], 11: [11], 12: [12], 9: [9], 7: [7], 5: [5], 1: [1, 2, 3, 4]},
        cs={
            1: [[1, 4, 3, 2, 1], [1, 4], [3, 4], [2, 3], [1, 2]]
        },
        bridges=[],
        cycles={
            1: [[6], [8], [10], [11], [9], [7], [5], [12]]
        },
        mader={
            1: [[4, 3, 2, 1], [4, 1], [4, 1], [3, 4], [2, 3], [1, 2]]
        }
    )
)

# Test graph: Tsin1 (34 vertices, complex graph)
Tsin1 = Example(
    name='tsin1',
    graph=Graph(
        edges=[
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 10],  # parallel
            [13, 10],  # parallel
            [12, 14],
            [14, 11],
            [14, 10],
            [9, 1],
            [9, 6],
            [8, 6],
            [8, 5],
            [7, 15],
            [15, 16],
            [16, 17],
            [16, 5],
            [17, 18],
            [18, 19],
            [19, 20],
            [20, 21],
            # [20, 21],  # TEST: this parallel is enough to merge two 3ecc
            [21, 22],  # parallel
            [22, 7],
            [22, 6],
            [22, 21],  # parallel
            [21, 23],
            [23, 24],
            [24, 7],
            [24, 5],
            [23, 7],
            [20, 25],
            [25, 26],
            [26, 18],
            [25, 18],
            [26, 19],
            [17, 27],
            [27, 16],
            [27, 15],
            [4, 28],
            [28, 29],
            [29, 4],
            [29, 30],
            [30, 31],
            [31, 29],
            [31, 4],
            [30, 28],
            [3, 32],
            [32, 33],
            [33, 2],
            [33, 1],
            [32, 34],
            [34, 1],
            [34, 2],
        ],
        num_vertices=34
    ),
    start_vertex=1,
    decomposition=Decomposition(
        components=[10, 18, 5, 4, 1],
        sigma={10: [10, 11, 12, 14, 13], 18: [18, 19, 20, 25, 26], 5: [5, 6, 7, 15, 16, 17, 27, 21, 22, 23, 24, 8, 9],
               4: [4, 28, 29, 30, 31], 1: [1, 2, 3, 32, 34, 33]},
        cs={
            1: [[1, 33, 32, 3, 2, 1], [1, 34, 32], [2, 34], [2, 33], [1, 3]],
            4: [[4, 31, 30, 29, 28, 4], [4, 29], [29, 31], [28, 30]],
            5: [[9, 8, 7, 6, 5], [5, 9], [5, 8], [6, 8], [6, 9], [5, 24, 23, 21, 17, 16, 15, 7], [7, 24], [7, 23],
                [6, 22, 21], [7, 22], [21, 22], [5, 16], [15, 27, 17], [16, 27]],
            10: [[10, 13, 12, 11, 10], [10, 13], [10, 14, 12], [11, 14]],
            18: [[20, 19, 18], [18, 20], [18, 26, 25, 20], [19, 26], [18, 25]]
        },
        bridges=[[9, 10]],
        cycles={
            5: [[18]],
            1: [[5, 4]]
        },
        mader={
            1: [[32, 3, 2, 1], [32, 33, 1], [32, 34, 1], [2, 34], [2, 33], [1, 3]],
            4: [[4, 31, 30, 29], [4, 28, 29], [4, 29], [29, 31], [28, 30]],
            5: [[8, 7, 6, 5], [8, 9, 5], [8, 5], [6, 8], [6, 9], [5, 24, 23, 21, 17, 16, 15, 7], [7, 24], [7, 23],
                [6, 22, 21], [7, 22], [21, 22], [5, 16], [15, 27, 17], [16, 27]],
            10: [[13, 12, 11, 10], [13, 10], [13, 10], [10, 14, 12], [11, 14]],
            18: [[20, 19, 18], [20, 18], [20, 25, 26, 18], [19, 26], [18, 25]]
        }
    )
)

# Modified Tsin1 with additional branches
Tsin1_mod = Example(
    name='tsin1_mod',
    graph=Graph(
        edges=[
            *Tsin1.graph.edges,  # original graph
            [3, 35],  # cut edge: v = 3, w = 35
            [35, 36],  # tree branch 35-36-37-38 of K_4 subgraph
            [36, 37],
            [37, 38],
            [37, 35],  # back edges
            [38, 36],
            [38, 35],  # anchor(w) back edge
            [35, 39],  # tree branch 35-39-40-41-42
            [39, 40],
            [40, 41],
            [41, 42],
            [40, 39],  # back edges
            [41, 39],
            [42, 39],
            [42, 35],
            [41, 43],  # tree branch 41-43-44-45, w_h = 44
            [43, 44],
            [44, 45],
            [45, 43],  # back edges
            [45, 35],  # anchor(w_h) back edge
            [44, 46],  # tree branch 44-46, wdd = 46

            [46, 47],  # demo
            [47, 43],  #
            [47, 44],  #

            [46, 43],  # back edge
            [46, 3],  # cut edge: back edge to the main tree
        ],
        num_vertices=47
    ),
    start_vertex=1,
    decomposition=Decomposition(
        components=[10, 18, 5, 4, 35, 1],  # new 35
        sigma={
            **Tsin1.decomposition.sigma,
            35: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
        },
        cs={
            **Tsin1.decomposition.cs,
            35: [
                [46, 44, 43, 41, 40, 39, 35],
                [35, 46],
                [35, 45, 44],
                [43, 45],
                [43, 47, 46],
                [44, 47],
                [43, 46],
                [35, 42, 41],
                [39, 42],
                [39, 41],
                [39, 40],
                [35, 38, 37, 36, 35],
                [35, 37],
                [36, 38],
            ]
        },
        bridges=Tsin1.decomposition.bridges,
        cycles={
            5: [[18]],
            1: [[5, 4], [35]]
        },
        mader={
            **Tsin1.decomposition.mader,
            35: [
                [44, 43, 41, 40, 39, 35],
                [44, 46, 35],
                [44, 45, 35],
                [43, 45],
                [43, 47, 46],
                [44, 47],
                [43, 46],
                [35, 42, 41],
                [39, 42],
                [39, 41],
                [39, 40],
                [35, 38, 37, 36, 35],
                [35, 37],
                [36, 38]
            ]
        }
    )
)

# Test graph: Tsin2 (14 vertices)
Tsin2 = Example(
    name='tsin2',
    graph=Graph(
        edges=[
            [1, 2],  # a
            [2, 3],  # b
            [3, 4],  # c
            [4, 1],  # d
            [4, 1],
            [3, 5],
            [5, 6],  # f
            [6, 7],  # g
            [7, 8],  # h
            [8, 5],  # i
            [6, 9],
            [9, 10],  # j
            [10, 11],  # k
            [10, 14],
            [11, 12],  # l
            [12, 13],  # m
            [13, 9],  # n
            [13, 10],
            [12, 9],
            [14, 2],  # o
        ],
        num_vertices=14
    ),
    start_vertex=1,
    decomposition=Decomposition(
        components=[8, 7, 11, 14, 9, 5, 2, 1],
        sigma={8: [8], 7: [7], 11: [11], 14: [14], 9: [9, 10, 12, 13], 5: [5, 6], 2: [2, 3], 1: [1, 4]},
        cs={
            1: [[1, 4, 1], [1, 4]],
            2: [[3, 2], [2, 3], [2, 3]],
            5: [[6, 5], [5, 6], [5, 6]],
            9: [[10, 9], [9, 10], [9, 13, 12, 10], [10, 13], [9, 12]]
        },
        bridges=[],
        cycles={
            9: [[11]],
            5: [[8, 7]],
            2: [[14, 9, 5]],
            1: [[2]]
        },
        mader={
            1: [[1, 4], [1, 4], [1, 4]],
            2: [[3, 2], [3, 2], [3, 2]],
            5: [[6, 5], [6, 5], [6, 5]],
            9: [[10, 9], [10, 9], [10, 12, 13, 9], [10, 13], [9, 12]]
        }
    )
)

# Modified Tsin2 with bridge and additional branches
Tsin2_mod = Example(
    name='tsin2_mod',
    graph=Graph(
        edges=[
            [1, 15],  # bridge
            [15, 16],  # right branch
            [16, 17],
            [17, 18],  # cut edge
            [18, 15],  # cut back-edge
            [17, 15],
            [16, 15],
            [15, 19],  # left branch
            [19, 20],
            [20, 21],  # cut edge
            [21, 15],  # cut back-edge
            [20, 15],
            [19, 15],
            # [15, 1],  # in the case we want a 2-cut instead of a bridge

            *Tsin2.graph.edges,
        ],
        num_vertices=21
    ),
    start_vertex=1,
    decomposition=Decomposition(
        components=[18, 21, 15] + Tsin2.decomposition.components,
        sigma={
            18: [18], 21: [21], 15: [15, 19, 20, 16, 17],
            **Tsin2.decomposition.sigma,
        },
        cs={
            15: [[15, 17, 16, 15], [15, 17], [15, 20, 19, 15], [15, 20], [15, 19], [15, 16]],
            **Tsin2.decomposition.cs,
        },
        bridges=[[1, 15]],
        cycles={
            15: [[21], [18]],
            **Tsin2.decomposition.cycles
        },
        mader={
            15: [[17, 16, 15], [17, 15], [17, 15], [15, 20, 19, 15], [15, 20], [15, 19], [15, 16]],
            **Tsin2.decomposition.mader,
        }
    )
)

# Test graph: Simple cycle (4 vertices)
SimpleCycle = Example(
    name='simple_cycle',
    graph=Graph(
        edges=[
            [1, 2],
            [2, 3],
            [3, 4],  # cut edge
            [4, 2],  # cut back-edge
            [3, 1],
            [2, 1],
        ],
        num_vertices=4
    ),
    start_vertex=1,
    decomposition=Decomposition(
        components=[4, 1],
        sigma={4: [4], 1: [1, 2, 3]},
        cs={
            1: [[1, 3, 2, 1], [1, 2], [2, 3]]
        },
        bridges=[],
        cycles={
            1: [[4]]
        },
        mader={
            1: [[1, 3, 2], [1, 2], [1, 2], [2, 3]]
        }
    )
)

# Test graph: Bchain at root (4 vertices)
BchainAtRoot = Example(
    name='bchain_at_root',
    graph=Graph(
        edges=[
            [1, 2],
            [2, 3],
            [3, 4],  # cut edge
            [4, 1],  # cut back-edge to root
            [3, 1],
            [2, 1],
        ],
        num_vertices=4
    ),
    start_vertex=1,
    decomposition=Decomposition(
        components=[4, 1],
        sigma={4: [4], 1: [1, 2, 3]},
        cs={
            1: [[1, 3, 2, 1], [1, 3], [1, 2]]
        },
        bridges=[],
        cycles={
            1: [[4]]
        },
        mader={
            1: [[3, 2, 1], [3, 1], [3, 1], [1, 2]]
        }
    )
)

all_examples = [
    Ladder,
    Shuriken,
    Tsin1,
    Tsin1_mod,
    Tsin2,
    Tsin2_mod,
    SimpleCycle,
    BchainAtRoot,
]

PositionsTsin1 = {
    1: (475, 629),
    2: (555, 550),
    3: (567, 439),
    4: (571, 346),
    5: (539, 255),
    6: (426, 284),
    7: (340, 353),
    8: (324, 456),
    9: (318, 583),
    10: (83, 585),
    11: (89, 467),
    12: (161, 363),
    13: (201, 469),
    14: (0, 417),
    15: (178, 298),
    16: (165, 205),
    17: (85, 113),
    18: (142, 41),
    19: (214, 0),
    20: (297, 0),
    21: (371, 0),
    22: (539, 85),
    23: (421, 76),
    24: (455, 135),
    25: (261, 94),
    26: (197, 109),
    27: (83, 226),
    28: (619, 181),
    29: (739, 157),
    30: (748, 237),
    31: (687, 313),
    32: (696, 508),
    33: (767, 660),
    34: (650, 718),
}

PositionsTsin2 = {
    1: (203, 91),
    2: (293, 93),
    3: (167, 224),
    4: (81, 213),
    5: (251, 233),
    6: (343, 231),
    7: (339, 313),
    8: (269, 305),
    9: (381, 146),
    10: (467, 146),
    11: (513, 237),
    12: (424, 301),
    13: (442, 214),
    14: (455, 44),
    15: (424, 50),
}
