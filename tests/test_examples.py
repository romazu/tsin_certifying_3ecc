#!/usr/bin/env python
import pytest
from utils import build_adjacency
from ecc_cert import Solver
from examples import all_examples


def run_example(example):
    graph = build_adjacency(example.graph.num_vertices, example.graph.edges)
    algo = Solver(graph, example.graph.num_vertices)
    algo.certifying_3_edge_connectivity(example.start_vertex)
    decomp = algo.materialize_decomposition()
    return decomp.is_equal_strict(example.decomposition)


@pytest.mark.parametrize("example", all_examples, ids=lambda e: e.name)
def test_example(example):
    assert run_example(example)
