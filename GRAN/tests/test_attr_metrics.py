"""Tests for attribute-aware evaluation metrics.

Standard MMD only tests structural statistics (degree, clustering,
spectral) and completely ignores node attrs. For bubble diagrams we need
metrics that test structure-semantics coupling:

  * ``endpoint_attr_pair_counts`` — for each edge, count the unordered
    (attr_u, attr_v) pair. Tells us "which attr combinations tend to be
    connected."
  * ``kl_divergence`` — KL(gen || ref) over sparse count dicts, with
    additive smoothing so disjoint supports don't blow up.
  * ``living_count_per_graph`` — for each graph, count how many nodes
    have a given attr class (e.g. Living). Answers "how many living
    rooms per generated floorplan."
  * ``per_class_degree_stats`` — for each attr class, aggregate node
    degrees. Tells us whether Living is a hub / Bathroom a leaf.
"""
import math
from collections import Counter

import networkx as nx
import pytest

from utils.attr_metrics import (
    endpoint_attr_pair_counts,
    kl_divergence,
    living_count_per_graph,
    per_class_degree_stats,
)


# --- endpoint_attr_pair_counts ------------------------------------------

def _triangle(attrs):
    """K3 with the given 3 attrs."""
    G = nx.complete_graph(3)
    return G, list(attrs)


def test_endpoint_pair_counts_simple_triangle():
    """A triangle with attrs [0, 1, 2] has edges (0,1), (0,2), (1,2)."""
    G, attrs = _triangle([0, 1, 2])
    counts = endpoint_attr_pair_counts([G], [attrs], num_classes=3)
    assert counts[(0, 1)] == 1
    assert counts[(0, 2)] == 1
    assert counts[(1, 2)] == 1
    assert sum(counts.values()) == 3


def test_endpoint_pair_counts_unordered():
    """Pairs are unordered — (0,1) and (1,0) collapse."""
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    counts = endpoint_attr_pair_counts([G], [[1, 0, 1]], num_classes=2)
    # edges: (0,1) attrs (1,0) -> (0,1); (1,2) attrs (0,1) -> (0,1)
    assert counts[(0, 1)] == 2
    assert (1, 0) not in counts


def test_endpoint_pair_counts_self_pair():
    """Edge between two nodes of same class -> (c, c) pair."""
    G = nx.Graph()
    G.add_edge(0, 1)
    counts = endpoint_attr_pair_counts([G], [[3, 3]], num_classes=7)
    assert counts[(3, 3)] == 1


def test_endpoint_pair_counts_multiple_graphs():
    """Counts accumulate across graphs."""
    G1 = nx.Graph(); G1.add_edge(0, 1)
    G2 = nx.Graph(); G2.add_edge(0, 1)
    counts = endpoint_attr_pair_counts(
        [G1, G2], [[0, 1], [0, 1]], num_classes=2)
    assert counts[(0, 1)] == 2


def test_endpoint_pair_counts_ignores_none_attrs():
    """Graphs with None attrs are skipped without crash."""
    G = nx.Graph(); G.add_edge(0, 1)
    counts = endpoint_attr_pair_counts([G], [None], num_classes=2)
    assert counts == {}


def test_endpoint_pair_counts_handles_short_attr_list():
    """If attr list shorter than node count, skip out-of-range nodes."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    counts = endpoint_attr_pair_counts([G], [[0, 1]], num_classes=2)
    # edge (0,1) OK; edge (1,2) skipped because attr[2] missing
    assert counts == {(0, 1): 1}


# --- kl_divergence ------------------------------------------------------

def test_kl_identical_dists_is_zero():
    counts = {(0, 1): 10, (1, 1): 5, (2, 3): 3}
    assert kl_divergence(counts, counts) == pytest.approx(0.0, abs=1e-6)


def test_kl_nonnegative():
    p = {(0, 0): 5, (0, 1): 3}
    q = {(0, 0): 3, (0, 1): 5}
    assert kl_divergence(p, q) > 0


def test_kl_handles_disjoint_support():
    """Keys only in gen (unseen in ref) should not cause inf."""
    p = {(0, 0): 10}
    q = {(1, 1): 10}
    result = kl_divergence(p, q)
    assert math.isfinite(result)
    assert result > 0


def test_kl_empty_p_is_zero():
    """If gen has no edges, KL should be 0 (no signal to compare)."""
    p = {}
    q = {(0, 0): 10}
    assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-6)


def test_kl_empty_q_does_not_crash():
    """If ref empty (edge case), smoothing still keeps result finite."""
    p = {(0, 0): 10}
    q = {}
    result = kl_divergence(p, q)
    assert math.isfinite(result)


# --- living_count_per_graph ---------------------------------------------

def test_living_count_basic():
    """3 graphs: 1 Living, 2 Living, 0 Living -> histogram {0:1, 1:1, 2:1}."""
    G = nx.Graph(); G.add_nodes_from([0, 1, 2])
    graphs = [G, G, G]
    attrs = [[0, 1, 2], [0, 0, 2], [1, 1, 2]]
    hist = living_count_per_graph(graphs, attrs, target_class=0)
    assert hist == Counter({1: 1, 2: 1, 0: 1})


def test_living_count_target_other_class():
    G = nx.Graph(); G.add_nodes_from([0, 1])
    hist = living_count_per_graph([G, G], [[1, 1], [1, 0]], target_class=1)
    assert hist == Counter({2: 1, 1: 1})


def test_living_count_skips_none():
    G = nx.Graph(); G.add_nodes_from([0, 1])
    hist = living_count_per_graph([G, G], [None, [0, 0]], target_class=0)
    assert hist == Counter({2: 1})


# --- per_class_degree_stats ---------------------------------------------

def test_per_class_degree_single_graph_triangle():
    """Triangle: all 3 nodes have degree 2. attrs [0,0,1]:
       class 0 has 2 nodes (deg 2); class 1 has 1 node (deg 2)."""
    G, attrs = _triangle([0, 0, 1])
    stats = per_class_degree_stats([G], [attrs], num_classes=2)
    assert stats[0]['count'] == 2
    assert stats[0]['mean'] == pytest.approx(2.0)
    assert stats[1]['count'] == 1
    assert stats[1]['mean'] == pytest.approx(2.0)


def test_per_class_degree_hub_vs_leaf():
    """Star K1,3: center deg 3, leaves deg 1.
       attrs = [Living=center, Bath, Bath, Bath]."""
    G = nx.star_graph(3)  # node 0 center, nodes 1-3 leaves
    attrs = [0, 2, 2, 2]
    stats = per_class_degree_stats([G], [attrs], num_classes=3)
    assert stats[0]['mean'] == pytest.approx(3.0)   # Living is hub
    assert stats[2]['mean'] == pytest.approx(1.0)   # Bath is leaf
    assert stats[1]['count'] == 0


def test_per_class_degree_missing_class_zero_count():
    G = nx.path_graph(2)
    stats = per_class_degree_stats([G], [[0, 0]], num_classes=7)
    assert stats[0]['count'] == 2
    for c in range(1, 7):
        assert stats[c]['count'] == 0
        assert stats[c]['mean'] is None


def test_per_class_degree_skips_none_attrs():
    G = nx.path_graph(2)
    stats = per_class_degree_stats(
        [G, G], [None, [0, 1]], num_classes=2)
    assert stats[0]['count'] == 1
    assert stats[1]['count'] == 1
