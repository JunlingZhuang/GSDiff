"""Unit tests for :class:`dataset.gran_data_v2.GRANDataV2`."""

import os
import pickle
import tempfile

import networkx as nx
import numpy as np
import torch
from easydict import EasyDict as edict

from dataset.gran_data_v2 import GRANDataV2


def _make_config(tmp_dir):
    return edict({
        'seed': 42,
        'model': edict({
            'name': 'GRANDataV2Test',
            'max_num_nodes': 10,
            'block_size': 1,
            'sample_stride': 1,
            'num_canonical_order': 1,
        }),
        'dataset': edict({
            'data_path': tmp_dir,
            'name': 'testset',
            'node_order': 'DFS',
            'num_fwd_pass': 1,
            'is_sample_subgraph': False,
            'num_subgraph_batch': 100,
            'is_overwrite_precompute': True,
        }),
    })


def _make_graph_with_attrs(n_nodes=5, seed=0):
    rng = np.random.RandomState(seed)
    G = nx.erdos_renyi_graph(n_nodes, 0.5, seed=seed)
    # Ensure connected: stitch all CCs together by adding an edge
    # from the first CC to one node of each other CC.
    if not nx.is_connected(G):
        components = [list(c) for c in nx.connected_components(G)]
        anchor = components[0][0]
        for cc in components[1:]:
            G.add_edge(anchor, cc[0])
    for i in range(n_nodes):
        G.nodes[i]['attr'] = int(rng.randint(0, 5))
    return G


def test_getitem_returns_node_attrs():
    """__getitem__ output dict should contain 'node_attrs' key."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(tmp)
        graphs = [_make_graph_with_attrs(n_nodes=5, seed=i) for i in range(3)]
        ds = GRANDataV2(config, graphs, tag='train')
        data_batch = ds[0]
        assert 'node_attrs' in data_batch[0]
        # Shape: (C, N_max)
        C = config.model.num_canonical_order
        N_max = config.model.max_num_nodes
        assert data_batch[0]['node_attrs'].shape == (C, N_max)


def test_attrs_match_original_graph():
    """Node attrs in output should match the original graph's 'attr' values
    (after canonical reordering)."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(tmp)
        G = _make_graph_with_attrs(n_nodes=5, seed=0)
        original_attrs = set(G.nodes[n]['attr'] for n in G.nodes())
        ds = GRANDataV2(config, [G], tag='train')
        data_batch = ds[0]
        # Only first n_nodes are meaningful (rest is padding)
        n_real = G.number_of_nodes()
        attrs_out = data_batch[0]['node_attrs'][0, :n_real].tolist()
        assert set(attrs_out) == original_attrs


def test_missing_attrs_default_to_zero():
    """If a graph has no 'attr' node attribute, should default to 0."""
    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(tmp)
        G = nx.path_graph(5)  # No 'attr' set
        ds = GRANDataV2(config, [G], tag='train')
        data_batch = ds[0]
        attrs = data_batch[0]['node_attrs'][0, :5].tolist()
        assert all(a == 0 for a in attrs)
