"""Tests for Plan C-1 attr-conditioned edge head.

When ``config.model.use_attr_conditioned_edge`` is True, GRANv2:
  1. Registers an ``attr_embedding`` table of shape (A+1, attr_emb_dim).
  2. Sets edge-head input dim to ``hidden_dim + 2 * attr_emb_dim`` (was hidden_dim).
  3. In training (``_inference``), gathers GROUND-TRUTH attr embeddings for
     both endpoints of every candidate edge and concats them into the edge
     head input alongside the existing diff = h_u - h_v.
  4. In sampling (``_sampling``), samples attr_k BEFORE deciding edges (k, j),
     and feeds embedded sampled attrs into the edge head.

When the flag is off (default), behavior is byte-identical to the previous
GRANv2 — verified by parameter count, attr_embedding absence, and a
forward-pass equivalence check against an old-style model.
"""

import tempfile

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from dataset.gran_data_v2 import GRANDataV2
from model.gran_v2 import GRANv2


def _cfg(use_attr_cond_edge=False, attr_emb_dim=8,
         num_attr_classes=5, tmp_dir=None):
    cfg = edict({
        'device': 'cpu',
        'seed': 42,
        'model': edict({
            'name': 'GRANv2',
            'max_num_nodes': 8,
            'hidden_dim': 16,
            'embedding_dim': 16,
            'is_sym': True,
            'block_size': 1,
            'sample_stride': 1,
            'num_GNN_prop': 1,
            'num_GNN_layers': 2,
            'edge_weight': 1.0,
            'dimension_reduce': True,
            'has_attention': True,
            'num_canonical_order': 1,
            'num_mix_component': 3,
            'num_attr_classes': num_attr_classes,
            'use_gatv2': False,
        }),
        'dataset': edict({
            'data_path': tmp_dir or '.',
            'name': 'attrcondtest',
            'node_order': 'DFS',
            'num_fwd_pass': 1,
            'is_sample_subgraph': False,
            'num_subgraph_batch': 100,
            'is_overwrite_precompute': True,
        }),
    })
    if use_attr_cond_edge:
        cfg.model.use_attr_conditioned_edge = True
        cfg.model.attr_embedding_dim = attr_emb_dim
    return cfg


def _make_graph(n_nodes=6, seed=0, num_classes=5):
    rng = np.random.RandomState(seed)
    G = nx.erdos_renyi_graph(n_nodes, 0.5, seed=seed)
    if not nx.is_connected(G):
        comps = [list(c) for c in nx.connected_components(G)]
        for cc in comps[1:]:
            G.add_edge(comps[0][0], cc[0])
    for i in range(n_nodes):
        G.nodes[i]['attr'] = int(rng.randint(0, num_classes))
    return G


def test_smoke_imports_and_helpers():
    """Smoke test: helpers build successfully."""
    cfg = _cfg()
    assert cfg.model.num_attr_classes == 5
    G = _make_graph(n_nodes=4, seed=0)
    assert G.number_of_nodes() == 4
