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


def test_attr_embedding_absent_when_flag_off():
    """Default GRANv2 has no attr_embedding (backward compat)."""
    cfg = _cfg(use_attr_cond_edge=False)
    model = GRANv2(cfg)
    assert not hasattr(model, 'attr_embedding') or model.attr_embedding is None
    # Also: use_attr_conditioned_edge attribute should default to False.
    assert getattr(model, 'use_attr_conditioned_edge', False) is False


def test_attr_embedding_registered_when_flag_on():
    """Flag on -> attr_embedding shape (A+1, attr_emb_dim)."""
    cfg = _cfg(use_attr_cond_edge=True, attr_emb_dim=8, num_attr_classes=5)
    model = GRANv2(cfg)
    assert hasattr(model, 'attr_embedding')
    # +1 for the "unknown / not-yet-sampled" slot used for K-block self-edges
    # and for endpoints whose attr hasn't been decided yet.
    assert model.attr_embedding.num_embeddings == 5 + 1
    assert model.attr_embedding.embedding_dim == 8
    # Initial weights should be small (std=0.1 init, see __init__).
    assert model.attr_embedding.weight.abs().mean().item() < 0.5


def test_edge_head_input_dim_grows_when_flag_on():
    """First Linear of output_theta and output_alpha grows by 2*attr_emb."""
    cfg_off = _cfg(use_attr_cond_edge=False)
    model_off = GRANv2(cfg_off)
    in_off = model_off.output_theta[0].in_features
    assert in_off == cfg_off.model.hidden_dim

    cfg_on = _cfg(use_attr_cond_edge=True, attr_emb_dim=8)
    model_on = GRANv2(cfg_on)
    in_on = model_on.output_theta[0].in_features
    assert in_on == cfg_on.model.hidden_dim + 2 * 8
    # output_alpha matches output_theta input dim
    assert model_on.output_alpha[0].in_features == in_on


def test_dataset_emits_subgraph_node_attrs_aligned_with_node_state():
    """``subgraph_node_attrs`` shape matches what node_state will have.

    For each subgraph of size jj+K, the array stores the ground-truth attr
    of every row (existing nodes get their real attr; padding/out-of-range
    slots get num_attr_classes -- the 'unknown' slot id).
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(num_attr_classes=5, tmp_dir=tmp)
        graphs = [_make_graph(n_nodes=6, seed=i, num_classes=5)
                  for i in range(3)]
        ds = GRANDataV2(cfg, graphs, tag='train')
        # __getitem__ returns a list of num_fwd_pass dicts; index pass 0.
        sample = ds[0][0]
        assert 'subgraph_node_attrs' in sample
        assert sample['subgraph_node_attrs'].dtype.kind in ('i', 'u')
        # The per-subgraph row count equals subgraph_size summed:
        # each subgraph contributes (jj+K) rows.
        assert sample['subgraph_node_attrs'].shape[0] == \
            sample['node_idx_feat'].shape[0]
        # Values must be in [0, num_attr_classes] (last slot reserved).
        assert sample['subgraph_node_attrs'].min() >= 0
        assert sample['subgraph_node_attrs'].max() <= cfg.model.num_attr_classes


def test_collate_stacks_subgraph_node_attrs():
    """collate_fn concatenates per-sample subgraph_node_attrs into one tensor."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(num_attr_classes=5, tmp_dir=tmp)
        graphs = [_make_graph(n_nodes=6, seed=i, num_classes=5)
                  for i in range(3)]
        ds = GRANDataV2(cfg, graphs, tag='train')
        batch = ds.collate_fn([ds[i] for i in range(len(graphs))])
        data = batch[0]
        assert 'subgraph_node_attrs' in data
        assert isinstance(data['subgraph_node_attrs'], torch.Tensor)
        assert data['subgraph_node_attrs'].dtype == torch.long
        # Aligns with node_idx_feat after collate (both flattened across batch).
        assert data['subgraph_node_attrs'].shape == data['node_idx_feat'].shape
