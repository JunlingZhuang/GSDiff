# tests/test_inference_modes.py
"""Tests for Task 7 inference APIs on GRANv2:

* Mode A — ``expand_graph``: continue autoregressive generation from a
  partial graph until a target size is reached.
* Mode B — ``predict_attr_with_edges``: given a partial graph and a
  user-specified connection pattern for the next node, return raw
  attribute logits for that new node.
* Sanity — unconditional generation still works unchanged.
"""
import numpy as np
import torch
from easydict import EasyDict as edict

from model.gran_v2 import GRANv2


def make_config(num_types=5):
    return edict({
        'device': 'cpu',
        'model': edict({
            'max_num_nodes': 20,
            'hidden_dim': 64,
            'embedding_dim': 64,
            'is_sym': True,
            'block_size': 1,
            'sample_stride': 1,
            'num_GNN_prop': 1,
            'num_GNN_layers': 2,
            'edge_weight': 1.0,
            'dimension_reduce': True,
            'has_attention': True,
            'num_canonical_order': 1,
            'num_mix_component': 5,
            'num_attr_classes': num_types,
            'use_gatv2': False,
        })
    })


def test_expand_graph_from_partial():
    """Mode A: partial graph -> generate N more nodes with edges and attrs."""
    config = make_config()
    model = GRANv2(config)
    model.eval()

    B = 1
    n_partial = 4
    # Chain graph as partial input
    partial_A = torch.zeros(B, n_partial, n_partial)
    for i in range(n_partial - 1):
        partial_A[:, i, i + 1] = 1.0
        partial_A[:, i + 1, i] = 1.0
    partial_attrs = torch.tensor([[0, 1, 1, 3]])  # e.g. Living, Bedroom, Bedroom, Kitchen

    A_new, attrs_new = model.expand_graph(
        partial_A=partial_A,
        partial_attrs=partial_attrs,
        num_target_nodes=6,
    )

    assert A_new.shape == (B, 6, 6)
    assert attrs_new.shape == (B, 6)
    # Partial structure preserved
    assert torch.allclose(
        A_new[:, :n_partial, :n_partial],
        partial_A[:, :n_partial, :n_partial])
    # Partial attrs preserved
    assert torch.equal(attrs_new[:, :n_partial], partial_attrs)


def test_predict_attr_with_fixed_edges():
    """Mode B: given partial graph + new node connection pattern, predict only attr."""
    config = make_config()
    model = GRANv2(config)
    model.eval()

    B = 1
    n_partial = 5
    partial_A = torch.zeros(B, n_partial, n_partial)
    for i in range(n_partial - 1):
        partial_A[:, i, i + 1] = 1.0
        partial_A[:, i + 1, i] = 1.0
    partial_attrs = torch.tensor([[0, 1, 1, 3, 2]])

    # User: new 6th node connects to nodes 0 and 2
    fixed_edges = torch.zeros(B, n_partial)
    fixed_edges[:, 0] = 1
    fixed_edges[:, 2] = 1

    attr_logits = model.predict_attr_with_edges(
        partial_A=partial_A,
        partial_attrs=partial_attrs,
        fixed_edges=fixed_edges,
    )

    assert attr_logits.shape == (B, 5)
    probs = torch.softmax(attr_logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_unconditional_still_works_after_task7():
    """Ensure Task 7 changes don't break unconditional generation."""
    config = make_config()
    model = GRANv2(config)
    model.eval()

    pmf = np.array([0.0] * 5 + [0.3, 0.4, 0.3] + [0.0] * 12)
    A_list, attr_list = model({
        'is_sampling': True,
        'batch_size': 2,
        'num_nodes_pmf': pmf,
    })
    assert len(A_list) == 2
    assert len(attr_list) == 2
