"""Tests for class-weighted attribute cross-entropy (v2, plan A).

When ``config.model.attr_class_weight`` is set, :class:`GRANv2` exposes a
buffer ``self.attr_class_weights`` of shape ``(num_attr_classes,)`` which
is passed as the ``weight=`` argument to ``F.cross_entropy`` every forward
pass.

Three modes:
  * ``None`` / unset     -> uniform weights (behaves like plain CE)
  * ``list`` of length A -> explicit per-class weights
  * ``'auto'``           -> uniform placeholder, runner computes
                            inverse-frequency weights from training graphs
                            and calls ``model.set_attr_class_weights(w)``

These tests verify:
  1. Default (unset) -> buffer is uniform ones.
  2. Explicit list  -> buffer copies the list verbatim.
  3. ``'auto'``      -> buffer initialised to ones (runner fills later).
  4. Setter replaces buffer in place, keeps device/dtype.
  5. Invalid list length raises.
  6. Training forward actually uses the weights (non-uniform weights
     produce a different attr_loss than uniform weights).
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


def _cfg(attr_class_weight=None, num_attr_classes=5, tmp_dir=None):
    """Build an EasyDict config for a tiny GRANv2."""
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
            'name': 'attrweighttest',
            'node_order': 'DFS',
            'num_fwd_pass': 1,
            'is_sample_subgraph': False,
            'num_subgraph_batch': 100,
            'is_overwrite_precompute': True,
        }),
    })
    if attr_class_weight is not None:
        cfg.model.attr_class_weight = attr_class_weight
    return cfg


def test_default_weights_are_uniform_ones():
    """Unset attr_class_weight -> all-ones buffer (no-op under CE)."""
    cfg = _cfg(attr_class_weight=None)
    model = GRANv2(cfg)
    assert hasattr(model, 'attr_class_weights')
    expected = torch.ones(cfg.model.num_attr_classes)
    assert torch.allclose(model.attr_class_weights, expected)


def test_explicit_list_weights_copied_verbatim():
    """List-of-length-A -> weights copied into the buffer."""
    weights = [0.5, 1.0, 2.0, 1.5, 0.8]
    cfg = _cfg(attr_class_weight=weights)
    model = GRANv2(cfg)
    expected = torch.tensor(weights, dtype=torch.float32)
    assert torch.allclose(model.attr_class_weights, expected)


def test_auto_mode_initial_weights_uniform():
    """``'auto'`` starts uniform; runner is expected to overwrite via setter."""
    cfg = _cfg(attr_class_weight='auto')
    model = GRANv2(cfg)
    # At construction time we don't know the frequencies yet.
    assert torch.allclose(
        model.attr_class_weights,
        torch.ones(cfg.model.num_attr_classes))
    # Mode attribute is retained so the runner can test for it.
    assert model.attr_class_weight_mode == 'auto'


def test_setter_overwrites_buffer_in_place():
    """``set_attr_class_weights`` must edit the buffer in place so that
    DataParallel replicas pick up the change via broadcast_buffers."""
    cfg = _cfg(attr_class_weight='auto')
    model = GRANv2(cfg)
    buf_ptr = model.attr_class_weights.data_ptr()

    new_w = torch.tensor([0.2, 1.0, 1.0, 3.0, 0.5])
    model.set_attr_class_weights(new_w)

    # Same underlying storage (in-place copy_, not reassignment).
    assert model.attr_class_weights.data_ptr() == buf_ptr
    assert torch.allclose(model.attr_class_weights, new_w)


def test_setter_rejects_wrong_shape():
    cfg = _cfg(attr_class_weight='auto')
    model = GRANv2(cfg)
    with pytest.raises(ValueError, match='shape'):
        model.set_attr_class_weights(torch.ones(3))  # wrong length


def test_invalid_list_length_raises_at_init():
    """List whose length != num_attr_classes must fail fast."""
    cfg = _cfg(attr_class_weight=[1.0, 2.0])  # length 2, but A=5
    with pytest.raises(ValueError, match='length'):
        GRANv2(cfg)


def test_invalid_mode_raises_at_init():
    cfg = _cfg(attr_class_weight='magic')  # unknown string
    with pytest.raises(ValueError, match='Invalid'):
        GRANv2(cfg)


def _make_graph(n_nodes=6, seed=0, num_classes=5):
    """Small connected ER graph with random integer node attrs."""
    rng = np.random.RandomState(seed)
    G = nx.erdos_renyi_graph(n_nodes, 0.5, seed=seed)
    if not nx.is_connected(G):
        comps = [list(c) for c in nx.connected_components(G)]
        for cc in comps[1:]:
            G.add_edge(comps[0][0], cc[0])
    for i in range(n_nodes):
        G.nodes[i]['attr'] = int(rng.randint(0, num_classes))
    return G


def test_weights_actually_affect_attr_loss():
    """Non-uniform weights must produce a different attr_loss than uniform.

    We build one model with uniform weights and another with the same
    random init but non-uniform weights, then run both forward passes on
    the same batch and check the attr losses differ.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Build two identical models, one plain, one with heavy weights on
        # class 0. We copy state_dict from the first into the second so the
        # ONLY difference is the attr_class_weights buffer.
        cfg_uniform = _cfg(attr_class_weight=None, tmp_dir=tmp)
        model_uniform = GRANv2(cfg_uniform)

        cfg_weighted = _cfg(
            attr_class_weight=[5.0, 1.0, 1.0, 1.0, 1.0], tmp_dir=tmp)
        model_weighted = GRANv2(cfg_weighted)
        # Copy all params + buffers EXCEPT attr_class_weights.
        src_state = model_uniform.state_dict()
        dst_state = model_weighted.state_dict()
        for k, v in src_state.items():
            if k != 'attr_class_weights':
                dst_state[k].copy_(v)

        # Build a real batch so packing matches runtime conventions.
        graphs = [_make_graph(n_nodes=6, seed=i, num_classes=5)
                  for i in range(3)]
        ds = GRANDataV2(cfg_uniform, graphs, tag='train')
        batch = ds.collate_fn([ds[i] for i in range(len(graphs))])
        data = batch[0]

        input_dict = {
            'is_sampling': False,
            'adj': data['adj'],
            'edges': data['edges'],
            'node_idx_gnn': data['node_idx_gnn'],
            'node_idx_feat': data['node_idx_feat'],
            'att_idx': data['att_idx'],
            'label': data['label'],
            'subgraph_idx': data['subgraph_idx'],
            'subgraph_idx_base': data['subgraph_idx_base'],
            'node_attrs': data['node_attrs'],
        }

        model_uniform.eval()
        model_weighted.eval()
        with torch.no_grad():
            _, loss_uniform = model_uniform(input_dict)
            _, loss_weighted = model_weighted(input_dict)

        # Different weights -> different attr loss (assuming at least one
        # class-0 node is in the batch, which the seed-based fixture makes
        # highly likely). If this flakes, raise the seed sweep.
        assert not torch.isclose(loss_uniform, loss_weighted), (
            "weighted CE produced identical loss to uniform CE — weights "
            "may not be threading through F.cross_entropy correctly")
