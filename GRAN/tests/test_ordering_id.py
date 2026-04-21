"""Tests for the optional ordering-id conditioning signal (v2, Path 1).

When ``config.model.use_ordering_id=True``, :class:`GRANv2` gains a
``self.ordering_embedding`` table indexed by canonical-ordering id
(0..C-1). The embedding is added to the initial node features before
GNN propagation, so the model can tell which of its ``C`` canonical
orderings a given training subgraph was produced from.

Motivation: with ``num_canonical_order > 1`` (multi-ordering training),
the same graph is seen under multiple orderings (DFS / BFS / k-core),
so the same "prefix" can predict different "next nodes" depending on
the ordering. Without an ordering-id signal, the model learns a mixture
distribution that does not match any single ordering at sampling time.
With this embedding, the model can condition its edge distribution on
the ordering explicitly, and we pin ``ordering_id=0`` at inference for
consistency.

These tests verify:
  1. The embedding module is created iff the flag is on, with the
     right ``num_embeddings = num_canonical_order``.
  2. Backward-compat: when the flag is absent, the module is not added
     (so old checkpoints still load cleanly).
  3. A full forward pass (training + sampling paths) still runs
     end-to-end when the feature is enabled with multi-ordering.
"""

import tempfile

import networkx as nx
import numpy as np
import torch
from easydict import EasyDict as edict

from dataset.gran_data_v2 import GRANDataV2
from model.gran_v2 import GRANv2


def _cfg(use_ordering_id=True, num_canonical_order=3, tmp_dir=None):
    """Build an EasyDict config for a small GRANv2 with multi-ordering.

    ``tmp_dir`` is only required by the training-path test because it
    builds a real :class:`GRANDataV2` shard on disk to avoid hand-rolling
    the complex (node_idx_feat, subgraph_idx, ...) packing format.
    """
    return edict({
        'device': 'cpu',
        'seed': 42,
        'model': edict({
            'name': 'GRANv2',
            'max_num_nodes': 10,
            'hidden_dim': 32,
            'embedding_dim': 32,
            'is_sym': True,
            'block_size': 1,
            'sample_stride': 1,
            'num_GNN_prop': 1,
            'num_GNN_layers': 2,
            'edge_weight': 1.0,
            'dimension_reduce': True,
            'has_attention': True,
            'num_canonical_order': num_canonical_order,
            'num_mix_component': 3,
            'num_attr_classes': 5,
            'use_gatv2': False,
            'use_degree_feature': False,
            'use_ordering_id': use_ordering_id,
        }),
        'dataset': edict({
            'data_path': tmp_dir or '.',
            'name': 'orderingidtest',
            'node_order': 'DFS+BFS+k_core',
            'num_fwd_pass': 1,
            'is_sample_subgraph': False,
            'num_subgraph_batch': 100,
            'is_overwrite_precompute': True,
        }),
    })


def test_ordering_embedding_exists():
    """When use_ordering_id=True, model must have an ordering_embedding
    submodule sized to num_canonical_order."""
    cfg = _cfg(use_ordering_id=True, num_canonical_order=3)
    model = GRANv2(cfg)
    assert hasattr(model, 'ordering_embedding')
    # One embedding row per canonical ordering (C=3 here).
    assert model.ordering_embedding.num_embeddings == 3


def test_ordering_embedding_off_by_default():
    """When the flag is absent or False, no ordering_embedding is added
    (keeps backward-compat with existing checkpoints)."""
    cfg = _cfg(use_ordering_id=False, num_canonical_order=3)
    # Remove the flag entirely to simulate an old config.
    del cfg.model.use_ordering_id
    model = GRANv2(cfg)
    assert not hasattr(model, 'ordering_embedding')


def _make_graph(n_nodes=6, seed=0):
    rng = np.random.RandomState(seed)
    G = nx.erdos_renyi_graph(n_nodes, 0.5, seed=seed)
    if not nx.is_connected(G):
        comps = [list(c) for c in nx.connected_components(G)]
        for cc in comps[1:]:
            G.add_edge(comps[0][0], cc[0])
    for i in range(n_nodes):
        G.nodes[i]['attr'] = int(rng.randint(0, 5))
    return G


def test_forward_with_ordering_id():
    """Full forward pass (training + sampling) with ordering-id ON and
    num_canonical_order=3."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(use_ordering_id=True, num_canonical_order=3, tmp_dir=tmp)
        model = GRANv2(cfg)

        # ---- training path --------------------------------------------
        # Build a real GRANDataV2 batch so node_idx_feat / subgraph_idx
        # packing matches exactly what the runtime produces.
        graphs = [_make_graph(n_nodes=6, seed=i) for i in range(2)]
        ds = GRANDataV2(cfg, graphs, tag='train')
        batch = ds.collate_fn([ds[i] for i in range(len(graphs))])
        data = batch[0]

        out = model({
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
        })
        assert isinstance(out, tuple) and len(out) == 2
        edge_loss, attr_loss = out
        assert torch.is_tensor(edge_loss)
        assert torch.is_tensor(attr_loss)
        # Losses should be finite (ordering-id path didn't NaN out).
        assert torch.isfinite(edge_loss).item()
        assert torch.isfinite(attr_loss).item()

        # ---- sampling path --------------------------------------------
        # Fixed ordering_id=0 (first canonical ordering = DFS) during
        # generation. Caller doesn't need to pass anything extra.
        model.eval()
        A_list, attr_list = model({
            'is_sampling': True,
            'batch_size': 2,
            'num_nodes_pmf': np.array(
                [0.0] * 3 + [0.3, 0.4, 0.3] + [0.0] * 4,
                dtype=np.float32),
        })
        assert len(A_list) == 2
        assert len(attr_list) == 2
