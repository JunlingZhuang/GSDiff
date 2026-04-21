"""Tests for the optional degree-rank structural feature (v2).

When ``config.model.use_degree_feature=True``, :class:`GRANv2` gains a
``self.degree_embedding`` table indexed by each node's current degree
rank (0 = highest-degree in the already-generated prefix). This is an
ordering-invariant structural signal that the model can use to learn
patterns like "the highest-degree node tends to be Living Room"
without relying on (brittle) generation-order position.

These tests verify:
  1. The embedding module is created iff the flag is on.
  2. Backward-compat: when the flag is absent, the module is not added
     (so old checkpoints still load cleanly).
  3. A full forward pass (sampling path) still runs end-to-end when
     the feature is enabled.
"""

import numpy as np
import torch
from easydict import EasyDict as edict

from model.gran_v2 import GRANv2


def _cfg(use_degree_feat=True):
    return edict({
        'device': 'cpu',
        'model': edict({
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
            'num_canonical_order': 1,
            'num_mix_component': 3,
            'num_attr_classes': 5,
            'use_gatv2': False,
            'use_degree_feature': use_degree_feat,
        })
    })


def test_degree_embedding_module_exists_when_enabled():
    """When config.model.use_degree_feature=True, the model must have a
    degree_embedding submodule sized to max_num_nodes."""
    model = GRANv2(_cfg(use_degree_feat=True))
    assert hasattr(model, 'degree_embedding')
    # The embedding table should be indexable up to max_num_nodes-1.
    assert model.degree_embedding.num_embeddings == 10


def test_degree_feature_off_by_default():
    """When the flag is absent or False, no degree_embedding is added
    (keeps backward-compat with existing checkpoints)."""
    cfg = _cfg(use_degree_feat=False)
    # Remove the flag entirely to simulate old config.
    del cfg.model.use_degree_feature
    model = GRANv2(cfg)
    assert not hasattr(model, 'degree_embedding')


def test_forward_still_works_with_degree_feature():
    """Full forward pass (sampling path) with the feature on."""
    cfg = _cfg(use_degree_feat=True)
    model = GRANv2(cfg)
    model.eval()
    out = model({
        'is_sampling': True,
        'batch_size': 2,
        'num_nodes_pmf': np.array([0.0] * 3 + [0.3, 0.4, 0.3] + [0.0] * 4,
                                  dtype=np.float32),
    })
    assert isinstance(out, tuple) and len(out) == 2
    A_list, attr_list = out
    assert len(A_list) == 2
