# tests/test_partial_graph.py
import numpy as np
import torch
from easydict import EasyDict as edict
from model.gran_v2 import GRANv2

def make_config():
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
            'num_attr_classes': 5,
            'use_gatv2': False,
        })
    })

def test_partial_graph_completion():
    """Given a partial adjacency, generation should start from after the partial graph."""
    config = make_config()
    model = GRANv2(config)
    model.eval()

    B = 2
    n_partial = 5
    # Create a small chain graph as partial input
    partial_A = torch.zeros(B, n_partial, n_partial)
    for i in range(n_partial - 1):
        partial_A[:, i, i + 1] = 1.0
        partial_A[:, i + 1, i] = 1.0
    partial_attrs = torch.randint(0, 5, (B, n_partial))

    input_dict = {
        'is_sampling': True,
        'batch_size': B,
        'num_nodes_pmf': np.array([0.0] * 8 + [0.5, 0.5] + [0.0] * 10),
        'partial_A': partial_A,
        'partial_attrs': partial_attrs,
        'start_idx': n_partial,
    }
    A_list, attr_list = model(input_dict)

    for A in A_list:
        n = A.shape[0]
        assert n >= n_partial, "Generated graph should be at least as large as partial"
        # Partial structure should be preserved in the lower-left triangle
        partial_sub = A[:n_partial, :n_partial]
        # Check the chain edges exist (allowing for symmetrization)
        for i in range(n_partial - 1):
            assert partial_sub[i, i+1] == 1.0 or partial_sub[i+1, i] == 1.0, \
                f"Partial edge ({i},{i+1}) not preserved"


def test_attribute_only_prediction():
    """Given a complete graph, predict attributes without generating new nodes.
    Set start_idx = num_nodes so no new edges are generated."""
    config = make_config()
    model = GRANv2(config)
    model.eval()

    B = 1
    N = 8
    # Complete graph (fully connected)
    A = torch.ones(B, N, N) - torch.eye(N).unsqueeze(0)

    input_dict = {
        'is_sampling': True,
        'batch_size': B,
        'num_nodes_pmf': np.array([0.0] * (N - 1) + [1.0] + [0.0] * (20 - N)),
        'partial_A': A,
        'start_idx': N,
    }
    A_list, attr_list = model(input_dict)
    assert len(attr_list) == 1
    # Attributes should be predicted for all nodes
    assert attr_list[0].shape[0] >= N


def test_unconditional_still_works():
    """Without partial graph, GRANv2 should work like original GRAN."""
    config = make_config()
    model = GRANv2(config)
    model.eval()

    input_dict = {
        'is_sampling': True,
        'batch_size': 3,
        'num_nodes_pmf': np.array([0.0] * 4 + [0.3, 0.4, 0.3] + [0.0] * 13),
    }
    A_list, attr_list = model(input_dict)
    assert len(A_list) == 3
    assert len(attr_list) == 3
